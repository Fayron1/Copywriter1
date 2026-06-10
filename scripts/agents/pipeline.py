"""
Pipeline — Оркестратор мультиагентной генерации статей.

Управляет полным циклом:
  Topic → Brain → Fact-Finder → Scout → Engineer → Heart
  → Sheriff ↔ Heart (до 3 итераций)
  → Mirror ↔ Heart (до 2 итераций)
  → Booster → Artist → Финальная статья

Использование:
    from agents.pipeline import Pipeline
    pipe = Pipeline(openai_api_key="sk-...")
    result = pipe.run(topic="Как открыть ООО в 2026", article_type="case_study")
"""
import json
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .registry import get_agent, AGENTS
from .prompts import get_system_prompt
from .rag import query_knowledge, format_rag_context
from .patterns import PATTERNS

logger = logging.getLogger("agents.pipeline")


def _extract_image_url(data: Any) -> Optional[str]:
    """Универсальный извлекатель URL из ответа API картинок."""
    if not data:
        return None
    if isinstance(data, str) and (data.startswith("http://") or data.startswith("https://")):
        return data
    if isinstance(data, list):
        for item in data:
            url = _extract_image_url(item)
            if url:
                return url
    if isinstance(data, dict):
        # Ищем стандартные ключи
        for key in ["url", "URL", "image", "image_url", "uri", "link"]:
            val = data.get(key)
            if val and isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                return val
            if val and isinstance(val, dict):
                url = _extract_image_url(val)
                if url:
                    return url
        # Ищем стандартные контейнеры
        for key in ["data", "images", "output", "results", "response"]:
            val = data.get(key)
            if val:
                url = _extract_image_url(val)
                if url:
                    return url
        # Рекурсивный обход всех остальных полей
        for key, val in data.items():
            if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                return val
            elif isinstance(val, (dict, list)):
                url = _extract_image_url(val)
                if url:
                    return url
    return None


def _extract_b64_data(data: Any) -> Optional[str]:
    """Универсальный извлекатель Base64 данных изображения из ответа API."""
    if not data:
        return None
    if isinstance(data, str):
        # Проверяем, похоже ли на Base64
        if len(data) > 100 and not data.startswith("http") and " " not in data:
            return data
        # Data URL: "data:image/png;base64,iVBORw0KGgo..."
        if data.startswith("data:image/") and ";base64," in data:
            try:
                return data.split(";base64,")[1]
            except IndexError:
                pass
    if isinstance(data, list):
        for item in data:
            b64 = _extract_b64_data(item)
            if b64:
                return b64
    if isinstance(data, dict):
        # Ищем стандартные ключи
        for key in ["b64_json", "b64", "base64", "image_data", "image", "content"]:
            val = data.get(key)
            if val and isinstance(val, str):
                if not (val.startswith("http://") or val.startswith("https://")):
                    b64 = _extract_b64_data(val)
                    if b64:
                        return b64
        # Ищем в контейнерах
        for key in ["data", "images", "output", "results", "response"]:
            val = data.get(key)
            if val:
                b64 = _extract_b64_data(val)
                if b64:
                    return b64
        # Рекурсивный обход
        for key, val in data.items():
            if isinstance(val, (dict, list)):
                b64 = _extract_b64_data(val)
                if b64:
                    return b64
    return None


def _extract_meta_from_text(text: str) -> Dict[str, Any]:
    """Резервный извлекатель метаданных из сырого текста с помощью регулярных выражений."""
    meta = {}
    if not text or not isinstance(text, str):
        return meta

    import re
    # 0. Попытка вытащить данные, если текст содержит псевдо-JSON ключи (когда сломался JSON от Booster)
    for key in ["title", "description"]:
        if key not in meta or not meta[key]:
            # Паттерн для поиска строкового значения в кавычках
            pat = r'"' + key + r'"\s*:\s*"(.*?)"'
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if not m:
                pat = r"'" + key + r"'\s*:\s*'(.*?)'"
                m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if not m:
                pat = r"'" + key + r"'\s*:\s*\"(.*?)\""
                m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if not m:
                pat = r"\"" + key + r"\"\s*:\s*'(.*?)'"
                m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m and m.group(1).strip():
                val = m.group(1).strip()
                val = val.replace('\\"', '"').replace("\\'", "'")
                meta[key] = val

    if "keywords" not in meta or not meta["keywords"]:
        kw_json_m = re.search(r'"keywords"\s*:\s*\[(.*?)\]', text, re.IGNORECASE | re.DOTALL)
        if not kw_json_m:
            kw_json_m = re.search(r"'keywords'\s*:\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
        if kw_json_m:
            kw_str = kw_json_m.group(1).strip()
            parts = re.findall(r'["\'](.*?)["\']', kw_str)
            if parts:
                meta["keywords"] = [p.strip() for p in parts if p.strip()]

    # 1. Попытка распарсить YAML frontmatter (между первыми --- и ---)
    fm_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL | re.MULTILINE)
    if fm_match:
        fm_text = fm_match.group(1)
        title_m = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', fm_text, re.MULTILINE | re.IGNORECASE)
        if title_m and not meta.get("title"):
            meta["title"] = title_m.group(1)
        desc_m = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', fm_text, re.MULTILINE | re.IGNORECASE)
        if desc_m and not meta.get("description"):
            meta["description"] = desc_m.group(1)
        kw_m = re.search(r'^keywords:\s*(.*?)\s*$', fm_text, re.MULTILINE | re.IGNORECASE)
        if kw_m and not meta.get("keywords"):
            kw_val = kw_m.group(1).strip()
            if kw_val.startswith("[") and kw_val.endswith("]"):
                try:
                    meta["keywords"] = json.loads(kw_val)
                except Exception:
                    pass

    # 2. Ищем явные маркеры в тексте
    if "title" not in meta or not meta["title"]:
        title_patterns = [
            r'(?:title|заголовок|meta title)\s*:\s*["\']?(.*?)["\']?\s*(?:\n|$)',
            r'\*\*(?:title|заголовок|meta title)\*\*:\s*(.*?)(?:\n|$)',
            r'^#\s*(.*?)(?:\n|$)'
        ]
        for pat in title_patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m and m.group(1).strip():
                candidate = m.group(1).strip().strip('"').strip("'")
                if len(candidate) > 5 and len(candidate) < 150:
                    meta["title"] = candidate
                    break

    if "description" not in meta or not meta["description"]:
        desc_patterns = [
            r'(?:description|описание|meta description)\s*:\s*["\']?(.*?)["\']?\s*(?:\n|$)',
            r'\*\*(?:description|описание|meta description)\*\*:\s*(.*?)(?:\n|$)'
        ]
        for pat in desc_patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m and m.group(1).strip():
                candidate = m.group(1).strip().strip('"').strip("'")
                if len(candidate) > 10:
                    meta["description"] = candidate
                    break

    if "keywords" not in meta or not meta["keywords"]:
        kw_patterns = [
            r'(?:keywords|ключевые слова|ключи)\s*:\s*(.*?)(?:\n|$)',
            r'\*\*(?:keywords|ключевые слова|ключи)\*\*:\s*(.*?)(?:\n|$)'
        ]
        for pat in kw_patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m and m.group(1).strip():
                kw_str = m.group(1).strip()
                parts = [k.strip().strip('"').strip("'") for k in kw_str.split(",") if k.strip()]
                if parts:
                    meta["keywords"] = parts
                    break

    return meta


def _save_image_from_response(api_data: Any, save_path: 'Path') -> bool:
    """Извлекает изображение (URL или Base64) и сохраняет его на диск."""
    import base64
    import urllib.request
    
    # 1. Пробуем Base64
    b64_data = _extract_b64_data(api_data)
    if b64_data:
        try:
            image_bytes = base64.b64decode(b64_data.strip())
            save_path.write_bytes(image_bytes)
            logger.info(f"   ✅ Изображение декодировано из Base64 и сохранено: {save_path.name}")
            return True
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка декодирования Base64: {e}. Попробуем найти URL...")
            
    # 2. Пробуем URL
    url = _extract_image_url(api_data)
    if url:
        try:
            logger.info(f"   📥 Скачиваю изображение по URL...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                save_path.write_bytes(response.read())
            logger.info(f"   ✅ Изображение скачано и сохранено: {save_path.name}")
            return True
        except Exception as e:
            logger.error(f"   ❌ Ошибка скачивания по URL: {e}")
            
    return False


# ============================================================
# Состояние pipeline
# ============================================================

@dataclass
class PipelineState:
    """Полное состояние pipeline между шагами."""
    provider: str = "deepseek"          # deepseek / kie / openai
    model: Optional[str] = None         # кастомная модель
    description: str = ""               # ручное описание/акценты темы
    style_nuances: str = ""             # ручная стилистика и нюансы
    additional_instructions: str = ""    # ручные дополнительные инструкции
    topic: str = ""
    article_type: str = "analysis"
    direction: str = ""
    style_id: str = ""                  # ID стиля из styles.py
    custom_chars: int = 0               # Пользовательский объём (из дашборда)
    output_dir: str = ""                # Папка для сохранения результатов (включая картинки)
    seo_budget: int = 0                 # Бюджет символов для Booster (SEO-резерв)

    # Выходы агентов
    brain_output: Dict = field(default_factory=dict)
    facts: Dict = field(default_factory=dict)
    scout_data: Dict = field(default_factory=dict)
    blueprint: Dict = field(default_factory=dict)
    draft: str = ""
    sheriff_review: Dict = field(default_factory=dict)
    mirror_review: Dict = field(default_factory=dict)
    seo_package: Dict = field(default_factory=dict)
    image_prompts: Dict = field(default_factory=dict)

    # Счётчики итераций
    sheriff_iterations: int = 0
    mirror_iterations: int = 0

    # Финальный результат
    final_article: str = ""
    final_meta: Dict = field(default_factory=dict)

    # Статус
    status: str = "pending"  # pending / running / completed / failed / budget_exhausted
    error: str = ""
    steps_completed: list = field(default_factory=list)

    # Счётчики токенов
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    tokens_by_agent: Dict = field(default_factory=dict)


# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    """
    Оркестратор мультиагентной генерации.

    Args:
        openai_api_key: ключ OpenAI API
        qdrant_client: клиент Qdrant (опционально)
        style_fingerprint: стилевой паспорт клиента (опционально)
        max_sheriff_iterations: макс. итераций Sheriff ↔ Heart
        max_mirror_iterations: макс. итераций Mirror ↔ Heart
    """

    def __init__(
        self,
        openai_api_key: str,
        qdrant_client=None,
        style_fingerprint: Optional[Dict] = None,
        max_sheriff_iterations: int = 2,
        max_mirror_iterations: int = 2,
    ):
        import os
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=openai_api_key, timeout=120.0)
        self.client = self.openai_client  # fallback
        
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "sk-23fc8150d8b14854ba3008c9e9e327f5")
        deepseek_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.deepseek_client = OpenAI(api_key=deepseek_key, base_url=deepseek_base, timeout=120.0)
        
        kie_key = os.getenv("KIE_API_KEY", "b21a40b5c6610f89d77bfa811e34d76a")
        kie_base = os.getenv("KIE_API_BASE", "https://api.kie.ai/v1")
        self.kie_client = OpenAI(api_key=kie_key, base_url=kie_base, timeout=120.0)
        
        self.qdrant = qdrant_client
        self.style = style_fingerprint
        self.max_sheriff = 2  # Hard cap
        self.max_mirror = 2   # Hard cap

    # ────────────────────────────────────────────
    # Главный метод
    # ────────────────────────────────────────────

    def run(
        self,
        topic: str,
        article_type: str = "analysis",
        direction: str = "",
        skip_scout: bool = False,
        skip_images: bool = False,
        style_id: str = "",
        custom_chars: int = 0,
        output_dir: Optional[str] = None,
        provider: str = "deepseek",
        model: Optional[str] = None,
        description: str = "",
        style_nuances: str = "",
        additional_instructions: str = "",
    ) -> PipelineState:
        """
        Запустить полный pipeline генерации статьи.

        Args:
            topic: тема статьи
            article_type: тип (checklist/case_study/law_review/reference/analysis/custom)
            direction: направление (налоги/юридическое/бизнес/финансы/экономика)
            skip_scout: пропустить Scout (если нет SearXNG)
            skip_images: пропустить Artist
            style_id: ID стиля из styles.py (auto-detect по article_type если пустой)
            custom_chars: пользовательский объём символов (0 = по умолчанию из стиля)
            output_dir: папка для сохранения результатов

        Returns:
            PipelineState с финальной статьёй и метаданными
        """
        # Авто-определение стиля по типу статьи если не указан
        effective_style = style_id or article_type

        # Динамическое переопределение для free_style
        if effective_style == "free_style":
            from .styles import get_style
            try:
                style = get_style("free_style")
                style.description = "Произвольная статья по свободной структуре с заданным объемом."
                style.tone = "деловой, экспертный"
                style.heart_instruction = None
                
                if description:
                    style.description = description
                if style_nuances:
                    style.tone = style_nuances
                    style.heart_instruction = f"Стиль написания и индивидуальные нюансы: {style_nuances}\n"
                if additional_instructions:
                    style.heart_instruction = (style.heart_instruction or "") + f"Дополнительные инструкции оператора: {additional_instructions}\n"
            except Exception as e:
                logger.warning(f"Ошибка динамического переопределения стиля: {e}")

        state = PipelineState(
            topic=topic,
            article_type=article_type,
            direction=direction,
            style_id=effective_style,
            custom_chars=custom_chars,
            output_dir=output_dir or "",
            provider=provider,
            model=model,
            description=description,
            style_nuances=style_nuances,
            additional_instructions=additional_instructions,
            status="running",
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"ð PIPELINE: {topic}")
        logger.info(f"   Тип: {article_type} | Направление: {direction}")
        logger.info(f"{'='*60}")

        try:
            # 1. Brain — декомпозиция
            self._step_brain(state)

            # 2. Fact-Finder — факты из RAG
            self._step_fact_finder(state)

            # 3. Scout — тренды (опционально)
            if not skip_scout:
                self._step_scout(state)

            # 4. Engineer — структура
            self._step_engineer(state)

            # 5. Heart — написание черновика
            self._step_heart(state)
            self._log_draft_length("Heart (Черновик)", state.draft)

            # 6. Объединённый review loop (Sheriff + условный Mirror → единая ревизия)
            for i in range(2):  # макс. 2 итерации объединённого review
                # Sheriff проверяет
                self._step_sheriff(state)
                sheriff_verdict = state.sheriff_review.get("verdict", "revision_needed")
                sheriff_score = state.sheriff_review.get("turing_score", 0)
                if isinstance(sheriff_score, str):
                    try:
                        sheriff_score = int(sheriff_score)
                    except (ValueError, TypeError):
                        sheriff_score = 0

                # Mirror проверяет ТОЛЬКО если Sheriff не дал высокий балл
                mirror_verdict = "pass"
                if sheriff_score < 85:
                    self._step_mirror(state)
                    mirror_verdict = state.mirror_review.get("verdict", "pass")
                else:
                    logger.info(f"   ⏭️ Mirror пропущен (Turing Score: {sheriff_score} ≥ 85)")

                # Если оба одобрили — выходим
                if sheriff_verdict == "approved" and mirror_verdict == "pass":
                    logger.info(f"   ✅ Текст одобрен (итерация {i+1})")
                    break

                # Объединяем фидбек и делаем ОДНУ ревизию
                logger.info(f"🔄 Объединённая ревизия #{i+1}")
                self._step_combined_revision(state)
                self._log_draft_length(f"Объединённая ревизия {i+1}", state.draft)

            # 7. Booster — SEO/GEO (только если есть текст)
            if state.draft and len(state.draft) > 100:
                self._step_booster(state)
                self._log_draft_length("Booster (SEO)", state.final_article)
                self._apply_smart_hard_cut(state)
                self._log_draft_length("Smart Hard-Cut (Финал)", state.final_article)

                # Финальная валидация (без API)
                validation_warnings = self._validate_final(state)
                for w in validation_warnings:
                    logger.warning(w)
            else:
                logger.error(f"❌ Draft пустой ({len(state.draft or '')} символов) — пропускаем Booster")
                state.final_article = state.draft or ""

            # 8. Artist (опционально)
            if not skip_images:
                self._step_artist(state)

            state.status = "completed"
            logger.info(f"\n✅ PIPELINE ЗАВЕРШЁН: {topic}")
            logger.info(f"   Sheriff итераций: {state.sheriff_iterations}")
            logger.info(f"   Mirror итераций: {state.mirror_iterations}")

            # Итого токенов
            if state.total_tokens > 0:
                logger.info(f"\n{'─'*60}")
                logger.info(f"📊 РАСХОД ТОКЕНОВ:")
                logger.info(f"   Prompt:     {state.total_prompt_tokens:>10,}")
                logger.info(f"   Completion: {state.total_completion_tokens:>10,}")
                logger.info(f"   ИТОГО:      {state.total_tokens:>10,}")
                if state.tokens_by_agent:
                    logger.info(f"   {'─'*29}")
                    for agent_id, data in state.tokens_by_agent.items():
                        total = data['prompt'] + data['completion']
                        logger.info(f"   {agent_id:>15}: {total:>8,} ({data['calls']} вызовов)")
                logger.info(f"{'─'*60}")

        except Exception as e:
            error_str = str(e).lower()
            if "insufficient" in error_str or "402" in error_str:
                state.status = "budget_exhausted"
                state.error = f"Баланс OpenAI исчерпан: {e}"
                logger.error(f"💸 БАЛАНС ИСЧЕРПАН: {e}")
            else:
                state.status = "failed"
                state.error = str(e)
                logger.error(f"❌ PIPELINE ОШИБКА: {e}")

        return state

    # ────────────────────────────────────────────
    # Шаги pipeline
    # ────────────────────────────────────────────

    def _step_brain(self, state: PipelineState):
        """Шаг 1: Brain — декомпозиция задачи."""
        logger.info("🧠 [1/8] Brain: декомпозиция задачи...")
        user_msg = (
            f"Тема: {state.topic}\n"
            f"Тип статьи: {state.article_type}\n"
            f"Направление: {state.direction}\n"
            f"Создай ТЗ для каждого агента."
        )
        state.brain_output = self._call_agent("brain", user_msg, state=state)
        state.steps_completed.append("brain")

    def _step_fact_finder(self, state: PipelineState):
        """Шаг 2: Fact-Finder — сбор фактов из RAG."""
        logger.info("🔎 [2/8] Fact-Finder: поиск фактов...")

        # RAG-запрос
        rag_context = ""
        task = state.brain_output.get("fact_finder_task", state.topic)
        # Safety: Claude может вернуть dict/list вместо строки
        if not isinstance(task, str):
            task = json.dumps(task, ensure_ascii=False) if isinstance(task, (dict, list)) else str(task)
        chunks = query_knowledge(task, "fact_finder", self.qdrant)
        rag_context = format_rag_context(chunks)

        user_msg = (
            f"Задание от Оркестратора: {task}\n"
            f"Тема: {state.topic}\n"
            f"Направление: {state.direction}\n\n"
            f"{rag_context}\n\n"
            f"Найди и структурируй все релевантные факты."
        )
        state.facts = self._call_agent("fact_finder", user_msg, state=state)
        state.steps_completed.append("fact_finder")

    def _step_scout(self, state: PipelineState):
        """Шаг 3: Scout — тренды и актуальность."""
        logger.info("📡 [3/8] Scout: анализ трендов через интернет...")
        
        # 1. Поиск через SearXNG
        from .searxng import search_web
        
        # Ищем по теме, либо по запросу, который мог сгенерировать Оркестратор
        search_query = state.brain_output.get("search_query", state.topic)
        if not isinstance(search_query, str):
            search_query = str(search_query)
        search_results = search_web(search_query)
        
        # Формируем текст сниппетов
        snippets_text = ""
        if search_results:
            snippets_text = "\n".join([f"[{i+1}] {r['title']}\nСниппет: {r['snippet']}\nИсточник: {r['url']}\n" for i, r in enumerate(search_results)])
        else:
            snippets_text = "Свежих данных в сети не найдено. Работай по базовой фактуре."

        scout_task = state.brain_output.get('scout_task', 'Найти актуальный угол подачи')
        if not isinstance(scout_task, str):
            scout_task = str(scout_task)

        user_msg = (
            f"Тема: {state.topic}\n"
            f"Направление: {state.direction}\n"
            f"Задание: {scout_task}\n\n"
            f"=== СВЕЖИЕ ДАННЫЕ ИЗ ИНТЕРНЕТА (SearXNG) ===\n"
            f"{snippets_text}\n"
            f"==========================================\n\n"
            f"Проанализируй эти сниппеты. Выдай hot_queries, угол подачи (angle) и оцени конкурентов."
        )
        state.scout_data = self._call_agent("scout", user_msg, state=state)
        state.steps_completed.append("scout")

    def _step_engineer(self, state: PipelineState):
        """Шаг 4: Engineer — структура статьи."""
        logger.info("🏗️ [4/8] Engineer: создание структуры...")
        pattern = PATTERNS.get(state.article_type, PATTERNS["free_style"])
        
        # Загружаем индивидуальный промпт из настроек стиля
        from .styles import get_style
        try:
            style = get_style(state.style_id or state.article_type)
            engineer_inst = style.engineer_instruction or pattern['engineer_structure']
        except Exception:
            engineer_inst = pattern['engineer_structure']

        user_msg = (
            f"Тема: {state.topic}\n"
            f"Тип статьи: {state.article_type}\n"
            f"Направление: {state.direction}\n\n"
            f"ЭТАЛОННАЯ СТРУКТУРА:\n{engineer_inst}\n\n"
            f"ФАКТЫ ОТ ИССЛЕДОВАТЕЛЯ:\n{json.dumps(state.facts, ensure_ascii=False, indent=2)}\n\n"
            f"УГОЛ ПОДАЧИ ОТ РАЗВЕДЧИКА:\n{json.dumps(state.scout_data, ensure_ascii=False, indent=2)}\n\n"
            f"Создай детальный Blueprint."
        )

        # RAG — шаблоны и фреймворки
        chunks = query_knowledge(f"шаблон {state.article_type}", "engineer", self.qdrant)
        if chunks:
            user_msg += f"\n\n{format_rag_context(chunks, max_chars=4000)}"

        state.blueprint = self._call_agent("engineer", user_msg, state=state)
        
        # Quality Gate: Проверка количества пунктов для чек-листа (строго 10 пунктов)
        if state.article_type == "checklist":
            sections = self._extract_sections(state.blueprint)
            if len(sections) < 10:
                logger.warning(f"⚠️ [Quality Gate] Engineer сгенерировал только {len(sections)} разделов вместо 10. Запрашиваем структуру заново.")
                retry_msg = (
                    user_msg + 
                    f"\n\nSYSTEM ERROR: You generated only {len(sections)} sections in the blueprint. "
                    f"You MUST generate strictly exactly 10 sections in the blueprint. "
                    f"Rewrite the blueprint with exactly 10 sections."
                )
                state.blueprint = self._call_agent("engineer", retry_msg, state=state)

        # Hard Fail: Проверка Хронотопа (только для кейсов)
        if state.article_type == "case_study":
            blueprint_str = json.dumps(state.blueprint, ensure_ascii=False)
            if "[CHRONOTOPE_SCENE]" not in blueprint_str:
                logger.warning("⚠️ [Hard Fail] В структуре нет тега [CHRONOTOPE_SCENE]. Запрашиваем структуру заново.")
                retry_msg = user_msg + "\n\nSYSTEM ERROR: You forgot to include the mandatory [CHRONOTOPE_SCENE] tag in the section titles. Rewrite the blueprint and include it."
                state.blueprint = self._call_agent("engineer", retry_msg, state=state)
                
        state.steps_completed.append("engineer")

    def _step_heart(self, state: PipelineState):
        """Шаг 5: Heart — написание черновика.

        Для лонгридов (>8000 целевых символов) использует
        посекционную генерацию: Heart пишет каждый раздел
        из blueprint отдельным вызовом, затем собирает.
        """
        logger.info("✍️ [5/8] Heart: написание текста...")
        style_block = self._get_style_block(state)

        # Определяем целевой и максимальный объём
        target_chars = state.custom_chars
        max_chars = 0
        if state.style_id:
            try:
                from .styles import get_style
                style = get_style(state.style_id)
                if not target_chars:
                    target_chars = style.target_chars
                max_chars = style.max_chars
            except Exception:
                pass
        if not target_chars:
            target_chars = 8000
        if not max_chars:
            max_chars = int(target_chars * 1.15)

        # SEO-резерв: 7% бюджета для Booster
        seo_reserve_pct = 0.07
        heart_target = int(target_chars * (1 - seo_reserve_pct))
        state.seo_budget = target_chars - heart_target
        logger.info(f"   🎯 Бюджет: Heart={heart_target} + SEO={state.seo_budget} = {target_chars}")

        # RAG — стилистические примеры
        rag_block = ""
        chunks = query_knowledge(state.topic, "heart", self.qdrant)
        if chunks:
            rag_block = format_rag_context(chunks, max_chars=4000)

        if heart_target > 20000 or state.article_type == "checklist":
            state.draft = self._heart_sectional(state, style_block, rag_block, heart_target)
        else:
            state.draft = self._heart_single(state, style_block, rag_block, heart_target)

        # Применяем Sanity-постпроцессор очистки артефактов
        state.draft = self._clean_leaked_ai_artifacts(state.draft)

        # Sanity-проверка начала статьи на утечку ИИ-методологии
        paragraphs = [p.strip() for p in state.draft.split("\n\n") if p.strip()]
        if paragraphs:
            first_two_paragraphs = " ".join(paragraphs[:2]).lower()
            
            ai_buzzwords = ["реестр", "статус", "колонка", "таблиц", "акт", "баз", "шаблон", "агент", "промпт", "стоп-строк", "генерац", "инструкц"]
            matched_buzzwords = [w for w in ai_buzzwords if w in first_two_paragraphs]
            
            if len(matched_buzzwords) >= 2:
                logger.warning(f"   ⚠️ [Sanity Check Failed] В начале статьи обнаружена служебная ИИ-лексика: {matched_buzzwords}. Перезапуск генерации...")
                warning_msg = (
                    f"\n\nSYSTEM WARNING: Your previous draft started with internal AI methodology words: {matched_buzzwords}. "
                    f"DO NOT write anything about internal regulations, columns, tables, registries, databases, checklists, rules, or instructions. "
                    f"Start the article directly with a highly engaging paragraph answering: what, who, and what benefit. "
                    f"Re-write the text, starting completely fresh and adhering strictly to this rule."
                )
                
                if heart_target > 20000 or state.article_type == "checklist":
                    state.draft = self._heart_sectional(state, style_block + warning_msg, rag_block, heart_target)
                else:
                    state.draft = self._heart_single(state, style_block + warning_msg, rag_block, heart_target)
                    
                state.draft = self._clean_leaked_ai_artifacts(state.draft)

        # HARD CAP
        actual = len(state.draft)
        if actual > max_chars and actual > 2000:
            logger.warning(f"   ⚠️ Превышение: {actual} vs лимит {max_chars}. Сокращаю...")
            state.draft = self._heart_condense(state, state.draft, heart_target)

        state.steps_completed.append("heart")
        logger.info(f"   🎯 Draft: {len(state.draft)} символов")

    def _heart_single(self, state, style_block, rag_block, target_chars):
        """Heart: генерация статьи одним вызовом."""
        min_chars = int(target_chars * 0.9)
        max_chars = int(target_chars * 1.1)
        user_msg = (
            f"BLUEPRINT ОТ СТРУКТУРИРОВЩИКА:\n{json.dumps(state.blueprint, ensure_ascii=False, indent=2)}\n\n"
            f"ФАКТЫ ОТ ИССЛЕДОВАТЕЛЯ:\n{json.dumps(state.facts, ensure_ascii=False, indent=2)}\n\n"
            f"{style_block}\n\n"
            f"⚠️ СТРОГОЕ ТРЕБОВАНИЕ К ОБЪЕМУ:\n"
            f"- Точный диапазон: от {min_chars} до {max_chars} символов\n"
            f"- Это примерно {target_chars // 6} слов\n"
            f"- ЗАПРЕЩЕНО писать больше {max_chars} символов\n"
            f"- ЗАПРЕЩЕНО писать меньше {min_chars} символов\n"
            f"- Если не укладываешься — сокращай менее важные разделы\n\n"
            f"{rag_block}\n\n"
            f"Напиши полный текст статьи в Markdown."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars)
        return result

    def _generate_clean_heart_text(self, user_msg: str, max_retries: int = 3, target_chars: int = 0) -> str:
        """Внутренняя обертка с Hard Fails для Писателя."""
        from .stopwords import ALL_STOP_WORDS
        stop_words = ALL_STOP_WORDS
        
        current_msg = user_msg
        for attempt in range(max_retries):
            result = self._call_agent("heart", current_msg, parse_json=False, target_chars=target_chars)
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            
            lower_text = text.lower()
            found_words = [w for w in stop_words if w in lower_text]
            if not found_words:
                text = text.replace("â€”", "-").replace("—", "-")
                return text
                
            logger.warning(f"⚠️ [Soft-Regex] Найден штамп: {found_words}. Просим переписать (попытка {attempt+1}/{max_retries})")
            current_msg += f"\n\nSYSTEM ALERT: Найден штамп: {found_words}. Перепиши этот абзац, убрав шаблон и используя точную, деловую формулировку."
            
        text = text.replace("â€”", "-").replace("—", "-")
        return text

    def _heart_sectional(self, state, style_block, rag_block, target_chars):
        """Heart: посекционная генерация лонгрида.

        Каждый раздел из blueprint пишется отдельным вызовом,
        затем все части собираются в единую статью.
        """
        sections = self._extract_sections(state.blueprint)
        if not sections:
            logger.warning("⚠️ Не удалось извлечь разделы из blueprint, fallback на single")
            return self._heart_single(state, style_block, rag_block, target_chars)

        if state.article_type == "checklist":
            chars_per_section = int((target_chars * 0.75) // len(sections))
        elif len(sections) > 5:
            chars_per_section = int((target_chars * 0.8) // len(sections))
        else:
            chars_per_section = target_chars // len(sections)
        words_per_section = chars_per_section // 6

        logger.info(f"   🎯 Посекционная генерация: {len(sections)} разделов × ~{chars_per_section} символов")

        parts = []
        for i, section in enumerate(sections, 1):
            logger.info(f"   ✍️ Раздел {i}/{len(sections)}: {section[:60]}...")

            section_msg = (
                f"Ты пишешь раздел {i} из {len(sections)} большой аналитической статьи.\n\n"
                f"ТЕМА СТАТЬИ: {state.topic}\n\n"
                f"ПОЛНЫЙ ПЛАН СТАТЬИ (для контекста):\n{json.dumps(state.blueprint, ensure_ascii=False, indent=2)}\n\n"
                f"ТЕКУЩИЙ РАЗДЕЛ: {section}\n\n"
                f"ФАКТЫ ОТ ИССЛЕДОВАТЕЛЯ:\n{json.dumps(state.facts, ensure_ascii=False, indent=2)}\n\n"
                f"{style_block}\n\n"
                f"⚠️ СТРОГОЕ ТРЕБОВАНИЕ К ОБЪЕМУ ЭТОГО РАЗДЕЛА:\n"
                f"- РОВНО {chars_per_section} символов (±10%, т.е. {int(chars_per_section*0.9)}-{int(chars_per_section*1.1)})\n"
                f"- Это примерно {words_per_section} слов — НЕ БОЛЬШЕ\n"
                f"- ЗАПРЕЩЕНО писать больше {int(chars_per_section*1.1)} символов\n"
                f"- Используй конкретные факты, цифры, ссылки на законы\n"
                f"- Избегай общих фраз типа 'оптимизировать расходы' — указывай КАК ИМЕННО\n"
                f"- Если это раздел с кейсом - придумай реалистичную историю с именами и городом. Пиши ПЕРЕСКАЗОМ от 3-го лица (не используй прямую речь, кроме 2-3 коротких бытовых реплик по 5-7 слов для атмосферы)\n\n"
                f"{rag_block}\n\n"
            )

            if parts:
                prev_context = "\n\n".join(parts[-2:])
                if len(prev_context) > 5000:
                    prev_context = prev_context[-5000:]
                section_msg += f"ПРЕДЫДУЩИЕ РАЗДЕЛЫ (для связности):\n{prev_context}\n\n"

            section_msg += (
                f"Напиши ТОЛЬКО раздел «{section}» в Markdown.\n"
                f"Начни с заголовка ## и уложись СТРОГО в {chars_per_section} символов."
            )

            result = self._generate_clean_heart_text(section_msg, target_chars=chars_per_section)
            parts.append(result)

        full_article = "\n\n".join(parts)

        if state.article_type == "checklist":
            logger.info("   ✍️ Генерирую Послесловие (заключение) для чек-листа...")
            conclusion_msg = (
                f"Ты пишешь Послесловие (заключение) для чек-листа.\n\n"
                f"ТЕМА ЧЕК-ЛИСТА: {state.topic}\n\n"
                f"ТЕКСТ ВСЕХ {len(sections)} ПУНКТОВ (для контекста):\n{full_article}\n\n"
                f"{style_block}\n\n"
                f"⚠️ СТРОГОЕ ТРЕБОВАНИЕ:\n"
                f"- Напиши ровно 1 лаконичный абзац заключения с выводами.\n"
                f"- Объем: до 500 символов (примерно 30-50 слов).\n"
                f"- ЗАПРЕЩЕНО писать больше 500 символов!\n"
                f"- Начни сразу с текста заключения, без заголовка H2/H3, без слов 'Послесловие' или 'Заключение'.\n\n"
                f"Напиши Послесловие:"
            )
            conclusion_text = self._generate_clean_heart_text(conclusion_msg, target_chars=500)
            full_article += "\n\n" + conclusion_text

        actual_chars = len(full_article)
        logger.info(f"   📊 Итого: {actual_chars} символов (цель: {target_chars})")

        if actual_chars < target_chars * 0.5:
            logger.warning(f"   ⚠️ Статья слишком короткая ({actual_chars}/{target_chars}), расширяю...")
            full_article = self._heart_expand(state, full_article, target_chars)

        return full_article

    def _heart_condense(self, state, draft, target_chars):
        """Сократить статью, если она превысила лимит."""
        import re as _re
        overflow = len(draft) - target_chars
        max_chars = int(target_chars * 1.1)
        logger.info(f"   ✂️ Сокращаю: {len(draft)} → {target_chars} (убрать ~{overflow} символов)")

        h2_count_before = 0
        if state.article_type == "checklist":
            h2_count_before = len(_re.findall(r'^## \d+\.', draft, _re.MULTILINE))
        
        extra_instruction = ""
        if state.article_type == "checklist":
            extra_instruction = (
                "\n\nПРИМЕЧАНИЕ ДЛЯ ЧЕК-ЛИСТА:\n"
                "- Сохраняй структуру (все заголовки ## должны остаться).\n"
                "- Сокращай только текст внутри пунктов, не удаляй сами пункты."
            )
            
        user_msg = (
            f"ЧЕРНОВИК СТАТЬИ ДЛЯ СОКРАЩЕНИЯ:\n{draft}\n\n"
            f"Сейчас в тексте: {len(draft)} символов. Нужно убрать ~{overflow} символов, чтобы уложиться в {target_chars} (максимум {max_chars}).\n\n"
            f"КАК СОКРАЩАТЬ:\n"
            f"- Убери повторы, водянистые фразы и общие рассуждения.\n"
            f"- Сократи слишком раздутые примеры и кейсы.\n"
            f"- Не трогай ключевые факты, цифры и ссылки на законы.\n"
            f"- Сохрани структуру (все заголовки ## должны остаться).\n"
            f"{extra_instruction}\n\n"
            f"Верни ПОЛНЫЙ сокращённый текст статьи в Markdown."
        )
        
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars)
        
        if len(result) > len(draft) or len(result) < target_chars * 0.3:
            logger.warning(f"   ⚠️ Сокращение не удалось ({len(result)} символов), оставляю оригинал")
            return draft

        if state.article_type == "checklist" and h2_count_before > 0:
            h2_count_after = len(_re.findall(r'^## \d+\.', result, _re.MULTILINE))
            if h2_count_after < h2_count_before:
                logger.warning(f"   ⚠️ Condense удалил пункты ({h2_count_before} → {h2_count_after}). Откат.")
                return draft

        return result

    def _heart_expand(self, state, draft, target_chars):
        """Расширить слишком короткую статью."""
        deficit = target_chars - len(draft)
        user_msg = (
            f"Статья слишком короткая. Нужно добавить ещё ~{deficit} символов ({deficit // 6} слов).\n\n"
            f"ТЕКУЩИЙ ТЕКСТ:\n{draft}\n\n"
            f"ФАКТЫ:\n{json.dumps(state.facts, ensure_ascii=False, indent=2)}\n\n"
            f"ЗАДАНИЕ:\n"
            f"1. Добавь подробные примеры и кейсы в каждый раздел\n"
            f"2. Расширь анализ с конкретными цифрами и ссылками на законы\n"
            f"3. Добавь блоки 'Что делать' с пошаговыми инструкциями\n"
            f"4. НЕ повторяй уже написанное — РАСШИРЯЙ и УГЛУБЛЯЙ\n\n"
            f"Верни ПОЛНЫЙ расширенный текст статьи."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars)
        return result

    def _extract_sections(self, blueprint: Dict) -> list:
        """Извлечь список разделов из blueprint Engineer."""
        # Пробуем стандартные ключи
        for key in ["sections", "structure", "outline", "chapters", "разделы", "план"]:
            if key in blueprint:
                val = blueprint[key]
                if isinstance(val, list):
                    # Список может быть строками или dict-ами
                    result = []
                    for item in val:
                        if isinstance(item, str):
                            result.append(item)
                        elif isinstance(item, dict):
                            # Берём title/name/heading
                            title = item.get("title") or item.get("name") or item.get("heading") or item.get("section", "")
                            if title:
                                result.append(str(title))
                    if result:
                        return result

        # Фолбэк: ищем любой список строк в blueprint
        for key, val in blueprint.items():
            if isinstance(val, list) and len(val) >= 3:
                strings = [str(item) if isinstance(item, str) else
                           item.get("title", item.get("name", str(item)))
                           if isinstance(item, dict) else str(item)
                           for item in val]
                if all(len(s) > 3 for s in strings):
                    return strings

        return []

    def _step_heart_revision(self, state: PipelineState):
        """Heart — доработка по фидбеку Sheriff."""
        logger.info("✍️ Heart: доработка по фидбеку Sheriff...")
        original_len = len(state.draft)
        target_chars = state.custom_chars or 8000
        
        user_msg = (
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"ФИДБЕК ОТ РЕДАКТОРА (Sheriff):\n{json.dumps(state.sheriff_review, ensure_ascii=False, indent=2)}\n\n"
            f"Внеси исправления, но не меняй структуру. Верни полный текст статьи."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars)
        if len(result) > original_len * 0.3:
            state.draft = result

    def _step_heart_humanize(self, state: PipelineState):
        """Heart — humanization по фидбеку Mirror."""
        logger.info("✍️ Heart: humanization...")
        original_len = len(state.draft)
        target_chars = state.custom_chars or 8000
        
        user_msg = (
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"ФИДБЕК ОТ ЗЕРКАЛА (Mirror):\n{json.dumps(state.mirror_review, ensure_ascii=False, indent=2)}\n\n"
            f"Внеси исправления для слома ИИ-ритма. Верни полный текст статьи."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars)
        if len(result) > original_len * 0.3:
            state.draft = result

    def _step_combined_revision(self, state: PipelineState):
        """Heart — объединенная доработка по замечаниям Sheriff и Mirror."""
        logger.info("✍️ Heart: объединенная доработка по замечаниям...")
        original_len = len(state.draft)
        target_chars = state.custom_chars or 8000
        
        user_msg = (
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"ФИДБЕК ОТ РЕДАКТОРА (Sheriff):\n{json.dumps(state.sheriff_review, ensure_ascii=False, indent=2)}\n\n"
            f"ФИДБЕК ОТ ЗЕРКАЛА (Mirror):\n{json.dumps(state.mirror_review, ensure_ascii=False, indent=2)}\n\n"
            f"Внеси все указанные исправления за один проход и верни ВЕСЬ текст статьи целиком."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars)
        if len(result) > original_len * 0.3:
            state.draft = result
            state.sheriff_iterations += 1
        logger.info(f"   📏 Draft после объединённой ревизии: {len(state.draft)} символов")

    def _step_sheriff(self, state: PipelineState):
        """Шериф (Редактор) — проверка качества и фактов."""
        logger.info("👮 [6/8] Sheriff: проверка черновика статьи...")
        user_msg = (
            f"ТЕМА СТАТЬИ: {state.topic}\n"
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"Выполни строгую проверку качества черновика."
        )
        response = self._call_agent("sheriff", user_msg, parse_json=True, state=state)
        state.sheriff_review = response
        
        if isinstance(response, dict):
            is_approved = response.get("approved", False)
            state.sheriff_review["verdict"] = "approved" if is_approved else "revision_needed"
            quality_gate = response.get("quality_gate", {})
            score = quality_gate.get("actionability_score", 10) * 10
            state.sheriff_review["turing_score"] = response.get("turing_score", score)
        else:
            state.sheriff_review["verdict"] = "revision_needed"
            state.sheriff_review["turing_score"] = 0
            
        logger.info(f"   👮 Sheriff вердикт: {state.sheriff_review['verdict']} (Turing Score: {state.sheriff_review.get('turing_score', 0)})")

    def _step_mirror(self, state: PipelineState):
        """Зеркало (Стилистический аналитик) — проверка человечности."""
        logger.info("🪞 [7/8] Mirror: стилистический аудит...")
        user_msg = (
            f"ТЕМА СТАТЬИ: {state.topic}\n"
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"Проанализируй текст на естественность ритма."
        )
        response = self._call_agent("mirror", user_msg, parse_json=True, state=state)
        state.mirror_review = response
        
        if isinstance(response, dict):
            turing_score = response.get("turing_score", 95)
            state.mirror_review["verdict"] = "pass" if turing_score >= 80 else "fail"
        else:
            state.mirror_review["verdict"] = "pass"
            
        logger.info(f"   🪞 Mirror вердикт: {state.mirror_review['verdict']} (Turing Score: {state.mirror_review.get('turing_score', 95)})")

    def _step_booster(self, state: PipelineState):
        """Шаг 8: Booster — SEO/GEO оптимизация."""
        logger.info("🚀 [8/9] Booster: SEO/GEO оптимизация...")
        
        user_msg = (
            f"ТЕМА СТАТЬИ: {state.topic}\n"
            f"ТИП СТАТЬИ: {state.article_type}\n"
            f"НАПРАВЛЕНИЕ: {state.direction}\n\n"
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"- Твой БЮДЖЕТ на SEO-добавки: ровно {state.seo_budget} символов. Это всё, что ты можешь добавить.\n"
            f"- Citation Bait: вплетай в ПОСЛЕДНЕЕ предложение перед каждым H2 (не создавай новые абзацы).\n"
            f"- LSI-ключи: перефразируй существующие предложения, не добавляя новых.\n"
            f"- FAQ: добавь в JSON-поле 'faq' (для Schema.org), но НЕ вставляй блок FAQ в тело статьи.\n"
            f"- Категорически ЗАПРЕЩЕНО добавлять новые разделы H2/H3.\n\n"
            f"Оптимизируй статью и подготовить SEO-пакет."
        )
        
        raw_response = self._call_agent("booster", user_msg, parse_json=False, state=state)
        
        import re
        import json
        
        metadata_match = re.search(r'<seo_metadata>\s*({.*?})\s*</seo_metadata>', raw_response, re.DOTALL)
        seo_package = {}
        if metadata_match:
            try:
                seo_package = json.loads(metadata_match.group(1).strip())
            except Exception as e:
                logger.warning(f"   ⚠️ Не удалось распарсить JSON в <seo_metadata>: {e}")
                cleaned_json = re.sub(r'^```(?:json)?\s*', '', metadata_match.group(1).strip())
                cleaned_json = re.sub(r'\s*```$', '', cleaned_json)
                try:
                    seo_package = json.loads(cleaned_json)
                except Exception:
                    pass
        else:
            try:
                cleaned_json = re.sub(r'^```(?:json)?\s*', '', raw_response.strip())
                cleaned_json = re.sub(r'\s*```$', '', cleaned_json)
                seo_package = json.loads(cleaned_json)
            except Exception:
                pass
                
        state.seo_package = seo_package
        
        article_match = re.search(r'<optimized_article>\s*(.*?)\s*</optimized_article>', raw_response, re.DOTALL)
        optimized_text = ""
        if article_match:
            optimized_text = article_match.group(1).strip()
        else:
            if "</seo_metadata>" in raw_response:
                parts = raw_response.split("</seo_metadata>")
                optimized_text = parts[1].strip()
                optimized_text = re.sub(r'<optimized_article>\s*', '', optimized_text)
                optimized_text = re.sub(r'</optimized_article>\s*$', '', optimized_text).strip()
            else:
                optimized_text = seo_package.get("optimized_text") or seo_package.get("article_text") or ""
                
        state.final_article = optimized_text if (optimized_text and len(optimized_text) > 100) else state.draft
        state.final_article = self._clean_leaked_ai_artifacts(state.final_article)
        
        fallback_meta = _extract_meta_from_text(state.final_article)
        state.final_meta = state.seo_package.get("meta", {})
        for k in ["title", "description", "keywords"]:
            if not state.final_meta.get(k) and fallback_meta.get(k):
                state.final_meta[k] = fallback_meta[k]
                logger.info(f"      📝 Извлечено {k} из текста: '{fallback_meta[k][:50]}...'")
 
        logger.info(f"   📊 Final article: {len(state.final_article)} символов")
        state.steps_completed.append("booster")

    def _step_artist(self, state: PipelineState):
        """Шаг 9: Artist — генерация и интеграция изображений через GPT Image 2."""
        logger.info("🎨 [9/9] Artist: генерация и интеграция изображений...")
        import urllib.request
        from pathlib import Path
        import re
        import httpx

        # 1. Загрузка визуального стиля и ключевого слова
        from .styles import get_style
        try:
            style = get_style(state.style_id or state.article_type)
            style_ref = getattr(style, "style_reference_prompt", None)
            default_word = getattr(style, "text_overlay_word", None)
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки стиля: {e}. Fallback на общие параметры.")
            style_ref = "slate grey background, bright neon acid green lines, dark corporate high-contrast style"
            default_word = "БИЗНЕС"

        if not style_ref:
            style_ref = "slate grey background, bright neon acid green lines, dark corporate high-contrast style"
        if not default_word:
            default_word = "БИЗНЕС"

        # 2. Определение русского слова для наложения на изображение
        text_overlay = default_word
        keywords_to_find = ["УСН", "НДС", "115-ФЗ", "ЕФС-1", "ИП", "ООО", "НДФЛ", "КоАП", "ФНС"]
        combined_source = (state.topic + " " + state.final_article).upper()
        for kw in keywords_to_find:
            if kw in combined_source:
                text_overlay = kw
                break

        logger.info(f"   🎨 Текстовый паспорт стиля: '{style_ref}'")
        logger.info(f"   🎨 Слово на картинке: '{text_overlay}'")

        # 3. Подсчет количества маркеров [картинка] или [IMAGE_PROMPT_HERE] в статье
        article_text = state.final_article
        marker_pattern = r"\[(?:картинка|IMAGE_PROMPT_HERE)(?::\s*.*?)?\]"
        markers = re.findall(marker_pattern, article_text)
        num_markers = len(markers)

        # Автоматическая вставка маркеров картинок, если они отсутствуют
        if num_markers == 0:
            logger.info("   🎨 Маркеры картинок не найдены в тексте. Вставляем их автоматически...")
            h2_matches = list(re.finditer(r'^##\s+(.*?)$', article_text, re.MULTILINE))
            if h2_matches:
                if state.article_type == "checklist":
                    target_indices = [3, 5, 7, 9]
                    inserted_count = 0
                    new_text = ""
                    last_pos = 0
                    h2_index = 0
                    for match in h2_matches:
                        h2_index += 1
                        start, end = match.span()
                        new_text += article_text[last_pos:start]
                        if h2_index in target_indices:
                            h2_title = match.group(1)
                            new_text += f"[IMAGE_PROMPT_HERE: Illustration for the section about {h2_title}]\n\n"
                            inserted_count += 1
                        new_text += article_text[start:end]
                        last_pos = end
                    new_text += article_text[last_pos:]
                    article_text = new_text
                    logger.info(f"   🎨 Автоматически вставлено {inserted_count} маркеров перед пунктами {target_indices}.")
                else:
                    inserted_count = 0
                    new_text = ""
                    last_pos = 0
                    h2_index = 0
                    for match in h2_matches:
                        h2_index += 1
                        start, end = match.span()
                        new_text += article_text[last_pos:start]
                        if h2_index > 1 and h2_index % 2 == 0:
                            h2_title = match.group(1)
                            new_text += f"[IMAGE_PROMPT_HERE: Illustration for the section about {h2_title}]\n\n"
                            inserted_count += 1
                        new_text += article_text[start:end]
                        last_pos = end
                    new_text += article_text[last_pos:]
                    article_text = new_text
                    logger.info(f"   🎨 Автоматически вставлено {inserted_count} маркеров перед H2 заголовками.")
                
                state.final_article = article_text
                markers = re.findall(marker_pattern, article_text)
                num_markers = len(markers)
        logger.info(f"   🎨 Найдено {num_markers} маркеров разделов в тексте.")

        # 4. Запрос к агенту Artist для генерации промптов к каждой сцене
        user_msg = (
            f"ТЕМА СТАТЬИ: {state.topic}\n"
            f"ТЕКСТ СТАТЬИ:\n{article_text[:6000]}\n\n"
            f"Мы используем модель GPT Image 2 для генерации обложки и разделительных иллюстраций.\n"
            f"Сгенерируй детальные художественные описания (промпты) для сцен.\n"
            f"Сцены должны соответствовать теме и содержанию разделов.\n\n"
            f"ОБЯЗАТЕЛЬНО верни результат строго в формате JSON (без лишнего текста, только валидный JSON):\n"
            f"{{\n"
            f"  \"cover_scene\": \"Детальное описание композиции для обложки (на английском языке, до 80 слов, описывай только объекты и фон, БЕЗ упоминания текста на картинке)\",\n"
            f"  \"section_scenes\": [\n"
            f"    \"Детальное описание композиции для первой разделительной картинки (на английском, до 80 слов, БЕЗ упоминания текста)\",\n"
            f"    \"Детальное описание композиции для второй разделительной картинки (на английском, до 80 слов, БЕЗ упоминания текста)\"\n"
            f"  ]\n"
            f"}}\n\n"
            f"Количество промптов в section_scenes должно быть строго равно {num_markers}."
        )

        artist_response = self._call_agent("artist", user_msg, parse_json=True, state=state)
        state.image_prompts = artist_response

        # 5. Определение папки для сохранения изображений
        output_dir_str = state.output_dir
        if not output_dir_str:
            timestamp = time.strftime("%Y%m%d_%H%M")
            slug = "".join([c if c.isalnum() or c == "_" else "_" for c in state.topic.lower()])[:40].strip("_")
            output_dir_str = f"output/{timestamp}_{slug}"

        output_dir = Path(output_dir_str)
        images_dir = output_dir / "images"
        
        # 6. Генерация и скачивание изображений с фолбеком
        section_paths = []
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            api_key = self.client.api_key
            base_url = str(self.client.base_url).rstrip('/')
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            def _generate_image_with_fallback(prompt: str, size: str, model: str = "gpt-image-2") -> Any:
                """Внутренний хелпер для генерации картинок с авто-фолбеком на DALL-E 3."""
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "n": 1
                }
                try:
                    logger.info(f"   🚀 Запрос к API изображений (Модель: {model}, Размер: {size})...")
                    with httpx.Client(timeout=120.0) as http_client:
                        resp = http_client.post(f"{base_url}/images/generations", json=payload, headers=headers)
                        resp.raise_for_status()
                        return resp.json()
                except Exception as err:
                    if model == "gpt-image-2":
                        fallback_model = "dall-e-3"
                        fallback_size = "1792x1024"
                        logger.warning(f"   ⚠️ Ошибка {model} ({size}): {err}. Пробуем оригинальный {fallback_model} ({fallback_size})...")
                        fallback_payload = {
                            "model": fallback_model,
                            "prompt": prompt,
                            "size": fallback_size,
                            "n": 1
                        }
                        try:
                            with httpx.Client(timeout=120.0) as http_client:
                                resp = http_client.post(f"{base_url}/images/generations", json=fallback_payload, headers=headers)
                                resp.raise_for_status()
                                return resp.json()
                        except Exception as err2:
                            logger.error(f"   ❌ Ошибка оригинального {fallback_model}: {err2}")
                            raise err2
                    else:
                        raise err
            
            # А. Генерация обложки (Размер 1536x768)
            cover_scene = artist_response.get("cover_scene", f"Conceptual cover art representing the theme: {state.topic}")
            full_cover_prompt = (
                f"{cover_scene}, style reference: {style_ref}. "
                f"Ensure a very bold, clean, minimalistic and large text overlay in Russian language reads exactly: '{text_overlay}'. "
                f"The text must be the primary design element and perfectly integrated, with absolutely zero grammatical or spelling errors."
            )
            
            cover_data = _generate_image_with_fallback(full_cover_prompt, "1536x768", "gpt-image-2")
            logger.info(f"   🔍 DEBUG: Сырой ответ API (обложка): {cover_data}")
            cover_path = images_dir / "main.png"
            if not _save_image_from_response(cover_data, cover_path):
                raise ValueError("Не удалось извлечь и сохранить обложку (ни из Base64, ни из URL)")
  
            # Б. Генерация разделителей разделов (Размер 1536x384)
            section_scenes = artist_response.get("section_scenes", [])
            for idx, scene in enumerate(section_scenes):
                if idx >= num_markers:
                    break
                full_section_prompt = (
                    f"{scene}, style reference: {style_ref}. "
                    f"Include a small, elegant and crisp Russian text reads: '{text_overlay}' as a subtle background detail. "
                    f"High contrast, beautiful lighting, wide landscape format."
                )
                
                sec_data = _generate_image_with_fallback(full_section_prompt, "1536x384", "gpt-image-2")
                logger.info(f"   🔍 DEBUG: Сырой ответ API (разделитель {idx+1}): {sec_data}")
                sec_path = images_dir / f"section_{idx+1}.png"
                if not _save_image_from_response(sec_data, sec_path):
                    raise ValueError(f"Не удалось извлечь и сохранить разделитель {idx+1}")
                section_paths.append(f"images/section_{idx+1}.png")

        except Exception as img_err:
            logger.error(f"❌ Ошибка генерации или сохранения изображений: {img_err}")
            logger.warning("⚠️ Продолжаем работу пайплайна без физических картинок.")

        # 7. Интеграция картинок в Markdown
        # А. Встраивание обложки в начало статьи (под заголовок H1)
        h1_match = re.search(r"^(#\s+.*?)$", article_text, re.MULTILINE)
        if h1_match:
            h1_line = h1_match.group(1)
            article_text = article_text.replace(h1_line, f"{h1_line}\n\n![Обложка](images/main.png)", 1)
        else:
            article_text = f"![Обложка](images/main.png)\n\n" + article_text

        # Б. Замена текстовых маркеров на Markdown-теги
        matches = list(re.finditer(marker_pattern, article_text))
        for idx, match in enumerate(matches):
            if idx < len(section_paths):
                markdown_img = f"![Иллюстрация]({section_paths[idx]})"
                article_text = article_text.replace(match.group(0), markdown_img, 1)
            else:
                article_text = article_text.replace(match.group(0), "", 1)

        state.final_article = article_text
        state.steps_completed.append("artist")

    def _apply_smart_hard_cut(self, state: PipelineState):
        """
        Интеллектуальный спасательный механизм сжатия и очистки от воды.
        Если финальная статья превышает динамический лимит объема,
        система сжимает её через LLM с сохранением H2-заголовков и структуры.
        """
        source_text = state.final_article if state.final_article else state.draft
        if not source_text:
            return

        target = state.custom_chars
        if not target:
            if state.style_id:
                try:
                    from .styles import get_style
                    target = get_style(state.style_id).target_chars
                except Exception:
                    pass
        if not target:
            from .patterns import PATTERNS
            target = PATTERNS.get(state.article_type, PATTERNS.get("free_style", {})).get("target_chars", 8000)

        # Вычисляем динамический margin
        if target < 5000:
            margin_pct = 0.30
        else:
            margin_pct = 0.20

        limit = int(target * (1.0 + margin_pct))
        if len(source_text) <= limit:
            if not state.final_article:
                state.final_article = source_text
            return

        logger.info(f"   ⚠️ [Intelligent Reduction] Статья превышает лимит ({len(source_text)} > {limit} симв.). Запускаю интеллектуальное сжатие...")

        # Запускаем интеллектуальное сжатие через LLM
        condensed = self._heart_condense(state, source_text, target)
        
        # Если сжатие прошло успешно и мы уложились в лимит
        if len(condensed) < len(source_text) and len(condensed) <= limit:
            logger.info(f"   ✅ [Intelligent Reduction] Статья успешно сжата с помощью LLM до {len(condensed)} символов.")
            state.final_article = condensed
            return
        
        # Если LLM не справилась (или вернула некорректный размер), 
        # применяем резервный аккуратный срез по структуре (fallback)
        logger.warning("   ⚠️ [Intelligent Reduction] Сжатие LLM не уложилось в лимит. Применяю резервное структурное сокращение...")
        
        cutoff_limit = limit
        cutoff_text = source_text[:cutoff_limit]

        # Ищем последний заголовок ## или H3/H2 в пределах cutoff_limit
        last_section = cutoff_text.rfind('\n## ')
        if last_section == -1:
            last_section = cutoff_text.rfind('\n### ')
        
        if last_section > target * 0.4:
            source_text = cutoff_text[:last_section].rstrip()
            logger.info(f"   ✂️ [Fallback Cut] Текст обрезан по заголовку раздела. Новый объем: {len(source_text)} символов.")
        else:
            last_paragraph = cutoff_text.rfind('\n\n')
            if last_paragraph > target * 0.4:
                source_text = cutoff_text[:last_paragraph].rstrip()
                logger.info(f"   ✂️ [Fallback Cut] Текст обрезан по концу абзаца. Новый объем: {len(source_text)} символов.")
            else:
                last_sentence = max(cutoff_text.rfind('. '), cutoff_text.rfind('! '), cutoff_text.rfind('? '))
                if last_sentence > target * 0.3:
                    source_text = cutoff_text[:last_sentence + 1].rstrip()
                    logger.info(f"   ✂️ [Fallback Cut] Текст обрезан по предложению. Новый объем: {len(source_text)} символов.")
                else:
                    source_text = cutoff_text.rstrip()
                    logger.info(f"   ✂️ [Fallback Cut] Грубая обрезка текста. Новый объем: {len(source_text)} символов.")

        # Генерируем адекватное завершение
        if len(source_text) > 500:
            try:
                last_300 = source_text[-300:]
                closing_msg = (
                    f"Тема статьи: {state.topic}\n"
                    f"Последние 300 символов статьи:\n{last_300}\n\n"
                    f"Напиши РОВНО один завершающий абзац-вывод (20-30 слов) для раздела 'ИТОГ'. "
                    f"Без банальностей, без вводных слов, без 'Таким образом'. Напиши конкретный, осязаемый итог по теме."
                )
                closing = self._call_agent("heart", closing_msg, parse_json=False, target_chars=200, state=state)
                if closing and isinstance(closing, str) and len(closing.strip()) < 300:
                    source_text += "\n\n## ИТОГ\n\n" + closing.strip()
            except Exception as e:
                logger.warning(f"   ⚠️ [Fallback Cut] Не удалось сгенерировать заключение: {e}")
                
        state.final_article = source_text

    def _clean_leaked_ai_artifacts(self, text: str) -> str:
        """Sanity-постпроцессор для очистки текста от служебных ИИ-артефактов и меток."""
        if not text:
            return ""
            
        import re
        
        first_header = re.search(r'^(?:#|##)\s', text, re.MULTILINE)
        if first_header:
            start_pos = first_header.start()
            text = text[start_pos:]
            
        markers_to_clean = [
            r'\[CHRONOTOPE_SCENE.*?\]',
            r'\[TIME_ANCHOR.*?\]',
            r'\[ВРЕМЕННОЙ\s+КОНТЕКСТ.*?\]',
            r'\[METHODOLOGY.*?\]',
            r'\[AI_INSTRUCTION.*?\]',
        ]
        for marker in markers_to_clean:
            text = re.sub(marker, '', text, flags=re.IGNORECASE)
            
        text = re.sub(r'\n*(?:Вот\s+черновик|Вот\s+статья|Надеюсь,\s+вам|Спецификация\s+генерации).*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
        return text.strip()

    def _log_draft_length(self, step_name: str, text: str):
        """Подсчитать и залогировать длину текста в символах и абзацах."""
        if not text:
            logger.info(f"📊 [Длина статьи] {step_name}: Текст пуст")
            return
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        logger.info(f"📊 [Длина статьи] {step_name} | Символов: {len(text)} | Абзацев: {len(paragraphs)}")

    # ────────────────────────────────────────────
    # Вспомогательные методы
    # ────────────────────────────────────────────

    def _call_agent(
        self,
        agent_id: str,
        user_message: str,
        parse_json: bool = True,
        target_chars: int = 0,
        state: 'PipelineState' = None,
    ) -> Any:
        """
        Вызвать агента через OpenAI API.

        Args:
            agent_id: ID агента
            user_message: пользовательское сообщение
            parse_json: пытаться ли парсить ответ как JSON
            target_chars: целевой объём в символах
            state: PipelineState для аккумуляции токенов (опционально)

        Returns:
            Dict (если JSON) или str (если текст)
        """
        agent = get_agent(agent_id)
        system_prompt = get_system_prompt(agent_id)

        # Динамический временной контекст для исключения устаревшего года (2025)
        import datetime
        current_year = datetime.datetime.now().year
        time_anchor = (
            f"\n\n[ВРЕМЕННОЙ КОНТЕКСТ]:\n"
            f"Текущий год — {current_year}. Все рекомендации, правила, лимиты, налоги, риски и метаданные (SEO Title, Description, H1) "
            f"должны генерироваться и быть актуальными исключительно для {current_year} года.\n"
            f"Любые упоминания {current_year - 1} года и более ранних периодов допускаются только в прошедшем времени "
            f"(как исторический контекст или сравнение). Категорически запрещено указывать прошлые годы (например, {current_year - 1}) "
            f"в качестве текущего или будущего времени в заголовках, мета-тегах и основном тексте."
        )
        extended_system_prompt = system_prompt + time_anchor

        # Обрезаем user_message если слишком длинный
        max_input = 50000  # ~12k токенов
        if len(user_message) > max_input:
            user_message = user_message[:max_input] + "\n\n[...контекст обрезан...]"

        # Динамический расчет лимита токенов
        max_tokens_for_call = agent.max_tokens
        if target_chars > 0 and agent_id in ("heart", "mirror", "booster"):
            # Для русского языка 1 токен ≈ 1.8 символов (в отличие от английского 4)
            # Дополнительно закладываем 45% запас на JSON-структуру, разметку и reasoning
            calculated_tokens = int(target_chars * 0.7)
            # Не превышаем максимум агента, но даем не менее 1500 токенов для корректного завершения
            max_tokens_for_call = max(1500, min(calculated_tokens, agent.max_tokens))
            logger.info(f"   ⚙️ Динамический лимит токенов для {agent.name}: {max_tokens_for_call} (базовый: {agent.max_tokens})")

        # Динамический выбор клиента и модели на основе провайдера
        current_client = self.deepseek_client  # По умолчанию DeepSeek
        model_name = agent.model  # По умолчанию модель агента
        
        if state is not None:
            provider = getattr(state, "provider", "deepseek").lower()
            custom_model = getattr(state, "model", None)
            
            if provider == "kie":
                current_client = self.kie_client
                model_name = custom_model if custom_model else "claude-4.7"
            elif provider == "openai":
                current_client = self.openai_client
                model_name = custom_model if custom_model else "gpt-4o"
            elif provider == "deepseek":
                current_client = self.deepseek_client
                if custom_model:
                    model_name = custom_model

        response = current_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": extended_system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=agent.temperature,
            max_completion_tokens=max_tokens_for_call,
        )
        # Аккумуляция токенов
        if hasattr(response, 'usage') and response.usage:
            p_tokens = response.usage.prompt_tokens or 0
            c_tokens = response.usage.completion_tokens or 0
            logger.info(f"   📊 [{agent_id}] {p_tokens:,}+{c_tokens:,} токенов")
            if state is not None:
                state.total_prompt_tokens += p_tokens
                state.total_completion_tokens += c_tokens
                state.total_tokens += p_tokens + c_tokens
                if agent_id not in state.tokens_by_agent:
                    state.tokens_by_agent[agent_id] = {"prompt": 0, "completion": 0, "calls": 0}
                state.tokens_by_agent[agent_id]["prompt"] += p_tokens
                state.tokens_by_agent[agent_id]["completion"] += c_tokens
                state.tokens_by_agent[agent_id]["calls"] += 1

        raw = response.choices[0].message.content
        if raw is None:
            raw = ""
            logger.warning(f"⚠️ [{agent_id}] ответ пустой (content=None)")
        raw = raw.strip()

        if parse_json:
            return self._parse_json_response(raw, agent_id)
        return raw

    def _parse_json_response(self, raw: str, agent_id: str) -> Dict:
        """Извлечь JSON из ответа агента (с обработкой markdown)."""
        import re
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Пробуем найти JSON внутри текста
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"⚠️ [{agent_id}] не удалось распарсить JSON, возвращаю как текст")
            return {"raw_response": raw}

    def _get_style_block(self, state: PipelineState = None) -> str:
        """
        Сформировать блок стилевых инструкций для Heart.

        Приоритет:
        1. Style Fingerprinting (из styles.py) — если style_id указан
        2. Клиентский стилевой паспорт (self.style) — если передан
        3. Пустой блок — агент работает по умолчанию
        """
        # 1. Style Fingerprinting из styles.py
        if state and state.style_id:
            try:
                from .styles import get_style_prompt
                custom = state.custom_chars if state.custom_chars > 0 else None
                return get_style_prompt(state.style_id, custom) + "\n\n"
            except (ValueError, ImportError):
                pass  # Стиль не найден — пробуем клиентский паспорт

        # 2. Правила из паттернов (patterns.py)
        pattern = PATTERNS.get(state.article_type, PATTERNS["free_style"])
        lines = [
            "СТИЛЕВОЙ ПАСПОРТ (обязательно соблюдать):",
            f"- {pattern['heart_style']}"
        ]
        
        # 3. Клиентский стилевой паспорт
        if self.style:
            if self.style.get("tone"):
                lines.append(f"- Тональность: {self.style['tone']}")
            if self.style.get("avg_sentence_length"):
                lines.append(f"- Средняя длина предложений: {self.style['avg_sentence_length']} слов")
            if self.style.get("transition_phrases"):
                lines.append(f"- Характерные переходы: {', '.join(self.style['transition_phrases'])}")
            if self.style.get("forbidden_patterns"):
                lines.append(f"- ЗАПРЕЩЕНО: {', '.join(self.style['forbidden_patterns'])}")
            if self.style.get("expertise_signals"):
                lines.append(f"- Сигналы экспертности: {', '.join(self.style['expertise_signals'])}")

        return "\n".join(lines) + "\n\n"

    def _validate_final(self, state: PipelineState) -> list:
        """
        Финальная валидация статьи (без API).
        Проверяет наличие штампов и лимиты объема с учетом динамического маржина.
        """
        warnings = []
        text = state.final_article or state.draft
        if not text:
            warnings.append("⚠️ Финальный текст статьи пуст.")
            return warnings

        # 1. Проверка на стоп-слова
        from .stopwords import ALL_STOP_WORDS
        lower_text = text.lower()
        found_words = [w for w in ALL_STOP_WORDS if w in lower_text]
        if found_words:
            unique_words = list(set(found_words))
            warnings.append(f"⚠️ В финальной статье обнаружены стоп-слова: {unique_words[:10]}")

        # 2. Проверка объема
        target = state.custom_chars
        if not target:
            if state.style_id:
                try:
                    from .styles import get_style
                    target = get_style(state.style_id).target_chars
                except Exception:
                    pass
        if not target:
            from .patterns import PATTERNS
            target = PATTERNS.get(state.article_type, PATTERNS.get("free_style", {})).get("target_chars", 8000)

        # Вычисляем динамический margin
        if target < 5000:
            margin_pct = 0.30
        else:
            margin_pct = 0.20

        min_allowed = int(target * (1.0 - margin_pct))
        max_allowed = int(target * (1.0 + margin_pct))
        actual = len(text)

        if actual < min_allowed:
            warnings.append(f"⚠️ Объем статьи ({actual} симв.) ниже допустимого минимума ({min_allowed} симв., допуск -{int(margin_pct*100)}%).")
        elif actual > max_allowed:
            warnings.append(f"⚠️ Объем статьи ({actual} симв.) выше допустимого максимума ({max_allowed} симв., допуск +{int(margin_pct*100)}%).")

        return warnings

