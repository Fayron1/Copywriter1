"""
Pipeline — Оркестратор мультиагентной генерации статей.

Управляет полным циклом (единый режим QUALITY_MODE):
  Topic → Brain → Fact-Finder → Freshness → Scout → Engineer
  → Plan-Critique (Ревизор) → Heart
  → Surgical Edit Loop (Ревизор ↔ хирургические правки, до 3 итераций, Accept-Best,
    + одноразовая жёсткая эскалация при низком балле правдоподобия)
  → Booster → Statistical Humanize → Smart Hard-Cut → Artist → Финальная статья

Persona & Scale Lock: Brain фиксирует паспорт ЦА (роль, масштаб, реалистичные ставки,
primary_intent, topic_class/legal_density). Паспорт наследуется Heart (стиль-блок) и
Ревизором (проверки правдоподобия/масштаба + перенесённые проверки Sheriff/Mirror).

Веб-поиск: SearXNG c retry+backoff и пулом инстансов; при отказе — fallback на
Gemini Grounding; деградация прозрачна через state.degraded_search.

Примечание: простой режим (Sheriff/Mirror/_heart_patch) выведен из эксплуатации.
Методы _step_sheriff / _step_mirror / _heart_patch оставлены в коде, но не вызываются;
их ценные проверки перенесены в системный промпт Ревизора в Surgical Edit Loop.

Использование:
    from agents.pipeline import Pipeline
    pipe = Pipeline(openai_api_key="sk-...")
    result = pipe.run(topic="Как открыть ООО в 2026", article_type="case_study")
"""
import json
import time
import os
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .registry import get_agent, AGENTS, MODELS, get_text_model_ids
from .prompts import get_system_prompt
from .rag import query_knowledge, format_rag_context
from .patterns import PATTERNS
from .freshness import check_facts as _freshness_check
from . import factcheck as _factcheck

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
    min_chars: int = 0                  # Минимальный лимит символов статьи
    max_chars: int = 0                  # Максимальный лимит символов статьи
    output_dir: str = ""                # Папка для сохранения результатов (включая картинки)
    seo_budget: int = 0                 # Бюджет символов для Booster (SEO-резерв)
    size: str = "short"                 # Размер: short (до 10k) / long (до 30k)
    keywords: List[str] = field(default_factory=list) # Ключевые слова для SEO
    seo_instructions: Dict[str, str] = field(default_factory=dict) # Динамический SEO-слой
    density_config: Dict[str, str] = field(default_factory=dict) # Динамическая плотность текста
    num_checklist_items: int = 10       # Количество пунктов для чек-листа
    quality_mode: bool = False          # Включение режима супер-качества QUALITY_MODE

    # Выходы агентов
    brain_output: Dict = field(default_factory=dict)
    persona_lock: Dict = field(default_factory=dict)  # Паспорт ЦА (Persona & Scale Lock): роль, масштаб, реалистичные ставки
    degraded_search: bool = False       # True, если веб-поиск деградировал (SearXNG недоступен / сработал fallback)
    facts: Dict = field(default_factory=dict)
    facts_original: Dict = field(default_factory=dict)  # бэкап фактов до проверки актуальности
    freshness_changes: list = field(default_factory=list)  # журнал правок актуальности
    verified_facts: Dict = field(default_factory=dict)  # Вариант B: результаты мульти-источниковой верификации {claim_norm: {verdict, sources, needs_hedging}}
    unsupported_claims: list = field(default_factory=list)  # Вариант A: неподтверждённые утверждения (для аудита)
    invalid_references: list = field(default_factory=list)  # Reference Validator: проблемные отсылки к статьям закона (для аудита)
    future_laws: list = field(default_factory=list)  # Темпоральный чек: нормы, ещё не вступившие в силу (для аудита)
    references: list = field(default_factory=list)  # финальные источники [{title, url, note}] для блока «Источники»
    compact_mode: bool = False  # Compact Mode: статьи до 5000 символов — упрощённая структура/SEO
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
    humanize_report: Dict = field(default_factory=dict)  # отчёт статистической хуманизации
    best_of_summary: str = ""

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

        # OpenAI — ключ приходит от вызывающей стороны (generate.py читает из окружения)
        if not openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY не задан. Укажите ключ в окружении (.env) — "
                "хардкод ключей в коде запрещён."
            )
        self.openai_client = OpenAI(api_key=openai_api_key, timeout=120.0)
        self.client = self.openai_client  # fallback

        # DeepSeek — основной провайдер генерации текста (обязателен).
        # Ключ берётся ТОЛЬКО из окружения; fallback-хардкод убран.
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY не задан. Укажите ключ в окружении (.env) — "
                "хардкод ключей в коде запрещён."
            )
        deepseek_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.deepseek_client = OpenAI(api_key=deepseek_key, base_url=deepseek_base, timeout=120.0)

        # KIE — провайдер генерации изображений (опционален; нужен только при provider="kie").
        kie_key = os.getenv("KIE_API_KEY")
        kie_base = os.getenv("KIE_API_BASE", "https://api.kie.ai/v1")
        self.kie_client = (
            OpenAI(
                api_key=kie_key,
                base_url=kie_base,
                timeout=120.0,
                default_headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
            ) if kie_key else None
        )
        # Гибрид-роутинг: kie.ai chat использует ПОМОДЕЛЬНЫЙ путь
        # https://api.kie.ai/<model>/v1/chat/completions (а не плоский /v1).
        # Клиенты создаём лениво по имени модели и кэшируем (см. _get_kie_client).
        self._kie_api_key = kie_key
        self._kie_host = os.getenv("KIE_API_HOST", "https://api.kie.ai").rstrip("/")
        self._kie_model_clients: Dict[str, Any] = {}

        self.qdrant = qdrant_client
        self.style = style_fingerprint
        self.max_sheriff = 2  # Hard cap
        self.max_mirror = 2   # Hard cap

    # ────────────────────────────────────────────
    # KIE: помодельный OpenAI-совместимый клиент (Гибрид)
    # ────────────────────────────────────────────

    def _get_kie_client(self, model_name: str):
        """OpenAI-клиент для kie.ai с ПОМОДЕЛЬНЫМ базовым путём.

        kie.ai chat ожидает per-model base: https://api.kie.ai/<model>/v1
        (SDK сам добавит /chat/completions). Плоский /v1 не работает для текста.
        Клиенты кэшируются по имени модели.
        """
        if not self._kie_api_key:
            raise RuntimeError(
                'Выбран provider="kie", но KIE_API_KEY не задан в окружении (.env).'
            )
        if not model_name:
            raise RuntimeError('provider="kie": не задано имя модели.')
        if model_name not in self._kie_model_clients:
            from openai import OpenAI
            base = f"{self._kie_host}/{model_name}/v1"
            self._kie_model_clients[model_name] = OpenAI(
                api_key=self._kie_api_key,
                base_url=base,
                timeout=120.0,
                default_headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
            )
        return self._kie_model_clients[model_name]

    # ────────────────────────────────────────────
    # Preflight: проверка моделей
    # ────────────────────────────────────────────

    def preflight(self, check_image: bool = True, provider: str = "deepseek",
                  text_model: Optional[str] = None) -> None:
        """Проверяет доступность всех моделей ДО запуска генерации (fail-fast).

        Использует листинг моделей провайдера (models.list) без платных вызовов.
        Собирает ВСЕ недоступные ID и поднимает единый RuntimeError со списком,
        чтобы неверные имена моделей вскрывались на старте, а не в середине прогона.
        """
        errors = []
        cache = {}

        def available(client):
            key = id(client)
            if key not in cache:
                try:
                    cache[key] = {m.id for m in client.models.list().data}
                except Exception:
                    cache[key] = None  # провайдер не поддерживает листинг
            return cache[key]

        def is_ok(client, model_id):
            if client is None or not model_id:
                return False
            ids = available(client)
            if ids is None:
                try:
                    client.models.retrieve(model_id)
                    return True
                except Exception:
                    return False
            return model_id in ids

        # Проверяем модели в зависимости от выбранного провайдера
        # OpenAI остаётся только для эмбеддингов (copywriter_kb/loader.py),
        # поэтому его текстовую модель проверяем лишь при provider="openai".
        prov = (provider or "deepseek").lower()
        if prov == "openai":
            om = text_model or MODELS["openai_text"]
            if not is_ok(self.openai_client, om):
                errors.append(f"OpenAI текст: модель '{om}' недоступна")
        elif prov == "kie":
            km = text_model or MODELS["kie_text"]
            try:
                if not is_ok(self._get_kie_client(km), km):
                    errors.append(f"KIE текст: модель '{km}' недоступна")
            except Exception as e:
                errors.append(f"KIE текст: {e}")
        else:  # deepseek (по умолчанию)
            ids = set(get_text_model_ids())
            if text_model:
                ids.add(text_model)
            # Исключаем KIE модели, так как они проверяются отдельно/не через DeepSeek
            kie_models = {MODELS["external_reviewer"], MODELS["kie_text"], "gpt-5.5", "gemini-3.1-pro", "gpt-5-5-openai-resp", "claude-opus-4-8"}
            ids = ids - kie_models
            for model_id in sorted(ids):
                if not is_ok(self.deepseek_client, model_id):
                    errors.append(f"DeepSeek (агент): модель '{model_id}' недоступна")

        # Изображения: достаточно, чтобы работала ХОТЯ БЫ ОДНА из пары primary/fallback
        if check_image:
            img_p = MODELS["openai_image_primary"]
            img_f = MODELS["openai_image_fallback"]
            
            img_p_ok = False
            if self._kie_api_key:
                if img_p.startswith("gpt-image"):
                    img_p_ok = True
                else:
                    try:
                        img_p_ok = is_ok(self._get_kie_client(img_p), img_p)
                    except Exception:
                        img_p_ok = False
            else:
                img_p_ok = is_ok(self.openai_client, img_p)
                
            img_f_ok = is_ok(self.openai_client, img_f)
            
            if not (img_p_ok or img_f_ok):
                errors.append(f"Изображения: ни '{img_p}' через KIE, ни '{img_f}' через OpenAI недоступны")

        # Health-check веб-поиска (не фатально): предупреждаем заранее, что статья
        # может быть написана без свежих данных, и есть ли рабочий fallback.
        try:
            from .searxng import health_check, _grounding_available
            if not health_check():
                if _grounding_available():
                    logger.warning("   ⚠️ SearXNG недоступен на preflight — будет использован fallback (Gemini Grounding).")
                else:
                    logger.warning("   ⚠️ SearXNG недоступен и fallback (KIE_API_KEY) не задан — статья может быть без свежих данных (degraded_search).")
            else:
                logger.info("   ✅ SearXNG доступен")
        except Exception as _e:
            logger.warning(f"   ⚠️ Не удалось проверить доступность SearXNG: {_e}")

        # Проверка размеров и лимитов из PATTERNS (загруженных из styles_config.json)
        from .patterns import PATTERNS
        for atype, pat in PATTERNS.items():
            target = pat.get("target_chars", 0)
            min_c = pat.get("min_chars", 0)
            max_c = pat.get("max_chars", 0)
            
            if min_c <= 0:
                errors.append(f"Лимиты {atype}: min_chars должен быть больше 0 (сейчас {min_c})")
            if target < min_c:
                errors.append(f"Лимиты {atype}: target_chars ({target}) не может быть меньше min_chars ({min_c})")
            if max_c < target:
                errors.append(f"Лимиты {atype}: max_chars ({max_c}) не может быть меньше target_chars ({target})")

        if errors:
            raise RuntimeError(
                "Preflight не пройден, исправьте конфигурацию:\n  - " + "\n  - ".join(errors)
            )
        logger.info("   ✅ Preflight моделей и размеров пройден")

    # ────────────────────────────────────────────
    # Главный метод
    # ────────────────────────────────────────────

    def run(
        self,
        topic: str,
        article_type: str = "analysis",
        direction: str = "",
        skip_scout: bool = False,
        skip_images: bool = True,
        style_id: str = "",
        custom_chars: int = 0,
        output_dir: Optional[str] = None,
        provider: str = "deepseek",
        model: Optional[str] = None,
        description: str = "",
        style_nuances: str = "",
        additional_instructions: str = "",
        size: str = "short",
        keywords: List[str] = None,
        quality_mode: Optional[bool] = None,
    ) -> PipelineState:
        """
        Запустить полный pipeline генерации статьи.
        """
        # QUALITY_MODE — единственный режим работы системы. Простой режим (Sheriff/Mirror)
        # выведен из эксплуатации. Параметр quality_mode и переменная окружения QUALITY_MODE
        # сохранены для обратной совместимости, но всегда трактуются как True.
        quality_mode = True

        # Вызываем смарт-роутер для выбора типа, лимитов и SEO-инструкций
        from .router import ArticleRouter
        kw_list = keywords or []
        routing_res = ArticleRouter.route(
            topic=topic,
            description=description,
            size=size,
            keywords=kw_list,
            custom_chars=custom_chars
        )
        
        article_type = routing_res["article_type"]
        custom_chars = routing_res["target_chars"]
        seo_instructions = routing_res["seo_instructions"]
        density_config = routing_res.get("density_config", {})

        # Если стиль задан явно (--style), он имеет приоритет над определением роутера.
        # Это устраняет рассогласование style_id vs article_type, когда роутер
        # классифицирует тему как analysis, а пользователь явно запросил checklist.
        if style_id and style_id != article_type:
            logger.info(f"   🔀 Стиль задан явно: {style_id}. Перекрываем тип статьи роутера ({article_type} → {style_id}).")
            article_type = style_id

        # Preflight: проверяем доступность моделей до начала генерации (fail-fast)
        self.preflight(check_image=not skip_images)

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
            min_chars=routing_res["min_chars"],
            max_chars=routing_res["max_chars"],
            output_dir=output_dir or "",
            provider=provider,
            model=model,
            description=description,
            style_nuances=style_nuances,
            additional_instructions=additional_instructions,
            status="running",
            size=size,
            keywords=kw_list,
            seo_instructions=seo_instructions,
            density_config=density_config,
            num_checklist_items=routing_res.get("num_checklist_items", 10),
            compact_mode=routing_res.get("compact_mode", False),
            quality_mode=quality_mode,
        )


        logger.info(f"\n{'='*60}")
        logger.info(f"ð PIPELINE: {topic}")
        logger.info(f"   Тип: {article_type} | Направление: {direction}")
        logger.info(f"   Режим качества (QUALITY_MODE): {'ВКЛ' if quality_mode else 'выкл'}")
        logger.info(f"{'='*60}")

        try:
            # 1. Brain — декомпозиция
            self._step_brain(state)

            # 2. Fact-Finder — факты из RAG
            self._step_fact_finder(state)

            # 2.5 Freshness — авто-проверка актуальности фактов (после Fact-Finder, до написания)
            self._step_freshness(state)

            # 2.6 Fact Verify — мульти-источниковая верификация (≥2 ист., только строгие темы)
            self._step_fact_verify(state)

            # 3. Scout — тренды (опционально)
            if not skip_scout:
                self._step_scout(state)

            # 4. Engineer — структура
            self._step_engineer(state)

            # ШАГ 2: Критика Blueprint (плана) Ревизором в QUALITY_MODE
            if state.quality_mode:
                self._step_plan_critique(state)

            # 5. Heart — написание черновика
            self._step_heart(state)
            self._log_draft_length("Heart (Черновик)", state.draft)

            # 5.5 Claim Check — извлечение утверждений + хеджирование неподтверждённых (Вариант A)
            self._step_claim_check(state)

            # 5.6 Reference Validator — валидация отсылок к статьям закона (баг ст.185.1)
            self._step_validate_references(state)

            # 5.7 Temporal Check — будущие законы не выдавать за действующие
            self._step_temporal_check(state)

            # 6. Ревью и правки — единый Surgical Edit Loop (QUALITY_MODE по умолчанию).
            # Простой режим (Sheriff + Mirror + _heart_patch) выведен из эксплуатации:
            # все статьи проходят только через хирургический цикл точечных правок.
            self._step_quality_edit_loop(state)

            # 7. Booster — SEO/GEO (только если есть текст)
            if state.draft and len(state.draft) > 100:
                self._step_booster(state)
                self._log_draft_length("Booster (SEO)", state.final_article)
                self._step_statistical_humanize(state)   # точечная хуманизация ПОСЛЕ SEO
                self._apply_smart_hard_cut(state)
                self._log_draft_length("Smart Hard-Cut (Финал)", state.final_article)

                # Финальная валидация (без API)
                validation_warnings = self._validate_final(state)
                for w in validation_warnings:
                    logger.warning(w)

                # 7.5 Сборка блока «Источники» (2–5 естественных ссылок)
                self._step_assemble_references(state)
            else:
                logger.error(f"❌ Draft пустой ({len(state.draft or '')} символов) — пропускаем Booster")
                state.final_article = state.draft or ""

            # 8. Artist (опционально)
            if not skip_images:
                self._step_artist(state)
            else:
                # Если генерация картинок отключена, вычищаем все маркеры [IMAGE_PROMPT_HERE] или [картинка]
                import re as _re_artist
                marker_pattern = r"\[(?:картинка|IMAGE_PROMPT_HERE)(?::\s*.*?)?\]"
                state.final_article = _re_artist.sub(marker_pattern, '', state.final_article)
                # Также вычищаем маркеры из seo_package (alt_texts[].marker и любые строковые поля),
                # иначе они утекают в финальный seo_package.json и помечаются как «утёкшие теги» в логах.
                self._clean_image_markers_from_seo(state)

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
        description_str = f"Описание/ТЗ от заказчика: {state.description}\n" if state.description else ""
        user_msg = (
            f"Тема: {state.topic}\n"
            f"Тип статьи: {state.article_type}\n"
            f"Направление: {state.direction}\n"
            f"{description_str}"
            f"Создай ТЗ для каждого агента."
        )
        state.brain_output = self._call_agent("brain", user_msg, state=state)
        # Persona & Scale Lock: фиксируем паспорт ЦА один раз — дальше его наследуют все агенты.
        pl = state.brain_output.get("persona_lock")
        if isinstance(pl, dict) and pl:
            state.persona_lock = pl
        logger.info(f"   📌 Persona Lock: {self._persona_summary(state.persona_lock)}")
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

        # Резервный веб-поиск в QUALITY_MODE при нехватке RAG-фактов
        if state.quality_mode and len(chunks) < 3:
            logger.info("      ⚠️ RAG вернул мало фактов (< 3). Запускаем резервный веб-поиск в SearXNG...")
            search_prompt = (
                f"Мы пишем юридическую/B2B статью.\n"
                f"Тема: {state.topic}\n"
                f"Направление: {state.direction}\n"
                f"Задание: {task}\n\n"
                f"Нам не хватает информации в локальной базе знаний. Сгенерируй 2-3 точечных поисковых запроса "
                f"для Яндекса/Google/SearXNG, чтобы найти точные юридические факты, законы, статьи кодексов "
                f"или судебные кейсы по этой теме.\n"
                f"Ответь строго в формате JSON: \n"
                f"{{\n"
                f"  \"queries\": [\"запрос 1\", \"запрос 2\", \"запрос 3\"]\n"
                f"}}\n"
                f"Не используй тире в ответах. Верни только JSON."
            )
            try:
                queries_res = self._call_agent("fact_finder", search_prompt, parse_json=True, state=state)
                queries = []
                if isinstance(queries_res, dict) and "queries" in queries_res:
                    queries = queries_res["queries"]
                elif isinstance(queries_res, list):
                    queries = queries_res
                
                if not queries:
                    queries = [state.topic, f"{state.topic} закон", f"{state.topic} судебная практика"]
                
                import time
                from .searxng import web_search
                web_snippets = []
                for idx_q, q in enumerate(queries[:3]):
                    if idx_q > 0:
                        time.sleep(1.5)  # задержка против rate limit SearXNG
                    logger.info(f"      🔍 Веб-поиск запрос: {q}")
                    results, meta = web_search(q)
                    if meta.get("degraded"):
                        state.degraded_search = True
                    if results:
                        for idx, r in enumerate(results[:5]):
                            web_snippets.append({
                                "text": f"Заголовок: {r['title']}\nСниппет: {r['snippet']}",
                                "source_file": r['url'],
                                "source_type": "web_search",
                                "score": 1.0
                            })
                
                if web_snippets:
                    logger.info(f"      ✅ Найдено {len(web_snippets)} веб-сниппетов для RAG-контекста.")
                    rag_context += "\n\n=== РЕЗЕРВНЫЙ ВЕБ-ПОИСК (SearXNG) ===\n"
                    for ws in web_snippets:
                        rag_context += f"\nИсточник: {ws['source_file']}\nТекст:\n{ws['text']}\n---"
            except Exception as e:
                logger.warning(f"      ⚠️ Ошибка при выполнении резервного веб-поиска: {e}")

        user_msg = (
            f"Задание от Оркестратора: {task}\n"
            f"Тема: {state.topic}\n"
            f"Направление: {state.direction}\n\n"
            f"{rag_context}\n\n"
            f"Найди и структурируй все релевантные факты."
        )
        state.facts = self._call_agent("fact_finder", user_msg, state=state)

        # Постфильтр: отбрасываем факты с очень низкой надёжностью (secondary без URL).
        # Это детерминированная защита от мусорных сниппетов до того, как они попадут в Heart.
        self._filter_low_reliability_facts(state)

        state.steps_completed.append("fact_finder")

    def _filter_low_reliability_facts(self, state: PipelineState):
        """Постфильтр фактов: отбрасываем низконадёжные (reliability < 0.5, secondary без URL).

        Детерминированная защита от мусорных сниппетов/UGC, которые Fact-Finder
        мог собрать из интернета. Факты с пометкой _filtered=True не удаляются
        полностью (для аудита), но Heart получает очищенный набор.
        """
        facts = state.facts
        if not isinstance(facts, dict):
            return
        items = facts.get("facts")
        if not isinstance(items, list):
            return

        filtered_count = 0
        for f in items:
            if not isinstance(f, dict):
                continue
            reliability = _factcheck._to_float(f.get("reliability", 1.0))
            source_class = str(f.get("source_class", "")).strip().lower()
            source_url = str(f.get("source_url", "")).strip()

            # Отбрасываем: низкая надёжность + вторичный источник + без URL
            if reliability < 0.5 and source_class == "secondary" and not source_url.startswith("http"):
                f["_filtered"] = True
                filtered_count += 1
            # Также помечаем secondary без URL с middling надёжностью
            elif reliability < 0.7 and source_class == "secondary" and not source_url.startswith("http"):
                # Понижаем reliability и ставим риск high, но не удаляем
                f["reliability"] = 0.4
                f["risk"] = "high"
                f["_demoted"] = True

        if filtered_count > 0:
            logger.info(f"   🧹 [factcheck] отфильтровано {filtered_count} низконадёжных фактов (reliability < 0.5, secondary, без URL)")

    def _step_freshness(self, state: PipelineState):
        """Шаг 2.5: авто-проверка актуальности фактов через kie.ai + Google Search.

        Полностью автоматический, без ручного участия. Заменяет устаревшие
        значения только при высокой уверенности и наличии источника. Любая
        ошибка не ломает пайплайн — факты остаются исходными.
        """
        try:
            if not state.facts or not isinstance(state.facts, dict):
                return
            updated, changes = _freshness_check(state.facts)
            state.freshness_changes = changes
            if changes:
                state.facts_original = state.facts  # бэкап для аудита
                state.facts = updated
            state.steps_completed.append("freshness")
        except Exception as e:
            logger.warning(f"⚠️ [freshness] шаг пропущен из-за ошибки: {e}")

    def _step_fact_verify(self, state: PipelineState):
        """Шаг 2.6: мульти-источниковая верификация ключевых фактов (≥2 источника).

        Только для строгих тем (юр/налог/фин). Проверяет числовые факты (ставки,
        лимиты, суммы, статьи законов) по нескольким независимым источникам через
        Google Search Grounding. Факт считается verified только при ≥2 источниках
        из ≥2 разных доменов. Все остальные помечаются needs_hedging=True.

        Безопасно: при любом сбое — шаг пропускается, пайплайн не ломается.
        """
        if not self._is_strict_topic(state.topic, state.description):
            logger.info("   ℹ️ [factcheck] тема не строгая — мульти-источниковая верификация пропущена.")
            return
        try:
            if not state.facts or not isinstance(state.facts, dict):
                return
            result = _factcheck.verify_facts(state.facts)
            if result:
                state.verified_facts = result
                # Помечаем факты, которые не прошли верификацию
                for fact in (state.facts.get("facts") or []):
                    if not isinstance(fact, dict):
                        continue
                    claim_norm = _factcheck._normalize_claim(str(fact.get("claim", "")))
                    for key, info in result.items():
                        if key in claim_norm or claim_norm in key:
                            if info.get("needs_hedging"):
                                fact["_factcheck_needs_hedging"] = True
                            break
            state.steps_completed.append("fact_verify")
        except Exception as e:
            logger.warning(f"⚠️ [factcheck] верификация пропущена из-за ошибки: {e}")

    def _step_scout(self, state: PipelineState):
        """Шаг 3: Scout — тренды и актуальность."""
        logger.info("📡 [3/8] Scout: анализ трендов через интернет...")
        
        # 1. Поиск через веб (SearXNG → fallback Gemini Grounding)
        from .searxng import web_search
        
        # Ищем по теме, либо по запросу, который мог сгенерировать Оркестратор
        search_query = state.brain_output.get("search_query", state.topic)
        if not isinstance(search_query, str):
            search_query = str(search_query)
        search_results, _search_meta = web_search(search_query)
        if _search_meta.get("degraded"):
            state.degraded_search = True
            logger.warning(f"      ⚠️ Поиск деградировал (источник: {_search_meta.get('source')}). Ставлю флаг degraded_search.")
        
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
        pattern = PATTERNS.get(state.article_type) or PATTERNS.get("seo")
        
        # Загружаем индивидуальный промпт из настроек стиля
        from .styles import get_style
        try:
            style = get_style(state.style_id or state.article_type)
            engineer_inst = style.engineer_instruction or pattern['engineer_structure']
        except Exception:
            engineer_inst = pattern['engineer_structure']

        if state.style_id == "checklist" and hasattr(state, "num_checklist_items"):
            num = state.num_checklist_items
            engineer_inst = engineer_inst.replace("10 нумерованных", f"{num} нумерованных")
            engineer_inst = engineer_inst.replace("10 пунктов", f"{num} пунктов")
            engineer_inst = engineer_inst.replace("## 10.", f"## {num}.")
            engineer_inst = engineer_inst.replace("## 1. ... ## 10.", f"## 1. ... ## {num}.")

        seo_prompt = state.seo_instructions.get("engineer_instruction", "") if state.seo_instructions else ""
        density = state.density_config
        density_prompt = ""
        simplification_prompt = ""

        # Эффективный целевой объём: custom_chars (если задан) или target_chars из state.
        # Compact Mode активен при target_chars < 5000 (см. router); в pipeline он выражается
        # через state.compact_mode. Но simplification срабатывает и при явно заданном
        # custom_chars < 6000 (старое поведение) — берём объединение условий.
        eff_target = state.custom_chars if state.custom_chars > 0 else 0
        is_small = state.compact_mode or (state.custom_chars > 0 and state.custom_chars < 6000)

        # Упрощение структуры для малых объемов
        if is_small:
            vol_label = state.custom_chars if state.custom_chars > 0 else "до 5000"
            simplification_prompt = (
                f"\n\n⚠️ ВНИМАНИЕ (МАЛЫЙ ОБЪЕМ): Задан очень малый общий объем статьи: {vol_label} символов.\n"
                f"- Чтобы статья не получилась скомканной и логика не пострадала, спроектируй МАКСИМАЛЬНО лаконичную структуру.\n"
                f"- Допускается строго 2-3 содержательных раздела H2 (не более).\n"
                f"- Категорически исключи любые избыточные таблицы и кейсы (оставь только один простой пример/кейс в тексте без сложных диалогов).\n"
                f"- Каждый раздел должен быть небольшим, но законченным."
            )

        if density and (state.custom_chars > 0 or state.compact_mode):
            vol_for_quota = state.custom_chars if state.custom_chars > 0 else eff_target
            h2_limit = f"строго {density.get('h2_count', '3-4')}"
            if state.style_id == "checklist":
                h2_limit = f"строго ровно {state.num_checklist_items} разделов H2"
            elif is_small:
                h2_limit = "строго 2-3 раздела H2"

            density_prompt = (
                f"\n\n⚠️ ТРЕБОВАНИЕ К КВОТАМ СИМВОЛОВ И СТРУКТУРЕ:\n"
                f"- Общий целевой объем статьи: {vol_for_quota} символов.\n"
                f"- Количество содержательных разделов H2 (не считая H1 и FAQ): {h2_limit}.\n"
                f"- Для каждого сгенерированного подзаголовка/пункта в твоем Blueprint ОБЯЗАТЕЛЬНО укажи целевой объем символов в поле `target_chars`.\n"
                f"- Распредели суммарно {vol_for_quota} символов между всеми разделами.\n"
                f"- Ориентир: вводный хук-кейс должен планироваться на {density.get('hook_size', '1-2 абзаца')}.\n"
            )

        user_msg = (
            f"Тема: {state.topic}\n"
            f"Тип статьи: {state.article_type}\n"
            f"Направление: {state.direction}\n\n"
            f"ЭТАЛОННАЯ СТРУКТУРА:\n{engineer_inst}{seo_prompt}{density_prompt}{simplification_prompt}\n\n"
            f"ФАКТЫ ОТ ИССЛЕДОВАТЕЛЯ:\n{self._compact_json(state.facts, 5000)}\n\n"
            f"УГОЛ ПОДАЧИ ОТ РАЗВЕДЧИКА:\n{self._compact_json(state.scout_data, 2500)}\n\n"
            f"Создай детальный Blueprint."
        )

        # RAG — шаблоны и фреймворки
        chunks = query_knowledge(f"шаблон {state.article_type}", "engineer", self.qdrant)
        if chunks:
            user_msg += f"\n\n{format_rag_context(chunks, max_chars=4000)}"

        state.engineer_user_msg = user_msg
        state.blueprint = self._call_agent("engineer", user_msg, state=state)
        
        # Quality Gate: Проверка количества пунктов для чек-листа
        if state.style_id == "checklist":
            sections = self._extract_sections(state.blueprint)
            if len(sections) < state.num_checklist_items:
                logger.warning(f"⚠️ [Quality Gate] Engineer сгенерировал только {len(sections)} разделов вместо {state.num_checklist_items}. Запрашиваем структуру заново.")
                retry_msg = (
                    user_msg + 
                    f"\n\nSYSTEM ERROR: You generated only {len(sections)} sections in the blueprint. "
                    f"You MUST generate strictly exactly {state.num_checklist_items} sections in the blueprint. "
                    f"Rewrite the blueprint with exactly {state.num_checklist_items} sections."
                )
                state.blueprint = self._call_agent("engineer", retry_msg, state=state)

        # Hard Fail: Проверка Хронотопа (только для кейсов)
        if state.style_id == "case_study":
            blueprint_str = json.dumps(state.blueprint, ensure_ascii=False)
            if "[CHRONOTOPE_SCENE]" not in blueprint_str:
                logger.warning("⚠️ [Hard Fail] В структуре нет тега [CHRONOTOPE_SCENE]. Запрашиваем структуру заново.")
                retry_msg = user_msg + "\n\nSYSTEM ERROR: You forgot to include the mandatory [CHRONOTOPE_SCENE] tag in the section titles. Rewrite the blueprint and include it."
                state.blueprint = self._call_agent("engineer", retry_msg, state=state)
                
        state.steps_completed.append("engineer")

    def _step_plan_critique(self, state: PipelineState):
        """ШАГ 2: Критика Blueprint (плана структуры) ревизором до написания статьи."""
        logger.info("🕵️ [4.5] Ревизор: оценка плана статьи (Blueprint)...")
        
        target_chars = state.custom_chars or 8000
        keywords_str = ", ".join(state.keywords) if state.keywords else "нет"
        
        user_msg = (
            f"ТЕМА СТАТЬИ: {state.topic}\n"
            f"ОПИСАНИЕ/ТЗ ОТ КЛИЕНТА: {state.description or 'нет'}\n"
            f"ЦЕЛЕВОЙ ОБЪЕМ: {target_chars} символов\n"
            f"КЛЮЧЕВЫЕ СЛОВА: {keywords_str}\n\n"
            f"ТЕКУЩИЙ BLUEPRINT (ПЛАН СТРУКТУРЫ):\n"
            f"{self._compact_json(state.blueprint, 5000)}\n\n"
            f"Выполни критический анализ этого плана. Проверь:\n"
            f"1. Нет ли дублирования тем между H2-разделами.\n"
            f"2. Раскрывает ли план ТЗ и описание от клиента (все ли важные юридические/бизнес-аспекты на месте).\n"
            f"3. Подходит ли количество разделов под целевой объем (для малых объемов < 6000 должно быть 2-3 раздела, для checklist - ровно {state.num_checklist_items} разделов).\\n\\n"
            f"Верни ответ строго в формате JSON:\n"
            f"{{\n"
            f"  \"ok\": true | false,\n"
            f"  \"issues\": [\n"
            f"    {{\"section\": \"название H2 или индекс\", \"problem\": \"описание проблемы\", \"fix\": \"как исправить\"}}\n"
            f"  ],\n"
            f"  \"missing_topics\": [\"пропущенные важные темы, которые нужно добавить\"],\n"
            f"  \"redundant_sections\": [\"лишние или дублирующие разделы, которые нужно убрать\"]\n"
            f"}}"
        )
        
        try:
            response = self._call_agent("external_reviewer", user_msg, parse_json=True, state=state)
        except Exception as e:
            logger.warning(f"   ⚠️ Не удалось вызвать внешнего ревизора для критики плана: {e}. Пропускаю.")
            response = {"ok": True}
        
        if isinstance(response, dict) and not response.get("ok", True):
            issues = response.get("issues", [])
            missing = response.get("missing_topics", [])
            redundant = response.get("redundant_sections", [])
            
            if issues or missing or redundant:
                logger.info(f"   🕵️ Ревизор нашёл замечания к плану. Пересобираем Blueprint с учётом правок...")
                
                # Загружаем сохраненный исходный промпт для повторного вызова
                original_user_msg = getattr(state, "engineer_user_msg", None)
                if not original_user_msg:
                    original_user_msg = (
                        f"Тема: {state.topic}\n"
                        f"Направление: {state.direction}\n\n"
                        f"ФАКТЫ:\n{self._compact_json(state.facts, 5000)}\n"
                    )
                
                retry_msg = (
                    original_user_msg +
                    f"\n\nSYSTEM ALERT: Ревизор забраковал твой первоначальный Blueprint. "
                    f"Пожалуйста, перепиши его, исправив следующие замечания (и СТРОГО сохрани все исходные ограничения структуры, включая количество разделов H2):\n"
                )
                for issue in issues:
                    retry_msg += f"- Раздел '{issue.get('section')}': {issue.get('problem')}. Рекомендация: {issue.get('fix')}\n"
                if missing:
                    retry_msg += f"- Пропущенные важные темы (ОБЯЗАТЕЛЬНО добавь их в структуру): {', '.join(missing)}\n"
                if redundant:
                    retry_msg += f"- Лишние/дублирующие темы (ОБЯЗАТЕЛЬНО удали или объедини): {', '.join(redundant)}\n"
                    
                retry_msg += "\nСделай один чистый повторный проход и верни новый скорректированный Blueprint в JSON."
                
                state.blueprint = self._call_agent("engineer", retry_msg, state=state)
                logger.info("   ✅ Blueprint успешно пересобран с учётом замечаний Ревизора.")
        else:
            logger.info("   ✅ Ревизор одобрил план статьи.")
            
        state.steps_completed.append("plan_critique")

    def _is_strict_topic(self, topic: str, description: str = "") -> bool:
        """True, если тема строгая (юр/налог/фин) — для неё нужна мульти-источниковая верификация.

        Нормализация ё→е: в темах пишут «учёте»/«счёт», а в словаре может быть «учет».
        Без нормализации трудовые темы вроде «Оплата сверхурочных при суммированном учёте»
        не распознавались как строгие → Fact Verify и Reference Validator отключались.
        """
        # Нормализуем ё→е для устойчивости сравнения
        topic_lower = (topic or "").lower().replace("ё", "е")
        desc_lower = (description or "").lower().replace("ё", "е")
        strict_keywords = [
            "налог", "ндс", "ндфл", "уфнс", "фнс", "закон", "кодекс", "115-фз",
            "54.1", "нк рф", "тк рф", "субсидиар", "ооо", "ип", "трудов", "спор", "суд",
            "коап", "штраф", "проверк", "банкрот", "сделок", "договор", "контрагент",
            "бухгалтер", "учет", "финанс", "аудит", "блокировк", "пени", "пошлин",
            "ставка", "лимит", "порог", "мрот", "усн", "патент", "самозанят", "нпд",
            # Трудовое право (для тем про сверхурочные, отпуска, зарплаты):
            "сверхурочн", "переработ", "оплат", "зарплат", "рабочего времени",
            "суммированн", "отпуск", "увольнен", "компенсаци", "надбавк",
            "преми", "выплат", "график", "сменн", "ночных", "больничн",
        ]
        return any(kw in topic_lower or kw in desc_lower for kw in strict_keywords)

    def _suggest_draft_model(self, topic: str, article_type: str, description: str) -> dict:
        """Определяет оптимального провайдера и модель на основе темы, стиля и ТЗ."""
        topic_lower = topic.lower()
        desc_lower = (description or "").lower()

        is_strict = self._is_strict_topic(topic, description)

        # Список ключевых слов для мягких тем (HR, управление, мотивация)
        soft_keywords = [
            "управлен", "делегирова", "мотивац", "hr", "кадр", "сотрудник", "команд",
            "лидер", "руководител", "психолог", "атмосфер", "выгоран", "опыт", "истори",
            "карьер", "собеседован", "бизнес-журнал", "колонка", "мнение", "эксперт"
        ]

        is_soft = any(kw in topic_lower or kw in desc_lower for kw in soft_keywords)

        # Стили по умолчанию
        if article_type in ("checklist", "reference", "seo") or is_strict:
            # Для строгих тем, чек-листов и справочников - DeepSeek Pro
            return {"provider": "deepseek", "model": None}
        elif article_type in ("case_study", "opinion") or is_soft:
            # Для кейсов, колонок эксперта и мягких тем - Claude
            if self._kie_api_key:
                return {"provider": "kie", "model": "claude-opus-4-8"}
            else:
                # Fallback при отсутствии KIE API ключа
                return {"provider": "deepseek", "model": None}
        else:
            # По умолчанию - DeepSeek Pro
            return {"provider": "deepseek", "model": None}

    def _step_heart(self, state: PipelineState):
        """Шаг 5: Heart — написание черновика.

        Для лонгридов (>20000 целевых символов после резерва) использует
        посекционную генерацию: Heart пишет каждый раздел
        из blueprint отдельным вызовом, затем собирает.
        """
        logger.info("✍️ [5/8] Heart: написание текста...")
        style_block = self._get_style_block(state)
        
        # Накладываем SEO-инструкции для Heart
        if state.seo_instructions and state.seo_instructions.get("heart_instruction"):
            style_block += state.seo_instructions["heart_instruction"]

        # Определяем целевой и максимальный объём напрямую из состояния
        target_chars = state.custom_chars or 8000
        min_chars = state.min_chars or int(target_chars * 0.85)
        max_chars = state.max_chars or int(target_chars * 1.15)

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

        # Дополнительно извлекаем премиальные B2B образцы стиля из базы знаний (Few shot anchors)
        style_chunks = query_knowledge("пример премиального B2B текста", "heart", self.qdrant, extra_filters={"chunk_type": "style_anchor"})
        if not style_chunks:
            style_chunks = query_knowledge("B2B экспертный стиль копирайтинга примеры", "heart", self.qdrant)
        if style_chunks:
            anchors = []
            for idx, c in enumerate(style_chunks[:3], 1):
                anchors.append(f"Образец {idx}:\n{c['text']}")
            rag_block += "\n\n=== РЕАЛЬНЫЕ ОБРАЗЦЫ ПРЕМИАЛЬНОГО B2B СТИЛЯ (ДЛЯ КОПИРОВАНИЯ ИНТОНАЦИИ) ===\n" + "\n".join(anchors)

        # Автоподбор модели для черновика
        suggested = self._suggest_draft_model(state.topic, state.article_type, state.description)
        is_default_run = (state.provider == "deepseek" and state.model is None)
        
        if is_default_run:
            active_provider = suggested["provider"]
            active_model = suggested["model"]
            logger.info(f"   🤖 Автоподбор модели черновика: {active_provider} / {active_model or 'default'}")
        else:
            active_provider = state.provider
            active_model = state.model
            logger.info(f"   🤖 Явно заданная модель черновика: {active_provider} / {active_model or 'default'}")

        feedback_msg = ""
        for attempt in range(1, 4):
            logger.info(f"   🎬 Попытка генерации черновика {attempt}/3...")
            current_style_block = style_block + feedback_msg
            
            if heart_target > 20000:
                state.draft = self._heart_sectional(
                    state, current_style_block, rag_block, heart_target,
                    override_model=active_model, override_provider=active_provider
                )
            else:
                state.draft = self._heart_single(
                    state, current_style_block, rag_block, heart_target,
                    override_model=active_model, override_provider=active_provider
                )

            # Применяем Sanity-постпроцессор очистки артефактов
            state.draft = self._clean_leaked_ai_artifacts(state.draft)

            # 1. Проверяем баззворды в начале
            paragraphs = [p.strip() for p in state.draft.split("\n\n") if p.strip()]
            matched_buzzwords = []
            if paragraphs:
                first_two_paragraphs = " ".join(paragraphs[:2]).lower()
                import re as _re_buzz
                ai_buzzword_patterns = [
                    r"\bпромпт\w*",
                    r"\bprompt\w*",
                    r"\bстоп[-\s]?(?:строк|слов|фраз)\w*",
                    r"\bреестр\w*\s+статус\w*",
                    r"\bколонк\w*\s+таблиц\w*",
                    r"\bбаз\w*\s+знаний",
                    r"\bшаблон\w*\s+(?:промпт|генерац|стать)\w*",
                    r"\b(?:ии|ai)[-\s]?агент\w*",
                    r"\bструктурировщик\w*",
                    r"\bblueprint\w*",
                    r"\bгенерац\w*\s+(?:стать|контент|текст)\w*",
                    r"\bsystem\s+prompt\b",
                ]
                for _bp in ai_buzzword_patterns:
                    _bm = _re_buzz.search(_bp, first_two_paragraphs, _re_buzz.IGNORECASE)
                    if _bm:
                        matched_buzzwords.append(_bm.group(0))

            if len(matched_buzzwords) >= 2:
                logger.warning(f"   ⚠️ [Sanity Check Failed] В начале статьи обнаружена служебная ИИ-лексика: {matched_buzzwords}.")
                feedback_msg = (
                    f"\n\nSYSTEM WARNING: Your previous draft started with internal AI methodology words: {matched_buzzwords}. "
                    f"DO NOT write anything about internal regulations, columns, tables, registries, databases, checklists, rules, or instructions. "
                    f"Start the article directly with a highly engaging paragraph answering: what, who, and what benefit. "
                    f"Вот начало текста, который ты сгенерировал:\n"
                    f"=== НАЧАЛО ПРЕДЫДУЩЕГО ТЕКСТА ===\n{state.draft[:2000]}...\n=== КОНЕЦ ПРЕДЫДУЩЕГО ТЕКСТА ===\n\n"
                    f"Перепиши текст заново на основе этого фрагмента, исправив начало согласно правилу."
                )
                continue

            # 2. Проверяем критические ошибки через _validate_final
            # Временно записываем draft в final_article
            old_final = state.final_article
            state.final_article = state.draft
            try:
                validation_warnings = self._validate_final(state)
            finally:
                state.final_article = old_final

            red_errors = [w for w in validation_warnings if w.startswith("🔴")]
            if red_errors:
                logger.warning(f"   ⚠️ [Validation Check Failed] Найдено {len(red_errors)} критических ошибок: {red_errors}")
                feedback_msg = (
                    f"\n\nSYSTEM WARNING (Предыдущая попытка генерации не прошла валидацию):\n"
                    f"При генерации текста возникли следующие критические ошибки, которые необходимо исправить:\n"
                    f"- " + "\n- ".join(red_errors) + "\n"
                    f"Вот текст, который ты сгенерировал:\n"
                    f"=== НАЧАЛО ПРЕДЫДУЩЕГО ТЕКСТА ===\n{state.draft}\n=== КОНЕЦ ПРЕДЫДУЩЕГО ТЕКСТА ===\n\n"
                    f"Напиши текст заново на основе предыдущего текста, строго исправив данные ошибки и сохранив все разделы!"
                )
                continue

            # Если прошли обе проверки — выходим из цикла
            logger.info("   ✅ Черновик успешно прошел валидацию и sanity-проверку")
            break
        else:
            logger.error("   ❌ Не удалось сгенерировать черновик без ошибок за 3 попытки. Оставляем последний вариант.")

        # Глобальный condense отключен во избежание "качелей". Длина контролируется посекционно.

        # Нормализация markdown-маркеров списков в чек-листах:
        # Heart иногда пишет пункты действий с отступом из пробелов вместо "- ".
        state.draft = self._normalize_checklist_bullets(state.draft, state)

        state.steps_completed.append("heart")
        logger.info(f"   🎯 Draft: {len(state.draft)} символов")

    def _step_claim_check(self, state: PipelineState):
        """Шаг 5.5: извлечение утверждений из черновика + хеджирование неподтверждённых.

        Вариант A (всегда включён). Три фазы:
        1. LLM извлекает все проверяемые утверждения (цифры, ставки, законы, даты).
        2. Программная сверка с state.facts и state.verified_facts (0 токенов).
        3. Если есть unsupported — LLM переписывает их в осторожную (хеджированную) форму.
        Текст заменяется точечно (original→hedged), H2-структура не трогается.

        Безопасно: при любом сбое — шаг пропускается, черновик не меняется.
        """
        try:
            draft = state.draft or ""
            if not draft or len(draft) < 200:
                return

            # Фаза 1: извлечение утверждений
            claims = _factcheck.extract_claims(draft)
            if not claims:
                logger.info("   ℹ️ [factcheck] Claim Extractor: проверяемых утверждений не найдено.")
                state.steps_completed.append("claim_check")
                return

            # Фаза 2: сверка с фактами (детерминированная, 0 токенов)
            supported, unsupported = _factcheck.match_claims_to_facts(
                claims, state.facts, state.verified_facts
            )

            # Фаза 3: хеджирование неподтверждённых
            if unsupported:
                # В компактном режиме ограничиваем число хеджей, чтобы не захлебнуть короткий текст.
                hedge_batch = unsupported[:4] if getattr(state, "compact_mode", False) else unsupported
                logger.info(f"   🛡️ [factcheck] хеджирование {len(hedge_batch)} неподтверждённых утверждений...")
                hedges = _factcheck.hedge_claims(hedge_batch)

                if hedges:
                    applied = 0
                    for h in hedges:
                        orig = h["original"]
                        hedged = h["hedged"]
                        # Заменяем первое вхождение точной строки (не глобально —
                        # одна формулировка может встретиться один раз)
                        if orig in draft:
                            draft = draft.replace(orig, hedged, 1)
                            applied += 1
                        else:
                            # Fallback: пробуем менее строгую замену (без крайних пробелов)
                            orig_stripped = orig.strip()
                            if orig_stripped in draft and orig_stripped != orig:
                                draft = draft.replace(orig_stripped, hedged, 1)
                                applied += 1

                    if applied > 0:
                        state.draft = draft
                        logger.info(f"   ✅ [factcheck] применено {applied} хеджирований в черновике.")
                        # Сохраняем для аудита
                        state.unsupported_claims = [
                            {"original": h["original"], "hedged": h["hedged"]}
                            for h in hedges
                        ]
                else:
                    logger.info("   ℹ️ [factcheck] хеджирование не дало результатов.")
            else:
                logger.info(f"   ✅ [factcheck] все {len(claims)} утверждений подтверждены фактами.")

            state.steps_completed.append("claim_check")
        except Exception as e:
            logger.warning(f"⚠️ [factcheck] Claim Check пропущен из-за ошибки: {e}")

    def _step_validate_references(self, state: PipelineState):
        """Шаг 5.6: валидация отсылок к статьям закона на предмет-тематическое соответствие.

        Reference Validator. Только для строгих тем (юр/налог/фин), т.к. только там
        встречаются отсылки к статьям закона. Ловит баги вида «допдень за переработку
        по ст. 185.1 ТК РФ», где 185.1 — это диспансеризация, а не сверхурочные.

        Алгоритм:
        1. Детерминированное извлечение всех отсылок (0 токенов).
        2. Один grounded LLM-вызов проверяет каждую: существует ли статья и регулирует
           ли тему контекста. Возвращаются только проблемные (wrong_topic/not_found).
        3. Хеджирование: заменяем неверную статью на correction (если найдена) или
           смягчаем до общей отсылки к кодексу. Точечный text.replace().

        Безопасно: при любом сбое — шаг пропускается, черновик не меняется.
        """
        if not self._is_strict_topic(state.topic, state.description):
            logger.info("   ℹ️ [factcheck] тема не строгая — валидация отсылок пропущена.")
            return
        try:
            draft = state.draft or ""
            if not draft or len(draft) < 200:
                return

            # Фазы 1+2: извлечение и валидация отсылок
            problems = _factcheck.validate_law_references(draft)
            if not problems:
                logger.info("   ✅ [factcheck] проблемных отсылок к статьям закона не найдено.")
                state.steps_completed.append("validate_references")
                return

            logger.info(f"   ⚠️ [factcheck] найдено {len(problems)} проблемных отсылок к статьям закона.")

            # Сохраняем для аудита
            state.invalid_references = [
                {
                    "citation": p.get("citation", ""),
                    "verdict": p.get("verdict", ""),
                    "actual_topic": p.get("actual_topic", ""),
                    "correction": p.get("correction", ""),
                    "confidence": p.get("confidence", 0),
                }
                for p in problems
            ]

            # Фаза 3: хеджирование проблемных отсылок
            hedges = _factcheck.hedge_references(problems)
            if hedges:
                applied = 0
                for h in hedges:
                    orig = h["original"]
                    hedged = h["hedged"]
                    if orig in draft:
                        draft = draft.replace(orig, hedged, 1)
                        applied += 1
                    else:
                        # Fallback: пробуем без крайних пробелов
                        orig_stripped = orig.strip()
                        if orig_stripped in draft and orig_stripped != orig:
                            draft = draft.replace(orig_stripped, hedged, 1)
                            applied += 1

                if applied > 0:
                    state.draft = draft
                    logger.info(f"   ✅ [factcheck] применено {applied} исправлений отсылок к статьям закона.")
            else:
                logger.info("   ℹ️ [factcheck] хеджирование отсылок не дало результатов.")

            state.steps_completed.append("validate_references")
        except Exception as e:
            logger.warning(f"⚠️ [factcheck] Reference Validator пропущен из-за ошибки: {e}")

    def _step_temporal_check(self, state: PipelineState):
        """Шаг 5.7: темпоральный чек — будущие законы не выдавать за действующие.

        Только для строгих тем. Находит в черновике правовые нормы, вступающие в силу
        ПОЗЖЕ сегодня (ещё не действующие), и хеджирует их в будущее время:
        «статья действует в новой редакции» → «статья вступит в силу в новой редакции
        с <дата>; до этого применяется <текущий порядок>».

        Безопасно: при любом сбое — шаг пропускается, черновик не меняется.
        """
        if not self._is_strict_topic(state.topic, state.description):
            logger.info("   ℹ️ [factcheck] тема не строгая — темпоральный чек пропущен.")
            return
        try:
            draft = state.draft or ""
            if not draft or len(draft) < 200:
                return

            import datetime
            today = datetime.datetime.now().strftime("%Y-%m-%d")

            problems = _factcheck.check_future_laws(draft, today)
            if not problems:
                logger.info("   ✅ [factcheck] норм, ещё не вступивших в силу, не найдено.")
                state.steps_completed.append("temporal_check")
                return

            logger.info(f"   ⚠️ [factcheck] найдено {len(problems)} будущих норм (ещё не в силе).")
            # Сохраняем для аудита
            state.future_laws = [
                {
                    "citation": p.get("citation", ""),
                    "effective_date": p.get("effective_date", ""),
                    "current_rule": p.get("current_rule", ""),
                    "confidence": p.get("confidence", 0),
                }
                for p in problems
            ]

            hedges = _factcheck.hedge_future_laws(problems, today)
            if hedges:
                applied = 0
                for h in hedges:
                    orig = h["original"]
                    hedged = h["hedged"]
                    if orig in draft:
                        draft = draft.replace(orig, hedged, 1)
                        applied += 1
                    else:
                        orig_stripped = orig.strip()
                        if orig_stripped in draft and orig_stripped != orig:
                            draft = draft.replace(orig_stripped, hedged, 1)
                            applied += 1

                if applied > 0:
                    state.draft = draft
                    logger.info(f"   ✅ [factcheck] применено {applied} темпоральных хеджей (будущее время).")
            else:
                logger.info("   ℹ️ [factcheck] хеджирование будущих норм не дало результатов.")

            state.steps_completed.append("temporal_check")
        except Exception as e:
            logger.warning(f"⚠️ [factcheck] темпоральный чек пропущен из-за ошибки: {e}")

    def _step_assemble_references(self, state: PipelineState):
        """Шаг 7.5: сборка блока «Источники» (2–5 естественных ссылок).

        Собирает пул URL из verified_facts, facts (source_url) и scout_data,
        ранжирует по авторитетности домена, берёт top 2–5, формирует markdown-блок
        ## Источники в конце статьи. НЕ ГОСТ — естественный формат: markdown-ссылки
        с кратким описанием.

        Безопасно: при любом сбое — блок не добавляется.
        """
        try:
            pool: Dict[str, dict] = {}  # url → {title, note, authority}

            # 1. Из verified_facts (высший приоритет — прошли мульти-источниковую проверку)
            for key, info in (state.verified_facts or {}).items():
                if not isinstance(info, dict):
                    continue
                for s in (info.get("sources") or []):
                    if not isinstance(s, dict):
                        continue
                    url = str(s.get("url", "")).strip()
                    if not url.startswith("http"):
                        continue
                    auth = _factcheck.domain_authority(url)
                    title = str(s.get("title", "")).strip()
                    snippet = str(s.get("snippet", "")).strip()
                    if url not in pool or auth > pool[url].get("authority", 0):
                        pool[url] = {"title": title, "note": _factcheck._truncate_at_word(snippet, 120), "authority": auth}

            # 2. Из facts (source_url из Fact-Finder)
            for fact in ((state.facts or {}).get("facts") or []):
                if not isinstance(fact, dict):
                    continue
                url = str(fact.get("source_url", "")).strip()
                if not url.startswith("http"):
                    continue
                auth = _factcheck.domain_authority(url)
                title = str(fact.get("source", "")).strip()  # именованное описание
                claim = _factcheck._truncate_at_word(str(fact.get("claim", "")).strip(), 120)
                if url not in pool or auth > pool[url].get("authority", 0):
                    pool[url] = {"title": title, "note": claim, "authority": auth}

            # 3. Из scout_data (источники из интернета)
            for s in ((state.scout_data or {}).get("sources") or []):
                if not isinstance(s, dict):
                    continue
                url = str(s.get("url", "")).strip()
                if not url.startswith("http"):
                    continue
                auth = _factcheck.domain_authority(url)
                title = str(s.get("title", "")).strip()
                if url not in pool or auth > pool[url].get("authority", 0):
                    pool[url] = {"title": title, "note": "", "authority": auth}

            if len(pool) < 2:
                logger.info("   ℹ️ [references] недостаточно источников для блока (<2). Пропускаем.")
                state.steps_completed.append("references")
                return

            # Ранжирование по авторитетности, дедуп по домену (берём лучший)
            seen_domains: Dict[str, str] = {}  # domain → url
            for url, info in sorted(pool.items(), key=lambda x: x[1].get("authority", 0), reverse=True):
                try:
                    host = (url.split("//")[1].split("/")[0]).lower().lstrip("www.")
                except (IndexError, ValueError):
                    host = ""
                if host and host in seen_domains:
                    continue  # уже есть источник с этого домена — пропускаем
                if host:
                    seen_domains[host] = url

            # Берём топ 2–5 (в компактном режиме — максимум 3)
            max_refs = 3 if getattr(state, "compact_mode", False) else 5
            top_urls = list(seen_domains.values())[:max_refs]
            if len(top_urls) < 2:
                logger.info("   ℹ️ [references] после дедупа осталось <2 уникальных доменов. Пропускаем.")
                state.steps_completed.append("references")
                return

            # Формируем markdown-блок
            lines = ["## Источники", ""]
            for url in top_urls:
                info = pool[url]
                title = info.get("title", url)
                note = info.get("note", "")
                if note:
                    lines.append(f"- [{title}]({url}) — {note}")
                else:
                    lines.append(f"- [{title}]({url})")

            ref_block = "\n".join(lines)
            state.final_article = state.final_article.rstrip() + "\n\n" + ref_block

            # Сохраняем для seo_package и HTML
            state.references = [
                {"title": pool[url].get("title", ""), "url": url, "note": pool[url].get("note", "")}
                for url in top_urls
            ]

            logger.info(f"   ✅ [references] добавлен блок «Источники» ({len(top_urls)} ссылок).")
            state.steps_completed.append("references")
        except Exception as e:
            logger.warning(f"⚠️ [references] сборка источников пропущена из-за ошибки: {e}")

    def _evaluate_draft_score(self, state: PipelineState, draft: str) -> dict:
        """Оценка черновика внешним ревизором (Turing Score и детальный вердикт)."""
        user_msg = (
            f"Проведи экспертную оценку юридического/B2B текста статьи.\n"
            f"Тема: {state.topic}\n"
            f"ТЗ/Описание: {state.description or 'нет'}\n\n"
            f"ТЕКСТ СТАТЬИ:\n{draft}\n\n"
            f"Оцени статью по шкале от 0 до 100 по следующим критериям:\n"
            f"1. Точность юридических формулировок и фактов.\n"
            f"2. Стиль изложения (строгий B2B, без «воды», без ИИ-клише и навязчивых фраз, без неуместных тире в качестве связок).\n"
            f"3. Соответствие ТЗ и полнота раскрытия темы.\n\n"
            f"Верни ответ строго в формате JSON:\n"
            f"{{\n"
            f"  \"score\": 85,  // итоговая общая оценка (0-100)\n"
            f"  \"critique\": \"краткий вердикт по качеству текста\"\n"
            f"}}"
        )
        try:
            res = self._call_agent("external_reviewer", user_msg, parse_json=True, state=state)
            if isinstance(res, dict):
                score = res.get("score", 0)
                try:
                    score = int(score)
                except (ValueError, TypeError):
                    score = 0
                return {"score": score, "critique": res.get("critique", "")}
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при оценке черновика ревизором: {e}")
        
        return {"score": 50, "critique": "Fallback score"}

    def _step_heart_best_of(self, state: PipelineState, style_block: str, rag_block: str, heart_target: int) -> str:
        """Генерация 3 черновиков на разных моделях/температурах и выбор лучшего ревизором."""
        logger.info("   🏆 [QUALITY_MODE] Запускаем генерацию Best-of-3 разнородных черновиков...")
        
        evaluated_drafts = []
        
        # Если KIE API ключ задан, применяем хитрый перебор моделей по кругу (DeepSeek Pro, Gemini 3.1 Pro, Claude Opus 4.8)
        if self._kie_api_key:
            models_pool = [
                {"provider": "kie", "model": "claude-opus-4-8", "temperature": 0.7, "label": "Claude Opus 4.8 (KIE)"},
                {"provider": "kie", "model": "gpt-5-5-openai-resp", "temperature": 0.8, "label": "GPT-5.5 (KIE)"},
                {"provider": "kie", "model": "gemini-3.1-pro", "temperature": 0.85, "label": "Gemini 3.1 Pro (KIE)"},
                {"provider": "deepseek", "model": MODELS["deepseek_pro"], "temperature": 0.85, "label": "DeepSeek Pro"}
            ]
            
            import time
            N = len(models_pool)
            for idx in range(3):
                draft_text = None
                used_cfg = None
                success = False
                
                # Мы делаем 3 круга попыток (всего до 3 попыток на каждую модель)
                for round_num in range(1, 4):
                    if success:
                        break
                    
                    # На каждом круге пробуем модель idx, затем по кругу
                    for shift in range(N):
                        model_idx = (idx + shift) % N
                        cfg = models_pool[model_idx]
                        
                        logger.info(
                            f"   ✍️ Генерация черновика #{idx+1} (попытка круг {round_num}, модель: {cfg['label']})..."
                        )
                        try:
                            # Делаем паузу перед запросом, чтобы избежать rate limit
                            time.sleep(2.0)
                            
                            if heart_target > 20000:
                                draft_text = self._heart_sectional(
                                    state, style_block, rag_block, heart_target,
                                    override_model=cfg["model"],
                                    override_provider=cfg["provider"],
                                    override_temperature=cfg["temperature"]
                                )
                            else:
                                draft_text = self._heart_single(
                                    state, style_block, rag_block, heart_target,
                                    override_model=cfg["model"],
                                    override_provider=cfg["provider"],
                                    override_temperature=cfg["temperature"]
                                )
                            
                            draft_text = self._clean_leaked_ai_artifacts(draft_text)
                            used_cfg = cfg
                            success = True
                            break
                        except Exception as e:
                            logger.warning(
                                f"      ⚠️ Сбой модели {cfg['label']} на круге {round_num}: {e}. Пробуем следующую модель."
                            )
                            # Задержка между попытками
                            time.sleep(3.0)
                
                # Если все модели по 3 попытки упали, используем DeepSeek Pro в качестве аварийного fallback
                if not success:
                    fallback_cfg = models_pool[0]  # DeepSeek Pro
                    logger.error(
                        f"      ❌ Все 3 модели по 3 попытки завершились ошибкой для черновика #{idx+1}. Используем аварийный DeepSeek Pro."
                    )
                    try:
                        time.sleep(2.0)
                        if heart_target > 20000:
                            draft_text = self._heart_sectional(
                                state, style_block, rag_block, heart_target,
                                override_model=fallback_cfg["model"],
                                override_provider=fallback_cfg["provider"],
                                override_temperature=fallback_cfg["temperature"]
                            )
                        else:
                            draft_text = self._heart_single(
                                state, style_block, rag_block, heart_target,
                                override_model=fallback_cfg["model"],
                                override_provider=fallback_cfg["provider"],
                                override_temperature=fallback_cfg["temperature"]
                            )
                        draft_text = self._clean_leaked_ai_artifacts(draft_text)
                        used_cfg = fallback_cfg
                        success = True
                    except Exception as e:
                        logger.error(f"      ❌ Аварийный fallback на DeepSeek Pro тоже завершился ошибкой: {e}")
                        continue
                
                if success and draft_text:
                    # Оцениваем черновик объективно по метрикам человечности
                    from .humanizer import analyze_article
                    sections = self._split_markdown_sections(draft_text)
                    eval_res = analyze_article(sections)
                    score = eval_res["article_human_score"]
                    critique = f"Объективный скор человечности: {score}/100"
                    
                    logger.info(f"      📈 Оценка черновика #{idx+1}: {score}/100. Создан моделью: {used_cfg['label']}")
                    evaluated_drafts.append({
                        "text": draft_text,
                        "score": score,
                        "config": used_cfg,
                        "critique": critique
                    })
        else:
            # При отсутствии KIE API ключа черновики генерирует только DeepSeek Pro с разными температурами
            drafts_configs = [
                {"provider": "deepseek", "model": MODELS["deepseek_pro"], "temperature": 0.85, "label": "DeepSeek Pro Temp 0.85"},
                {"provider": "deepseek", "model": MODELS["deepseek_pro"], "temperature": 0.6, "label": "DeepSeek Pro Temp 0.6"},
                {"provider": "deepseek", "model": MODELS["deepseek_pro"], "temperature": 0.4, "label": "DeepSeek Pro Temp 0.4"}
            ]
            for idx, cfg in enumerate(drafts_configs, 1):
                logger.info(f"   ✍️ Генерация черновика #{idx} ({cfg['label']}): model={cfg['model']}, temp={cfg['temperature']}...")
                try:
                    if heart_target > 20000:
                        draft_text = self._heart_sectional(
                            state, style_block, rag_block, heart_target,
                            override_model=cfg["model"],
                            override_provider=cfg["provider"],
                            override_temperature=cfg["temperature"]
                        )
                    else:
                        draft_text = self._heart_single(
                            state, style_block, rag_block, heart_target,
                            override_model=cfg["model"],
                            override_provider=cfg["provider"],
                            override_temperature=cfg["temperature"]
                        )
                    
                    draft_text = self._clean_leaked_ai_artifacts(draft_text)
                    
                    from .humanizer import analyze_article
                    sections = self._split_markdown_sections(draft_text)
                    eval_res = analyze_article(sections)
                    score = eval_res["article_human_score"]
                    critique = f"Объективный скор человечности: {score}/100"
                    
                    logger.info(f"      📈 Оценка черновика #{idx}: {score}/100. Вердикт: {critique}")
                    evaluated_drafts.append({
                        "text": draft_text,
                        "score": score,
                        "config": cfg,
                        "critique": critique
                    })
                except Exception as e:
                    logger.error(f"      ❌ Ошибка при генерации/оценке черновика #{idx}: {e}")

        if not evaluated_drafts:
            raise RuntimeError("Все попытки генерации черновиков завершились ошибкой.")
            
        # Выбираем лучший по score
        best_draft_data = max(evaluated_drafts, key=lambda x: x["score"])
        logger.info(f"   🏆 Победитель: черновик {best_draft_data['config']['label']} с баллом {best_draft_data['score']}/100!")

        # Составляем краткую сводку генерации черновиков
        summary_lines = []
        summary_lines.append("Сводка генерации черновиков (Best of 3):")
        for idx, d in enumerate(evaluated_drafts, 1):
            summary_lines.append(f"  Черновик {idx} создан моделью {d['config']['label']} (модель: {d['config']['model']}), оценка человечности: {d['score']}/100")
        summary_lines.append(f"  Победитель: {best_draft_data['config']['label']} (оценка: {best_draft_data['score']}/100)")
        state.best_of_summary = "\n".join(summary_lines)
        
        logger.info("   📋 " + " \n   📋 ".join(summary_lines))

        return best_draft_data["text"]

    def _step_quality_edit_loop(self, state: PipelineState):
        """Surgical Edit Loop в QUALITY_MODE с памятью одобренных разделов и стратегией Accept-Best."""
        logger.info("🕵️ [6/8] Ревизор: запуск Surgical Edit Loop в QUALITY_MODE...")
        
        best_draft = state.draft
        best_score = 0
        approved_sections = set()
        MAX_QUALITY_ITERS = 3
        # True, если текущий state.draft содержит правки, которые ещё не получили оценку.
        # Нужно, чтобы правки последней итерации не терялись при Accept-Best.
        pending_unscored_patch = False
        # Снимок последних правок ревизора — нужен для одноразовой жёсткой эскалации вне цикла.
        last_edits = []
        
        best_score = 0
        
        for iteration in range(MAX_QUALITY_ITERS):
            logger.info(f"\n   🔄 Surgical Edit Loop: итерация {iteration+1} из {MAX_QUALITY_ITERS} (лучший балл: {best_score})")
            
            # Разбираем черновик на разделы
            sections = self._split_markdown_sections(state.draft)
            editable = [i for i, b in enumerate(sections) if b["level"] == 2]
            if len(editable) < 2:
                logger.info("      ℹ️ Мало H2-разделов для хирургических правок. Прерываем цикл.")
                break
                
            # Формируем каталог для ревизора
            lines = []
            for i, b in enumerate(sections):
                body = b["raw"].strip().replace("\n", " ")
                preview = body[:400] + ("…" if len(body) > 400 else "")
                lines.append(f"[{i}] {b['heading'] if b['level'] == 2 else '(вступление)'}: {preview}")
            catalog = "\n".join(lines)
            
            # Накладываем дополнительные требования в зависимости от стиля
            extra_critique_rules = ""
            if state.style_id == "checklist":
                extra_critique_rules = (
                    f"\n⚠️ ТРЕБОВАНИЯ ДЛЯ ЧЕК-ЛИСТА:\n"
                    f"- Статья должна содержать строго ровно {state.num_checklist_items} содержательных разделов H2 (не считая Введение, Самопроверку и Послесловие).\n"
                    f"- Каждый раздел H2 должен быть разбит на короткие абзацы (по 2-3 предложения) и маркированные списки действий.\n"
                    f"- Внутри каждого раздела H2 должен быть короткий абзац ошибки, начинающийся строго со слова «Ошибка — » или «Ошибка: ».\n"
                    f"- Внутри разделов H2 категорически запрещены подзаголовки H3 (###).\n"
                    f"- В конце статьи обязательно должны быть разделы «Быстрая самопроверка» (5 пунктов списка с флажками Markdown) и «Послесловие» (короткий вывод до 500 символов).\n"
                    f"- Если количество разделов H2 не равно {state.num_checklist_items} или в них отсутствуют описания ошибок/списки действий — статья НЕ должна одобряться (approved: false)."
                )

            # Запрашиваем у ревизора план точечных правок
            system_prompt = (
                "Ты — планировщик точечных правок юридической/B2B статьи на базе Gemini 3 Pro.\n"
                "Тебе дают статью, разбитую на пронумерованные разделы, и описание/ТЗ.\n"
                "Определи, какие разделы нуждаются в доработке (юридическая точность, полнота, стиль).\n"
                f"{extra_critique_rules}\n\n"
                "⚠️ СТРОГИЕ ПРАВИЛА ХУМАНИЗАЦИИ:\n"
                "- Тон строго экспертный, деловой, авторитетный B2B. Без панибратства, сленга.\n"
                "- Никаких ИИ-клише и штампов (например: 'в данной статье мы рассмотрим', 'важно отметить', 'подводя итоги').\n"
                "- Контролируй частоту использования тире: не ставь его почти в каждом предложении. Используй тире естественно, но избегай монотонности текста.\n"
                "- Избегай художественных описаний природы, погоды или эмоций. Должна быть сухая деловая конкретика (номера статей ТК/НК РФ, законы, сроки).\n\n"
                "🎯 ПРОВЕРКА ПРАВДОПОДОБИЯ (Ревизор реальности — высший приоритет; твоя ЦА — собственники и топ-менеджеры с критическим мышлением, они мгновенно видят фальшь):\n"
                "1. ПЛАУЗИБЕЛЬНОСТЬ РОЛИ. Делает ли персонаж то, что человек его роли и масштаба реально делает? "
                "(Собственник на обороте сотен млн НЕ проверяет лично ссылки на ячейки Excel и бухгалтерские проводки — это работа главбуха/аудитора.) Несоответствие → требуй правку.\n"
                "2. ПРОПОРЦИОНАЛЬНОСТЬ СТАВОК. Соответствует ли ущерб/выгода масштабу бизнеса из Паспорта ЦА? Потеря десятков млн из-за одной таблицы при скромном обороте — гипербола. Требуй приземлить.\n"
                "3. ПРИЧИННО-СЛЕДСТВЕННАЯ ЛОГИКА. Нет ли натянутых связок (типа «знание судебной практики ВС РФ держит собственника в операционке»)? Настоящая причина обычно проще. Требуй убрать притянутое.\n"
                "4. КОНСИСТЕНТНОСТЬ МАСШТАБА. Все рекомендации в одном масштабе бизнеса? Нельзя одновременно «делегировать канцелярию» и «вводить операционного директора на 100 млн».\n"
                "5. УМЕСТНОСТЬ ИМЁН/КОНТЕКСТА. Анахронизмы и геополитически заряженные имена собственные (напр. «Северный поток») → требуй заменить на нейтральный отраслевой пример.\n"
                "6. ОДИН ЧИТАТЕЛЬ — ОДНА ЗАДАЧА. Не расщепляется ли текст на лайфстайл-пост + юрсправку + операционную инструкцию? Держи единый primary_intent из Паспорта ЦА.\n\n"
                "🔍 ПЕРЕНЕСЁННЫЕ ПРОВЕРКИ SHERIFF/MIRROR (фактура и анти-ИИ):\n"
                "- CLAIM-LEVEL: каждая ключевая цифра/ставка/срок должна выглядеть проверяемой и не противоречить остальному тексту. Подозрительные «оценочные» цифры без маркировки — флаг.\n"
                "- AI-ПЛАСТИК: лови пластиковые штампы (экосистема, фундамент, синергия, комплексный подход, в современных реалиях) и канцелярит.\n"
                "- BURSTINESS/РИТМ: монотонная длина предложений и однотипные зачины абзацев — флаг на переработку ритма.\n\n"
                "Формат ответа СТРОГО JSON:\n"
                "{\n"
                "  \"approved\": false,  // true, если статья не требует доработки и полностью соответствует всем требованиям\n"
                "  \"score\": 85,        // оценка статьи от 0 до 100\n"
                "  \"edits\": [\n"
                "    {\"section_index\": <индекс раздела>, \"reason\": \"почему нужна правка\", \"instruction\": \"что конкретно сделать в этом разделе\"}\n"
                "  ]\n"
                "}"
            )
            
            user_msg = (
                f"ТЕМА СТАТЬИ: {state.topic}\n"
                f"ТЗ/ОПИСАНИЕ: {state.description or 'нет'}\n\n"
                f"{self._persona_block(state)}"
                f"РАЗДЕЛЫ СТАТЬИ:\n{catalog}\n\n"
                f"ПОЛНЫЙ ТЕКСТ СТАТЬИ:\n{state.draft}\n\n"
                f"Верни JSON с оценкой и списком правок."
            )
            
            try:
                response = self._call_agent("external_reviewer", user_msg, parse_json=True, state=state)
            except Exception as e:
                logger.warning(f"      ⚠️ Ревизор недоступен на итерации {iteration+1} ({e}). Прерываем цикл правок.")
                break
                
            if not isinstance(response, dict):
                logger.warning("      ⚠️ Некорректный формат ответа от ревизора. Пропускаем.")
                continue
                
            score = response.get("score", 0)
            try:
                score = int(score)
            except (ValueError, TypeError):
                score = 0
                
            edits = response.get("edits", [])
            logger.info(f"      📈 Текущий балл: {score}/100. Найдено правок: {len(edits)}")
            
            if score > best_score:
                best_score = score
                best_draft = state.draft
            # Текущий черновик только что получил свежую оценку — он больше не "неоценённый".
            pending_unscored_patch = False
                
            if (response.get("approved") == True and score >= 85) or score >= 90 or not edits:
                logger.info(f"      ✅ Ревизор одобрил статью (балл {score} >= 85 и approved=True).")
                break
                
            # Фильтруем правки с учетом памяти одобренных/замороженных разделов
            valid_edits = []
            for e in edits:
                idx = e.get("section_index")
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    continue
                if idx in approved_sections:
                    logger.info(f"      ⏭️ Пропускаем раздел [{idx}], так как он заморожен (был одобрен ранее).")
                    continue
                valid_edits.append({
                    "section_index": idx,
                    "reason": e.get("reason", ""),
                    "instruction": e.get("instruction", "")
                })

            # Запоминаем последние правки ревизора для возможной жёсткой эскалации.
            if valid_edits:
                last_edits = list(valid_edits)

            if not valid_edits:
                logger.info("      ℹ️ Все запрошенные правки относятся к замороженным разделам. Выходим.")
                break
                
            # Выполняем точечные правочки
            new_sections = list(sections)
            changed = 0
            
            instr_by_idx = {}
            for e in valid_edits:
                instr_by_idx.setdefault(e["section_index"], []).append(e["instruction"])
                
            for idx in instr_by_idx:
                sec = sections[idx]
                prev_raw = sections[idx - 1]["raw"] if idx - 1 >= 0 else ""
                next_raw = sections[idx + 1]["raw"] if idx + 1 < len(sections) else ""
                prev_tail = prev_raw.strip()[-400:]
                next_head = next_raw.strip()[:400]
                instruction = "\n".join(f"- {t}" for t in instr_by_idx[idx] if t)
                
                # Добавляем требование о тире
                instruction += "\n- Контролируй частоту использования тире: не ставь его почти в каждом предложении. Используй тире естественно, но избегай монотонности текста."
                
                rewritten = self._rewrite_section(
                    state, sec, prev_tail, next_head, instruction,
                    override_model=MODELS["deepseek_pro"],
                    override_temperature=0.85
                )
                
                if self._section_ok(sec, rewritten):
                    trailing = sec["raw"][len(sec["raw"].rstrip()):]
                    new_sections[idx] = {**sec, "raw": rewritten.rstrip() + trailing}
                    changed += 1
                else:
                    logger.warning(f"      ⚠️ Раздел [{idx}] переписан некорректно. Оставляем оригинал.")
                    
            if changed == 0:
                logger.info("      ℹ️ Ни один раздел не был изменен. Прерываем цикл.")
                break
                
            # Собираем статью
            new_draft = self._reassemble_sections(new_sections)
            
            # Проверяем целостность структуры
            import re as _re
            h2_before = len(_re.findall(r"(?m)^##\s", state.draft))
            h2_after = len(_re.findall(r"(?m)^##\s", new_draft))
            if h2_after < h2_before or len(new_draft) < len(state.draft) * 0.6:
                logger.warning("      ⚠️ Нарушена структура статьи после правок. Откатываемся.")
                continue
                
            # Фиксируем разделы, которые НЕ правились на этой итерации, как одобренные
            for idx in range(len(sections)):
                if idx not in instr_by_idx:
                    approved_sections.add(idx)
                    
            state.draft = new_draft
            state.sheriff_iterations += 1
            # Эти правки ещё не оценены ревизором — пометим, чтобы не потерять при Accept-Best.
            pending_unscored_patch = True
            logger.info(f"      🩹 Успешно пропатчено разделов: {changed}. Длина: {len(state.draft)} символов.")
            
        # Если последняя итерация применила правки, которые ещё не оценивались ревизором,
        # оцениваем финальный черновик и сравниваем с лучшим. Раньше эти правки терялись:
        # скоринг шёл в начале итерации, а патч последней итерации оставался без оценки.
        if pending_unscored_patch:
            final_eval = self._evaluate_draft_score(state, state.draft)
            final_score = final_eval.get("score", 0)
            try:
                final_score = int(final_score)
            except (ValueError, TypeError):
                final_score = 0
            logger.info(f"   🔎 Финальная оценка пропатченного черновика: {final_score}/100 (лучший ранее: {best_score})")
            if final_score >= best_score and len(state.draft) > 100:
                best_score = final_score
                best_draft = state.draft

        # ─── Жёсткая ОДНОРАЗОВАЯ эскалация (строго ВНЕ цикла, без рекурсии) ───
        # Если после всех итераций лучший черновик всё ещё неправдоподобен (ниже порога),
        # один раз полностью переписываем худший раздел и берём лучшее из двух (accept-best).
        # Новых итераций ревизора это НЕ порождает — система по кругу не гоняется.
        BELIEVABILITY_THRESHOLD = 75
        # Эскалация чинит до 2 худших разделов за ОДИН проход (без новых итераций ревизора),
        # затем единожды переоценивает и применяет accept-best. Так система не зацикливается.
        MAX_ESCALATION_SECTIONS = 2
        if best_score < BELIEVABILITY_THRESHOLD and last_edits and len(best_draft) > 100:
            esc_sections = self._split_markdown_sections(best_draft)

            # Берём до 2 уникальных валидных H2-разделов из последних правок ревизора
            target_idxs = []
            for e in last_edits:
                try:
                    idx = int(e.get("section_index"))
                except (TypeError, ValueError):
                    continue
                if idx in target_idxs:
                    continue
                if 0 <= idx < len(esc_sections) and esc_sections[idx]["level"] == 2:
                    target_idxs.append(idx)
                if len(target_idxs) >= MAX_ESCALATION_SECTIONS:
                    break

            instr_by_idx = {}
            for e in last_edits:
                try:
                    idx = int(e.get("section_index"))
                except (TypeError, ValueError):
                    continue
                if idx in target_idxs:
                    instr_by_idx.setdefault(idx, []).append(e.get("instruction", ""))

            if target_idxs:
                logger.info(
                    f"   🚨 Эскалация: балл {best_score} < {BELIEVABILITY_THRESHOLD}. "
                    f"Однократный полный переписк худших разделов {target_idxs}."
                )
                rewritten_any = False
                for idx in target_idxs:
                    sec = esc_sections[idx]
                    prev_raw = esc_sections[idx - 1]["raw"] if idx - 1 >= 0 else ""
                    next_raw = esc_sections[idx + 1]["raw"] if idx + 1 < len(esc_sections) else ""
                    extra = "\n".join(f"- {t}" for t in instr_by_idx.get(idx, []) if t)
                    instruction = (
                        "ПОЛНЫЙ ПЕРЕПИСК РАЗДЕЛА ради правдоподобия (последний шанс).\n"
                        f"{extra}\n"
                        "- Приземли кейс, цифры и поведение персонажа под Паспорт ЦА (роль, масштаб, реалистичные ставки).\n"
                        "- Убери гиперболы, анахронизмы, смешение масштабов и натянутую причинно-следственную логику."
                    )
                    rewritten = self._rewrite_section(
                        state, sec, prev_raw.strip()[-400:], next_raw.strip()[:400], instruction,
                        override_model=MODELS["deepseek_pro"], override_temperature=0.8
                    )
                    if self._section_ok(sec, rewritten):
                        trailing = sec["raw"][len(sec["raw"].rstrip()):]
                        esc_sections[idx] = {**sec, "raw": rewritten.rstrip() + trailing}
                        rewritten_any = True
                    else:
                        logger.warning(f"   ⚠️ Эскалация: раздел [{idx}] не прошёл проверку. Оставляем оригинал раздела.")

                if rewritten_any:
                    esc_draft = self._reassemble_sections(esc_sections)
                    esc_eval = self._evaluate_draft_score(state, esc_draft)
                    try:
                        esc_score = int(esc_eval.get("score", 0))
                    except (ValueError, TypeError):
                        esc_score = 0
                    logger.info(f"   🚨 После эскалации: {esc_score}/100 (было {best_score}).")
                    # accept-best: принимаем только если не стало хуже
                    if esc_score >= best_score and len(esc_draft) > 100:
                        best_score = esc_score
                        best_draft = esc_draft
                else:
                    logger.warning("   ⚠️ Эскалация: ни один раздел не переписан. Оставляем лучший черновик.")

        # Применяем стратегию Accept-Best
        if len(best_draft) > 100:
            state.draft = best_draft
            logger.info(f"   🏆 [Surgical Edit Loop] Завершено. Восстановлен лучший черновик с баллом {best_score}/100.")

    def _heart_single(self, state, style_block, rag_block, target_chars, override_model=None, override_provider=None, override_temperature=None):
        """Heart: генерация статьи одним вызовом."""
        min_chars = state.min_chars or int(target_chars * 0.85)
        max_chars = state.max_chars or int(target_chars * 1.15)
        
        # Динамический расчет лимита на разделы для лаконичности СЕО-статей
        sections = self._extract_sections(state.blueprint)
        num_sections = len(sections) if sections else 5
        chars_per_section = int(target_chars / num_sections)
        
        conciseness_instruction = ""
        density = state.density_config
        if density:
            conciseness_instruction = (
                f"- Требования к плотности и ритму текста:\n"
                f"  1) Длина каждого абзаца: строго {density.get('sentences', '2-3')} предложения.\n"
                f"  2) Стиль изложения: {density.get('tone_style', 'сбалансированная бизнес-проза')}.\n"
                f"  3) Бюджет на один раздел H2: ориентировочно {chars_per_section} символов.\n"
                f"  4) Объем вступления (сценки-крючка): {density.get('hook_size', '1-2 абзаца')}.\n"
                f"- ⚠️ ВАЖНО: Статья должна оставаться полноценной статьей, написанной связной и плавной журнальной прозой. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать сухими тезисами или превращать текст в список правил (телеграфный стиль). Списки используй только для перечислений.\n"
            )

        user_msg = (
            f"BLUEPRINT ОТ СТРУКТУРИРОВЩИКА:\n{self._compact_json(state.blueprint, 5000)}\n\n"
            f"ФАКТЫ ОТ ИССЛЕДОВАТЕЛЯ:\n{self._compact_json(state.facts, 5000)}\n\n"
            f"{style_block}\n\n"
            f"⚠️ КРИТИЧЕСКОЕ ТРЕБОВАНИЕ К ОБЪЕМУ (МАКСИМАЛЬНЫЙ ПРИОРИТЕТ):\n"
            f"- Твой текст должен содержать строго от {min_chars} до {max_chars} символов (включая пробелы).\n"
            f"- Идеальный ориентир: {target_chars} символов.\n"
            f"- Это примерно {target_chars // 6} слов.\n"
            f"- Превышение лимита в {max_chars} символов КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО. Если ты напишешь больше, текст будет механически обрезан, что полностью разрушит структуру статьи.\n"
            f"- Если в структуре слишком много разделов, сокращай объём каждого из них, но уложись в рамки.\n"
            f"{conciseness_instruction}"
            f"- Если не укладываешься — сокращай менее важные разделы\n\n"
            f"{rag_block}\n\n"
            f"Напиши полный текст статьи в Markdown."
        )
        result = self._generate_clean_heart_text(
            user_msg, target_chars=target_chars, state=state,
            override_model=override_model,
            override_provider=override_provider,
            override_temperature=override_temperature
        )
        return result

    def _apply_stopwords_cleanup(self, text: str) -> str:
        """Авто-очистка ТОЛЬКО безопасных многословных канцелярских оборотов (0 токенов).

        #7: одиночные слова больше не заменяются вслепую (искажало смысл) — см. stopwords.py.
        """
        import re

        replacements = {
            "из таблицы видно": "таблица показывает",
            "эти примеры показывают": "примеры показывают",
            "важно отметить, что": "отметим, что",
            "важно отметить что": "отметим, что",
            "важно отметить": "отметим",
            "таким образом,": "в итоге,",
            "таким образом": "в итоге",
            "подводя итоги,": "в итоге,",
            "подводя итоги": "в итоге",
            "проработка динамики": "анализ динамики",
            "важно уделять внимание": "следует учитывать",
            "является ключевым": "ключевой",
            "важно понимать": "отметим",
            "в заключение": "в итоге",
            "в современном мире": "сейчас",
            "позволяет оптимизировать": "оптимизирует",
            "закладывает фундамент": "готовит основу",
            "в современных реалиях": "сейчас",
            "на сегодняшний день": "сейчас",
            "комплексный подход": "подход",
            "ключевой фактор успеха": "фактор успеха",
            "играет важную роль": "важен",
            "необходимо учитывать, что": "учтите, что",
            "необходимо учитывать что": "учтите, что",
            "необходимо учитывать": "учтите",
            "следует отметить, что": "отметим, что",
            "следует отметить что": "отметим, что",
            "следует отметить": "отметим",
            "стоит подчеркнуть, что": "подчеркнем, что",
            "стоит подчеркнуть что": "подчеркнем, что",
            "стоит подчеркнуть": "подчеркнем",
            "нельзя недооценивать": "важно учитывать",
            "является неотъемлемой частью": "входит в",
            "в контексте текущих изменений": "сейчас",
            "системный подход к решению": "решение",
            "в условиях неопределенности": "при неопределенности",
            "трансформация бизнес-процессов": "изменение процессов",
            "высокий уровень экспертизы": "экспертиза",
            "в долгосрочной перспективе": "в будущем",
            "оптимизация процессов": "оптимизация",
            "динамично развивающийся": "развивающийся",
        }

        # #7: НЕ затираем стоп-слова вслепую. Одиночные слова (экосистема, фундамент,
        # масштабирование, синергия, конвейер) НЕ авто-заменяем — это искажало смысл и
        # ломало падежи. Их ловит только ревизор (REVIEW_FLAG_WORDS) для точечной правки.
        # Авто-замена применяется ТОЛЬКО к безопасным многословным оборотам выше.

        # Сортируем ключи по длине в убывающем порядке
        sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
        for pattern in sorted_keys:
            repl = replacements[pattern]
            regex = r'(?i)(?<![а-яА-ЯёЁ])' + re.escape(pattern) + r'(?![а-яА-ЯёЁ])'
            text = re.sub(regex, repl, text)

        text = text.replace("â€”", "—")
        return text

    def _programmatic_trim(self, text: str, max_chars: int) -> str:
        """Программная обрезка текста до max_chars по последнему полному абзацу.

        Не ломает H2-структуру: обрезает только целые абзацы (разделённые \\n\\n),
        а если превышение внутри абзаца — по последнему предложению.
        """
        if len(text) <= max_chars:
            return text

        import re

        # 1. Разбиваем на абзацы (по \n\n)
        paragraphs = text.split("\n\n")

        # 2. Накапливаем абзацы, пока укладываемся в лимит
        result_parts = []
        current_len = 0
        for i, para in enumerate(paragraphs):
            candidate_len = current_len + len(para) + (2 if result_parts else 0)  # \n\n
            if candidate_len <= max_chars:
                result_parts.append(para)
                current_len = candidate_len
            else:
                # Этот абзац не помещается целиком
                remaining = max_chars - current_len - (2 if result_parts else 0)
                if remaining > 200:
                    # Обрезаем по последнему предложению (. ! ? ...)
                    trimmed = para[:remaining]
                    # Ищем последнее предложение-границу
                    last_sentence = max(
                        trimmed.rfind("."),
                        trimmed.rfind("!"),
                        trimmed.rfind("?"),
                        trimmed.rfind("—"),
                    )
                    if last_sentence > len(trimmed) * 0.5:
                        result_parts.append(trimmed[:last_sentence + 1])
                break

        trimmed_text = "\n\n".join(result_parts)

        # 3. Если ничего не осталось — хотя бы вернуть начало
        if not trimmed_text.strip():
            trimmed_text = text[:max_chars]

        # 4. Убедимся, что последний символ — пунктуация (не обрыв)
        trimmed_text = trimmed_text.rstrip()
        if trimmed_text and trimmed_text[-1] not in ".!?;:—)»":
            trimmed_text += "."

        return trimmed_text

    def _normalize_checklist_bullets(self, text: str, state: 'PipelineState' = None) -> str:
        """Нормализация markdown-списков в чек-листах.

        Проблема: Heart генерирует строки с отступом из пробелов ("  Проводим аудит...")
        вместо markdown-маркеров ("- Проводим аудит..."). Рендер показывает их как
        обычный текст без буллетов.

        Постпроцессор детектит ГРУППЫ подряд идущих строк с одинаковым отступом ≥2 пробелов
        (≥2 строки в группе — это визуально список) и добавляет валидный маркер "- ".
        Применяется ТОЛЬКО к стилю checklist, чтобы не задеть обычные абзацы других стилей.
        """
        import re

        # Применяем только к чек-листам — для остальных стилей это небезопасно.
        if state is not None:
            sid = getattr(state, "style_id", "") or ""
            atype = getattr(state, "article_type", "") or ""
            if sid != "checklist" and atype != "checklist":
                return text

        lines = text.split("\n")
        n = len(lines)
        fixed = 0

        def is_skippable(s: str) -> bool:
            return (not s
                    or s.startswith(("#", ">", "|", "---", "***"))
                    or s.startswith("Ошибка")
                    or s.startswith("- [ ]")
                    or s.startswith("- ")
                    or s.startswith("* ")
                    or s.startswith("+ "))

        i = 0
        while i < n:
            line = lines[i]
            stripped = line.lstrip(" ")
            indent = len(line) - len(stripped)

            # Нужен отступ ≥2 и не пропускаемая строка
            if indent >= 2 and not is_skippable(stripped):
                # Соберём группу подряд идущих строк с тем же отступом
                group_start = i
                j = i
                while j < n:
                    g_line = lines[j]
                    g_stripped = g_line.lstrip(" ")
                    g_indent = len(g_line) - len(g_stripped)
                    if g_indent == indent and not is_skippable(g_stripped):
                        j += 1
                    else:
                        break
                group_len = j - group_start

                # Список = ≥2 строки в группе с одинаковым отступом.
                # Это защищает от ложного срабатывания на одиночный абзац-цитату.
                if group_len >= 2:
                    for k in range(group_start, j):
                        lines[k] = "- " + lines[k].lstrip(" ")
                    fixed += group_len
                i = j
            else:
                i += 1

        if fixed > 0:
            logger.info(f"   🔧 [Нормализация] Добавлено {fixed} markdown-маркеров списков в чек-листе.")
        return "\n".join(lines)


    def _generate_clean_heart_text(self, user_msg: str, max_retries: int = 2, target_chars: int = 0, state: PipelineState = None, override_model=None, override_provider=None, override_temperature=None) -> str:
        """Внутренняя обертка с мягкими проверками и авто-очисткой для Писателя."""
        from .stopwords import ALL_STOP_WORDS
        import re
        stop_words = ALL_STOP_WORDS
        
        current_msg = user_msg
        for attempt in range(max_retries):
            result = self._call_agent(
                "heart", current_msg, parse_json=False, target_chars=target_chars, state=state,
                override_model=override_model,
                override_provider=override_provider,
                override_temperature=override_temperature
            )
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            
            # 1. Проверяем штампы: просим переписать только при сильном загрязнении (>= 3 штампа) и только 1 раз (attempt < 1)
            lower_text = text.lower()
            found_words = [w for w in stop_words if w in lower_text]
            if len(found_words) >= 3 and attempt < 1:
                logger.warning(f"   ⚠️ [Soft-Regex] Найдено много штампов: {found_words}. Просим переписать (попытка {attempt+1}/{max_retries})")
                current_msg = user_msg + (
                    f"\n\nSYSTEM ALERT: В тексте обнаружено много штампов: {found_words}. "
                    f"Вот текст, который ты сгенерировал:\n"
                    f"=== НАЧАЛО ПРЕДЫДУЩЕГО ТЕКСТА ===\n{text}\n=== КОНЕЦ ПРЕДЫДУЩЕГО ТЕКСТА ===\n\n"
                    f"Перепиши этот текст заново, убрав указанные штампы и используя точную, деловую B2B-лексику, но сохрани все разделы оглавления и структуры!"
                )
                continue
                
            break

        # 2. Проверяем оверобъем после завершения всех ретраев генерации.
        # Двухуровневая стратегия вместо единого LLM-condense:
        #   - Умеренное превышение (<1.4× target): программная обрезка по последнему
        #     абзацу/разделу, чтобы не тратить токены на condense и не ломать структуру.
        #   - Сильное превышение (>=1.4× target): LLM-condense (единственный вызов).
        is_full_article = False
        if state is not None and state.min_chars > 0:
            is_full_article = target_chars >= state.min_chars * 0.8
            
        current_max_chars = state.max_chars if is_full_article else int(target_chars * 1.15)
        over_limit = current_max_chars > 0 and len(text) > current_max_chars

        if over_limit:
            over_ratio = len(text) / target_chars if target_chars > 0 else 1.0

            if over_ratio < 1.4:
                # --- Умеренное превышение: программная обрезка ---
                logger.info(f"   ✂️ [Volume Check] Лёгкое превышение ({len(text)} симв. vs лимит {current_max_chars}, ratio={over_ratio:.2f}). Программная обрезка...")
                text = self._programmatic_trim(text, current_max_chars)
                logger.info(f"   ✅ [Volume Check] Текст программно обрезан до {len(text)} символов.")
            else:
                # --- Сильное превышение: LLM-condense ---
                logger.warning(f"   ⚠️ [Volume Check] Финальный черновик сильно превышен ({len(text)} симв. vs лимит {current_max_chars}, ratio={over_ratio:.2f}). Запускаем LLM-сжатие...")
                compress_msg = (
                    f"ТЕКСТ СТАТЬИ:\n{text}\n\n"
                    f"Этот текст превышает лимит. Пожалуйста, сожми статью строго до {target_chars} символов. "
                    f"Убери любые повторы, размышления, вводные слова. Сохрани все разделы (H2), ссылки на законы, таблицы и факты. "
                    f"Текст должен остаться связным и легко читаемым. Верни только измененный текст статьи."
                )
                try:
                    compressed_text = self._call_agent(
                        "heart", compress_msg, parse_json=False, target_chars=target_chars, state=state,
                        override_model=override_model,
                        override_provider=override_provider,
                        override_temperature=override_temperature
                    )
                    if len(compressed_text) < len(text) and len(compressed_text) >= int(target_chars * 0.70):
                        text = compressed_text
                        logger.info(f"   ✅ [Volume Check] Текст успешно сжат через LLM до {len(text)} символов.")
                    else:
                        logger.warning(f"   ⚠️ [Volume Check] LLM-сжатие не привело к уменьшению длины ({len(compressed_text)} симв.). Пробуем программную обрезку.")
                        text = self._programmatic_trim(text, current_max_chars)
                except Exception as e:
                    logger.warning(f"   ⚠️ [Volume Check] Ошибка при LLM-сжатии: {e}. Пробуем программную обрезку.")
                    text = self._programmatic_trim(text, current_max_chars)

        return self._apply_stopwords_cleanup(text)

    def _heart_sectional(self, state, style_block, rag_block, target_chars, override_model=None, override_provider=None, override_temperature=None):
        """Heart: посекционная генерация лонгрида.

        Каждый раздел из blueprint пишется отдельным вызовом,
        затем все части собираются в единую статью.
        """
        sections = self._extract_sections(state.blueprint)
        if not sections:
            logger.warning("⚠️ Не удалось извлечь разделы из blueprint, fallback на single")
            return self._heart_single(state, style_block, rag_block, target_chars)

        # Делаем маппинг: заголовок -> dict раздела из blueprint, чтобы узнать целевой объем target_chars
        bp_sections = []
        for key in ["sections", "structure", "outline", "chapters", "разделы", "план"]:
            if key in state.blueprint and isinstance(state.blueprint[key], list):
                bp_sections = state.blueprint[key]
                break

        section_dicts = {}
        for item in bp_sections:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or item.get("heading") or item.get("section", "")
                if title:
                    section_dicts[str(title).strip().lower()] = item

        logger.info(f"   🎯 Посекционная генерация: {len(sections)} разделов")

        parts = []
        for i, section in enumerate(sections, 1):
            # Определяем chars_per_section
            item_dict = section_dicts.get(section.strip().lower(), {})
            custom_section_chars = item_dict.get("target_chars")
            if custom_section_chars and isinstance(custom_section_chars, (int, float)):
                # Зарезервировать SEO-бюджет (уменьшить на 7%), так как в черновике структурировщика
                # указана полная длина без учета SEO-резерва.
                chars_per_section = int(custom_section_chars * 0.93)
            else:
                # target_chars здесь уже = heart_target (с вычтенным SEO-резервом)
                chars_per_section = target_chars // len(sections)
            words_per_section = chars_per_section // 6

            logger.info(f"   ✍️ Раздел {i}/{len(sections)}: {section[:60]}... (~{chars_per_section} символов)")

            # Требования к плотности и ритму текста в этом разделе
            conciseness_instruction = ""
            density = state.density_config
            if density:
                sentences_req = density.get('sentences', '3-4')
                tone_req = density.get('tone_style', 'бизнес-проза')
                conciseness_instruction = (
                    f"- Требования к плотности и ритму текста в этом разделе:\n"
                    f"  1) Длина каждого абзаца: строго {sentences_req} предложения.\n"
                    f"  2) Стиль изложения: {tone_req}.\n"
                    f"  3) Бюджет на этот раздел: ориентировочно {chars_per_section} символов.\n"
                )
                if i == 1 and density.get('hook_size'):
                    conciseness_instruction += f"  4) Объем вступления (сценки-крючка) в этом (первом) разделе: {density.get('hook_size')}.\n"
                conciseness_instruction += (
                    f"- ⚠️ ВАЖНО: Раздел должен быть написан связной и плавной журнальной прозой. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать сухими тезисами или превращать текст в список правил (телеграфный стиль). Списки используй только для перечислений.\n"
                )

            section_msg = (
                f"Ты пишешь раздел {i} из {len(sections)} большой аналитической статьи.\n\n"
                f"ТЕМА СТАТЬИ: {state.topic}\n\n"
                f"ПОЛНЫЙ ПЛАН СТАТЬИ (для контекста):\n{self._compact_json(state.blueprint, 4000)}\n\n"
                f"ТЕКУЩИЙ РАЗДЕЛ: {section}\n\n"
                f"ФАКТЫ ОТ ИССЛЕДОВАТЕЛЯ:\n{self._compact_json(state.facts, 5000)}\n\n"
                f"{style_block}\n\n"
                f"⚠️ КРИТИЧЕСКОЕ ТРЕБОВАНИЕ К ОБЪЕМУ РАЗДЕЛА (МАКСИМАЛЬНЫЙ ПРИОРИТЕТ):\n"
                f"- РОВНО {chars_per_section} символов (±10%, т.е. {int(chars_per_section*0.9)}-{int(chars_per_section*1.1)})\n"
                f"- Это примерно {words_per_section} слов — НЕ БОЛЬШЕ\n"
                f"- ЗАПРЕЩЕНО писать больше {int(chars_per_section*1.1)} символов. Превышение лимита приведет к браку.\n"
                f"{conciseness_instruction}"
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

            result = self._generate_clean_heart_text(
                section_msg, target_chars=chars_per_section, state=state,
                override_model=override_model,
                override_provider=override_provider,
                override_temperature=override_temperature
            )
            
            # Посекционный контроль: если раздел превысил бюджет более чем на 15%, сжимаем его
            if len(result) > chars_per_section * 1.15:
                logger.warning(f"   ⚠️ Раздел слишком длинный ({len(result)} vs {chars_per_section}), запускаем локальное сжатие раздела...")
                result = self._condense_single_section(
                    state, result, section, chars_per_section,
                    override_model=override_model,
                    override_provider=override_provider
                )
            parts.append(result)

        full_article = "\n\n".join(parts)

        if state.style_id == "checklist":
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
            conclusion_text = self._generate_clean_heart_text(
                conclusion_msg, target_chars=500, state=state,
                override_model=override_model,
                override_provider=override_provider,
                override_temperature=override_temperature
            )
            full_article += "\n\n" + conclusion_text

        actual_chars = len(full_article)
        logger.info(f"   📊 Итого: {actual_chars} символов (цель: {target_chars})")

        # Убрана связка _heart_expand и глобальный condense во избежание качелей "недопис -> раздув"

        return full_article

    def _condense_single_section(self, state, text, section_title, target_chars, override_model=None, override_provider=None) -> str:
        """Сократить конкретный раздел до его целевого объема."""
        user_msg = (
            f"РАЗДЕЛ ДЛЯ СОКРАЩЕНИЯ:\n{text}\n\n"
            f"Текущая длина: {len(text)} символов. Целевая длина: {target_chars} символов.\n"
            f"Сократи этот текст до целевого объема, сохранив его структуру (заголовок ##, списки, если есть, абзацы).\n"
            f"Убери лишнюю воду и повторы. Не урезай юридические ссылки и цифры.\n"
            f"Верни только измененный раздел в формате Markdown."
        )
        result = self._generate_clean_heart_text(
            user_msg, target_chars=target_chars, state=state,
            override_model=override_model,
            override_provider=override_provider
        )
        # Проверяем инварианты
        if len(result) < len(text) and len(result) >= target_chars * 0.85:
            # Заголовок должен быть на месте
            if result.strip().startswith("##"):
                return result
        logger.warning("   ⚠️ Сокращение отдельного раздела не уложилось в рамки или сломало заголовок. Откат на оригинал.")
        return text

    def _heart_condense(self, state, draft, target_chars):
        """Сократить статью, если она превысила лимит."""
        import re as _re
        actual_len = len(draft)
        
        # Динамический целевой объем для сжатия:
        # Если превышение небольшое (< 135% от базового таргета), сжимаем не до минимума,
        # а до верхнего лимита (1.15 * target_chars)
        if actual_len < target_chars * 1.35:
            condense_target = int(target_chars * 1.15)
        else:
            condense_target = target_chars

        overflow = actual_len - condense_target
        max_chars = int(condense_target * 1.05)
        logger.info(f"   ✂️ Сокращаю: {actual_len} → {condense_target} (убрать ~{overflow} символов)")

        h2_count_before = len(_re.findall(r'(?m)^##\s', draft))
        
        extra_instruction = ""
        if state.style_id == "checklist":
            extra_instruction = (
                f"\n\nПРИМЕЧАНИЕ ДЛЯ ЧЕК-ЛИСТА:\n"
                f"- СТРОГО сохрани структуру: ровно {state.num_checklist_items} содержательных пунктов (## 1. ... ## 10.). Не удаляй и не объединяй пункты!\n"
                f"- Каждый пункт должен остаться состоящим из коротких абзацев, списков действий («что делать») и блока ошибки с префиксом «Ошибка — ».\n"
                f"- Обязательно сохрани в конце разделы «Быстрая самопроверка» и «Послесловие»."
            )
            
        user_msg = (
            f"ЧЕРНОВИК СТАТЬИ ДЛЯ СОКРАЩЕНИЯ:\n{draft}\n\n"
            f"Сейчас в тексте: {actual_len} символов. Нужно убрать ~{overflow} символов, чтобы уложиться в {condense_target} (максимум {max_chars}).\n\n"
            f"КАК СОКРАЩАТЬ:\n"
            f"- Убери повторы, водянистые фразы и общие рассуждения.\n"
            f"- Сократи слишком раздутые примеры и кейсы.\n"
            f"- Не трогай ключевые факты, цифры и ссылки на законы.\n"
            f"- Сохрани структуру (все заголовки ## должны остаться).\n"
            f"- Запрещено использовать тире («—») в качестве связки (типа «подлежащее — сказуемое»). Избегай тире вообще, перестраивай предложения!\n"
            f"{extra_instruction}\n\n"
            f"Верни ПОЛНЫЙ сокращённый текст статьи в Markdown."
        )
        
        result = self._generate_clean_heart_text(user_msg, target_chars=condense_target, state=state)
        
        # Минимальный порог объема после сжатия: 85% для чек-листов, 80% для остальных стилей
        if state.style_id in ("checklist", "reference"):
            min_allowed = int(target_chars * 0.85)
        else:
            min_allowed = int(target_chars * 0.80)

        if len(result) > len(draft) or len(result) < min_allowed:
            logger.warning(f"   ⚠️ Сокращение не удалось ({len(result)} символов, требуется не менее {min_allowed}), оставляю оригинал")
            return draft

        if h2_count_before > 0:
            h2_count_after = len(_re.findall(r'(?m)^##\s', result))
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
            f"ФАКТЫ:\n{self._compact_json(state.facts, 5000)}\n\n"
            f"ЗАДАНИЕ:\n"
            f"1. Добавь подробные примеры и кейсы в каждый раздел\n"
            f"2. Расширь анализ с конкретными цифрами и ссылками на законы\n"
            f"3. Добавь блоки 'Что делать' с пошаговыми инструкциями\n"
            f"4. НЕ повторяй уже написанное — РАСШИРЯЙ и УГЛУБЛЯЙ\n\n"
            f"Верни ПОЛНЫЙ расширенный текст статьи."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars, state=state)
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

    def _blueprint_outline(self, blueprint: Dict, max_chars: int = 500) -> str:
        """Компактный outline плана — только H2-заголовки (~300-500 символов).

        Используется в _rewrite_section вместо полного JSON-дампа blueprint (3000 симв.),
        т.к. для точечной правки одного раздела нужна только структура статьи для контекста,
        а не всё содержимое плана. Экономия: ~2500 символов × N вызовов _rewrite_section.
        """
        if not isinstance(blueprint, dict):
            return ""
        headings = self._extract_sections(blueprint)
        if not headings:
            return ""
        # Нумерованный список заголовков
        lines = [f"{i+1}. {h}" for i, h in enumerate(headings)]
        outline = "\n".join(lines)
        if len(outline) > max_chars:
            outline = outline[:max_chars].rsplit("\n", 1)[0] + " …"
        return outline

    def _step_heart_revision(self, state: PipelineState):
        """Heart — доработка по фидбеку Sheriff."""
        logger.info("✍️ Heart: доработка по фидбеку Sheriff...")
        original_len = len(state.draft)
        target_chars = state.custom_chars or 8000
        min_chars = state.min_chars or int(target_chars * 0.85)
        max_chars = state.max_chars or int(target_chars * 1.15)
        
        style_block = self._get_style_block(state)
        
        conciseness_instruction = ""
        density = state.density_config
        if density:
            conciseness_instruction = (
                f"- Требования к плотности и ритму текста:\n"
                f"  1) Длина каждого абзаца: ориентировочно {density.get('sentences', '3-4')} предложения.\n"
                f"  2) Стиль изложения: {density.get('tone_style', 'бизнес-проза')}.\n"
                f"  3) Вступление (сценки-крючок): {density.get('hook_size', '1-2 абзаца')}.\n"
                f"- ⚠️ ВАЖНО: Статья должна оставаться полноценной журнальной прозой. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать сухими списками или перечислениями (телеграфный стиль).\n"
            )

        reduction_warning = ""
        if len(state.draft) > max_chars:
            reduction_warning = (
                f"\n- ⚠️ ВНИМАНИЕ: Текущий черновик имеет объем {len(state.draft)} символов, "
                f"что превышает лимит в {max_chars} символов. "
                f"Внося замечания Sheriff, ты ОБЯЗАН сократить текст статьи, чтобы итоговый объем "
                f"уложился в диапазон от {min_chars} до {max_chars} символов! "
                f"Вырезай воду, пиши лаконичнее.\n"
            )

        user_msg = (
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"ФИДБЕК ОТ РЕДАКТОРА (Sheriff):\n{self._compact_json(state.sheriff_review, 3000)}\n\n"
            f"{style_block}\n\n"
            f"⚠️ СТРОГОЕ ТРЕБОВАНИЕ К ОБЪЕМУ ПРИ РЕВИЗИИ:\n"
            f"- Диапазон: от {min_chars} до {max_chars} символов.\n"
            f"- Текст статьи ПОСЛЕ всех правок должен укладываться в этот объем.{reduction_warning}\n"
            f"{conciseness_instruction}"
            f"Внеси исправления, но не меняй структуру. Верни полный измененный текст статьи."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars, state=state)
        if len(result) > original_len * 0.3:
            state.draft = result

    def _step_heart_humanize(self, state: PipelineState):
        """Heart — humanization по фидбеку Mirror."""
        logger.info("✍️ Heart: humanization...")
        original_len = len(state.draft)
        target_chars = state.custom_chars or 8000
        min_chars = state.min_chars or int(target_chars * 0.85)
        max_chars = state.max_chars or int(target_chars * 1.15)
        
        style_block = self._get_style_block(state)
        
        conciseness_instruction = ""
        density = state.density_config
        if density:
            conciseness_instruction = (
                f"- Требования к плотности и ритму текста:\n"
                f"  1) Длина каждого абзаца: ориентировочно {density.get('sentences', '3-4')} предложения.\n"
                f"  2) Стиль изложения: {density.get('tone_style', 'бизнес-проза')}.\n"
                f"  3) Вступление (сценки-крючок): {density.get('hook_size', '1-2 абзаца')}.\n"
                f"- ⚠️ ВАЖНО: Статья должна оставаться полноценной журнальной прозой. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать сухими списками или перечислениями (телеграфный стиль).\n"
            )

        reduction_warning = ""
        if len(state.draft) > max_chars:
            reduction_warning = (
                f"\n- ⚠️ ВНИМАНИЕ: Текущий черновик имеет объем {len(state.draft)} символов, "
                f"что превышает лимит в {max_chars} символов. "
                f"Внося замечания Mirror, ты ОБЯЗАН сократить текст статьи, чтобы итоговый объем "
                f"уложился в диапазон от {min_chars} до {max_chars} символов! "
                f"Вырезай воду, пиши лаконичнее.\n"
            )

        user_msg = (
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"ФИДБЕК ОТ ЗЕРКАЛА (Mirror):\n{self._compact_json(state.mirror_review, 3000)}\n\n"
            f"{style_block}\n\n"
            f"⚠️ СТРОГОЕ ТРЕБОВАНИЕ К ОБЪЕМУ ПРИ РЕВИЗИИ:\n"
            f"- Диапазон: от {min_chars} до {max_chars} символов.\n"
            f"- Текст статьи ПОСЛЕ всех правок должен укладываться в этот объем.{reduction_warning}\n"
            f"{conciseness_instruction}"
            f"Внеси исправления для слома ИИ-ритма. Верни полный измененный текст статьи."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars, state=state)
        if len(result) > original_len * 0.3:
            state.draft = result

    def _step_combined_revision(self, state: PipelineState):
        """Heart — объединенная доработка по замечаниям Sheriff и Mirror."""
        logger.info("✍️ Heart: объединенная доработка по замечаниям...")
        original_len = len(state.draft)
        target_chars = state.custom_chars or 8000
        min_chars = state.min_chars or int(target_chars * 0.85)
        max_chars = state.max_chars or int(target_chars * 1.15)
        
        style_block = self._get_style_block(state)
        
        conciseness_instruction = ""
        density = state.density_config
        if density:
            conciseness_instruction = (
                f"- Требования к плотности и ритму текста:\n"
                f"  1) Длина каждого абзаца: ориентировочно {density.get('sentences', '3-4')} предложения.\n"
                f"  2) Стиль изложения: {density.get('tone_style', 'бизнес-проза')}.\n"
                f"  3) Вступление (сценки-крючок): {density.get('hook_size', '1-2 абзаца')}.\n"
                f"- ⚠️ ВАЖНО: Статья должна оставаться полноценной журнальной прозой. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать сухими списками или перечислениями (телеграфный стиль).\n"
            )

        reduction_warning = ""
        if len(state.draft) > max_chars:
            reduction_warning = (
                f"\n- ⚠️ ВНИМАНИЕ: Текущий черновик имеет объем {len(state.draft)} символов, "
                f"что превышает лимит в {max_chars} символов. "
                f"Внося замечания Sheriff и Mirror, ты ОБЯЗАН сократить текст статьи, чтобы итоговый объем "
                f"уложился в диапазон от {min_chars} до {max_chars} символов! "
                f"Вырезай воду, пиши лаконичнее.\n"
            )

        user_msg = (
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"ФИДБЕК ОТ РЕДАКТОРА (Sheriff):\n{self._compact_json(state.sheriff_review, 3000)}\n\n"
            f"ФИДБЕК ОТ ЗЕРКАЛА (Mirror):\n{self._compact_json(state.mirror_review, 3000)}\n\n"
            f"{style_block}\n\n"
            f"⚠️ СТРОГОЕ ТРЕБОВАНИЕ К ОБЪЕМУ ПРИ РЕВИЗИИ:\n"
            f"- Диапазон: от {min_chars} до {max_chars} символов.\n"
            f"- Текст статьи ПОСЛЕ всех правок должен укладываться в этот объем.{reduction_warning}\n"
            f"{conciseness_instruction}"
            f"Внеси все указанные исправления за один проход и верни ВЕСЬ измененный текст статьи целиком."
        )
        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars, state=state)
        if len(result) > original_len * 0.3:
            state.draft = result
            state.sheriff_iterations += 1
        logger.info(f"   📏 Draft после объединённой ревизии: {len(state.draft)} символов")

    # ────────────────────────────────────────────
    # Хирургический редактор: правки по разделам, а не всей статьи
    # ────────────────────────────────────────────

    def _split_markdown_sections(self, text: str) -> list:
        """Разбить markdown на блоки по H2 (## ). Срезы непрерывны: join(raw)==text.

        Возвращает список {"level": 0|2, "heading": str, "raw": str}. level 0 —
        вступление до первого H2. H3/H4 остаются внутри своего H2-блока.
        """
        import re
        matches = list(re.finditer(r"(?m)^##\s+.*$", text))
        if not matches:
            return [{"level": 0, "heading": "", "raw": text}]
        blocks = []
        if matches[0].start() > 0:
            blocks.append({"level": 0, "heading": "", "raw": text[:matches[0].start()]})
        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            heading = m.group().lstrip("#").strip()
            blocks.append({"level": 2, "heading": heading, "raw": text[start:end]})
        return blocks

    def _reassemble_sections(self, blocks: list) -> str:
        return "".join(b["raw"] for b in blocks)

    def _raw_json_call(self, system_prompt: str, user_message: str, state=None,
                       max_tokens: int = 2000, temperature: float = 0.1) -> dict:
        """Вспомогательный строгий JSON-вызов (не зарегистрированный агент).

        Идёт через _chat_completion (retry/backoff + json-mode + fallback) и
        устойчивый парсер. Выбор провайдера/модели — как в _call_agent.
        """
        base_agent = get_agent("sheriff")
        current_client = self.deepseek_client
        model_name = base_agent.model
        if state is not None:
            provider = getattr(state, "provider", "deepseek").lower()
            custom_model = getattr(state, "model", None)
            if provider == "kie":
                model_name = custom_model or MODELS["kie_text"]
                current_client = self._get_kie_client(model_name)
            elif provider == "openai":
                current_client = self.openai_client
                model_name = custom_model or MODELS["openai_text"]
            elif provider == "deepseek" and custom_model:
                model_name = custom_model
        resp = self._chat_completion(
            current_client,
            response_format={"type": "json_object"},
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt + " Ответ строго в формате JSON."},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = self._parse_json_response(raw, "edit_planner")
        return parsed if isinstance(parsed, dict) else {}

    def _build_edit_plan(self, state: 'PipelineState', sections: list) -> list:
        """Сопоставить замечания Sheriff/Mirror с конкретными разделами (строгий JSON).

        Возвращает отвалидированный список {"section_index","reason","instruction"}
        только для существующих H2-разделов. Пустой список = править нечего.
        """
        lines = []
        for i, b in enumerate(sections):
            body = b["raw"].strip().replace("\n", " ")
            if b["level"] == 2:
                preview = body[:400] + ("…" if len(body) > 400 else "")
                lines.append(f"[{i}] {preview}")
            else:
                preview = body[:120] + ("…" if len(body) > 120 else "")
                lines.append(f"[{i}] (вступление) {preview}")
        catalog = "\n".join(lines)

        sheriff = json.dumps(state.sheriff_review or {}, ensure_ascii=False)
        mirror = json.dumps(getattr(state, "mirror_review", {}) or {}, ensure_ascii=False)
        if len(sheriff) > 4000:
            sheriff = sheriff[:4000] + "…"
        if len(mirror) > 4000:
            mirror = mirror[:4000] + "…"

        system = (
            "Ты — планировщик точечных правок статьи. Тебе дают статью, разбитую на "
            "пронумерованные разделы, и замечания редактора (Sheriff) и стилиста (Mirror). "
            "Определи, какие КОНКРЕТНЫЕ разделы нужно поправить и что именно изменить. "
            "НЕ переписывай текст. Включай ТОЛЬКО разделы, где правка реально нужна "
            "(чем меньше, тем лучше). Формат строго: "
            "{\"edits\":[{\"section_index\":<целое>,\"reason\":\"<кратко>\","
            "\"instruction\":\"<что именно сделать в этом разделе>\"}]}. "
            "Если правок нет — верни {\"edits\":[]}."
        )
        user = (
            f"РАЗДЕЛЫ СТАТЬИ (индекс в квадратных скобках):\n{catalog}\n\n"
            f"ЗАМЕЧАНИЯ РЕДАКТОРА (Sheriff):\n{sheriff}\n\n"
            f"ЗАМЕЧАНИЯ СТИЛИСТА (Mirror):\n{mirror}\n\n"
            f"Верни JSON со списком правок по разделам."
        )
        try:
            data = self._raw_json_call(system, user, state=state, max_tokens=2000, temperature=0.1)
        except Exception as e:
            logger.warning(f"   ⚠️ Планировщик правок недоступен ({e}); fallback на полную ревизию.")
            return []

        raw_edits = data.get("edits") if isinstance(data, dict) else None
        if not isinstance(raw_edits, list):
            return []
        edits = []
        for e in raw_edits:
            if not isinstance(e, dict):
                continue
            idx = e.get("section_index")
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(sections)) or sections[idx]["level"] != 2:
                continue
            reason = str(e.get("reason", "")).strip()
            instruction = str(e.get("instruction", "") or reason).strip()
            edits.append({"section_index": idx, "reason": reason, "instruction": instruction})
        return edits

    def _section_ok(self, orig: dict, new: str) -> bool:
        s = (new or "").strip()
        if len(s) < 40 or not s.startswith("##"):
            return False
        olen = len(orig["raw"])
        if len(s) < olen * 0.3 or len(s) > olen * 3 + 500:
            return False
        return True

    def _rewrite_section(self, state: 'PipelineState', sec: dict,
                         prev_tail: str, next_head: str, instruction: str,
                         override_model=None, override_provider=None, override_temperature=None) -> str:
        """Переписать ОДИН раздел с контекстом стыков и общим планом."""
        target = int(len(sec["raw"]) * 1.15)
        # Только заголовки плана — для точечной правки нужен контекст структуры, не весь план.
        # Раньше слался полный JSON blueprint (до 3000 симв.); экономия ~2500 симв. на вызов.
        outline = self._blueprint_outline(getattr(state, "blueprint", {}) or {})
        bp_block = outline if outline else json.dumps(getattr(state, "blueprint", {}) or {}, ensure_ascii=False)[:1500]
        msg = (
            f"Ты редактируешь ОДИН раздел статьи, не трогая остальные.\n\n"
            f"ТЕМА СТАТЬИ: {state.topic}\n\n"
            f"СТРУКТУРА СТАТЬИ (для логики; НЕ переписывай её):\n{bp_block}\n\n"
            f"КОНЕЦ ПРЕДЫДУЩЕГО РАЗДЕЛА (только для плавного стыка; НЕ повторяй и НЕ переписывай):\n…{prev_tail}\n\n"
            f"НАЧАЛО СЛЕДУЮЩЕГО РАЗДЕЛА (только для стыка; НЕ переписывай):\n{next_head}…\n\n"
            f"ТЕКУЩИЙ РАЗДЕЛ (именно его нужно переписать):\n{sec['raw']}\n\n"
            f"ЧТО ИСПРАВИТЬ В ЭТОМ РАЗДЕЛЕ:\n{instruction or '- общая шлифовка по замечаниям'}\n\n"
            f"ТРЕБОВАНИЯ:\n"
            f"- Верни ТОЛЬКО переписанный этот раздел в Markdown, начиная с того же заголовка '{'#'*sec.get('level',2)} {sec['heading']}'.\n"
            f"- Сохрани этот же заголовок и примерно тот же объём (~{len(sec['raw'])} символов).\n"
            f"- Текст должен логично продолжать предыдущий раздел и подводить к следующему.\n"
            f"- НЕ добавляй и НЕ удаляй заголовки H2; не пиши ничего вне этого раздела.\n"
        )
        result = self._generate_clean_heart_text(
            msg, target_chars=target, state=state,
            override_model=override_model, override_provider=override_provider,
            override_temperature=override_temperature)
        return (result or "").strip()

    def _heart_patch(self, state: 'PipelineState'):
        """Хирургический редактор статьи (замена полной перепиписи в цикле ревизий).

        Правит только проблемные H2-разделы, сохраняя цельность; при широких
        правках/сбое откатывается к _step_combined_revision.
        """
        import os
        import re as _re
        if os.getenv("HEART_PATCH_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
            return self._step_combined_revision(state)

        draft = state.draft or ""
        sections = self._split_markdown_sections(draft)
        editable = [i for i, b in enumerate(sections) if b["level"] == 2]
        if len(editable) < 2:
            logger.info("   ℹ️ Мало H2-разделов для хирургии — полная ревизия.")
            return self._step_combined_revision(state)

        plan = self._build_edit_plan(state, sections)
        if not plan:
            logger.info("   ℹ️ План правок пуст/нечитаем — полная ревизия.")
            return self._step_combined_revision(state)

        flagged = sorted({p["section_index"] for p in plan})
        if len(flagged) / max(1, len(editable)) > 0.5:
            logger.info(f"   ℹ️ Правок много ({len(flagged)}/{len(editable)} > 50%) — полная ревизия.")
            return self._step_combined_revision(state)

        instr_by_idx = {}
        for p in plan:
            instr_by_idx.setdefault(p["section_index"], []).append(p["instruction"])

        new_sections = list(sections)
        changed = 0
        for i in flagged:
            sec = sections[i]
            prev_raw = sections[i - 1]["raw"] if i - 1 >= 0 else ""
            next_raw = sections[i + 1]["raw"] if i + 1 < len(sections) else ""
            prev_tail = prev_raw.strip()[-400:]
            next_head = next_raw.strip()[:400]
            instruction = "\n".join(f"- {t}" for t in instr_by_idx.get(i, []) if t)
            rewritten = self._rewrite_section(state, sec, prev_tail, next_head, instruction)
            if self._section_ok(sec, rewritten):
                trailing = sec["raw"][len(sec["raw"].rstrip()):]  # сохранить исходный хвост (\n\n)
                new_sections[i] = {**sec, "raw": rewritten.rstrip() + trailing}
                changed += 1
            else:
                logger.warning(f"   ⚠️ Раздел [{i}] переписан некорректно — оставляю оригинал.")

        if changed == 0:
            logger.info("   ℹ️ Ни один раздел не изменён — полная ревизия (fallback).")
            return self._step_combined_revision(state)

        new_draft = self._reassemble_sections(new_sections)

        if not self.assert_invariants(draft, new_draft):
            logger.warning("   ⚠️ [Инвариант] Хирургические правки нарушили инварианты структуры. Откат к полной ревизии.")
            return self._step_combined_revision(state)

        state.draft = new_draft
        state.sheriff_iterations += 1
        logger.info(
            f"   🩹 Хирургические правки: {changed} из {len(editable)} разделов; "
            f"объём {len(draft)}→{len(new_draft)} символов."
        )

    def _get_sheriff_guidance(self, state: 'PipelineState') -> str:
        """Чек-лист для Sheriff: инструкция из стиля (приоритет) + чек-лист паттерна.

        Раньше sheriff_instruction (styles_config) и sheriff_checklist (patterns)
        НЕ доходили до агента. Теперь подмешиваются в его запрос.
        """
        parts = []
        sid = getattr(state, "style_id", "") or ""
        if sid:
            try:
                from .styles import get_style
                st = get_style(sid)
                instr = getattr(st, "sheriff_instruction", None)
                if instr:
                    parts.append(instr.strip())
            except Exception:
                pass
        key = sid if sid in PATTERNS else getattr(state, "article_type", "")
        pat = PATTERNS.get(key) or PATTERNS.get(getattr(state, "article_type", ""), {})
        if isinstance(pat, dict):
            chk = pat.get("sheriff_checklist")
            if chk:
                import re as _re
                # Динамически адаптируем требования объема в чек-листе Sheriff под текущие лимиты
                if state.min_chars > 0 and state.max_chars > 0:
                    chk = _re.sub(r"[Оо]бъём\s+\d+[\s\d–\-]*\s+символов\?", f"объём строго от {state.min_chars} до {state.max_chars} символов?", chk)
                if chk.strip() not in parts:
                    parts.append(chk.strip())

        # Добавляем SEO-инструкции для проверки (если они есть)
        seo_chk = state.seo_instructions.get("sheriff_instruction") if state.seo_instructions else None
        if seo_chk:
            parts.append(seo_chk.strip())

        # Добавляем требования к плотности (если они есть)
        density = getattr(state, "density_config", None)
        if density:
            sentences_val = density.get("sentences", "3-4")
            tone_val = density.get("tone_style", "бизнес-проза")
            hook_val = density.get("hook_size", "1-2 абзаца")
            
            # Извлекаем максимальное число предложений
            max_s = 4
            try:
                max_s = int(sentences_val.split('-')[-1])
            except Exception:
                pass
                
            density_chk = (
                f"Проверь плотность и ритмику текста на соответствие целевому объему:\n"
                f"  - Длина абзацев рекомендуется в среднем {sentences_val} предложения. Указывай замечания только на абзацы, содержащие более {max_s + 2} предложений (явно раздутые).\n"
                f"  - Тон и стиль изложения: {tone_val}.\n"
                f"  - Объем введения (сценки-крючка): {hook_val}.\n"
                f"  - Убедись, что статья написана плавной журнальной прозой, а не превратилась в сухой список правил (телеграфный стиль)."
            )
            parts.append(density_chk.strip())

        if state.style_id == "checklist" and hasattr(state, "num_checklist_items"):
            num = state.num_checklist_items
            for idx, part in enumerate(parts):
                part = part.replace("10 пунктов", f"{num} пунктов")
                part = part.replace("10 нумерованных", f"{num} нумерованных")
                part = part.replace("каждого из 10", f"каждого из {num}")
                part = part.replace("## 10.", f"## {num}.")
                part = part.replace("## 1. ... ## 10.", f"## 1. ... ## {num}.")
                parts[idx] = part

        if not parts:
            return ""
        return (
            "\n\nОБЯЗАТЕЛЬНЫЙ ЧЕК-ЛИСТ ПРОВЕРКИ (учти КАЖДЫЙ пункт в вердикте и комментариях):\n- "
            + "\n- ".join(parts)
        )

    def _step_sheriff(self, state: PipelineState):
        """Шериф (Редактор) — проверка качества и фактов."""
        logger.info("👮 [6/8] Sheriff: проверка черновика статьи...")
        guidance = self._get_sheriff_guidance(state)
        user_msg = (
            f"ТЕМА СТАТЬИ: {state.topic}\n"
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"Выполни строгую проверку качества черновика."
            f"{guidance}"
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
        try:
            from .humanizer import analyze_article
            _rep = analyze_article(self._split_markdown_sections(state.draft))
            user_msg += (
                f"\n\nIZMERENNYE METRIKI (obyektivno, poscitano kodom, ne na glaz): "
                f"human_score={_rep['article_human_score']}/100. "
                f"Opiraysya na eto chislo pri vystavlenii turing_score, a ne ugaday."
            )
        except Exception:
            pass
        response = self._call_agent("mirror", user_msg, parse_json=True, state=state)
        state.mirror_review = response
        
        if isinstance(response, dict):
            turing_score = response.get("turing_score", 95)
            state.mirror_review["verdict"] = "pass" if turing_score >= 80 else "fail"
        else:
            state.mirror_review["verdict"] = "pass"
            
        logger.info(f"   🪞 Mirror вердикт: {state.mirror_review['verdict']} (Turing Score: {state.mirror_review.get('turing_score', 95)})")

    def assert_invariants(self, before: str, after: str) -> bool:
        """Проверить, что структура и целостность текста не пострадали при изменении.
        Возвращает True, если инварианты соблюдены, иначе False.
        """
        import re
        
        # 1. Проверка числа H2-заголовков
        h2_before = re.findall(r'^##\s+(.+)$', before, re.MULTILINE)
        h2_after = re.findall(r'^##\s+(.+)$', after, re.MULTILINE)
        if len(h2_before) != len(h2_after):
            logger.warning(f"   ⚠️ [Инвариант] Число H2 изменилось: {len(h2_before)} -> {len(h2_after)}")
            return False
            
        # 2. Проверка уровней заголовков: допускаются только H2 и H3
        all_headings = re.findall(r'^(#+)\s+.*$', after, re.MULTILINE)
        for h in all_headings:
            if len(h) > 3 or len(h) == 0:
                logger.warning(f"   ⚠️ [Инвариант] Обнаружен недопустимый уровень заголовка: {h}")
                return False
                
        # 3. Ни один раздел не вырос/сжался > ±15%
        def get_section_lengths(text):
            sections_content = re.split(r'^##\s+.+$', text, flags=re.MULTILINE)
            return [len(s) for s in sections_content]
            
        lens_before = get_section_lengths(before)
        lens_after = get_section_lengths(after)
        
        if len(lens_before) != len(lens_after):
            logger.warning("   ⚠️ [Инвариант] Не совпадает структура разделов при разделении по H2")
            return False
            
        for idx, (l_b, l_a) in enumerate(zip(lens_before, lens_after)):
            limit_pct = 0.20 if idx == 0 else 0.15
            if l_b > 0:
                diff_pct = abs(l_a - l_b) / l_b
                if diff_pct > limit_pct:
                    logger.warning(f"   ⚠️ [Инвариант] Раздел {idx} изменился более чем на {limit_pct*100}%: {l_b} -> {l_a} (изменение: {diff_pct:.1%})")
                    return False

        # 4. Нет обрыва на полуслове
        tail = after.rstrip()
        if tail and tail[-1] not in '.!?»")*':
            logger.warning(f"   ⚠️ [Инвариант] Текст обрывается на полуслове: '{tail[-20:]}'")
            return False
            
        # 5. Нет утёкших системных или XML тегов
        for tag in ["<seo_metadata>", "</seo_metadata>", "<optimized_article>", "</optimized_article>", "<structure_revision>", "</structure_revision>"]:
            if tag in after:
                logger.warning(f"   ⚠️ [Инвариант] Обнаружена утечка служебного тега: {tag}")
                return False
                
        return True

    def _step_statistical_humanize(self, state: PipelineState):
        """Точечная статистическая хуманизация ПОСЛЕ SEO.

        Анализ метрик (burstiness/σ/TTR/штампы) — 0 токенов.
        LLM-вызовов = только провальные секции, не более max_fix, каждая 1 раз.
        accept-best: переписанное принимается, только если human_score вырос.
        """
        from .humanizer import humanize_article

        text = state.final_article or state.draft
        if not text or len(text) < 100:
            return
        logger.info("🧪 [Humanizer] Точечная статистическая хуманизация...")

        # Динамический выбор модели для хуманизации секций
        suggested = self._suggest_draft_model(state.topic, state.article_type, state.description)
        hum_provider = "deepseek"
        hum_model = MODELS["deepseek_pro"]

        if self._kie_api_key:
            if suggested["provider"] == "kie" and suggested["model"] == "claude-opus-4-8":
                hum_provider = "kie"
                hum_model = "claude-opus-4-8"
            else:
                hum_provider = "kie"
                hum_model = "gemini-3.1-pro"

        logger.info(f"   🤖 Модель для хуманизации секций: {hum_provider} / {hum_model}")

        def _rewrite(section_raw, instruction, prev_tail, next_head):
            import re as _re
            mh = _re.match(r"^\s*(#{1,6})\s+(.*)", section_raw)
            level = len(mh.group(1)) if mh else 2
            heading = mh.group(2).strip() if mh else ""
            sec = {"raw": section_raw, "level": level, "heading": heading}
            return self._rewrite_section(
                state, sec, prev_tail, next_head, instruction,
                override_model=hum_model,
                override_provider=hum_provider,
                override_temperature=0.85,   # выше базовой 0.75 — для живого ритма при переписи
            )

        try:
            result = humanize_article(
                text,
                split_fn=self._split_markdown_sections,
                reassemble_fn=self._reassemble_sections,
                rewrite_fn=_rewrite,
                min_score=75, max_fix=3, editable_levels=(2, 3),
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"   ⚠️ [Humanizer] Ошибка хуманизации: {e}. Пропуск.")
            return

        state.humanize_report = {
            "score_before": result["score_before"],
            "score_after": result["score_after"],
            "rewrites": result["rewrites"],
            "details": result["details"],
        }
        
        # Хуманизация должна быть точечной и не терять объём.
        # Если результат незначительно превысил лимит — программно обрезаем
        # до лимита вместо полного отката (сохраняем score-улучшение).
        soft_overhead = 200  # допускаем перебор до 200 символов сверх max_chars
        if result["text"]:
            result_len = len(result["text"])
            orig_len = len(text)
            max_allowed = (state.max_chars or orig_len * 1.2) + soft_overhead

            if result_len >= orig_len * 0.95 and result_len <= max_allowed:
                if self.assert_invariants(text, result["text"]):
                    # Если чуть-чуть превысили max_chars — програмно обрежем
                    if result_len > (state.max_chars or orig_len * 1.2):
                        trimmed = self._programmatic_trim(result["text"], state.max_chars or int(orig_len * 1.2))
                        if len(trimmed) >= orig_len * 0.95:
                            state.final_article = self._clean_leaked_ai_artifacts(trimmed)
                            logger.info("   ✅ Хуманизация завершена (програмно обрезана до лимита, инварианты соблюдены).")
                        else:
                            # Обрезка слишком агрессивна — берём оригинал
                            logger.warning("   ⚠️ [Humanizer] Програмная обрезка слишком агрессивна. Откат на оригинал.")
                    else:
                        state.final_article = self._clean_leaked_ai_artifacts(result["text"])
                        logger.info("   ✅ Хуманизация успешно завершена (инварианты соблюдены).")
                else:
                    logger.warning("   ⚠️ [Humanizer] Результат нарушил инварианты структуры. Откат на оригинал.")
            else:
                logger.warning(
                    f"   ⚠️ [Humanizer] Результат слишком короткий ({result_len} < {orig_len * 0.95:.0f}) "
                    f"или сильно превысил лимит ({result_len} > {max_allowed}). Откат."
                )

    def _apply_booster_edits(self, text: str, citation_baits: list, lsi_replacements: list, state: PipelineState = None) -> str:
        """Программное наложение правок от Booster (Citation Bait и LSI-ключи) на текст статьи.

        Каждая правка применяется по отдельности с проверкой assert_invariants.
        Если правка нарушает инварианты — она пропускается (только логируется),
        остальные правки продолжают применяться. Это устраняет проблему,
        когда одна нарушающая правка отбрасывала все SEO-улучшения.
        """
        import re

        original_text = text
        applied_lsi = 0
        applied_baits = 0

        # 1. LSI replacements — по одной с проверкой инвариантов
        if lsi_replacements and isinstance(lsi_replacements, list):
            for rep in lsi_replacements:
                if not isinstance(rep, dict):
                    continue
                find_str = rep.get("find") or rep.get("original")
                replace_str = rep.get("replace") or rep.get("enhanced")
                if find_str and replace_str:
                    find_str = find_str.strip()
                    replace_str = replace_str.strip()
                    if not find_str:
                        continue
                    candidate = text.replace(find_str, replace_str)
                    if candidate != text:
                        # Проверяем, не сломала ли замена структуру
                        if self.assert_invariants(text, candidate):
                            text = candidate
                            applied_lsi += 1
                        else:
                            logger.info(f"      ⏭️ LSI-замена '{find_str[:30]}...' пропущена (нарушила инварианты).")
                 
        # 2. Citation Baits — по одной с проверкой инвариантов
        if citation_baits and isinstance(citation_baits, list):
            for bait in citation_baits:
                if not isinstance(bait, dict):
                    continue
                section_name = bait.get("section", "").strip()
                bait_text = bait.get("bait_text", "").strip()
                if not section_name or not bait_text:
                    continue

                # Очистка названия секции от префиксов вроде "## " или "H2: "
                clean_sec = section_name
                if clean_sec.startswith("##"):
                    clean_sec = clean_sec.lstrip("#").strip()
                elif clean_sec.lower().startswith("h2:"):
                    clean_sec = clean_sec[3:].strip()

                # Ищем заголовок H2 в тексте
                escaped_sec = re.escape(clean_sec)
                pattern = r"^##\s*" + escaped_sec + r"\s*$"
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    h2_pos = match.start()
                    before_text = text[:h2_pos]
                    lines = before_text.splitlines()

                    # Ищем последний непустой абзац, не являющийся заголовком или картинкой
                    target_line_idx = -1
                    for idx in range(len(lines) - 1, -1, -1):
                        line_strip = lines[idx].strip()
                        if line_strip and not line_strip.startswith("#") and not line_strip.startswith("[") and not line_strip.startswith("|"):
                            target_line_idx = idx
                            break

                    if target_line_idx != -1:
                        orig_line = lines[target_line_idx]
                        if orig_line.endswith(".") or orig_line.endswith("?") or orig_line.endswith("!"):
                            lines[target_line_idx] = orig_line + " " + bait_text
                        else:
                            lines[target_line_idx] = orig_line + ". " + bait_text

                        candidate = "\n".join(lines) + text[h2_pos:]
                        if self.assert_invariants(text, candidate):
                            text = candidate
                            applied_baits += 1
                        else:
                            logger.info(f"      ⏭️ Citation Bait для '{clean_sec[:30]}...' пропущен (нарушил инварианты).")

        if applied_lsi or applied_baits:
            logger.info(f"      ✅ Booster-правки применены: {applied_lsi} LSI + {applied_baits} Citation Baits.")
        else:
            logger.info(f"      ℹ️ Ни одна Booster-правка не прошла проверку инвариантов.")
        return text

    def _derive_description_from_article(self, article: str, max_len: int = 160) -> str:
        """Вывести meta description из первого содержательного абзаца (а не заглушки)."""
        if not article:
            return ""
        import re as _re
        for raw in article.split("\n"):
            line = raw.strip()
            if not line:
                continue
            # пропускаем заголовки, изображения, списки, разметку
            if line.startswith(("#", "!", ">", "|", "-", "*", "`")):
                continue
            line = _re.sub(r"[*_`#>\[\]]", "", line).strip()
            if len(line) < 40:
                continue
            if len(line) > max_len:
                cut = line[:max_len].rsplit(" ", 1)[0]
                return cut.rstrip(",.;: ") + "…"
            return line
        return ""

    def _step_booster(self, state: PipelineState):
        """Шаг 8: Booster — SEO/GEO оптимизация."""
        logger.info("🚀 [8/9] Booster: SEO/GEO оптимизация...")
        
        user_msg = (
            f"ТЕМА СТАТЬИ: {state.topic}\n"
            f"ТИП СТАТЬИ: {state.article_type}\n"
            f"НАПРАВЛЕНИЕ: {state.direction}\n\n"
            f"ЧЕРНОВИК СТАТЬИ:\n{state.draft}\n\n"
            f"- Твой БЮДЖЕТ на SEO-добавки: ровно {state.seo_budget} символов. Это всё, что ты можешь добавить.\n"
            f"- Citation Bait: {'подготовь ОДНУ наживку (40-50 слов) на всю статью (компактный режим).' if getattr(state, 'compact_mode', False) else 'подготовь по одной наживке (40-50 слов) на каждый раздел H2.'}\n"
            f"- LSI-ключи: найди предложения в тексте и подготовь замены, чтобы встроить ключевые слова.{' Максимум 2 замены (компактный режим).' if getattr(state, 'compact_mode', False) else ''}\n"
            f"- FAQ: добавь в JSON-поле 'faq' (для Schema.org), но НЕ вставляй блок FAQ в тело статьи.\n"
            f"- Категорически ЗАПРЕЩЕНО добавлять новые разделы H2/H3.\n\n"
            f"Верни точечные правки в формате JSON."
        )
        
        seo_package = self._call_agent("booster", user_msg, parse_json=True, state=state)
        state.seo_package = seo_package
        
        booster_truncated = bool(getattr(state, "last_call_truncated", False))
        
        if booster_truncated or not seo_package or not isinstance(seo_package, dict):
            logger.warning("   ⚠️ Booster: вызов оборван или некорректный ответ. Откат на черновик.")
            optimized_text = state.draft
        else:
            citation_baits = seo_package.get("citation_baits", [])
            lsi_replacements = seo_package.get("lsi_insertions") or seo_package.get("lsi_replacements") or []

            # Применяем правки программно — каждая проверяется отдельно на инварианты.
            # Если хоть одна правка прошла — считаем Booster успешным.
            optimized_text = self._apply_booster_edits(state.draft, citation_baits, lsi_replacements, state=state)

            # Финальная проверка длины (инварианты уже проверены внутри _apply_booster_edits)
            if len(optimized_text) > (state.max_chars or len(state.draft) * 1.2):
                logger.warning(f"   ⚠️ [Booster] Объём после правок превысил лимит ({len(optimized_text)} > {state.max_chars}). Откат.")
                optimized_text = state.draft
            elif len(optimized_text) == len(state.draft):
                # Ни одна правка не прошла
                logger.info("   ℹ️ Booster: ни одна правка не прошла (исходный текст без изменений).")
                optimized_text = state.draft
            else:
                logger.info("   ✅ SEO-правки Booster успешно наложены.")

        state.final_article = optimized_text
        state.final_article = self._clean_leaked_ai_artifacts(state.final_article)
        state.final_article = self._apply_stopwords_cleanup(state.final_article)
        
        # Извлекаем и дополняем метаданные
        state.final_meta = (state.seo_package.get("meta", {}) or {}) if state.seo_package else {}
        fallback_meta = _extract_meta_from_text(state.final_article)
        for k in ["title", "description", "keywords"]:
            if not state.final_meta.get(k) and fallback_meta.get(k):
                state.final_meta[k] = fallback_meta[k]
                logger.info(f"      📝 Извлечено {k} из текста: '{str(fallback_meta[k])[:50]}...'")

        # #3 SEO-мета: гарантируем непустые keywords/description в HTML (раньше пробрасывались пустыми).
        # keywords: meta.keywords → seo_package.keywords → ключи из контракта (state.keywords)
        if not state.final_meta.get("keywords"):
            sp_keywords = state.seo_package.get("keywords") if isinstance(state.seo_package, dict) else None
            if sp_keywords:
                state.final_meta["keywords"] = sp_keywords
                logger.info("      📝 keywords взяты из seo_package (top-level).")
            elif getattr(state, "keywords", None):
                state.final_meta["keywords"] = list(state.keywords)
                logger.info(f"      📝 keywords взяты из контракта (state.keywords): {len(state.keywords)} шт.")
        # description: если пусто — берём первый содержательный абзац статьи (не заглушку).
        if not state.final_meta.get("description"):
            derived = self._derive_description_from_article(state.final_article)
            if derived:
                state.final_meta["description"] = derived
                logger.info(f"      📝 description выведено из первого абзаца: '{derived[:50]}...'")

        # Генерация подзаголовка статьи на основе темы
        if not state.final_meta.get("subtitle"):
            try:
                sub_msg = f"Напиши один вовлекающий, лаконичный подзаголовок (лид-абзац, 15-25 слов) для B2B-статьи на тему: '{state.topic}'. Без кавычек, без двоеточий, без тире в качестве связок. Сразу пиши суть."
                subtitle_text = self._call_agent("heart", sub_msg, parse_json=False, state=state)
                cleaned_sub = subtitle_text
                cleaned_sub = re.sub(r'<.*?>', '', cleaned_sub)
                cleaned_sub = re.sub(r'\{.*?\}', '', cleaned_sub, flags=re.DOTALL)
                state.final_meta["subtitle"] = cleaned_sub.strip().strip('"«» ')
                logger.info(f"      📝 Подзаголовок успешно сгенерирован: '{state.final_meta['subtitle'][:50]}...'")
            except Exception as e:
                logger.warning(f"   ⚠️ Не удалось сгенерировать подзаголовок: {e}")
                state.final_meta["subtitle"] = state.final_meta.get("description", "")
        
        # Фиксация красивого заголовка H1
        best_title = state.final_meta.get("title") or state.final_meta.get("h1") or state.topic
        h_match = re.search(r"^(#{1,2}\s+.*?)$", state.final_article, re.MULTILINE)
        if h_match:
            h_line = h_match.group(1)
            state.final_article = state.final_article.replace(h_line, f"# {best_title}", 1)
            logger.info(f"      📝 Заголовок H1 заменен на красивый заголовок: '# {best_title}'")
        else:
            state.final_article = f"# {best_title}\n\n" + state.final_article
            logger.info(f"      📝 Заголовок H1 добавлен в начало статьи: '# {best_title}'")
 
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
                if state.style_id == "checklist":
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
            if not self._kie_api_key:
                logger.warning("   ⚠️ KIE_API_KEY не задан в окружении. Пропускаю генерацию картинок.")
                raise ValueError("KIE_API_KEY не задан")
                
            api_key = self._kie_api_key
            base_url = "https://api.kie.ai/v1"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            def _generate_image_with_fallback(prompt: str, size: str, model: str = MODELS["openai_image_primary"]) -> Any:
                """Внутренний хелпер для генерации картинок через KIE.ai с поддержкой повторных попыток."""
                
                # Если используется модель gpt-image, идем через официальное асинхронное KIE API
                if model.startswith("gpt-image"):
                    # Определяем aspect_ratio на основе запрашиваемого размера
                    # Обложка: 1536x768 (соотношение 2:1)
                    # Разделители: 1536x384 (соотношение 4:1, используем 3:1 как наиболее близкий поддерживаемый)
                    aspect_ratio = "2:1" if size == "1536x768" else "3:1"
                    
                    payload = {
                        "model": model,
                        "input": {
                            "prompt": prompt,
                            "aspect_ratio": aspect_ratio,
                            "resolution": "1K"
                        }
                    }
                    
                    max_img_retries = 3
                    img_delay = 5.0
                    for attempt in range(1, max_img_retries + 1):
                        try:
                            logger.info(f"   🚀 Создание задачи генерации KIE (Модель: {model}, Aspect: {aspect_ratio}, попытка {attempt}/{max_img_retries})...")
                            with httpx.Client(timeout=120.0) as http_client:
                                resp = http_client.post("https://api.kie.ai/api/v1/jobs/createTask", json=payload, headers=headers)
                                resp.raise_for_status()
                                resp_data = resp.json()
                                
                                code = resp_data.get("code")
                                if code != 200:
                                    raise RuntimeError(f"KIE createTask вернул код {code}: {resp_data.get('msg')}")
                                    
                                task_id = resp_data.get("data", {}).get("taskId")
                                if not task_id:
                                    raise RuntimeError(f"KIE createTask не вернул taskId: {resp_data}")
                                    
                                logger.info(f"   ⏳ Задача KIE создана: {task_id}. Начинаю опрос статуса...")
                                
                                # Цикл опроса (polling)
                                poll_attempts = 24  # 24 * 5 секунд = 120 секунд максимум
                                for poll_idx in range(poll_attempts):
                                    import time
                                    time.sleep(5.0)
                                    
                                    status_resp = http_client.get(f"https://api.kie.ai/api/v1/jobs/recordInfo?taskId={task_id}", headers=headers)
                                    status_resp.raise_for_status()
                                    status_data = status_resp.json()
                                    
                                    status_code = status_data.get("code")
                                    if status_code != 200:
                                        logger.warning(f"   ⚠️ recordInfo вернул код {status_code}: {status_data.get('msg')}. Продолжаю опрос...")
                                        continue
                                        
                                    task_data = status_data.get("data", {})
                                    success_flag = task_data.get("successFlag")
                                    state_val = task_data.get("state") or task_data.get("status")
                                    
                                    # Успешное завершение
                                    if success_flag == 1 or state_val == "success":
                                        logger.info(f"   ✅ Генерация изображения KIE успешно завершена за {(poll_idx + 1) * 5} сек.")
                                        return task_data
                                    # Ошибка
                                    elif success_flag in (-1, 2) or state_val in ("fail", "failed", "error"):
                                        raise RuntimeError(f"Задача KIE завершилась с ошибкой: {status_data.get('msg') or 'unknown error'}")
                                    
                                    # В процессе
                                    logger.info(f"   ... статус: {state_val or 'waiting'} (прошло {(poll_idx + 1) * 5} сек)...")
                                    
                                raise TimeoutError("Превышено время ожидания генерации изображения KIE (120 секунд)")
                                
                        except Exception as err:
                            logger.error(f"   ❌ Ошибка генерации изображений KIE на попытке {attempt}: {err}")
                            if attempt == max_img_retries:
                                raise err
                            import time
                            time.sleep(img_delay)
                else:
                    # Фолбек на стандартный OpenAI-совместимый эндпоинт (например, для DALL-E)
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "size": size,
                        "n": 1
                    }
                    
                    model_base_url = f"https://api.kie.ai/{model}/v1"
                    max_img_retries = 3
                    img_delay = 5.0
                    for attempt in range(1, max_img_retries + 1):
                        try:
                            logger.info(f"   🚀 Запрос к API изображений (Модель: {model}, Размер: {size}, попытка {attempt}/{max_img_retries})...")
                            with httpx.Client(timeout=120.0) as http_client:
                                resp = http_client.post(f"{model_base_url}/images/generations", json=payload, headers=headers)
                                resp.raise_for_status()
                                return resp.json()
                        except Exception as err:
                            logger.error(f"   ❌ Ошибка API изображений на попытке {attempt}: {err}")
                            if attempt == max_img_retries:
                                raise err
                            import time
                            time.sleep(img_delay)
            
            # А. Генерация обложки (Размер 1536x768)
            cover_scene = artist_response.get("cover_scene", f"Conceptual cover art representing the theme: {state.topic}")
            full_cover_prompt = (
                f"{cover_scene}, style reference: {style_ref}. "
                f"Ensure a very bold, clean, minimalistic and large text overlay in Russian language reads exactly: '{text_overlay}'. "
                f"The text must be the primary design element and perfectly integrated, with absolutely zero grammatical or spelling errors."
            )
            
            cover_data = _generate_image_with_fallback(full_cover_prompt, "1536x768", MODELS["openai_image_primary"])
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
                
                sec_data = _generate_image_with_fallback(full_section_prompt, "1536x384", MODELS["openai_image_primary"])
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
        Аварийный спасательный механизм (Умный структурный hard-cut).
        Если финальная статья превышает лимит символов более чем на 20%,
        система аккуратно сжимает или обрезает ее.
        """
        # Работаем с final_article, если она пустая - берем draft
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
            # Если целевой объем не задан, берем из паттернов
            from .patterns import PATTERNS
            pattern = PATTERNS.get(state.article_type, PATTERNS.get("free_style", {}))
            target = pattern.get("target_chars", 8000)

        limit = int(target * 1.2)  # Допускаем перебор до 20%
        if len(source_text) <= limit:
            if not state.final_article:
                state.final_article = source_text
            return

        # 1. Чек-листы/справочники: грубая обрезка запрещена → только бережный condense.
        if state.style_id in ("checklist", "reference"):
            logger.info(f"   ⚠️ [Hard-Cut] Статья длинная ({len(source_text)} символов). Лимит: {limit}. Бережное сжатие (condense)...")
            condensed = self._heart_condense(state, source_text, target)
            min_allowed = int(target * 0.85)
            if len(condensed) < len(source_text) and len(condensed) >= min_allowed:
                source_text = condensed
                logger.info(f"   ✅ [Hard-Cut] Статья сжата через condense до {len(source_text)} символов.")
            else:
                logger.warning(f"   ⚠️ [Hard-Cut] Condense не увенчался успехом или сжатие слишком агрессивно. Оставляем исходный текст.")
            logger.warning("   ⚠️ [Hard-Cut] Для чек-листов/справочников грубая обрезка запрещена. Оставляем сжатый/исходный текст.")
            state.final_article = source_text
            return

        # 2. Остальные форматы: ОДНО экспресс-сжатие.
        logger.info(f"   ⚠️ [Hard-Cut] Статья длинная ({len(source_text)} символов). Лимит: {limit}. Сжатие через модель...")
        try:
            old_model = get_agent("heart").model
            # В QUALITY_MODE используем основную модель для сохранения качества слога, иначе переключаемся на flash
            if not state.quality_mode:
                get_agent("heart").model = MODELS["deepseek_flash"]  # временно переключаем на flash
            
            # Динамический целевой объем сжатия:
            # Если превышение небольшое (< 135% от базового таргета), сжимаем до лимита
            if len(source_text) < target * 1.35:
                condense_target = limit
            else:
                condense_target = target

            compress_msg = (
                f"ТЕКСТ СТАТЬИ:\n{source_text}\n\n"
                f"Этот текст превышает лимит. Пожалуйста, сожми статью строго до {condense_target} символов. "
                f"Убери любые повторы, размышления, вводные слова. Сохрани все разделы (H2), ссылки на законы, таблицы и факты. "
                f"Текст должен остаться связным и легко читаемым. Верни только измененный текст статьи."
            )
            compressed_text = self._call_agent("heart", compress_msg, parse_json=False, target_chars=condense_target, state=state)
            
            if not state.quality_mode:
                get_agent("heart").model = old_model  # возвращаем модель
            
            min_allowed = int(target * 0.70)
            if len(compressed_text) < len(source_text) and len(compressed_text) >= min_allowed:
                source_text = compressed_text
                logger.info(f"   ✅ [Hard-Cut] Статья успешно сжата до {len(source_text)} символов.")
                if len(source_text) <= limit:
                    state.final_article = source_text
                    return
            else:
                logger.warning(f"   ⚠️ [Hard-Cut] Сжатие отвергнуто (длина: {len(compressed_text)}, минимум: {min_allowed})")
        except Exception as e:
            try:
                get_agent("heart").model = old_model
            except Exception:
                pass
            logger.warning(f"   ⚠️ [Hard-Cut] Сжатие завершилось ошибкой: {e}")

        # 4. БЕЗ ЖЁСТКОЙ ОБРЕЗКИ. Раньше здесь текст резался по последнему заголовку
        #    (source_text[:cutoff]) и к нему приклеивался синтетический "## ИТОГ".
        #    Это удаляло финальные разделы (Выводы/Послесловие) и ломало логику статьи —
        #    оглавление обещало разделы, которых в теле уже не было.
        #    Теперь структуру НЕ трогаем: если модельный condense выше не уложил текст
        #    в лимит — оставляем ПОЛНУЮ статью. Лучше длиннее цели, чем без выводов.
        logger.warning(
            f"   ⚠️ [Hard-Cut ОТКЛЮЧЁН] Текст ({len(source_text)} симв.) превышает "
            f"лимит {limit}, но жёсткая обрезка удалена (ломала структуру). "
            f"Оставляю полный текст без удаления разделов."
        )
        state.final_article = source_text

    def _clean_image_markers_from_seo(self, state: 'PipelineState'):
        """Вычистить маркеры [IMAGE_PROMPT_HERE: ...] / [картинка: ...] из seo_package.

        При выключенном Artist маркеры удаляются из текста статьи, но остаются в
        state.seo_package (alt_texts[].marker, а иногда в citation_baits/lsi_insertions),
        что приводит к «утёкшим тегам» в финальном seo_package.json. Этот метод
        рекурсивно очищает все строковые значения seo_package.
        """
        import re
        if not getattr(state, "seo_package", None):
            return
        marker_re = re.compile(r"\[(?:картинка|IMAGE_PROMPT_HERE)(?::\s*.*?)?\]")

        def _scrub(obj):
            if isinstance(obj, str):
                return marker_re.sub("", obj).strip()
            if isinstance(obj, list):
                return [_scrub(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _scrub(v) for k, v in obj.items()}
            return obj

        state.seo_package = _scrub(state.seo_package)

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

    def _validate_final(self, state: 'PipelineState') -> list:
        """Финальная валидация статьи (без API). Возвращает список предупреждений."""
        import re
        warnings = []
        text = state.final_article or ""
        if not text:
            return ["🔴 Финальная статья пуста!"]

        # 1. Проверка длины
        target = state.custom_chars
        if not target:
            if state.style_id:
                try:
                    from .styles import get_style
                    target = get_style(state.style_id).target_chars
                except Exception:
                    pass
        if not target:
            target = 8000

        if len(text) > target * 1.3:
            warnings.append(f"⚠️ Статья слишком длинная: {len(text)} vs цель {target}")
        if len(text) < target * 0.5:
            warnings.append(f"⚠️ Статья слишком короткая: {len(text)} vs цель {target}")

        # 2. Утёкшие теги
        leaked_tags = re.findall(r'\[(картинка|IMAGE_PROMPT_HERE|TABLE|CHRONOTOPE_SCENE)[^\]]*\]', text)
        if leaked_tags:
            warnings.append(f"⚠️ Утёкшие теги в тексте: {leaked_tags[:5]}")

        # 3. Стоп-слова
        try:
            from .stopwords import ALL_STOP_WORDS
            found = [w for w in ALL_STOP_WORDS if w in text.lower()]
            if found:
                warnings.append(f"⚠️ Стоп-слова в финальном тексте: {found[:5]}")
        except ImportError:
            pass

        # 4. Структурная целостность (новые жёсткие проверки целостности статьи).
        h2_titles = re.findall(r'^##\s+(.+)$', text, re.MULTILINE)
        if len(h2_titles) < 2:
            warnings.append(f"🔴 Слишком мало H2-разделов ({len(h2_titles)}) — структура могла быть обрезана.")

        # 5. Признак аварийного среза: последний раздел называется 'ИТОГ'.
        if h2_titles and h2_titles[-1].strip().upper() == "ИТОГ":
            warnings.append("🔴 Последний раздел = машинный 'ИТОГ' (признак старой жёсткой обрезки).")

        # 6. Обрыв на полуслове (нет финальной пунктуации в конце статьи).
        tail = text.rstrip()
        # Очищаем от Markdown-ограждений в конце (например, *, _, |, ~, пробелы)
        tail = re.sub(r'[\s\*_\|`~]+$', '', tail)
        if tail and tail[-1] not in '.!?»")…':
            warnings.append("🔴 Статья обрывается на полуслове (нет финальной пунктуации) — возможен обрыв по токенам.")

        # 7. Оглавление обещает больше разделов, чем есть в теле.
        toc_match = re.search(r'(?:^|\n)#{1,3}\s*Содержание\s*\n(.+?)(?:\n#{1,3}\s|\Z)', text, re.DOTALL)
        if toc_match:
            toc_lines = [l.strip(" -*•\t") for l in toc_match.group(1).split("\n") if l.strip(" -*•\t")]
            
            def clean_header_text(h_text: str) -> str:
                # Удаляем нумерацию в начале (например, "1. ", "1) ", "1.2. ", "Пункт 1: ")
                h_text = re.sub(r'^(?:\d+[\.)]\s*|пункт\s+\d+[:\.]?\s*)+', '', h_text, flags=re.IGNORECASE)
                # Оставляем только буквы и цифры в нижнем регистре
                return re.sub(r'[^\w]', '', h_text).lower()

            body_heads_cleaned = {clean_header_text(h) for h in (h2_titles + re.findall(r'^###\s+(.+)$', text, re.MULTILINE))}
            
            clean_toc_lines = []
            for l in toc_lines:
                # Очищаем синтаксис ссылок Markdown [Заголовок](#ссылка) -> Заголовок
                clean_l = re.sub(r'\[(.+?)\]\(.*?\)', r'\1', l).strip()
                clean_toc_lines.append(clean_l)
                
            missing = []
            for t in clean_toc_lines:
                if not t:
                    continue
                t_clean = clean_header_text(t)
                if not t_clean:
                    continue
                matched = False
                for b in body_heads_cleaned:
                    if t_clean in b or b in t_clean or t_clean[:15] in b or b[:15] in t_clean:
                        matched = True
                        break
                if not matched:
                    missing.append(t)
                    
            if len(missing) >= 2:
                warnings.append(f"🔴 Оглавление обещает разделы, отсутствующие в теле: {missing[:3]} — статья могла быть обрезана.")

        return warnings

    def _compact_json(self, obj, limit: int = 4000) -> str:
        """Компактный JSON без отступов с ограничением длины.

        indent=2 раздувал входные токены почти вдвое (пробелы/переводы строк);
        для передачи модели структура читается и без них.
        """
        try:
            s = json.dumps(obj, ensure_ascii=False)
        except Exception:
            s = str(obj)
        if len(s) > limit:
            s = s[:limit] + " …"
        return s

    def _chat_completion(self, client, *, response_format=None, **kwargs):
        """chat.completions.create с retry/backoff на транзиентных ошибках.

        Повторяет ТОЛЬКО временные сбои (429 rate limit, таймаут, разрыв
        соединения, 5xx). Ошибки клиента (4xx, кроме 429) поднимаются сразу.
        Если провайдер не понимает response_format — один раз повторяет без него.
        Чистый stdlib, без внешних зависимостей.
        """
        import os, time, random
        try:
            from openai import (
                RateLimitError, APITimeoutError, APIConnectionError,
                InternalServerError, APIError, BadRequestError,
            )
        except Exception:  # на случай иной версии SDK — деградируем без ретраев
            RateLimitError = APITimeoutError = APIConnectionError = ()
            InternalServerError = BadRequestError = ()
            APIError = ()

        max_retries = int(os.getenv("AGENT_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("AGENT_RETRY_BASE_DELAY", "2"))
        max_delay = float(os.getenv("AGENT_RETRY_MAX_DELAY", "30"))
        use_rf = (
            response_format is not None
            and os.getenv("AGENT_JSON_MODE", "true").lower() in ("1", "true", "yes", "on")
        )

        attempt = 0
        token_param_swapped = False
        while True:
            call_kwargs = dict(kwargs)
            if use_rf:
                call_kwargs["response_format"] = response_format
            try:
                return client.chat.completions.create(**call_kwargs)
            except Exception as e:
                is_kie = hasattr(client, 'base_url') and 'kie.ai' in str(client.base_url)
                if is_kie:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(f"   ❌ KIE API недоступен после {max_retries} повторов: {e}")
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 1)
                    logger.warning(
                        f"   ⏳ Ошибка KIE API ({type(e).__name__}: {e}); "
                        f"повтор {attempt}/{max_retries} через {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                # Обработка для не-KIE клиентов:
                if BadRequestError and isinstance(e, BadRequestError):
                    emsg = str(e).lower()
                    if not token_param_swapped and ("max_tokens" in emsg or "max_completion_tokens" in emsg):
                        if "max_tokens" in kwargs:
                            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                        elif "max_completion_tokens" in kwargs:
                            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                        token_param_swapped = True
                        logger.warning(f"   ⚠️ Параметр лимита токенов не принят ({e}); меняю max_tokens↔max_completion_tokens и повторяю")
                        continue
                    if use_rf:
                        logger.warning(f"   ⚠️ response_format не поддержан ({e}); повтор без JSON-mode")
                        use_rf = False
                        continue
                    raise
                elif (RateLimitError and isinstance(e, RateLimitError)) or \
                     (APITimeoutError and isinstance(e, APITimeoutError)) or \
                     (APIConnectionError and isinstance(e, APIConnectionError)) or \
                     (InternalServerError and isinstance(e, InternalServerError)):
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(f"   ❌ API недоступен после {max_retries} повторов: {e}")
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 1)
                    logger.warning(
                        f"   ⏳ Транзиентная ошибка ({type(e).__name__}); "
                        f"повтор {attempt}/{max_retries} через {delay:.1f}s"
                    )
                    time.sleep(delay)
                elif APIError and isinstance(e, APIError):
                    status = getattr(e, "status_code", None)
                    if status is not None and status >= 500:
                        attempt += 1
                        if attempt > max_retries:
                            logger.error(f"   ❌ Сервер {status} после {max_retries} повторов: {e}")
                            raise
                        delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 1)
                        logger.warning(
                            f"   ⏳ Серверная ошибка {status}; повтор {attempt}/{max_retries} через {delay:.1f}s"
                        )
                        time.sleep(delay)
                    else:
                        raise
                else:
                    raise

    def _call_agent(
        self,
        agent_id: str,
        user_message: str,
        parse_json: bool = True,
        target_chars: int = 0,
        state: 'PipelineState' = None,
        override_model: str = None,
        override_provider: str = None,
        override_temperature: float = None,
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

        # Подстановка адаптивного числа пунктов чек-листа в промпт Heart.
        # Раньше промпт жестко требовал "ровно 10", что конфликтовало с адаптивным
        # расчётом роутера (5/7/10) и гарантированно браковало короткие чек-листы.
        if agent_id == "heart" and "{NUM_CHECKLIST_ITEMS}" in system_prompt:
            num_items = getattr(state, "num_checklist_items", None) if state else None
            if not num_items or num_items <= 0:
                num_items = 10
            system_prompt = system_prompt.replace("{NUM_CHECKLIST_ITEMS}", str(num_items))

        # Динамическая адаптация промпта Heart под целевой объём
        if agent_id == "heart" and target_chars > 0 and target_chars < 12000:
            logger.info("   ⚙️ Адаптация системного промпта Heart под малый/средний объём...")
            system_prompt = system_prompt.replace(
                "распиши его МЕХАНИКУ на 3-4 предложения",
                "распиши его МЕХАНИКУ лаконично на 2-3 предложения"
            )
            system_prompt = system_prompt.replace(
                "- ОБЪЕМ РАЗДЕЛА И СТАТЬИ ЦЕЛИКОМ диктуется ТЗ и Blueprint. Соблюдай заданный лимит символов: лёгкое превышение (до 10%) допустимо, значительное (свыше 15%) — брак.",
                "- Пиши максимально компактно и плотно! Жёстко укладывайся в заданный лимит символов. Превышение лимита — критический брак."
            )
            system_prompt = system_prompt.replace(
                "должна быть законченным развернутым предложением (или двумя) с детальным описанием бизнес-механики, а не сухой краткой инструкцией в 3-5 слов",
                "должна быть емкой, но информативной, без избыточных подробностей"
            )
            system_prompt = system_prompt.replace(
                "должна представлять собой законченное развернутое предложение (или два) с описанием бизнес-механики, а не сухую краткую инструкцию в 3-5 слов",
                "должна быть емкой, но информативной, без избыточных подробностей"
            )

        # Динамический временной контекст для исключения устаревшего года (2025)
        import datetime
        now = datetime.datetime.now()
        current_year = now.year
        today_iso = now.strftime("%Y-%m-%d")
        today_ru = now.strftime("%d.%m.%Y")
        time_anchor = (
            f"\n\n[ВРЕМЕННОЙ КОНТЕКСТ]:\n"
            f"Сегодня — {today_ru} ({today_iso}). Текущий год — {current_year}.\n"
            f"Все рекомендации, правила, лимиты, налоги, риски и метаданные (SEO Title, Description, H1) "
            f"должны генерироваться и быть актуальными исключительно для {current_year} года.\n"
            f"Любые упоминания {current_year - 1} года и более ранних периодов допускаются только в прошедшем времени "
            f"(как исторический контекст или сравнение). Категорически запрещено указывать прошлые годы (например, {current_year - 1}) "
            f"в качестве текущего или будущего времени в заголовках, мета-тегах и основном тексте.\n\n"
            f"⚠️ БУДУЩИЕ ИЗМЕНЕНИЯ ЗАКОНА (критично):\n"
            f"- Любая правовая норма с датой вступления в силу ПОСЛЕ {today_ru} — ЕЩЁ НЕ ДЕЙСТВУЕТ.\n"
            f"- Если закон принят/опубликован, но вступает в силу позже сегодня ({today_ru}) — это БУДУЩЕЕ изменение.\n"
            f"- Пиши его СТРОГО в будущем времени: «вступит в силу с <дата>», «с <дата> будет действовать».\n"
            f"- ОБЯЗАТЕЛЬНО указывай дату вступления в силу. Не описывай будущую норму как действующую практику.\n"
            f"- Текущая (действующая) редакция — та, что вступила в силу ДО или В {today_ru}. Описывай действующий порядок как актуальный.\n"
            f"- Не пиши «статья действует в новой редакции» или «предусмотрен алгоритм», если редакция вступает позже сегодня."
        )
        extended_system_prompt = system_prompt + time_anchor
        if agent_id == "heart" and target_chars > 0:
            min_chars = int(target_chars * 0.85)
            max_chars = int(target_chars * 1.15)
            
            is_full_article = False
            if state is not None and state.min_chars > 0:
                is_full_article = target_chars >= state.min_chars * 0.8
                
            if is_full_article:
                min_chars = getattr(state, "min_chars", min_chars) or min_chars
                max_chars = getattr(state, "max_chars", max_chars) or max_chars
            
            limit_block = (
                f"\n\n[КРИТИЧЕСКОЕ ТРЕБОВАНИЕ К ОБЪЕМУ ТЕКСТА (ЖЕСТКИЙ ЛИМИТ)]:\n"
                f"Твой целевой объем текста: РОВНО {target_chars} символов (включая пробелы).\n"
                f"Допустимый коридор: от {min_chars} до {max_chars} символов.\n"
                f"Превышение лимита в {max_chars} символов или занижение менее {min_chars} символов является критическим браком и приведет к сбою системы.\n"
                f"Пожалуйста, жестко планируй плотность и длину текста. Распределяй объем по разделам. Пиши лаконично, без пустых вводных фраз и размусоливания."
            )
            extended_system_prompt += limit_block

        # Обрезаем user_message если слишком длинный
        max_input = 50000  # ~12k токенов
        if len(user_message) > max_input:
            user_message = user_message[:max_input] + "\n\n[...контекст обрезан...]"

        # max_tokens — это ПОТОЛОК БЕЗОПАСНОСТИ против обрыва ответа, а НЕ регулятор длины.
        # Длину задают промпт и бюджет раздела (chars_per_section).
        # РАНЬШЕ здесь стоял множитель `target_chars * 0.7`, который физически ОБРЕЗАЛ
        # ответ по лимиту токенов (finish_reason=length) — особенно у Booster, который
        # отдаёт весь текст статьи + JSON-метаданные. Это была одна из причин «обрезки».
        # Теперь используем полный, протестированный лимит агента из registry.py.
        max_tokens_for_call = agent.max_tokens
        if target_chars > 0 and agent_id in ("heart", "mirror", "booster"):
            logger.info(f"   ⚙️ Лимит токенов (потолок безопасности) для {agent.name}: {max_tokens_for_call}")

        # Динамический выбор клиента и модели на основе провайдера
        current_client = self.deepseek_client  # По умолчанию DeepSeek
        model_name = agent.model  # По умолчанию модель агента
        
        is_external_reviewer_kie = False
        if override_provider:
            provider = override_provider.lower()
            model_name = override_model if override_model else model_name
            if provider == "kie":
                current_client = self._get_kie_client(model_name)
            elif provider == "openai":
                current_client = self.openai_client
            elif provider == "deepseek":
                current_client = self.deepseek_client
        elif agent_id == "external_reviewer":
            if self._kie_api_key:
                model_name = MODELS["external_reviewer"]
                current_client = self._get_kie_client(model_name)
                is_external_reviewer_kie = True
            else:
                logger.warning("   ⚠️ [QUALITY_MODE] KIE_API_KEY не задан. Деградирую external_reviewer до DeepSeek Pro.")
                model_name = MODELS["deepseek_pro"]
                current_client = self.deepseek_client
        elif state is not None:
            provider = getattr(state, "provider", "deepseek").lower()
            custom_model = getattr(state, "model", None)
            
            if provider == "kie":
                # ...
                model_name = custom_model if custom_model else MODELS["kie_text"]
                current_client = self._get_kie_client(model_name)
            elif provider == "openai":
                current_client = self.openai_client
                model_name = custom_model if custom_model else MODELS["openai_text"]
            elif provider == "deepseek":
                current_client = self.deepseek_client
                if custom_model:
                    model_name = custom_model

        # JSON-mode (response_format) только для агентов, чей ответ парсится как JSON.
        # Если провайдер его не поддержит — _chat_completion повторит без него.
        response_format = {"type": "json_object"} if parse_json else None
        if parse_json and "json" not in extended_system_prompt.lower():
            extended_system_prompt += (
                "\n\nВерни ответ СТРОГО как один валидный JSON-объект, без markdown-ограждения."
            )

        # Модели o1/o3-mini требуют max_completion_tokens, все остальные модели требуют max_tokens
        is_o1_o3 = any(m in model_name.lower() for m in ["o1-", "o3-"])
        is_reasoning = "reasoning" in model_name.lower() or "r1" in model_name.lower() or "pro" in model_name.lower() or "opus" in model_name.lower() or is_o1_o3
        
        # Calculate dynamic max_tokens ceiling (insurance ceiling, not length controller)
        max_tokens_for_call = agent.max_tokens
        if target_chars > 0 and agent_id in ("heart", "mirror", "booster"):
            content_tokens = target_chars / 2.2          # реалистичная плотность контента
            headroom = 1.45                              # +45% запас, чтобы не рвать на полуслове
            reasoning_budget = 3500 if is_reasoning else 0   # отдельно на «мышление»
            max_tokens_for_call = min(agent.max_tokens, int(content_tokens * headroom) + reasoning_budget)
            max_tokens_for_call = max(1500, max_tokens_for_call) # Safe minimum ceiling
            logger.info(f"   ⚙️ Оптимизированный лимит токенов для {agent.name}: {max_tokens_for_call} (ожидаемые символы: {target_chars}, reasoning: {is_reasoning})")

        chat_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": extended_system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": override_temperature if override_temperature is not None else agent.temperature,
        }
        if getattr(agent, "top_p", None) is not None:
            chat_params["top_p"] = agent.top_p

        # Attempt API call with retry on length truncation
        response = None
        attempt = 0
        max_attempts = 2
        
        while attempt < max_attempts:
            if is_o1_o3:
                chat_params["max_completion_tokens"] = max_tokens_for_call
            else:
                chat_params["max_tokens"] = max_tokens_for_call

            try:
                response = self._chat_completion(
                    current_client,
                    response_format=response_format,
                    **chat_params
                )
                if not response or not getattr(response, "choices", None) or len(response.choices) == 0 or response.choices[0] is None:
                    raise RuntimeError("API returned empty choices or invalid response structure")
            except Exception as e:
                if attempt == 0 and is_external_reviewer_kie:
                    logger.warning(f"   ⚠️ [QUALITY_MODE] Ошибка KIE API ({e}). Деградирую external_reviewer до DeepSeek Pro.")
                    current_client = self.deepseek_client
                    model_name = MODELS["deepseek_pro"]
                    chat_params["model"] = model_name
                    is_o1_o3 = any(m in model_name.lower() for m in ["o1-", "o3-"])
                    # Adjust parameters for DeepSeek
                    if "max_completion_tokens" in chat_params:
                        chat_params.pop("max_completion_tokens")
                    attempt += 1
                    continue
                else:
                    raise
            
            _finish_reason = getattr(response.choices[0], "finish_reason", "stop")
            if _finish_reason != "length":
                break
                
            attempt += 1
            if attempt < max_attempts:
                logger.warning(
                    f"🔴 [{agent_id}] ОТВЕТ ОБОРВАН по лимиту токенов (finish_reason=length, max_tokens={max_tokens_for_call}). "
                    f"Попытка {attempt+1}/{max_attempts}: увеличиваем потолок и перезапрашиваем..."
                )
                max_tokens_for_call = min(int(max_tokens_for_call * 1.25), agent.max_tokens * 1.5 if agent.max_tokens * 1.5 <= 32000 else 32000)
            else:
                logger.error(
                    f"🔴 [{agent_id}] Ретраи исчерпаны. Ответ по-прежнему оборван (finish_reason=length). "
                    f"Результат неполный — фиксируем брак."
                )

        # Аккумуляция токенов
        if response and hasattr(response, 'usage') and response.usage:
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

        raw = response.choices[0].message.content if response else ""
        if raw is None:
            raw = ""
            logger.warning(f"⚠️ [{agent_id}] ответ пустой (content=None)")
        raw = raw.strip()

        _finish_reason = getattr(response.choices[0], "finish_reason", None) if response else None
        if state is not None:
            state.last_call_truncated = (_finish_reason == "length")
        if _finish_reason == "length":
            logger.warning(
                f"🔴 [{agent_id}] ОТВЕТ ОБОРВАН по лимиту токенов "
                f"(finish_reason=length, max_tokens={max_tokens_for_call}). "
                f"Результат может быть неполным — потребитель обязан это проверить."
            )

        if parse_json:
            return self._parse_json_response(raw, agent_id)
        return raw

    def _parse_json_response(self, raw: str, agent_id: str) -> Dict:
        """Извлечь JSON из ответа агента. Устойчиво к markdown/мусору; не бросает.

        Стратегия: (1) снять ```json``` ограждение; (2) прямой json.loads;
        (3) сбалансированный поиск первого {...}-объекта с учётом строк/экранов;
        (4) терпимость к висячим запятым. Последний шаг — {"raw_response": raw}.
        """
        import re

        def _try(s):
            try:
                return json.loads(s)
            except Exception:
                return None

        if not raw:
            return {"raw_response": ""}

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        for candidate in (cleaned, raw):
            obj = _try(candidate)
            if isinstance(obj, dict):
                return obj

        # Сбалансированный разбор: ищем первый завершённый объект {...}
        start = raw.find("{")
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(raw)):
                ch = raw[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = raw[start:i + 1]
                        obj = _try(snippet)
                        if obj is None:
                            obj = _try(re.sub(r",(\s*[}\]])", r"\1", snippet))
                        if isinstance(obj, dict):
                            return obj
                        break
            start = raw.find("{", start + 1)

        logger.warning(f"⚠️ [{agent_id}] не удалось распарсить JSON, возвращаю как текст")
        return {"raw_response": raw}

    def _persona_summary(self, pl: dict) -> str:
        """Короткая строка-резюме паспорта ЦА для логов."""
        if not isinstance(pl, dict) or not pl:
            return "не задан (Brain не вернул persona_lock)"
        role = pl.get("reader_role") or pl.get("role") or "?"
        rev = pl.get("revenue_tier") or pl.get("revenue") or "?"
        return f"роль={role}; масштаб={rev}"

    def _persona_block(self, state: PipelineState = None) -> str:
        """
        Рендер «Паспорта ЦА» (Persona & Scale Lock) для прокидывания в Heart и Ревизора.
        Возвращает пустую строку, если Brain не задал persona_lock (обратная совместимость).
        """
        pl = getattr(state, "persona_lock", None) if state else None
        bo = (getattr(state, "brain_output", {}) or {}) if state else {}
        if not isinstance(pl, dict):
            pl = {}
        # topic_class / legal_density могут лежать как в persona_lock, так и на верхнем уровне Brain
        topic_class = pl.get("topic_class") or bo.get("topic_class")
        legal_density = pl.get("legal_density")
        if legal_density in (None, ""):
            legal_density = bo.get("legal_density")
        if not pl and not topic_class and legal_density in (None, ""):
            return ""

        def g(*keys):
            for k in keys:
                v = pl.get(k)
                if v:
                    return v
            return ""

        lines = ["📌 ПАСПОРТ ЦА (Persona & Scale Lock — соблюдать во ВСЕХ разделах):"]
        role = g("reader_role", "role", "persona")
        if role:
            lines.append(f"- Кто читатель / герой кейса: {role}")
        rev = g("revenue_tier", "revenue", "scale")
        if rev:
            lines.append(f"- Масштаб бизнеса (оборот): {rev}")
        head = g("headcount_band", "headcount")
        if head:
            lines.append(f"- Штат: {head}")
        stakes = g("realistic_stakes_range", "stakes")
        if stakes:
            lines.append(f"- Реалистичный диапазон ставок (потери/выгоды): {stakes}")
        deleg = g("delegation_norms", "delegation")
        if deleg:
            lines.append(f"- Что обычно делегируется (персонаж НЕ делает лично): {deleg}")
        intent = g("primary_intent", "intent")
        if intent:
            lines.append(f"- Единый primary_intent (одна задача для одного читателя): {intent}")
        if topic_class:
            lines.append(f"- Класс темы: {topic_class}")
        if legal_density not in (None, ""):
            lines.append(f"- legal_density (плотность юр-ссылок, 0–1): {legal_density}")
        lines.append(
            "- ИНВАРИАНТ: советы, цифры и поведение персонажа ДОЛЖНЫ соответствовать роли и масштабу. "
            "Гиперболы, анахронизмы и смешение масштабов (микро ↔ холдинг) запрещены."
        )
        return "\n".join(lines) + "\n\n"

    def _get_style_block(self, state: PipelineState = None) -> str:
        """
        Сформировать блок стилевых инструкций для Heart.

        Приоритет:
        1. Style Fingerprinting (из styles.py) — если style_id указан
        2. Клиентский стилевой паспорт (self.style) — если передан
        3. Пустой блок — агент работает по умолчанию

        Префикс: Persona & Scale Lock (паспорт ЦА), если задан Brain.
        """
        persona_prefix = self._persona_block(state)
        # 1. Style Fingerprinting из styles.py
        if state and state.style_id:
            try:
                from .styles import get_style_prompt
                custom = state.custom_chars if state.custom_chars > 0 else None
                style_prompt = get_style_prompt(state.style_id, custom) + "\n\n"
                if state.style_id == "checklist" and hasattr(state, "num_checklist_items"):
                    num = state.num_checklist_items
                    style_prompt = style_prompt.replace("10 пунктов", f"{num} пунктов")
                    style_prompt = style_prompt.replace("10 нумерованных", f"{num} нумерованных")
                    style_prompt = style_prompt.replace("каждого из 10", f"каждого из {num}")
                    style_prompt = style_prompt.replace("## 10.", f"## {num}.")
                    style_prompt = style_prompt.replace("## 1. ... ## 10.", f"## 1. ... ## {num}.")
                return persona_prefix + style_prompt
            except (ValueError, ImportError):
                pass  # Стиль не найден — пробуем клиентский паспорт

        # 2. Правила из паттернов (patterns.py)
        pattern = PATTERNS.get(state.article_type) or PATTERNS.get("seo")
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

        return persona_prefix + "\n".join(lines) + "\n\n"
