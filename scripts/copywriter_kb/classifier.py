"""
Классификатор метаданных для Knowledge Base Pipeline.

Генерирует агенто-специфичные метаданные:
  - source_type, source_priority, chunk_type (для Исследователя)
  - rhythm_type, emotive_charge (для Писателя)
  - error_type, cliché_replacement (для Редактора)
  - logical_pattern, attention_level (для Структурировщика)
  - intent_type, html_tag (для SEO-оптимизатора)
  - image_style, brand_guidelines (для Визуализатора)

Стратегии: raw (только базовые мета), raw_classified (мета через GPT),
           distill (полная дистилляция + мета), hybrid (смешанная).
"""
import re
import json
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import (
    DISTILL_MODEL, API_DELAY, MAX_RETRIES, RETRY_DELAYS,
    DEEPSEEK_API_KEY, DEEPSEEK_API_BASE,
)

logger = logging.getLogger("kb.classifier")


class BudgetExhaustedError(Exception):
    """Баланс OpenAI исчерпан — graceful exit"""
    pass


# ============================================================
# OpenAI клиент (ленивая инициализация)
# ============================================================

_openai_client = None


def _get_openai():
    """Получить OpenAI клиент (singleton)"""
    global _openai_client
    if not _openai_client:
        from openai import OpenAI
        from .config import OPENAI_API_KEY
        _openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=90.0)
    return _openai_client


_deepseek_client = None


def _get_deepseek():
    """DeepSeek клиент (singleton) для дистилляции/классификации текста.

    Разводка: OpenAI — только embeddings, DeepSeek — только LLM-текст.
    DeepSeek имеет OpenAI-совместимый endpoint (base_url=api.deepseek.com/v1).
    """
    global _deepseek_client
    if not _deepseek_client:
        from openai import OpenAI
        _deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE, timeout=90.0)
    return _deepseek_client



# ============================================================
# Основная функция классификации
# ============================================================

def classify_chunk(
    chunk: str,
    agent_id: str,
    strategy: str,
    source_file: str,
    file_format: str,
) -> Optional[Dict[str, Any]]:
    """
    Классифицировать чанк и сгенерировать метаданные.

    Args:
        chunk: Текст чанка
        agent_id: ID агента (fact_finder, heart, sheriff и т.д.)
        strategy: Стратегия обработки (raw, raw_classified, distill, hybrid)
        source_file: Имя исходного файла
        file_format: Формат файла (.pdf, .fb2 и т.д.)

    Returns:
        Словарь метаданных или None (если чанк бесполезен)
    """
    # Базовые метаданные (всегда присутствуют)
    metadata = {
        "text": chunk,
        "source_file": source_file,
        "file_format": file_format,
        "language": _detect_language(chunk),
    }

    if strategy == "raw":
        # Только базовые метаданные + эвристическая классификация
        metadata.update(_heuristic_classify(chunk, agent_id, source_file))
        return metadata

    elif strategy == "raw_classified":
        # RAW текст + классификация через GPT
        gpt_meta = _gpt_classify(chunk, agent_id)
        if gpt_meta is None:
            return None  # GPT решил, что чанк бесполезен
        metadata.update(_heuristic_classify(chunk, agent_id, source_file))
        metadata.update(gpt_meta)
        return metadata

    elif strategy == "distill":
        # Полная дистилляция через GPT
        distilled = _gpt_distill(chunk, agent_id)
        if distilled is None:
            return None
        metadata.update(distilled)
        # Заменяем text на дистиллированный вариант для embedding
        embed_text = (
            f"{distilled.get('concept', '')}. "
            f"{distilled.get('description', '')}. "
            f"{distilled.get('application', '')}"
        )
        metadata["embed_text"] = embed_text
        return metadata

    elif strategy == "hybrid":
        # Решаем по длине и содержимому: короткие/справочные → RAW, длинные → distill
        if len(chunk) < 400 or _is_reference_content(chunk):
            metadata.update(_heuristic_classify(chunk, agent_id, source_file))
            return metadata
        else:
            distilled = _gpt_distill(chunk, agent_id)
            if distilled is None:
                return None
            metadata.update(distilled)
            embed_text = (
                f"{distilled.get('concept', '')}. "
                f"{distilled.get('description', '')}. "
                f"{distilled.get('application', '')}"
            )
            metadata["embed_text"] = embed_text
            return metadata

    return metadata


# ============================================================
# Эвристическая классификация (без GPT)
# ============================================================

# Маппинг признаков имени файла → кодекс РФ.
# (проверяем по подстроке имени файла в нижнем регистре)
_CODEX_BY_FILE_HINT = [
    (("трудовой", "trudovoy", "tk_rf", "tk-rf", "tk.", "тк рф", "тк_рф"), "ТК РФ"),
    (("nkodeks", "налоговый", "nk_rf", "nk-rf", "нк рф", "нк_рф"), "НК РФ"),
    (("grazhdanskij", "гражданский", "gk_rf", "gk-rf", "гк рф", "гк_рф"), "ГК РФ"),
    (("жилищный", "zhilishchnyy", "zhk_rf", "жк рф", "жк_рф"), "ЖК РФ"),
    (("семейный", "semejnyj", "sk_rf", "ск рф", "ск_рф"), "СК РФ"),
    (("уголовный", "ugolovnyj", "uk_rf", "ук рф", "ук_рф"), "УК РФ"),
    (("коап", "koap", "кодекс об административных"), "КоАП РФ"),
]

# Локальный regex извлечения номера статьи из текста чанка.
# (без импорта factcheck, чтобы не создавать циклическую зависимость)
_ARTICLE_NUM_RE = re.compile(
    r"ст(?:атья|атьи|атье|атью|атьё|\.)?\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
# Regex номера ФЗ из имени файла: "115-ФЗ", "ФЗ-115", "38 ФЗ"
_FZ_NUM_RE = re.compile(r"(\d+)\s*[-‑]?\s*фз|фз\s*[-‑]?\s*(\d+)", re.IGNORECASE)


def _law_metadata(chunk: str, filename_lower: str) -> Dict[str, Any]:
    """Детерминированная разметка law-чанка (0 токенов).

    Возвращает codex, article_number, authority, source_url.
    - codex: по имени файла (ТК РФ/НК РФ/ГК РФ/КоАП РФ/ФЗ-N).
    - article_number: номер статьи из текста чанка (первое совпадение «Статья 152»).
    - authority: pravo.gov.ru по умолчанию (официальный портал правовой информации).
    - source_url: пустая строка (URL конкретной статьи проставит авто-апдейтер).
    """
    meta: Dict[str, Any] = {}

    # 1. codex по имени файла
    codex = ""
    for hints, name in _CODEX_BY_FILE_HINT:
        if any(h in filename_lower for h in hints):
            codex = name
            break
    if not codex:
        # ФЗ-N: извлекаем номер из имени файла
        m = _FZ_NUM_RE.search(filename_lower)
        if m:
            num = m.group(1) or m.group(2)
            codex = f"ФЗ-{num}"
    meta["codex"] = codex

    # 2. article_number из текста чанка (берём первое совпадение)
    art_match = _ARTICLE_NUM_RE.search(chunk)
    meta["article_number"] = art_match.group(1) if art_match else ""

    # 3. authority: pravo.gov.ru — официальный портал правовой информации РФ.
    #    Не выдумываем конкретику; точный орган проставит апдейтер при наличии.
    meta["authority"] = "pravo.gov.ru"

    # 4. source_url: пустая строка — URL конкретной статьи проставит будущий
    #    авто-апдейтер; ручной PDF не содержит надёжного URL.
    meta["source_url"] = ""

    return meta


def _heuristic_classify(chunk: str, agent_id: str, source_file: str) -> Dict[str, Any]:
    """
    Базовая классификация на основе паттернов в тексте.
    Быстро, бесплатно, работает оффлайн.
    """
    meta = {}

    # Определяем source_type по содержимому
    lower = chunk.lower()
    filename_lower = source_file.lower()

    if any(kw in filename_lower for kw in ["кодекс", "kodeks", "фз", "fz", "закон", "zakon", "коап"]):
        meta["source_type"] = "law"
        meta["source_priority"] = 1
        # Новые поля для законов (детерминированно, 0 токенов).
        meta.update(_law_metadata(chunk, filename_lower))
    elif any(kw in filename_lower for kw in ["гайд", "guide", "template", "manual", "handbook"]):
        meta["source_type"] = "guide"
        meta["source_priority"] = 2
    elif filename_lower.endswith(".txt"):
        meta["source_type"] = "reference"
        meta["source_priority"] = 2
    else:
        meta["source_type"] = "book"
        meta["source_priority"] = 2

    # chunk_type по содержимому
    if re.search(r'(?:штраф|санкци|наказан|ответственност)', lower):
        meta["chunk_type"] = "sanction"
    elif re.search(r'(?:определени|понятие|означает|является|признаётся)', lower):
        meta["chunk_type"] = "definition"
    elif re.search(r'(?:порядок|инструкци|необходимо|следует|обязан)', lower):
        meta["chunk_type"] = "instruction"
    elif re.search(r'(?:пример|кейс|случай|ситуаци|практик)', lower):
        meta["chunk_type"] = "case_study"
    else:
        meta["chunk_type"] = "general"

    # Агенто-специфичные эвристики
    if agent_id == "heart":
        # Писатель: определяем ритм
        avg_sentence_len = _avg_sentence_length(chunk)
        if avg_sentence_len < 8:
            meta["rhythm_type"] = "short-punchy"
        elif avg_sentence_len < 15:
            meta["rhythm_type"] = "balanced"
        else:
            meta["rhythm_type"] = "explanatory"

    elif agent_id == "sheriff":
        # Редактор: определяем тип ошибки
        if any(kw in lower for kw in ["клише", "штамп", "канцеляр", "стоп-слов"]):
            meta["error_type"] = "cliché"
        elif any(kw in lower for kw in ["пассив", "страдательн"]):
            meta["error_type"] = "passive_voice"
        elif any(kw in lower for kw in ["плеоназм", "тавтолог", "избыточн"]):
            meta["error_type"] = "pleonasm"
        elif any(kw in lower for kw in ["логик", "противореч", "софизм", "заблужден"]):
            meta["error_type"] = "logic_gap"

    elif agent_id == "booster":
        # SEO: определяем intent_type
        if any(kw in lower for kw in ["купить", "заказать", "цена", "стоимость"]):
            meta["intent_type"] = "transactional"
        elif any(kw in lower for kw in ["как", "что такое", "почему", "зачем"]):
            meta["intent_type"] = "informational"
        else:
            meta["intent_type"] = "informational"

    elif agent_id == "artist":
        # Визуализатор: определяем стиль
        if any(kw in lower for kw in ["бренд", "brand", "логотип", "фирменн"]):
            meta["image_style"] = "strict_corporate"
        elif any(kw in lower for kw in ["промпт", "prompt", "генерац", "dall"]):
            meta["image_style"] = "prompt_reference"

    return meta


# ============================================================
# GPT классификация (лёгкая — без дистилляции)
# ============================================================

# Промпты для GPT-классификации по агентам
_CLASSIFY_PROMPTS = {
    "heart": """Проанализируй фрагмент текста из книги по копирайтингу/стилистике.
Определи его характеристики для AI-писателя.

Фрагмент:
---
{chunk}
---

Ответь ТОЛЬКО в JSON (без markdown):
{{
  "rhythm_type": "short-punchy" | "balanced" | "explanatory",
  "emotive_charge": "rational" | "inspiring" | "warning",
  "voice_key": "professional" | "startup-like" | "academic",
  "keywords": ["ключ1", "ключ2"]
}}

Если фрагмент бесполезен (оглавление, реклама) — верни: {{"skip": true}}""",

    "sheriff": """Проанализируй фрагмент текста из справочника по редактуре/корректуре.
Определи его характеристики для AI-редактора.

Фрагмент:
---
{chunk}
---

Ответь ТОЛЬКО в JSON (без markdown):
{{
  "error_type": "pleonasm" | "cliché" | "logic_gap" | "ai_marker" | "passive_voice" | "grammar" | "general",
  "expertise_level": "basic" | "expert" | "ultra-expert",
  "keywords": ["ключ1", "ключ2"]
}}

Если фрагмент бесполезен (оглавление, реклама) — верни: {{"skip": true}}""",

    "booster": """Проанализируй фрагмент текста по SEO/GEO оптимизации.
Определи его характеристики для AI-SEO-специалиста.

Фрагмент:
---
{chunk}
---

Ответь ТОЛЬКО в JSON (без markdown):
{{
  "intent_type": "informational" | "transactional" | "navigational",
  "html_tag": "title_tag" | "meta_description" | "h1_logic" | "faq_structure" | "schema_markup" | "general",
  "geo_relevance": true | false,
  "keywords": ["ключ1", "ключ2"]
}}

Если фрагмент бесполезен (оглавление, реклама) — верни: {{"skip": true}}""",
}


def _gpt_classify(chunk: str, agent_id: str) -> Optional[Dict]:
    """Классифицировать чанк через GPT (без дистилляции)"""
    prompt_template = _CLASSIFY_PROMPTS.get(agent_id)
    if not prompt_template:
        return {}  # Для агентов без GPT-классификации — пустые мета

    return _call_openai_with_retry(
        system="Ты — эксперт по классификации текстов для мультиагентной системы копирайтинга. Отвечай ТОЛЬКО в JSON.",
        user=prompt_template.format(chunk=chunk[:2000]),
        max_completion_tokens=300,
        context="classify",
    )


# ============================================================
# GPT дистилляция (полная — для стратегии distill/hybrid)
# ============================================================

_DISTILL_PROMPTS = {
    "engineer": """Из фрагмента книги по структурированию/копирайтингу извлеки полезный
шаблон или фреймворк для AI-Структурировщика текстов.

Фрагмент:
---
{chunk}
---

Ответь ТОЛЬКО в JSON (без markdown):
{{
  "concept": "Название концепта (2-5 слов)",
  "description": "Краткое описание (2-3 предложения)",
  "application": "Как применить в структурировании B2B-статей (1-2 предложения)",
  "logical_pattern": "problem-solution" | "pyramid-principle" | "comparison-table" | "storytelling-loop" | "general",
  "attention_level": "intro" | "body" | "conclusion" | "call-to-action" | "universal",
  "keywords": ["ключ1", "ключ2"]
}}

Если фрагмент бесполезен (оглавление, библиография, реклама) — верни: {{"skip": true}}""",

    "booster": """Из фрагмента по SEO/GEO извлеки ключевую стратегию или тактику
для AI-SEO-специалиста, работающего с контентом для бизнеса в РФ 2026.

Фрагмент:
---
{chunk}
---

Ответь ТОЛЬКО в JSON (без markdown):
{{
  "concept": "Название концепта (2-5 слов)",
  "description": "Краткое описание (2-3 предложения)",
  "application": "Как применить в SEO/GEO для B2B-сайта в РФ 2026 (1-2 предложения). Учитывай E-E-A-T, SGE, Perplexity.",
  "intent_type": "informational" | "transactional" | "navigational",
  "html_tag": "title_tag" | "meta_description" | "h1_logic" | "faq_structure" | "schema_markup" | "general",
  "keywords": ["ключ1", "ключ2"]
}}

Если фрагмент бесполезен — верни: {{"skip": true}}""",
}


def _gpt_distill(chunk: str, agent_id: str) -> Optional[Dict]:
    """Полная дистилляция чанка через GPT"""
    prompt_template = _DISTILL_PROMPTS.get(agent_id)
    if not prompt_template:
        # Фолбэк: универсальная дистилляция
        prompt_template = """Из следующего фрагмента извлеки ключевой концепт, полезный
для профессионального AI-копирайтера.

Фрагмент:
---
{chunk}
---

Ответь ТОЛЬКО в JSON (без markdown):
{{
  "concept": "Название концепта (2-5 слов)",
  "description": "Краткое описание (2-3 предложения)",
  "application": "Как применить в генерации B2B-контента (1-2 предложения)",
  "keywords": ["ключ1", "ключ2"]
}}

Если фрагмент бесполезен — верни: {{"skip": true}}"""

    return _call_openai_with_retry(
        system="Ты — эксперт по извлечению знаний для мультиагентной системы копирайтинга. Отвечай ТОЛЬКО в JSON.",
        user=prompt_template.format(chunk=chunk[:3000]),
        max_completion_tokens=500,
        context="distill",
    )


# ============================================================
# Retry-обёртка для OpenAI (защита от исчерпания баланса)
# ============================================================

def _call_openai_with_retry(
    system: str,
    user: str,
    max_completion_tokens: int = 500,
    context: str = "api",
) -> Optional[Dict]:
    """
    Вызов OpenAI с retry и exponential backoff.
    Обрабатывает:
      - 429 Rate Limit → retry с задержкой
      - 402 Insufficient Funds → BudgetExhaustedError (graceful exit)
      - Другие ошибки → retry, потом пропуск
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Разводка клиентов: дистилляция/классификация текста — через DeepSeek
            # (OpenAI используется только для embeddings в loader.py).
            response = _get_deepseek().chat.completions.create(
                model=DISTILL_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.2,
                max_completion_tokens=max_completion_tokens,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            data = json.loads(raw)

            if data.get("skip"):
                return None

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parse error ({context}): {e}")
            return {} if context == "classify" else None

        except Exception as e:
            error_str = str(e).lower()

            # Баланс исчерпан — немедленный выход
            if "insufficient" in error_str or "402" in error_str or "billing" in error_str:
                logger.error(f"💸 БАЛАНС OPENAI ИСЧЕРПАН: {e}")
                raise BudgetExhaustedError(f"Баланс OpenAI исчерпан: {e}")

            # Rate limit — retry с задержкой
            if "rate" in error_str or "429" in error_str or "limit" in error_str:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(f"⏳ Rate limit ({context}), попытка {attempt + 1}/{MAX_RETRIES}, жду {delay} сек...")
                time.sleep(delay)
                continue

            # Другие ошибки
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(f"⚠️ OpenAI {context} error (попытка {attempt + 1}): {e}, жду {delay} сек...")
                time.sleep(delay)
            else:
                logger.error(f"❌ OpenAI {context} error (все попытки): {e}")
                return {} if context == "classify" else None

    return {} if context == "classify" else None


# ============================================================
# Утилиты
# ============================================================

def _detect_language(text: str) -> str:
    """Простое определение языка по символам"""
    cyrillic = len(re.findall(r'[а-яА-ЯёЁ]', text[:500]))
    latin = len(re.findall(r'[a-zA-Z]', text[:500]))
    return "ru" if cyrillic > latin else "en"


def _avg_sentence_length(text: str) -> float:
    """Средняя длина предложения в словах"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return 10
    total_words = sum(len(s.split()) for s in sentences)
    return total_words / len(sentences)


def _is_reference_content(text: str) -> bool:
    """Определить, является ли текст справочным (таблицы, списки, формулы)"""
    # Высокая плотность спецсимволов = справочный контент
    special_chars = len(re.findall(r'[|:;\-\d{}\[\]]', text))
    return special_chars / max(len(text), 1) > 0.1
