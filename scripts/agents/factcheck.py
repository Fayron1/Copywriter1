"""
Анти-галлюцинационный модуль: верификация фактов и хеджирование утверждений.

Три публичные функции (все безопасны: при любом сбое возвращают пустой/исходный результат,
пайплайн не падает):

1. verify_facts(facts) — мульти-источниковая верификация ключевых числовых фактов
   через Google Search Grounding (gemini-3.1-pro). Вариант B. Только строгие темы.
   Факт = verified только при ≥2 источниках из ≥2 разных доменов.

2. extract_claims(text) — извлечение из черновика всех проверяемых утверждений
   (цифры, ставки, статьи закона, даты, суммы). Вариант A, фаза 1. Всегда.

3. hedge_claims(unsupported) — перепись неподтверждённых утверждений в осторожную
   (хеджированную) форму. Вариант A, фаза 3. Только если есть unsupported.

Переиспользует инфраструктуру freshness.py (_config, _loads_lenient, grounded_search).
Свой собственный _call_kie нужен, т.к. в freshness._call_kie системный промпт и схема
зашиты жёстко. Здесь — обобщённый вызыватель с параметрами system_prompt/schema.
"""
import os
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("agents.factcheck")

# Переиспользуем общие хелперы из freshness
from .freshness import _config as _freshness_config
from .freshness import _loads_lenient
from .freshness import _to_float


# ────────────────────────────────────────────────────────────
# Конфигурация
# ────────────────────────────────────────────────────────────
def is_verify_enabled() -> bool:
    """Включена ли мульти-источниковая верификация (env FACT_VERIFY_ENABLED, по умолчанию true)."""
    return os.getenv("FACT_VERIFY_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def is_claim_check_enabled() -> bool:
    """Включён ли Claim Extractor (env CLAIM_CHECK_ENABLED, по умолчанию true)."""
    return os.getenv("CLAIM_CHECK_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def _cfg() -> Dict[str, Any]:
    """Конфиг: переиспользует FRESHNESS_* env (тот же kie.ai / gemini backend)."""
    cfg = _freshness_config()
    # Свои лимиты, чтобы можно было тюнить независимо
    cfg["max_verify_facts"] = int(os.getenv("FACT_VERIFY_MAX_FACTS", "8"))
    cfg["min_sources"] = int(os.getenv("FACT_VERIFY_MIN_SOURCES", "2"))
    cfg["min_domains"] = int(os.getenv("FACT_VERIFY_MIN_DOMAINS", "2"))
    return cfg


# Авторитетные домены для ранжирования ссылок (переиспользуется в pipeline._step_assemble_references)
PRIMARY_DOMAINS = (
    "nalog.gov.ru", "pravo.gov.ru", "publication.pravo.gov.ru",
    "consultant.ru", "garant.ru", "minfin.gov.ru", "nalog.ru",
    "rosstat.gov.ru", "zakon.gov.ru",
)
MAJOR_MEDIA_DOMAINS = (
    "rbc.ru", "forbes.ru", "vedomosti.ru", "vc.ru", "kommersant.ru",
    "tass.ru", "ria.ru", "interfax.ru",
)


def domain_authority(url: str) -> int:
    """Балл авторитетности домена: 3 (первичник), 2 (крупное СМИ), 1 (прочее)."""
    try:
        host = (urlparse(url).netloc or "").lower().lstrip("www.")
    except Exception:
        return 0
    if any(host == d or host.endswith("." + d) for d in PRIMARY_DOMAINS):
        return 3
    if any(host == d or host.endswith("." + d) for d in MAJOR_MEDIA_DOMAINS):
        return 2
    return 1


# ────────────────────────────────────────────────────────────
# Обобщённый LLM-вызыватель (gemini-3.1-pro + grounding, по образцу freshness._call_kie)
# ────────────────────────────────────────────────────────────
def _truncate_at_word(text: str, max_chars: int = 120, ellipsis: str = "…") -> str:
    """Обрезать текст до max_chars, не ломая слово.

    Идём до последнего пробела в пределах лимита; если получилось осмысленно
    (хотя бы 40% от лимита) — добавляем ellipsis. Иначе возвращаем как есть
    (короткие строки не трогаем). Гарантирует, что описания источников не
    обрываются на полуслове («...кроме о»).
    """
    if not text:
        return ""
    s = str(text).strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    # последнее слово целиком: ищем последний пробел
    last_space = cut.rfind(" ")
    if last_space >= int(max_chars * 0.4):
        cut = cut[:last_space].rstrip(".,;:—- ")
    return cut.rstrip() + ellipsis


def _call_kie_generic(
    system_prompt: str,
    user_text: str,
    response_schema: Dict[str, Any],
    cfg: Dict[str, Any],
    use_grounding: bool = True,
) -> Optional[Any]:
    """
    Один вызов kie.ai gemini-3.1-pro (grounded, если use_grounding=True).
    Возвращает распарсенный объект (dict/list) или None при сбое.
    """
    import httpx

    base = os.getenv("FRESHNESS_API_BASE", "https://api.kie.ai").rstrip("/")
    url = cfg["url"]

    payload: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
        "include_thoughts": False,
        "reasoning_effort": cfg["reasoning_effort"],
        "response_format": response_schema,
    }
    if use_grounding:
        payload["tools"] = [{"type": "function", "function": {"name": "googleSearch"}}]

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    last_err: Optional[str] = None
    for attempt in range(1, cfg["max_retries"] + 1):
        try:
            with httpx.Client(timeout=cfg["timeout"]) as http:
                resp = http.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                raw = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                return _loads_lenient(raw) if raw else None
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                logger.warning(f"⚠️ [factcheck] невосстановимая ошибка API: {last_err}")
                return None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < cfg["max_retries"]:
            backoff = min(2 ** attempt, 10)
            time.sleep(backoff)

    logger.warning(f"⚠️ [factcheck] не удалось получить ответ за {cfg['max_retries']} попыток: {last_err}")
    return None


# ════════════════════════════════════════════════════════════
# ВАРИАНТ B: мульти-источниковая верификация фактов
# ════════════════════════════════════════════════════════════
_VERIFY_SYSTEM_PROMPT = (
    "Ты — строгий верификатор фактов для деловых статей по налогам, праву и финансам РФ.\n"
    "Тебе дают список ключевых фактов (ставки, лимиты, суммы, даты, номера статей закона).\n"
    "Используя доступный тебе Google Search, проверь КАЖДЫЙ факт по нескольким независимым источникам.\n\n"
    "ПРАВИЛА:\n"
    "1. Приоритет — официальные и авторитетные ресурсы (nalog.gov.ru, pravo.gov.ru, Минфин,\n"
    "   КонсультантПлюс, Гарант). СМИ и блоги — только как подтверждение.\n"
    "2. Для каждого факта верни список реально найденных источников (title, url, snippet).\n"
    "3. verdict:\n"
    "   - 'verified' — факт подтверждён;\n"
    "   - 'partial' — подтверждён частично / неточно сформулирован;\n"
    "   - 'unverified' — не удалось подтвердить;\n"
    "   - 'conflict' — источники противоречат друг другу.\n"
    "4. НЕ выдумывай источники и URL. Если ничего не нашёл — пустой sources и verdict='unverified'.\n"
    "5. Для conflict/unverified укажи correction — корректное значение, если удалось найти.\n"
)

_VERIFY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_verify",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "verdict": {"type": "string", "description": "verified | partial | unverified | conflict"},
                            "confidence": {"type": "number"},
                            "correction": {"type": "string", "description": "Корректное значение, если факт ошибочен"},
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"},
                                    },
                                    "required": ["title", "url", "snippet"],
                                },
                            },
                        },
                        "required": ["claim", "verdict", "confidence", "sources"],
                    },
                }
            },
            "required": ["results"],
        },
    },
}


def _select_verifiable_facts(facts: Dict[str, Any], max_facts: int = 8) -> List[str]:
    """Выбрать из state.facts ключевые числовые/правовые факты для проверки.

    Фильтруем по признаку «содержит число/процент/год/статью закона».
    Сортируем по reliability (первичники — вперёд), берём top-N.
    """
    items = facts.get("facts") if isinstance(facts, dict) else None
    if not isinstance(items, list):
        return []

    # Признаки проверяемого факта: цифры, %, годы (2020-2030), статьи закона
    numeric_re = re.compile(r"\d|процент|%|\b20[0-2]\d\b|ст\.?\s*\d|п\.?\s*\d|НК\s*РФ|КоАП|ФЗ|ТК\s*РФ", re.IGNORECASE)

    candidates = []
    for f in items:
        if not isinstance(f, dict):
            continue
        claim = str(f.get("claim", "")).strip()
        if not claim or len(claim) < 8:
            continue
        if not numeric_re.search(claim):
            continue
        reliability = _to_float(f.get("reliability", 0.8))
        candidates.append((reliability, claim))

    # Первичники (reliability высокая) вперёд
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in candidates[:max_facts]]


def verify_facts(facts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Мульти-источниковая верификация ключевых фактов.

    Возвращает словарь {нормализованный_claim: {verdict, confidence, correction, sources, needs_hedging}}.
    needs_hedging=True если НЕ verified по жёсткому критерию (<min_sources источников ИЛИ <min_domains доменов).

    Безопасно: при любом сбое возвращает {}.
    """
    if not facts or not isinstance(facts, dict):
        return {}
    if not is_verify_enabled():
        logger.info("   ⏭️ [factcheck] верификация отключена (FACT_VERIFY_ENABLED=false)")
        return {}

    cfg = _cfg()
    if not cfg["api_key"]:
        logger.warning("⚠️ [factcheck] KIE_API_KEY не задан — верификация пропущена")
        return {}

    claims = _select_verifiable_facts(facts, cfg["max_verify_facts"])
    if not claims:
        logger.info("   ℹ️ [factcheck] нет числовых/правовых фактов для верификации")
        return {}

    logger.info(f"🔬 [factcheck] верификация {len(claims)} фактов через {cfg['model']} + Google Search...")

    user_text = "Проверь каждый факт по нескольким независимым источникам:\n\n"
    user_text += "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

    parsed = _call_kie_generic(_VERIFY_SYSTEM_PROMPT, user_text, _VERIFY_SCHEMA, cfg, use_grounding=True)
    if not isinstance(parsed, dict):
        logger.warning("⚠️ [factcheck] не удалось распарсить ответ верификатора")
        return {}

    results = parsed.get("results") or []
    if not isinstance(results, list):
        return {}

    min_sources = cfg["min_sources"]
    min_domains = cfg["min_domains"]
    out: Dict[str, Dict[str, Any]] = {}
    verified_count = 0
    hedged_count = 0

    for r in results:
        if not isinstance(r, dict):
            continue
        claim = str(r.get("claim", "")).strip()
        verdict = str(r.get("verdict", "unverified")).strip().lower()
        sources = r.get("sources") or []
        if not isinstance(sources, list):
            sources = []

        # Жёсткий программный фильтр: verified только при ≥min_sources источников И ≥min_domains доменов
        domains = set()
        clean_sources = []
        for s in sources:
            if not isinstance(s, dict):
                continue
            url = str(s.get("url", "")).strip()
            if not url.startswith("http"):
                continue
            try:
                host = (urlparse(url).netloc or "").lower().lstrip("www.")
            except Exception:
                host = ""
            if host:
                domains.add(host)
            clean_sources.append(s)

        hard_verified = len(clean_sources) >= min_sources and len(domains) >= min_domains
        # needs_hedging: НЕ прошёл жёсткий фильтр ИЛИ модель сказала conflict/unverified
        needs_hedging = (not hard_verified) or verdict in ("conflict", "unverified", "partial")
        if not needs_hedging:
            verified_count += 1
        else:
            hedged_count += 1

        out[_normalize_claim(claim)] = {
            "claim": claim,
            "verdict": verdict if hard_verified else (verdict if verdict != "verified" else "partial"),
            "confidence": _to_float(r.get("confidence", 0)),
            "correction": str(r.get("correction", "") or "").strip(),
            "sources": clean_sources,
            "domains": sorted(domains),
            "needs_hedging": needs_hedging,
        }

    logger.info(
        f"   ✅ [factcheck] проверено {len(out)} фактов: "
        f"verified={verified_count}, needs_hedging={hedged_count} "
        f"(критерий: ≥{min_sources} ист. из ≥{min_domains} доменов)"
    )
    return out


# ════════════════════════════════════════════════════════════
# ВАРИАНТ A — фаза 1: извлечение утверждений из черновика
# ════════════════════════════════════════════════════════════
_CLAIMS_SYSTEM_PROMPT = (
    "Ты — извлекатель проверяемых утверждений из деловой B2B-статьи.\n"
    "Найди ВСЕ утверждения, которые можно фактчекнуть: конкретные цифры, проценты, ставки,\n"
    "лимиты, суммы штрафов, номера статей законов (НК РФ, ТК РФ, КоАП, ФЗ), даты вступления в силу,\n"
    "сроки, пороговые значения.\n\n"
    "НЕ включай: общие утверждения без чисел, оценки мнений, маркетинговые тезисы.\n"
    "Для каждого утверждения верни:\n"
    "- text: точная формулировка из текста (цитата);\n"
    "- type: rate | limit | penalty | date | law_article | amount | threshold | other;\n"
    "- value: нормализованное значение (например '20%', '50000', '2026-01-01', 'ст. 14.5 КоАП');\n"
    "- law: номер статьи/закона, если есть (например 'НК РФ ст. 346.20'), иначе пусто.\n"
)

_CLAIMS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "claims_extract",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "type": {"type": "string"},
                            "value": {"type": "string"},
                            "law": {"type": "string"},
                        },
                        "required": ["text", "type", "value"],
                    },
                }
            },
            "required": ["claims"],
        },
    },
}


def extract_claims(text: str) -> List[Dict[str, str]]:
    """
    Извлечь проверяемые утверждения из текста статьи.

    Возвращает список {text, type, value, law}. При сбое — [].
    """
    if not text or not text.strip():
        return []
    if not is_claim_check_enabled():
        logger.info("   ⏭️ [factcheck] Claim Extractor отключён (CLAIM_CHECK_ENABLED=false)")
        return []

    cfg = _cfg()
    if not cfg["api_key"]:
        logger.warning("⚠️ [factcheck] KIE_API_KEY не задан — извлечение утверждений пропущено")
        return []

    # Ограничиваем длину, чтобы не раздувать токены (достаточно для фактчекинга)
    snippet = text[:20000]
    logger.info(f"🔍 [factcheck] извлечение проверяемых утверждений из черновика ({len(snippet)} симв.)...")

    parsed = _call_kie_generic(
        _CLAIMS_SYSTEM_PROMPT, snippet, _CLAIMS_SCHEMA, cfg, use_grounding=False
    )
    if not isinstance(parsed, dict):
        logger.warning("⚠️ [factcheck] не удалось распарсить ответ экстрактора утверждений")
        return []

    claims = parsed.get("claims") or []
    if not isinstance(claims, list):
        return []

    clean: List[Dict[str, str]] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        t = str(c.get("text", "")).strip()
        v = str(c.get("value", "")).strip()
        if not t or not v:
            continue
        clean.append({
            "text": t,
            "type": str(c.get("type", "other")).strip(),
            "value": v,
            "law": str(c.get("law", "") or "").strip(),
        })

    logger.info(f"   ✅ [factcheck] извлечено утверждений: {len(clean)}")
    return clean


# ════════════════════════════════════════════════════════════
# Сверка claims с фактами (фаза 2, детерминированная, 0 токенов)
# ════════════════════════════════════════════════════════════
def _normalize_value(v: str) -> str:
    """Нормализация значения для сравнения: только цифры, точка (десятичная) и %.

    «50 000 ₽» → «50000», «20%» → «20%», «6%» → «6%».
    Буквы/валюты отбрасываются — значение нужно для матчинга чисел/ставок,
    закон сравнивается отдельно по номеру статьи.
    """
    if not v:
        return ""
    s = str(v).lower().strip()
    s = re.sub(r"[^\d.%]", "", s)
    return s


def _normalize_claim(c: str) -> str:
    """Нормализация текста факта/утверждения для ключа словаря."""
    if not c:
        return ""
    s = str(c).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s[:200]


def match_claims_to_facts(
    claims: List[Dict[str, str]],
    facts: Dict[str, Any],
    verified_facts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Сверить извлечённые claims с проверенными фактами. ДЕТЕРМИНИРОВАННО (0 токенов).

    Возвращает (supported, unsupported):
    - supported — claim подтверждён фактом (значение найдено в facts) ИЛИ верифицирован ≥2 ист.;
    - unsupported — не найден в фактах И не верифицирован (или needs_hedging=True).

    Логика:
    1. Строим индекс известных значений из facts["facts"][].claim + value.
    2. Если claim совпал по нормализованному value/law с фактом → supported.
    3. Если verified_facts содержит факт и needs_hedging=False → supported.
    4. Иначе — unsupported.
    """
    if not isinstance(claims, list):
        return [], []

    # Индекс известных значений и законов из facts
    known_values = set()
    known_laws = set()
    known_claims_norm = set()
    items = facts.get("facts") if isinstance(facts, dict) else None
    if isinstance(items, list):
        for f in items:
            if not isinstance(f, dict):
                continue
            fc = str(f.get("claim", ""))
            if fc:
                known_claims_norm.add(_normalize_claim(fc))
            # Извлекаем числа из факта как известные значения
            for m in re.findall(r"\d[\d\s]*[%]?(?:\s*₽)?", str(f.get("claim", "")) + " " + str(f.get("comment", ""))):
                known_values.add(_normalize_value(m))
            # Законы
            law = str(f.get("source", "")) + " " + str(f.get("claim", ""))
            for m in re.findall(r"(?:ст\.?\s*\d[\d.]*|п\.?\s*\d[\d.]*|НК\s*РФ|ТК\s*РФ|КоАП|ФЗ\s*[-]?\s*\d+)", law, re.IGNORECASE):
                known_laws.add(_normalize_value(m))

    # Индекс верифицированных фактов
    verified_norm = set()
    hedged_norm = set()
    if verified_facts and isinstance(verified_facts, dict):
        for key, info in verified_facts.items():
            if not isinstance(info, dict):
                continue
            if info.get("needs_hedging"):
                hedged_norm.add(key)
            else:
                verified_norm.add(key)

    supported: List[Dict[str, str]] = []
    unsupported: List[Dict[str, str]] = []

    for c in claims:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text", ""))
        value = _normalize_value(str(c.get("value", "")))
        law = _normalize_value(str(c.get("law", "")))
        text_norm = _normalize_claim(text)

        is_supported = False

        # 1. Прямое совпадение значения с известным
        if value and value in known_values:
            is_supported = True
        # 2. Совпадение по закону
        if not is_supported and law and law in known_laws:
            is_supported = True
        # 3. Текст факта близко к известному
        if not is_supported and text_norm and text_norm in known_claims_norm:
            is_supported = True
        # 4. Верифицирован внешним ревизором (≥2 ист.)
        if not is_supported:
            for vkey in verified_norm:
                if vkey and (vkey in text_norm or text_norm in vkey):
                    is_supported = True
                    break

        # 5. Помечен как needs_hedging внешним ревизором → принудительно unsupported
        forced_unsupported = False
        for hkey in hedged_norm:
            if hkey and (hkey in text_norm or text_norm in hkey):
                forced_unsupported = True
                break

        if forced_unsupported or not is_supported:
            unsupported.append(c)
        else:
            supported.append(c)

    logger.info(
        f"   🔍 [factcheck] сверка: {len(supported)} поддержано фактами, "
        f"{len(unsupported)} неподтверждено → на хеджирование"
    )
    return supported, unsupported


# ════════════════════════════════════════════════════════════
# ВАРИАНТ A — фаза 3: хеджирование неподтверждённых утверждений
# ════════════════════════════════════════════════════════════
_HEDGE_SYSTEM_PROMPT = (
    "Ты — редактор-стилист деловых B2B-текстов. Тебе дают список утверждений, которые\n"
    "НЕ подтверждены проверенными источниками (потенциальные галлюцинации или неточные формулировки).\n"
    "Твоя задача — переписать КАЖДОЕ утверждение в осторожную (хеджированную) форму, не теряя сути.\n\n"
    "ПРАВИЛА ХЕДЖИРОВАНИЯ:\n"
    "1. УБИРАЙ АБСОЛЮТЫ: «всегда», «невозможно», «точно», «гарантированно», «обязательно»\n"
    "   (кроме железобетонных фактов) → «как правило», «в большинстве случаев», «обычно», «как правило».\n"
    "2. РАЗДЕЛЯЙ ФАКТ И ВЫВОД: сначала что известно, потом интерпретация. Не выдавай оценку за факт.\n"
    "3. ПРАВОВЫЕ ОГОВОРКИ: для юридических тем добавляй контекст — страна (РФ), статус резидентства,\n"
    "   тип договора/режима, период действия нормы («по состоянию на 2026 год», «для налоговых резидентов РФ»).\n"
    "4. ТОН: редакторский, а не адвокатски-утверждающий. Меньше категоричности, больше точности.\n"
    "5. СОХРАНЯЙ длину близкой к оригиналу (±20%), стиль и падежи. Не меняй смысл — только уверенность.\n"
    "6. НЕ добавляй новые цифры, законы или факты. Только смягчай формулировку.\n\n"
    "ПРИМЕРЫ:\n"
    "«Ставка составляет 20%» → «По данным отрасли, ставка оценивается примерно в 20%»\n"
    "«Штраф 50 000 ₽ по ст. 14.5 КоАП» → «Штраф может достигать порядка 50 000 ₽ (ст. 14.5 КоАП РФ, для юрлиц)»\n"
    "«Это всегда приводит к блокировке счёта» → «Как правило, это может привести к блокировке счёта»\n"
)

_HEDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hedge_claims",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "hedges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string", "description": "Точная цитата из текста (как в claim.text)"},
                            "hedged": {"type": "string", "description": "Осторожная формулировка"},
                        },
                        "required": ["original", "hedged"],
                    },
                }
            },
            "required": ["hedges"],
        },
    },
}


def hedge_claims(unsupported: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Переписать неподтверждённые утверждения в хеджированную форму.

    Возвращает список {original, hedged}. original — точная цитата для text.replace().
    При сбое — [].
    """
    if not unsupported:
        return []

    cfg = _cfg()
    if not cfg["api_key"]:
        logger.warning("⚠️ [factcheck] KIE_API_KEY не задан — хеджирование пропущено")
        return []

    # Ограничиваем количество, чтобы не раздувать (не больше 12 за вызов)
    batch = unsupported[:12]
    logger.info(f"🛡️ [factcheck] хеджирование {len(batch)} неподтверждённых утверждений...")

    user_text = "Перепиши следующие утверждения в осторожную форму (правила выше):\n\n"
    for i, c in enumerate(batch, 1):
        user_text += f"{i}. {c.get('text', '')}\n"

    parsed = _call_kie_generic(_HEDGE_SYSTEM_PROMPT, user_text, _HEDGE_SCHEMA, cfg, use_grounding=False)
    if not isinstance(parsed, dict):
        logger.warning("⚠️ [factcheck] не удалось распарсить ответ хеджирования")
        return []

    hedges = parsed.get("hedges") or []
    if not isinstance(hedges, list):
        return []

    clean: List[Dict[str, str]] = []
    for h in hedges:
        if not isinstance(h, dict):
            continue
        orig = str(h.get("original", "")).strip()
        hedged = str(h.get("hedged", "")).strip()
        if not orig or not hedged or orig == hedged:
            continue
        clean.append({"original": orig, "hedged": hedged})

    logger.info(f"   ✅ [factcheck] подготовлено хеджей: {len(clean)}")
    return clean


# ════════════════════════════════════════════════════════════
# REFERENCE VALIDATOR — ловля перепутанных статей закона
# (баг ст.185.1: ст.185.1 ТК РФ = диспансеризация, а не сверхурочные)
# ════════════════════════════════════════════════════════════
def is_ref_validate_enabled() -> bool:
    """Включён ли Reference Validator (env REF_VALIDATE_ENABLED, по умолчанию true)."""
    return os.getenv("REF_VALIDATE_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


# Regex извлечения атомарных отсылок к нормативным актам РФ.
# Ловит: "ст. 152 ТК РФ", "статья 99", "в статье 185.1", "п. 2 ст. 54.1 НК РФ",
#        "КоАП ст. 14.5", "ст. 14.5 КоАП", "ФЗ-115", "115-ФЗ"
# "статье/статьи/статью/статья/ст." — любой падеж слова «статья» + сокращение «ст.»
_CITATION_RE = re.compile(
    r"(?:"
    r"(?:п\.?\s*\d+(?:\.\d+)*\s*ст(?:атья|атьи|атье|атью|атьё|\.)?\s*\d+(?:\.\d+)*)"  # п. 2 ст. 54.1
    r"|(?:ст(?:атья|атьи|атье|атью|атьё|\.)?\s*\d+(?:\.\d+)*)"                        # ст. 152 / в статье 185.1
    r"|(?:КоАП\s*(?:РФ\s*)?(?:ст\.?\s*)?\d+(?:\.\d+)*)"    # КоАП ст. 14.5
    r"|(?:ст\.?\s*\d+(?:\.\d+)*\s*КоАП)"                   # ст. 14.5 КоАП
    r"|(?:\d+(?:\.\d+)?\s*[-‑]?\s*ФЗ)"                     # 115-ФЗ
    r"|(?:ФЗ\s*[-‑]?\s*\d+(?:\.\d+)*)"                     # ФЗ-115
    r")"
    r"\s*(?:РФ)?"
    r"(?:\s+(?:НК|ТК|ГК|ЖК|СК|УК|АПК|ГрК)\s*РФ)?",
    re.IGNORECASE,
)

# Кодексы, которые часто привязываются: ТК РФ, НК РФ, ГК РФ, КоАП РФ
_CODEX_RE = re.compile(r"\b(НК|ТК|ГК|ЖК|СК|УК|АПК|ГрК)\s*РФ\b|\bКоАП(?:\s*РФ)?\b", re.IGNORECASE)


def _sentence_around(text: str, pos: int, length: int = 300) -> str:
    """Вернуть контекст-окно вокруг позиции pos (для проверки темы отсылки).

    Идём от pos назад до начала предложения (. ! ? с пробелом) и вперёд аналогично.
    Ограничиваем окно length символами. Надёжнее, чем _split_sentences, т.к. точки
    внутри номеров статей (п. 2 ст. 54.1) не ломают извлечение.
    """
    if not text:
        return ""
    # Назад: ищем конец предыдущего предложения
    start = pos
    for i in range(pos, max(-1, pos - length), -1):
        if i <= 0:
            start = 0
            break
        if i < len(text) and text[i] in ".!?" and (i + 1 >= len(text) or text[i + 1].isspace()):
            start = i + 1
            break
    # Вперёд: ищем конец текущего предложения
    end = pos
    for i in range(pos, min(len(text), pos + length)):
        if text[i] in ".!?" and (i + 1 >= len(text) or text[i + 1].isspace()):
            end = i + 1
            break
    else:
        end = min(len(text), pos + length)
    return text[start:end].strip()


def _extract_legal_citations(text: str, max_citations: int = 12) -> List[Dict[str, str]]:
    """Детерминированно извлечь отсылки к нормативным актам из текста (0 токенов).

    Возвращает список {citation, context}:
    - citation: нормализованная отсылка («ст. 185.1 ТК РФ»);
    - context: предложение/окно, в котором найдена отсылка (для проверки темы).

    Работает по finditer по всему тексту (не по предложениям — точки внутри
    номеров статей типа «п. 2 ст. 54.1» иначе ломают разбиение). Контекст берём
    окном вокруг совпадения.
    """
    if not text or not text.strip():
        return []

    seen = set()
    out: List[Dict[str, str]] = []

    for m in _CITATION_RE.finditer(text):
        citation = m.group(0).strip()
        citation = re.sub(r"\s+", " ", citation)
        # Дополним кодексом, если он есть рядом в тексте, но не в самой отсылке
        # (ищем кодекс в окне ±40 символов вокруг совпадения)
        window_start = max(0, m.start() - 40)
        window_end = min(len(text), m.end() + 40)
        window = text[window_start:window_end]
        codex_match = _CODEX_RE.search(window)
        if codex_match and not _CODEX_RE.search(citation):
            citation = f"{citation} {codex_match.group(0)}"
        context = _sentence_around(text, m.start(), 300)
        key = (citation.lower(), context[:80].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"citation": citation, "context": context})
        if len(out) >= max_citations:
            break

    return out


_REF_SYSTEM_PROMPT = (
    "Ты — строгий валидатор юридических отсылок в деловых статьях по праву и налогам РФ.\n"
    "Тебе дают список отсылок к нормативным актам (статьи законов) с контекстом — предложением,\n"
    "в котором они использованы. Твоя задача — используя Google Search, проверить КАЖДУЮ отсылку:\n"
    "1) существует ли статья с таким номером в указанном кодексе (ТК РФ, НК РФ, ГК РФ, КоАП, ФЗ);\n"
    "2) РЕГУЛИРУЕТ ли эта статья тему, заявленную в контексте.\n\n"
    "ГЛАВНОЕ ПРАВИЛО (именно его нарушения мы ловим):\n"
    "Если отсылка в контексте про тему А (напр. «сверхурочные / переработки»), а статья с этим\n"
    "номером фактически регулирует тему Б (напр. ст. 185.1 ТК РФ — это диспансеризация/медосмотры,\n"
    "НЕ сверхурочные), то verdict='wrong_topic' и в actual_topic укажи реальную тему статьи,\n"
    "а в correction — правильную статью для темы А, если её удалось найти.\n\n"
    "verdict:\n"
    "- 'correct' — статья существует и регулирует заявленную тему;\n"
    "- 'wrong_topic' — статья существует, но регулирует ДРУГУЮ тему (это баг!);\n"
    "- 'not_found' — не удалось подтвердить существование статьи с таким номером.\n\n"
    "ПРАВИЛА:\n"
    "- Приоритет источников: pravo.gov.ru, КонсультантПлюс, Гарант, nalog.gov.ru.\n"
    "- НЕ выдумывай номера статей в correction — только если нашёл реальную статью.\n"
    "- Если unsure — лучше 'correct' (не ломаем верную отсылку), чем ложный 'wrong_topic'.\n"
    "- confidence: 0..1, насколько уверен в verdict.\n"
)

_REF_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ref_validate",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "citation": {"type": "string", "description": "Отсылка как в тексте"},
                            "verdict": {"type": "string", "description": "correct | wrong_topic | not_found"},
                            "actual_topic": {"type": "string", "description": "Реальная тема статьи (для wrong_topic)"},
                            "correction": {"type": "string", "description": "Правильная статья для темы контекста, если найдена"},
                            "confidence": {"type": "number"},
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"},
                                    },
                                    "required": ["title", "url", "snippet"],
                                },
                            },
                        },
                        "required": ["citation", "verdict", "confidence"],
                    },
                }
            },
            "required": ["references"],
        },
    },
}


def validate_law_references(text: str) -> List[Dict[str, Any]]:
    """
    Проверить все отсылки к статьям закона в тексте на предмет-тематическое соответствие.

    Возвращает список ТОЛЬКО проблемных отсылок (verdict in (wrong_topic, not_found)),
    для каждой: {citation, context, verdict, actual_topic, correction, confidence, sources}.
    При любом сбое — [].

    Это закрывает слепое пятно Fact Verifier: тот проверяет «есть ли факт в источниках»,
    а здесь проверяем «существует ли статья и регулирует ли она заявленную тему».
    """
    if not text or not text.strip():
        return []
    if not is_ref_validate_enabled():
        logger.info("   ⏭️ [factcheck] Reference Validator отключён (REF_VALIDATE_ENABLED=false)")
        return []

    cfg = _cfg()
    if not cfg["api_key"]:
        logger.warning("⚠️ [factcheck] KIE_API_KEY не задан — валидация отсылок пропущена")
        return []

    # Фаза 1: детерминированное извлечение отсылок (0 токенов)
    citations = _extract_legal_citations(text)
    if not citations:
        logger.info("   ℹ️ [factcheck] отсылок к статьям закона не найдено — валидация не нужна.")
        return []

    logger.info(f"🔎 [factcheck] валидация {len(citations)} отсылок к статьям закона через {cfg['model']} + Google Search...")

    # Фаза 2: один grounded LLM-вызов для всех отсылок
    user_text = "Проверь каждую отсылку к статье закона: существует ли она и регулирует ли тему контекста.\n\n"
    for i, c in enumerate(citations, 1):
        user_text += f"{i}. Отсылка: {c['citation']}\n   Контекст: {c['context']}\n"

    parsed = _call_kie_generic(_REF_SYSTEM_PROMPT, user_text, _REF_SCHEMA, cfg, use_grounding=True)
    if not isinstance(parsed, dict):
        logger.warning("⚠️ [factcheck] не удалось распарсить ответ валидатора отсылок")
        return []

    refs = parsed.get("references") or []
    if not isinstance(refs, list):
        return []

    # Контекст-индекс для возврата (маппим обратно к исходному предложению)
    ctx_by_citation = {}
    for c in citations:
        ctx_by_citation.setdefault(c["citation"].lower(), c["context"])

    problems: List[Dict[str, Any]] = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        verdict = str(r.get("verdict", "")).strip().lower()
        if verdict not in ("wrong_topic", "not_found"):
            continue  # только проблемные
        citation = str(r.get("citation", "")).strip()
        if not citation:
            continue
        confidence = _to_float(r.get("confidence", 0))
        problems.append({
            "citation": citation,
            "context": ctx_by_citation.get(citation.lower(), ""),
            "verdict": verdict,
            "actual_topic": str(r.get("actual_topic", "") or "").strip(),
            "correction": str(r.get("correction", "") or "").strip(),
            "confidence": confidence,
            "sources": [s for s in (r.get("sources") or []) if isinstance(s, dict)],
        })

    logger.info(
        f"   ✅ [factcheck] проверено отсылок: {len(refs)}, проблемных: {len(problems)} "
        f"(wrong_topic/not_found)"
    )
    return problems


def hedge_references(problems: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Переписать предложения с проблемными отсылками к статьям закона в безопасную форму.

    Для каждой проблемной отсылки:
    - если есть correction (найдена правильная статья) → заменить неверную на правильную;
    - иначе → смягчить, убрать конкретный неверный номер статьи
      («...согласно положениям ТК РФ о сверхурочной работе»).

    Возвращает список {original, hedged} для точечного text.replace().
    При сбое — [].
    """
    if not problems:
        return []

    cfg = _cfg()
    if not cfg["api_key"]:
        logger.warning("⚠️ [factcheck] KIE_API_KEY не задан — хеджирование отсылок пропущено")
        return []

    hedge_system = (
        "Ты — редактор-юрист. Тебе дают предложения из деловой статьи, в которых обнаружены\n"
        "ОШИБОЧНЫЕ отсылки к статьям закона (номер статьи не соответствует теме, либо статья\n"
        "не найдена). Перепиши КАЖДОЕ предложение в безопасную форму:\n\n"
        "ПРАВИЛА:\n"
        "1. Если указано поле correction (найдена правильная статья) — замени неверный номер\n"
        "   статьи на correction, сохранив остальную формулировку.\n"
        "2. Если correction пустой — убери конкретный номер ошибочной статьи и замени на общую\n"
        "   отсылку к кодексу по теме контекста (напр. «согласно положениям ТК РФ о сверхурочной\n"
        "   работе» вместо «ст. 185.1 ТК РФ»).\n"
        "3. Сохрани смысл, длину (±25%), стиль и падежи. Не добавляй новых цифр/законов.\n"
        "4. Не меняй ничего, кроме отсылки к закону.\n"
    )

    hedge_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "ref_hedge",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "hedges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original": {"type": "string", "description": "Точное предложение из контекста"},
                                "hedged": {"type": "string", "description": "Безопасная формулировка"},
                            },
                            "required": ["original", "hedged"],
                        },
                    }
                },
                "required": ["hedges"],
            },
        },
    }

    user_text = "Перепиши предложения с ошибочными отсылками к статьям закона:\n\n"
    for i, p in enumerate(problems[:8], 1):
        corr = p.get("correction", "")
        note = f" (замени на: {corr})" if corr else " (убери конкретный номер, оставь общую отсылку к кодексу по теме)"
        user_text += f"{i}. Предложение: {p.get('context', '')}\n   Проблема: {p.get('citation', '')} — {p.get('actual_topic', 'не найдена')}{note}\n"

    parsed = _call_kie_generic(hedge_system, user_text, hedge_schema, cfg, use_grounding=False)
    if not isinstance(parsed, dict):
        logger.warning("⚠️ [factcheck] не удалось распарсить ответ хеджирования отсылок")
        return []

    hedges = parsed.get("hedges") or []
    if not isinstance(hedges, list):
        return []

    clean: List[Dict[str, str]] = []
    for h in hedges:
        if not isinstance(h, dict):
            continue
        orig = str(h.get("original", "")).strip()
        hedged = str(h.get("hedged", "")).strip()
        if not orig or not hedged or orig == hedged:
            continue
        clean.append({"original": orig, "hedged": hedged})

    logger.info(f"   ✅ [factcheck] подготовлено хеджей отсылок: {len(clean)}")
    return clean


# ════════════════════════════════════════════════════════════
# ТЕМПОРАЛЬНЫЙ ФАКТЧЕКИНГ — будущие законы не выдавать за действующие
# (баг: «ст. 152 действует в новой редакции» при effective_date 01.09.2026
#  и сегодня 25.06.2026 — норма ещё не в силе)
# ════════════════════════════════════════════════════════════
def is_temporal_check_enabled() -> bool:
    """Включён ли темпоральный чек (env TEMPORAL_CHECK_ENABLED, по умолчанию true)."""
    return os.getenv("TEMPORAL_CHECK_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


_FUTURE_SYSTEM_PROMPT = (
    "Ты — темпоральный фактчекер деловых статей по праву и налогам РФ.\n"
    "Тебе дают фрагменты статьи, где упоминаются правовые нормы с датами вступления в силу.\n"
    "Сегодня — {today}. Твоя задача: для каждой пары {норма, дата} определить, вступает ли норма\n"
    "в силу ПОЗЖЕ сегодня (т.е. ещё НЕ действует) на момент {today}.\n\n"
    "Используй Google Search, чтобы подтвердить дату вступления в силу из авторитетных источников\n"
    "(pravo.gov.ru, КонсультантПлюс, Гарант).\n\n"
    "verdict:\n"
    "- 'future' — дата вступления в силу ПОЗЖЕ {today} (норма ещё НЕ действует, это будущее изменение);\n"
    "- 'current' — норма вступила в силу ДО или В {today} (действует сейчас);\n"
    "- 'unknown' — дату установить не удалось.\n\n"
    "Для 'future' обязательно укажи effective_date (ISO) и current_rule — кратко текущий\n"
    "(пока действующий) порядок, чтобы статья не маскировала будущую норму под действующую.\n"
    "Возвращай ТОЛЬКО нормы с verdict='future' (проблемные) — они требуют хеджирования в будущее время.\n"
)

_FUTURE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "temporal_check",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "future_laws": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "citation": {"type": "string", "description": "Норма/статья как в тексте"},
                            "effective_date": {"type": "string", "description": "Дата вступления в силу (ISO YYYY-MM-DD)"},
                            "verdict": {"type": "string", "description": "future | current | unknown"},
                            "current_rule": {"type": "string", "description": "Кратко: текущий (пока действующий) порядок"},
                            "confidence": {"type": "number"},
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"},
                                    },
                                    "required": ["title", "url", "snippet"],
                                },
                            },
                        },
                        "required": ["citation", "verdict", "confidence"],
                    },
                }
            },
            "required": ["future_laws"],
        },
    },
}


def check_future_laws(text: str, today: str) -> List[Dict[str, Any]]:
    """
    Найти в тексте правовые нормы, вступающие в силу ПОЗЖЕ today (т.е. ещё не действующие).

    today — ISO-дата 'YYYY-MM-DD' (текущая дата пайплайна).
    Возвращает список ТОЛЬКО проблемных (verdict='future') норм:
      {citation, effective_date, verdict, current_rule, confidence, sources, context}.
    При любом сбое — [].

    Это закрывает дыру «будущий закон подан как действующая норма»: временный чекер
    находит пары {норма, дата в будущем} и подаёт их на хеджирование в будущее время.
    """
    if not text or not text.strip() or not today:
        return []
    if not is_temporal_check_enabled():
        logger.info("   ⏭️ [factcheck] темпоральный чек отключён (TEMPORAL_CHECK_ENABLED=false)")
        return []

    cfg = _cfg()
    if not cfg["api_key"]:
        logger.warning("⚠️ [factcheck] KIE_API_KEY не задан — темпоральный чек пропущен")
        return []

    snippet = text[:20000]
    logger.info(f"📅 [factcheck] темпоральный чек норм в черновике (сегодня {today})...")

    system_prompt = _FUTURE_SYSTEM_PROMPT.format(today=today)
    user_text = (
        f"Сегодня: {today}.\n"
        f"Найди в тексте правовые нормы с датами вступления в силу ПОЗЖЕ {today} "
        f"(ещё не действующие на {today}):\n\n{snippet}"
    )

    parsed = _call_kie_generic(system_prompt, user_text, _FUTURE_SCHEMA, cfg, use_grounding=True)
    if not isinstance(parsed, dict):
        logger.warning("⚠️ [factcheck] не удалось распарсить ответ темпорального чека")
        return []

    laws = parsed.get("future_laws") or []
    if not isinstance(laws, list):
        return []

    problems: List[Dict[str, Any]] = []
    for law in laws:
        if not isinstance(law, dict):
            continue
        verdict = str(law.get("verdict", "")).strip().lower()
        if verdict != "future":
            continue
        citation = str(law.get("citation", "")).strip()
        if not citation:
            continue
        problems.append({
            "citation": citation,
            "effective_date": str(law.get("effective_date", "") or "").strip(),
            "current_rule": str(law.get("current_rule", "") or "").strip(),
            "confidence": _to_float(law.get("confidence", 0)),
            "sources": [s for s in (law.get("sources") or []) if isinstance(s, dict)],
        })

    logger.info(f"   ✅ [factcheck] найдено норм, ещё не вступивших в силу: {len(problems)}")
    return problems


def hedge_future_laws(problems: List[Dict[str, Any]], today: str) -> List[Dict[str, str]]:
    """
    Переписать предложения с будущими законами в будущее время.

    Заменяет «статья действует в новой редакции» → «статья вступит в силу в новой редакции
    с <effective_date>; до этого применяется <current_rule>».
    Возвращает список {original, hedged} для точечного text.replace(). При сбое — [].
    """
    if not problems:
        return []

    cfg = _cfg()
    if not cfg["api_key"]:
        logger.warning("⚠️ [factcheck] KIE_API_KEY не задан — хеджирование будущих законов пропущено")
        return []

    hedge_system = (
        "Ты — редактор-юрист. Тебе дают предложения из статьи, где будущие изменения закона\n"
        f"(вступающие в силу ПОЗЖЕ {today}) ошибочно поданы как ДЕЙСТВУЮЩИЕ. Перепиши каждое\n"
        "в будущее время по правилам:\n"
        "1. Замени «действует»/«предусмотрен»/«применяется» на будущее время: «вступит в силу\n"
        "   с <effective_date>», «с <effective_date> будет действовать».\n"
        "2. ОБЯЗАТЕЛЬНО укажи дату вступления в силу.\n"
        "3. Если есть current_rule — добавь коротко: «до <даты> применяется <текущий порядок>».\n"
        "4. Сохрани смысл, длину (±25%), стиль, падежи. Не добавляй новые факты.\n"
        "5. Меняй только время/формулировку нормы, не трогай остальное.\n"
    )

    hedge_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "future_hedge",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "hedges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original": {"type": "string", "description": "Точное предложение из черновика"},
                                "hedged": {"type": "string", "description": "Формулировка в будущем времени"},
                            },
                            "required": ["original", "hedged"],
                        },
                    }
                },
                "required": ["hedges"],
            },
        },
    }

    user_text = f"Сегодня: {today}. Перепиши предложения с будущими законами в будущее время:\n\n"
    for i, p in enumerate(problems[:8], 1):
        ed = p.get("effective_date", "")
        cr = p.get("current_rule", "")
        note = f" (вступает с {ed}; текущий порядок: {cr})" if ed else ""
        user_text += f"{i}. Норма: {p.get('citation', '')}{note}\n"

    parsed = _call_kie_generic(hedge_system, user_text, hedge_schema, cfg, use_grounding=False)
    if not isinstance(parsed, dict):
        logger.warning("⚠️ [factcheck] не удалось распарсить ответ хеджирования будущих законов")
        return []

    hedges = parsed.get("hedges") or []
    if not isinstance(hedges, list):
        return []

    clean: List[Dict[str, str]] = []
    for h in hedges:
        if not isinstance(h, dict):
            continue
        orig = str(h.get("original", "")).strip()
        hedged = str(h.get("hedged", "")).strip()
        if not orig or not hedged or orig == hedged:
            continue
        clean.append({"original": orig, "hedged": hedged})

    logger.info(f"   ✅ [factcheck] подготовлено хеджей будущих законов: {len(clean)}")
    return clean
