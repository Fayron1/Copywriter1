"""
Freshness — автоматическая проверка актуальности фактов через kie.ai + Google Search Grounding.

Назначение:
    Перед написанием статьи (после Fact-Finder) проверяет собранные факты
    (ставки, лимиты, сроки, размеры пошлин/штрафов, номера статей, даты вступления
    в силу и т.п.) против актуальных данных из интернета и при необходимости
    заменяет устаревшие значения на свежие — ПОЛНОСТЬЮ АВТОМАТИЧЕСКИ, без ручного участия.

Бэкенд:
    kie.ai, модель gemini-3.1-pro (OpenAI-совместимый эндпоинт), с включённым
    инструментом googleSearch (Grounding with Google Search). См.
    https://kie.ai/gemini-3-1-pro

    Эндпоинт:  {FRESHNESS_API_BASE}/{FRESHNESS_MODEL}/v1/chat/completions
    Авторизация: Authorization: Bearer <KIE_API_KEY>

Безопасность (важно для налогов/права):
    Значение заменяется ТОЛЬКО при наличии авторитетного источника и высокой
    уверенности (>= FRESHNESS_CONFIDENCE_THRESHOLD). Иначе исходный факт остаётся
    без изменений. Все изменения (и применённые, и отклонённые) попадают в журнал
    (change-log) для последующего аудита.

Отказоустойчивость:
    Любая ошибка (нет ключа, сеть, таймаут, неверный JSON) НЕ ломает пайплайн —
    функция возвращает исходные факты и пустой журнал, ошибка пишется в лог.

Конфигурация (env):
    FRESHNESS_ENABLED                — "true"/"false" (по умолчанию true)
    KIE_API_KEY                      — ключ kie.ai (один на все модели)
    FRESHNESS_API_BASE               — база API (по умолчанию https://api.kie.ai)
    FRESHNESS_MODEL                  : модель (по умолчанию gemini-3.1-pro)
    FRESHNESS_REASONING_EFFORT       — "low"/"high" (по умолчанию low)
    FRESHNESS_CONFIDENCE_THRESHOLD   — порог уверенности 0..1 (по умолчанию 0.75)
    FRESHNESS_TIMEOUT                — таймаут запроса, сек (по умолчанию 120)
    FRESHNESS_MAX_RETRIES            — число повторов при сбое (по умолчанию 3)
"""
import os
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agents.freshness")


# ────────────────────────────────────────────────────────────
# Системный промпт фактчекера (порт логики Veritas, адаптирован под RU/право/налоги)
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Ты — строгий фактчекер для деловых статей по налогам, праву и финансам РФ.\n"
    "Тебе на вход дают JSON с фактами, собранными из базы знаний для будущей статьи.\n"
    "Твоя задача — с помощью поиска Google (он тебе доступен) проверить АКТУАЛЬНОСТЬ\n"
    "всех чувствительных ко времени фактов: налоговые ставки, лимиты и пороги, размеры\n"
    "пошлин/штрафов/взносов, сроки и даты, номера и редакции статей законов, МРОТ,\n"
    "ключевая ставка, лимиты УСН/патента и т.п.\n\n"
    "ПРАВИЛА:\n"
    "1. Проверяй каждый такой факт по актуальным источникам. Приоритет — официальные\n"
    "   и авторитетные ресурсы (nalog.gov.ru, gov.ru, publication.pravo.gov.ru,\n"
    "   КонсультантПлюс, Гарант, профильные госпорталы).\n"
    "2. Заменяй значение ТОЛЬКО если нашёл авторитетный источник И уверен, что данные\n"
    "   устарели. Указывай confidence (0..1) и source_url для каждого изменения.\n"
    "3. Если подтвердить не удалось или источник сомнительный — НИЧЕГО НЕ МЕНЯЙ,\n"
    "   оставь исходное значение (verdict='unverifiable').\n"
    "4. Не выдумывай источники и цифры. Лучше оставить как есть, чем подставить мусор.\n"
    "5. Текстовые/стилистические/неизмеряемые факты не трогай.\n\n"
    "ФОРМАТ ОТВЕТА (строго JSON по схеме):\n"
    "- verified_facts_json: строка с ПОЛНЫМ JSON фактов в ТОЙ ЖЕ структуре, что на входе,\n"
    "  где устаревшие значения заменены на актуальные (только при высокой уверенности и\n"
    "  наличии источника); все остальные поля — без изменений.\n"
    "- changes: массив всех проверок чувствительных фактов (и изменённых, и нет):\n"
    "  claim, old_value, new_value, verdict ('current'|'outdated'|'unverifiable'),\n"
    "  confidence (0..1), source_url."
)


_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "freshness_check",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "verified_facts_json": {
                    "type": "string",
                    "description": (
                        "Полный JSON фактов в исходной структуре, где устаревшие значения "
                        "заменены на актуальные (только при высокой уверенности и наличии "
                        "источника); всё остальное без изменений."
                    ),
                },
                "changes": {
                    "type": "array",
                    "description": "Журнал всех проверок чувствительных ко времени фактов.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "old_value": {"type": "string"},
                            "new_value": {"type": "string"},
                            "verdict": {
                                "type": "string",
                                "description": "current | outdated | unverifiable",
                            },
                            "confidence": {"type": "number"},
                            "source_url": {"type": "string"},
                        },
                        "required": ["claim", "old_value", "new_value", "verdict", "confidence", "source_url"],
                    },
                },
            },
            "required": ["verified_facts_json", "changes"],
            "title": "Freshness check result",
            "description": "Результат проверки актуальности фактов с журналом изменений.",
        },
    },
}


# ────────────────────────────────────────────────────────────
# Конфигурация из окружения
# ────────────────────────────────────────────────────────────
def is_enabled() -> bool:
    """Включена ли проверка актуальности (env FRESHNESS_ENABLED, по умолчанию true)."""
    return os.getenv("FRESHNESS_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def _config() -> Dict[str, Any]:
    base = os.getenv("FRESHNESS_API_BASE", "https://api.kie.ai").rstrip("/")
    model = os.getenv("FRESHNESS_MODEL", "gemini-3.1-pro").strip()
    return {
        "api_key": os.getenv("KIE_API_KEY"),
        "url": f"{base}/{model}/v1/chat/completions",
        "model": model,
        "reasoning_effort": os.getenv("FRESHNESS_REASONING_EFFORT", "low").strip().lower(),
        "threshold": float(os.getenv("FRESHNESS_CONFIDENCE_THRESHOLD", "0.75")),
        "timeout": float(os.getenv("FRESHNESS_TIMEOUT", "120")),
        "max_retries": int(os.getenv("FRESHNESS_MAX_RETRIES", "3")),
    }


# ────────────────────────────────────────────────────────────
# Вспомогательные парсеры
# ────────────────────────────────────────────────────────────
def _loads_lenient(raw: str) -> Optional[Any]:
    """Толерантный парсер JSON: снимает ```-обёртку и пытается выдрать объект."""
    if not raw or not isinstance(raw, str):
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def _call_kie(facts_text: str, cfg: Dict[str, Any]) -> Optional[str]:
    """
    Один вызов kie.ai gemini-3.1-pro с googleSearch + structured output.
    Возвращает строку message.content или None при сбое (с ретраями/бэкоффом).
    """
    import httpx

    payload = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": (
                    "Факты для статьи (проверь актуальность чувствительных ко времени значений "
                    "через поиск Google и верни результат строго по схеме):\n\n" + facts_text
                )}],
            },
        ],
        # Включаем Grounding with Google Search
        "tools": [{"type": "function", "function": {"name": "googleSearch"}}],
        "include_thoughts": False,
        "reasoning_effort": cfg["reasoning_effort"],
        "response_format": _RESPONSE_SCHEMA,
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    last_err: Optional[str] = None
    for attempt in range(1, cfg["max_retries"] + 1):
        try:
            with httpx.Client(timeout=cfg["timeout"]) as http:
                resp = http.post(cfg["url"], json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return (data.get("choices") or [{}])[0].get("message", {}).get("content")
            # 429/5xx — есть смысл повторить
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                logger.warning(f"⚠️ [freshness] невосстановимая ошибка API: {last_err}")
                return None
        except Exception as e:  # сеть/таймаут
            last_err = f"{type(e).__name__}: {e}"
        if attempt < cfg["max_retries"]:
            backoff = min(2 ** attempt, 10)
            logger.info(f"   ↻ [freshness] попытка {attempt} не удалась ({last_err}); повтор через {backoff}s")
            time.sleep(backoff)

    logger.warning(f"⚠️ [freshness] не удалось получить ответ за {cfg['max_retries']} попыток: {last_err}")
    return None


# ────────────────────────────────────────────────────────────
# Публичный API
# ────────────────────────────────────────────────────────────
def check_facts(facts: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Проверить актуальность фактов и вернуть (актуализированные_факты, журнал_изменений).

    При любой проблеме безопасно возвращает (исходные_факты, []) — пайплайн не ломается.
    """
    if not facts or not isinstance(facts, dict):
        return facts, []
    if not is_enabled():
        logger.info("   ⏭️ [freshness] проверка актуальности отключена (FRESHNESS_ENABLED=false)")
        return facts, []

    cfg = _config()
    if not cfg["api_key"]:
        logger.warning("⚠️ [freshness] KIE_API_KEY не задан — проверка актуальности пропущена")
        return facts, []

    try:
        facts_text = json.dumps(facts, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        logger.warning("⚠️ [freshness] факты несериализуемы — проверка пропущена")
        return facts, []
    if len(facts_text) > 40000:
        facts_text = facts_text[:40000] + "\n... [обрезано]"

    logger.info(f"🛡️ [freshness] проверка актуальности через {cfg['model']} + Google Search...")
    raw = _call_kie(facts_text, cfg)
    if not raw:
        return facts, []

    parsed = _loads_lenient(raw)
    if not isinstance(parsed, dict):
        logger.warning("⚠️ [freshness] не удалось распарсить ответ — оставляю исходные факты")
        return facts, []

    changes = parsed.get("changes") or []
    if not isinstance(changes, list):
        changes = []

    # Применённые изменения (для лога): прошли порог уверенности и имеют источник
    threshold = cfg["threshold"]
    applied = [
        c for c in changes
        if isinstance(c, dict)
        and str(c.get("verdict", "")).lower() == "outdated"
        and _to_float(c.get("confidence")) >= threshold
        and str(c.get("source_url", "")).startswith("http")
    ]

    # verified_facts_json — модель уже применила корректировки с учётом порога/источника
    verified = _loads_lenient(parsed.get("verified_facts_json", ""))
    if not isinstance(verified, dict) or not verified:
        # Структура сломана/пуста — безопасно оставляем исходные факты
        if applied:
            logger.warning("⚠️ [freshness] verified_facts некорректны — применяю исходные факты, но журнал сохранён")
        return facts, changes

    logger.info(
        f"   ✅ [freshness] проверено: всего записей в журнале {len(changes)}, "
        f"применено правок {len(applied)} (порог {threshold})"
    )
    for c in applied:
        logger.info(f"      • {c.get('claim','')}: '{c.get('old_value','')}' → '{c.get('new_value','')}' "
                    f"(conf={c.get('confidence')}, {c.get('source_url','')})")

    return verified, changes


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ────────────────────────────────────────────────────────────
# Локальный самотест (без пайплайна): python freshness.py
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    demo = {
        "facts": [
            {"topic": "Лимит дохода по УСН", "value": "200 млн руб."},
            {"topic": "Ставка НДС (общая)", "value": "20%"},
            {"topic": "МРОТ", "value": "19 242 руб."},
        ]
    }
    print("Исходные факты:")
    print(json.dumps(demo, ensure_ascii=False, indent=2))
    updated, log = check_facts(demo)
    print("\nАктуализированные факты:")
    print(json.dumps(updated, ensure_ascii=False, indent=2))
    print("\nЖурнал изменений:")
    print(json.dumps(log, ensure_ascii=False, indent=2))
