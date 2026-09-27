"""
Паспорт параметров статьи (NormParams): контроль «год x норма».

Проблема (кейс «НДС 5% при УСН», 2026-09-27): статья Beautiful-структуры
взяла параметры 2025 года (60 млн / 20%) из эталонных статей и веб-страниц,
хотя действующие на 2026 — 20 млн / 22% / 272,5 / 490,5.

Решение: ДО письма Heart извлекаем действующие на год темы ключевые
параметры (порог, ставки, лимиты, даты, порядок перехода) через Vane
(Яндекс-метапоиск, официальные источники) + DeepSeek-структуризацию.
Для каждого параметра просим также УСТАРЕВШЕЕ значение прошлых лет —
оно становится чёрным списком: если устаревшая цифра всплывает в черновике,
это критический конфликт версий (publish-gate).

Параметры инжектятся в промпт Heart как ЗАФИКСИРОВАННЫЕ (отклонение = брак)
и сверяются детерминированно в _validate_final.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger("agents.norm_params")

EXTRACT_SYSTEM = (
    "Ты — налоговый юрист-аналитик. Из брифинга извлеки КЛЮЧЕВЫЕ ЧИСЛОВЫЕ ПАРАМЕТРЫ темы, "
    "действующие НА УКАЗАННЫЙ ГОД, по официальным источникам (ФНС, НК РФ, Минфин). "
    "ВАЖНО: различай действующие и устаревшие значения. Если параметр менялся — "
    "укажи оба: value (действует на год статьи) и old_value (прошлые годы). "
    "ПОКРЫТИЕ: включи параметры ВСЕХ сценариев статьи, включая последствия ПОСЛЕ события — "
    "например, для темы НПД нужны не только лимит и ставки НПД, но и ставки/пороги "
    "альтернативных режимов (УСН, ОСНО: НДС, НДФЛ) и фиксированные взносы ИП ТОЧНЫМИ суммами. "
    "ЗАПРЕЩЕНО «около»: значение либо точное, либо пустое. "
    "Также построй СЦЕНАРНУЮ МАТРИЦУ: отдельные строки для каждого сочетания "
    "статус-до-события × событие × исход (успех/пропуск срока). "
    "Верни строго JSON: {\"params\": [{\"name\":..., \"value\":..., \"old_value\":..., "
    "\"source\":..., \"url\":...}], \"scenarios\": [{\"status_before\": \"ИП на НПД\", "
    "\"event\": \"превышение лимита 2,4 млн\", \"effect_date\": \"с даты превышения\", "
    "\"action\": \"уведомление о переходе на УСН\", \"deadline\": \"20 календарных дней\", "
    "\"form\": \"по актуальной форме ФНС (проверить в ЛК)\", \"regime_after\": \"ИП на УСН\", "
    "\"source\": \"422-ФЗ + разъяснение ФНС\"}]}. Максимум 8 параметров и 5 сценариев, "
    "только критичные для темы. Отделяй физлицо без ИП от ИП — это РАЗНЫЕ строки."
)


def get_norm_params(topic: str, year: int) -> List[Dict[str, Any]]:
    """Извлечь действующие на {year} ключевые параметры + сценарную матрицу.

    Возвращает [{"params": [...], "scenarios": [...]}] или [] при сбое.
    (Совместимость: возвращает список dict'ов с ключами params/scenarios.)
    """
    try:
        from .vane import research as vane_research
        vr = vane_research(
            f"Официальные разъяснения ФНС и НК РФ, ДЕЙСТВУЮЩИЕ на {year} год: {topic}. "
            f"Только актуальные на {year} год значения: пороги, ставки, лимиты, даты вступления, "
            f"порядки перехода, ТОЧНЫЕ суммы взносов. Значения прошлых лет помечай как устаревшие. "
            f"Разделяй сценарии для физлица без ИП и для ИП. "
            f"Приоритет: nalog.gov.ru, garant.ru, consultant.ru.")
        briefing = (vr.get("message") or "").strip()
        if not briefing:
            return []
        srcs = vr.get("sources", [])[:6]
        src_line = "\n".join(f"- {s.get('url', '')}" for s in srcs if s.get("url"))
        return _extract_params(briefing, src_line, year)
    except Exception as e:
        logger.warning(f"norm_params: Vane сбой: {e}")
        return []


def _extract_params(briefing: str, src_line: str, year: int) -> List[Dict[str, Any]]:
    """LLM-структуризация брифинга в параметры + сценарии (DeepSeek flash)."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            timeout=120.0,
        )
        user = (
            f"ГОД СТАТЬИ: {year} (сегодняшние правила = правила {year} года).\n\n"
            f"БРИФИНГ:\n{briefing[:7000]}\n\n"
            f"ИСТОЧНИКИ:\n{src_line[:800]}\n\n"
            f"Извлеки параметры и сценарную матрицу, действующие на {year} год."
        )
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_DEEPSEEK_FLASH", "deepseek-v4-flash"),
            messages=[{"role": "system", "content": EXTRACT_SYSTEM},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=120.0,
        )
        data = json.loads(resp.choices[0].message.content)

        clean_params = []
        for p in (data.get("params") or [])[:8]:
            if isinstance(p, dict) and p.get("name") and p.get("value"):
                clean_params.append({
                    "name": str(p["name"])[:120],
                    "value": str(p["value"])[:140],
                    "old_value": str(p.get("old_value", "") or "")[:140],
                    "source": str(p.get("source", ""))[:160],
                    "url": str(p.get("url", ""))[:300],
                })
        clean_scen = []
        for s in (data.get("scenarios") or [])[:5]:
            if isinstance(s, dict) and s.get("status_before") and s.get("event"):
                clean_scen.append({
                    k: str(s.get(k, "") or "")[:140]
                    for k in ("status_before", "event", "effect_date", "action",
                              "deadline", "form", "regime_after", "source")
                })
        logger.info(f"norm_params: {len(clean_params)} параметров, "
                    f"{len(clean_scen)} сценариев на {year} год")
        return [{"params": clean_params, "scenarios": clean_scen}]
    except Exception as e:
        logger.warning(f"norm_params: извлечение сбой: {e}")
        return []


def _unwrap(params_field) -> tuple:
    """Совместимость: state.norm_params может быть старым списком параметров
    или новым [{'params': [...], 'scenarios': [...]}]. Возвращает (params, scenarios)."""
    if not params_field:
        return [], []
    if isinstance(params_field, dict):
        return params_field.get("params", []), params_field.get("scenarios", [])
    if isinstance(params_field, list) and params_field and isinstance(params_field[0], dict) \
            and ("params" in params_field[0] or "scenarios" in params_field[0]):
        return params_field[0].get("params", []), params_field[0].get("scenarios", [])
    return list(params_field), []


def format_params_block(params_field, year: int) -> str:
    """Блок для промпта Heart: ЗАФИКСИРОВАННЫЕ ПАРАМЕТРЫ + СЦЕНАРНАЯ МАТРИЦА."""
    params, scenarios = _unwrap(params_field)
    if not params and not scenarios:
        return ""
    lines = [f"=== ЗАФИКСИРОВАННЫЕ ПАРАМЕТРЫ НА {year} ГОД (проверены по официальным источникам) ===",
             "ЭТИ И ТОЛЬКО ЭТИ значения используй в статье. Отклонение = критический брак.", ""]
    for i, p in enumerate(params, 1):
        lines.append(f"{i}. {p.get('name','')}: {p.get('value','')} "
                     f"(источник: {p.get('source') or 'брифинг'})")
        if p.get("old_value"):
            lines.append(f"   ❌ УСТАРЕЛО (запрещено в статье): {p['old_value']}")
    if scenarios:
        lines += ["", "=== СЦЕНАРНАЯ МАТРИЦА (строки НЕЛЬЗЯ смешивать) ===",
                  "Каждая строка — отдельная ситуация. Смешение статусов (физлицо vs ИП), "
                  "режимов (НПД/УСН/ОСНО) или дат последствий из разных строк = критический брак.",
                  "Пиши каждый сценарий своим блоком; даты и сроки бери ТОЛЬКО из строки.", ""]
        for j, s in enumerate(scenarios, 1):
            lines.append(f"Строка {j}: {s.get('status_before','?')} — {s.get('event','?')}")
            lines.append(f"   Дата последствий: {s.get('effect_date','?')}")
            lines.append(f"   Действие: {s.get('action','—')} | Срок: {s.get('deadline','—')} | "
                         f"Форма: {s.get('form','—')}")
            lines.append(f"   Режим после: {s.get('regime_after','?')} | Источник: {s.get('source','—')}")
        lines.append("=== КОНЕЦ МАТРИЦЫ ===")
    lines.append("=== КОНЕЦ ПАРАМЕТРОВ ===")
    return "\n".join(lines)


_UNIT_RE = re.compile(
    r"(\d[\d \t]*[.,]?\d*)\s*(млн|млрд|тыс\.?|%|руб\.?|₽|кв\.?\s?м|дн\.?|дней|дня|мес(?:яца|яцев)?)?",
    re.IGNORECASE)


def _num_unit_pairs(s: str) -> List[tuple]:
    """Пары (число, единица) из строки. Годы (19xx/20xx) исключаются — это даты,
    а не значения параметров (кейс: «(до 2026)» ловилось как устаревшее «2026»)."""
    pairs = []
    if not s:
        return pairs
    for m in _UNIT_RE.finditer(s):
        num = m.group(1).strip().replace(" ", "").replace("\t", "")
        unit = (m.group(2) or "").strip().rstrip(".").lower()
        if not num or not re.search(r"\d", num):
            continue
        if re.fullmatch(r"(19|20)\d{2}", num):
            continue
        variants = {num}
        if "." in num:
            variants.add(num.replace(".", ","))
        if "," in num:
            variants.add(num.replace(",", "."))
        for v in variants:
            pairs.append((v, unit))
    return pairs


def _in_text(num: str, unit: str, text: str) -> bool:
    """Число в тексте с учётом единицы: «20%» не матчит «20 млн»."""
    if unit:
        compact = f"{num}{unit}"
        spaced = f"{num} {unit}"
        spaced2 = f"{num}\u00a0{unit}"
        if compact in text or spaced in text or spaced2 in text:
            return True
        # единица может стоять через слово («20 % от», «60 млн руб.») —
        # для компактных единиц достаточно соседства в 6 символах
        for m in re.finditer(rf"(?<![\d.,]){re.escape(num)}(?![\d.,])", text):
            tail = text[m.end():m.end() + 8].lower()
            if tail.lstrip(" ").startswith(unit):
                return True
        return False
    return re.search(rf"(?<![\d.,]){re.escape(num)}(?![\d.,])", text) is not None


def check_draft(draft: str, params_field) -> List[str]:
    """Детерминированная сверка: устаревшие значения + сущностный линтер."""
    issues = []
    if not draft:
        return issues
    text = draft.replace("\u00a0", " ")
    params, _scenarios = _unwrap(params_field)
    for p in params:
        old_pairs = _num_unit_pairs(p.get("old_value", ""))
        cur_nums = {n for n, _ in _num_unit_pairs(p.get("value", ""))}
        for num, unit in old_pairs:
            if num in cur_nums:
                continue  # совпадает с действующим — не конфликт
            if _in_text(num, unit, text):
                issues.append(
                    f"🔴 КОНФЛИКТ ВЕРСИЙ: параметр «{p.get('name', '?')}» — в тексте устаревшее "
                    f"значение «{num}{unit}» (действующее: {p.get('value', '?')})")
    missing = []
    for p in params:
        cur = _num_unit_pairs(p.get("value", ""))
        if cur and not any(_in_text(n, u, text) for n, u in cur):
            missing.append(f"🟡 Параметр «{p.get('name','?')}» = {p.get('value','?')} "
                           f"не найден в тексте — проверь")
    issues.extend(missing[:3])
    issues.extend(lint_legal(text))
    return issues


# ============================================================
# Детерминированный юридический линтер (без LLM)
# ============================================================

_BROKEN_CITATION_PATTERNS = [
    (re.compile(r"\)\.\s*[0-9]+\s*(?:п\.|ч\.|ст\.)"), "оборванная ссылка: «). N ст.»"),
    (re.compile(r"(?:ст|п|ч|подп)\.\s*\d+[^\s)]{0,4}\.\s+\d+\s+(?:ст|п|ч)\."), "два номера норм подряд"),
    (re.compile(r"\((?:ч|п|подп)\.\s*[^)]*$", re.MULTILINE), "незакрытая скобка ссылки"),
]
_ABSOLUTES = ["обязательно", "автоматически", "всегда", "никогда", "гарантированно",
              "вне закона", "доначислят", "переквалифицирует", "штраф будет",
              "в каждом случае", "заказчик откажется"]
_PRACTICE_CLAIMS = ["в одном из кейсов", "в нашей практике был случай", "реальный кейс компании"]
_FORM_RE = re.compile(r"по\s+форме\s*[№N]?\s*[\w.-]+", re.IGNORECASE)


def lint_legal(text: str) -> List[str]:
    """Оборванные ссылки, абсолюты без источника, псевдо-практика, формы без оговорки."""
    issues = []
    for pat, desc in _BROKEN_CITATION_PATTERNS:
        for m in list(pat.finditer(text))[:3]:
            frag = text[max(0, m.start() - 20):m.end() + 25].replace("\n", " ")
            issues.append(f"🔴 БИТАЯ ССЫЛКА ({desc}): …{frag}…")
    low = text.lower()
    hits = [w for w in _ABSOLUTES if w in low]
    if hits:
        issues.append(f"🟡 АБСОЛЮТЫ без источника ({len(hits)} шт.): "
                      f"{', '.join(sorted(set(hits))[:6])} — замените на условную форму")
    for phrase in _PRACTICE_CLAIMS:
        idx = low.find(phrase)
        if idx >= 0:
            window = low[max(0, idx - 60):idx + 120]
            if "условный пример" not in window and "условн" not in window:
                issues.append(f"🟡 ПСЕВДО-ПРАКТИКА: «{phrase}» без маркировки "
                              f"«Условный пример» или источника — переформулируйте")
    for m in list(_FORM_RE.finditer(text))[:3]:
        window = text[max(0, m.start() - 80):m.end() + 80].lower()
        if "актуальн" not in window and "проверьт" not in window:
            issues.append(f"🟡 НОМЕР ФОРМЫ без оговорки актуальности: «{m.group(0)}» — "
                          f"добавь «по актуальной форме ФНС (проверьте в личном кабинете)»")
    return issues
