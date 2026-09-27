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
            f"порядки перехода, ТОЧНЫЕ суммы взносов. ОБЯЗАТЕЛЬНО включи ставки и пороги ВСЕХ "
            f"режимов, в которые читатель может попасть из темы статьи (ОСНО: ставка НДС, шкала "
            f"НДФЛ; УСН: лимиты; взносы ИП — фиксированная часть и срок уплаты 1% с превышения). "
            f"Для каждого параметра, менявшегося с прошлого года, укажи старое значение "
            f"(например НДС: было 20%, с 2026 — 22%). "
            f"Значения прошлых лет помечай как устаревшие. "
            f"Разделяй сценарии для физлица без ИП и для ИП. "
            f"ЗАКОНОПРОЕКТЫ и непринятые поправки ИСКЛЮЧАЙ полностью — только действующие нормы."
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


# ============================================================
# Правила для будущих лет: проектные значения НЕ подаются как факт
# ============================================================

# Ожидаемые изменения на 2027 (из разъяснений ФНС 2026) — для валидации паспортов,
# построенных на 2027-м. Если паспорт на 2027 содержит значение 2026 года — 🔴.
KNOWN_FUTURE_RULES = {
    2027: {
        "usn_vat_exemption_threshold": {
            "current_2026": "20 000 000",
            "expected_2027": "15 000 000",
            "note": "порог освобождения от НДС на УСН снижается поэтапно: 2027 — 15 млн, 2028 — 10 млн",
        },
    },
}

_QUARTER_VS_MONTH = [
    (re.compile(r"утрат\w+[^.]{0,60}(?:с\s+)?начал[ао]\s+квартала", re.I),
     "«с начала квартала» — неверно: утрата права на УСН наступает с 1-го числа месяца "
     "превышения (ст. 346.13 НК РФ)"),
    (re.compile(r"переход\w+\s+на\s+ОСНО[^.]{0,60}(?:с\s+)?начал[ао]\s+квартала", re.I),
     "«переход на ОСНО с начала квартала» — неверно: с 1-го числа месяца превышения"),
]


def lint_year_rules(text: str, year: int, params_field) -> List[str]:
    """Проверка годовой согласованности паспорта + квартал-vs-месяц + проектные дефляторы."""
    issues = []
    params, _ = _unwrap(params_field)
    if year in KNOWN_FUTURE_RULES:
        for rule_key, rule in KNOWN_FUTURE_RULES[year].items():
            # Число из «20 000 000» — первое число строки (пробелы не блокируют)
            cur_val_num = re.search(r"(\d[\d\s]*\d|\d)", rule["current_2026"])
            if cur_val_num and cur_val_num.group(1).replace(" ", "") in text.replace(" ", ""):
                all_param_values = " ".join(p.get("value", "") for p in params)
                if rule["expected_2027"][:3].replace(" ", "") not in all_param_values.replace(" ", ""):
                    issues.append(
                        f"🔴 ПАСПОРТ {year}: параметр «{rule_key}» — в паспорте нет значения "
                        f"{rule['expected_2027']} (действует {rule['current_2026']} только в 2026). "
                        f"Для {year} года — {rule['note']}.")
    # Квартал vs месяц
    for pat, msg in _QUARTER_VS_MONTH:
        if pat.search(text):
            issues.append(f"🔴 {msg}")
    # Проектный дефлятор как факт
    if re.search(r"дефлятор\w*[^.]{0,40}1,15", text, re.I):
        issues.append(
            "🟡 ДЕФЛЯТОР 1,15 — проектное значение, не утверждён приказом Минэкономразвития. "
            "Маркируй как «прогноз» или «стресс-сценарий», не как действующий лимит.")
    return issues


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
        # Обязательный минимум: основная ставка НДС и базовая ставка НДФЛ должны
        # быть ВСЕГДА (кейс: экстракция их теряла -> «20%» безнаказанно жил в тексте).
        have_vat = any("ндс" in p["name"].lower() for p in clean_params)
        have_ndfl = any("ндфл" in p["name"].lower() for p in clean_params)
        if not have_vat or not have_ndfl:
            fix = client.chat.completions.create(
                model=os.getenv("MODEL_DEEPSEEK_FLASH", "deepseek-v4-flash"),
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                    {"role": "user", "content":
                        f"В списке параметров {'НЕТ ставки НДС' if not have_vat else ''}"
                        f"{' и НЕТ ставки НДФЛ' if not have_ndfl else ''}. "
                        f"ДОБАВЬ недостающие параметры из брифинга (основная ставка НДС на {year} "
                        f"с old_value прошлых лет, например 20%→22%; базовые ставки НДФЛ). "
                        f"Верни полный JSON заново."},
                ],
                response_format={"type": "json_object"}, temperature=0.0, timeout=90.0)
            data = json.loads(fix.choices[0].message.content)
            clean_params = []
            for p in (data.get("params") or [])[:10]:
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


def enforce_params(text: str, params_field) -> tuple:
    """
    Принудительная петля: детерминированная замена устаревших чисел на действующие.
    Возвращает (исправленный_текст, список_замен). Затрагивает ТОЛЬКО числовые
    значения из чёрного списка (old_value), которых нет среди действующих.
    """
    if not text:
        return text, []
    params, _ = _unwrap(params_field)
    fixes = []
    for p in params:
        cur_nums = {n for n, _ in _num_unit_pairs(p.get("value", ""))}
        for num, unit in _num_unit_pairs(p.get("old_value", "")):
            if num in cur_nums:
                continue
            cur_val = p.get("value", "")
            cur_first = cur_val.split()[0] if cur_val else ""
            for needle in (f"{num}{unit}", f"{num} {unit}", f"{num}\u00a0{unit}"):
                if needle in text:
                    text = text.replace(needle, cur_first)
                    fixes.append(f"{needle} → {cur_first} ({p.get('name','?')})")
    # Детерминированное удаление псевдо-экспертности (Heart не убирает из промпта)
    _PSEUDO = [
        ("на моей практике как", "при аудите подобных систем"),
        ("на моей практике", "в практике отрасли"),
        ("в моей практике", "в практике отрасли"),
        ("я сопровождал", "специалисты сопровождали"),
        ("мы защищали", "специалисты защищали"),
    ]
    for old, new in _PSEUDO:
        if old in text.lower():
            import re as _re
            text = _re.sub(_re.escape(old), new, text, flags=_re.IGNORECASE)
            fixes.append(f"псевдо-эксперт: «{old}» → «{new}»")
    return text, fixes


def check_matrix_compliance(draft: str, params_field) -> List[str]:
    """LLM-сверка качественных правил матрицы (даты, сроки, порядок) с текстом.
    Детерминированный чек ловит только числа; «с месяца» vs «с даты превышения»
    ловится здесь. Один дешёвый вызов DeepSeek."""
    _, scenarios = _unwrap(params_field)
    if not scenarios or not draft:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            timeout=90.0,
        )
        scen_text = "\n".join(
            f"{i}. {s.get('status_before','?')} | событие: {s.get('event','?')} | "
            f"дата последствий: {s.get('effect_date','?')} | действие: {s.get('action','?')} | "
            f"срок: {s.get('deadline','?')} | режим после: {s.get('regime_after','?')}"
            for i, s in enumerate(scenarios, 1))
        system = (
            "Ты юридический валидатор. Сверь текст статьи со сценарной матрицей. "
            "Найди ТОЛЬКО противоречия качественных правил: неверные даты последствий "
            "(например «с месяца» вместо «с даты превышения»), неверные сроки, неверный порядок, "
            "смешение статусов/режимов из разных строк. НЕ оценивай стиль. "
            "Верни JSON: {\"issues\": [{\"quote\": \"фраза из текста\", "
            "\"matrix_rule\": \"правило из матрицы\", \"severity\": \"critical|warning\"}]}. "
            "Пустой issues, если противоречий нет."
        )
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_DEEPSEEK_FLASH", "deepseek-v4-flash"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": f"МАТРИЦА:\n{scen_text}\n\nТЕКСТ:\n{draft[:12000]}"}],
            response_format={"type": "json_object"},
            temperature=0.0, timeout=90.0)
        data = json.loads(resp.choices[0].message.content)
        issues = []
        for it in (data.get("issues") or [])[:6]:
            if not isinstance(it, dict):
                continue
            mark = "🔴" if str(it.get("severity")) == "critical" else "🟡"
            issues.append(f"{mark} ПРОТИВОРЕЧИЕ МАТРИЦЕ: «{(it.get('quote') or '')[:80]}» — "
                          f"по матрице: {(it.get('matrix_rule') or '')[:100]}")
        return issues
    except Exception as e:
        logger.warning(f"matrix compliance сбой: {e}")
        return []


def fix_matrix_violations(draft: str, issues: List[str], params_field) -> str:
    """Автоправка качественных противоречий матрице: один DeepSeek-вызов,
    «исправь ТОЛЬКО отмеченные места». Возвращает исправленный текст или исходный."""
    _, scenarios = _unwrap(params_field)
    if not draft or not issues or not scenarios:
        return draft
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            timeout=150.0)
        scen_text = "\n".join(
            f"{i}. {s.get('status_before','?')} | {s.get('event','?')} | "
            f"дата: {s.get('effect_date','?')} | срок: {s.get('deadline','—')} | "
            f"режим после: {s.get('regime_after','?')}"
            for i, s in enumerate(scenarios, 1))
        system = (
            "Ты юридический корректор. Тебе дан текст статьи и список ПРОТИВОРЕЧИЙ "
            "сценарной матрице. Исправь ТОЛЬКО отмеченные места, приведя их в соответствие "
            "с матрицей. НЕ переписывай остальное, сохраняй объём (±10%), стиль, заголовки "
            "и разметку. Верни полный исправленный текст статьи без комментариев."
        )
        user = (
            f"МАТРИЦА:\n{scen_text}\n\nПРОТИВОРЕЧИЯ ДЛЯ ИСПРАВЛЕНИЯ:\n"
            + "\n".join(f"- {i}" for i in issues)
            + f"\n\nТЕКСТ:\n{draft[:14000]}"
        )
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_DEEPSEEK_PRO", "deepseek-v4-pro"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1, timeout=150.0)
        fixed = (resp.choices[0].message.content or "").strip()
        # Защита от переписывания «всего»: объём должен остаться близким
        if fixed and 0.8 <= len(fixed) / max(1, len(draft)) <= 1.25:
            return fixed
        logger.warning("fix_matrix_violations: объём изменился слишком сильно — откат")
        return draft
    except Exception as e:
        logger.warning(f"fix_matrix_violations сбой: {e}")
        return draft


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


def fix_markdown_tables(text: str) -> tuple:
    """Исправить разделители Markdown-таблиц: Heart пишет | — | вместо | --- |.
    Возвращает (исправленный_текст, количество_замен)."""
    if not text:
        return text, 0
    fixes = 0
    lines = text.split('\n')
    for i in range(len(lines)):
        line = lines[i].strip()
        # Строка-разделитель: | — | — | или | - | - |  или смешанная
        if line.startswith('|') and line.endswith('|') and i > 0:
            prev = lines[i-1].strip()
            # Предыдущая строка — заголовок таблицы (тоже pipe)
            if prev.startswith('|') and prev.endswith('|'):
                # Проверяем, является ли текущая строка разделителем из тире
                cells = [c.strip() for c in line.strip('|').split('|')]
                if all(c in ('—', '–', '-', '') for c in cells) and len(cells) >= 2:
                    # Заменяем на правильный Markdown-разделитель
                    sep = '|' + '|'.join([' --- '] * len(cells)) + '|'
                    if lines[i] != sep:
                        lines[i] = sep
                        fixes += 1
    if fixes:
        text = '\n'.join(lines)
    return text, fixes


def lint_form_deadline(text: str, params_field=None) -> List[str]:
    """Проверка формы и срока перехода после утраты НПД (правила редактора 2026-09-27):
    - форма 26.2-1 не должна быть единственной в сценарии утраты НПД
    - срок «20-е число месяца» — неверный, должно быть «20 календарных дней»
    - «патент не уменьшается на взносы» — критическая ошибка
    """
    issues = []
    if not text:
        return issues
    low = text.lower()

    # 1. Форма 26.2-1 в контексте утраты НПД
    if '26.2-1' in low and ('ндп' in low or 'самозанят' in low):
        issues.append(
            "🔴 ФОРМА: «26.2-1» в контексте утраты НПД — для этого сценария ФНС "
            "рекомендует форму КНД 1150094 (уведомление о переходе на УСН в связи "
            "с утратой права на НПД). Проверьте и уточните.")

    # 2. Срок «20-е число месяца» вместо «20 календарных дней»
    if re.search(r'20[- ]?\w*\s*числ\w*\s*месяц', low):
        issues.append(
            "🔴 СРОК: «20-е число месяца» — неверная формула. Правильно: "
            "«20 календарных дней с даты снятия с учёта НПД».")

    # 3. Патент + взносы
    if re.search(r'патент\w*\s*[^.]*не\s*уменьш', low):
        issues.append(
            "🔴 ПСН: «патент не уменьшается на взносы» — КРИТИЧЕСКАЯ ошибка. "
            "Налог при ПСН уменьшается на страховые взносы: без работников — "
            "до 100%, с работниками — до 50%.")

    return issues


_STAT_WITHOUT_SRC = re.compile(
    r"(?:около|примерно|порядка|свыше|более)?\s*(\d{2,3})\s*%\s*(?:всех|ограничений|случаев|компаний|бизнес|организаций|предпринимател)", re.I)
_INCOMPLETE_CITATION = re.compile(
    r"(?:Приказ|Письмо|Постановление)\s*[№N]?\s*\d+[^\d]{0,5}(?:от\s*[\d.]+)?(?!\s*(?:\.|—|–)\s*(?:ФНС|Минфин|Росфинмониторинг|Правительства|Банка\s*России))", re.I)

_PLACEHOLDER_PATTERNS = [
    (re.compile(r"стать[уяе]\s+от\s+(?:(?:КоАП|НК|ГК|ТК)|$|\.)", re.I), "заглушка «статья от»"),
    (re.compile(r"ст\.\s*от\s+", re.I), "заглушка «ст. от»"),
    (re.compile(r"по\s+статье\s+от\s", re.I), "заглушка «по статье от»"),
    (re.compile(r"стать[уяе]\s+от\s*[^0-9]", re.I), "заглушка: после «от» нет номера"),
]
_PSEUDO_EXPERT = [
    "на моей практике", "в моей практике", "я сопровождал", "мы защищали",
    "я не раз видел", "моим клиентам",
]
_UNVERIFIED_COST = re.compile(
    r"(?:обходится|стоит|стоит порядка|составляет|потребует|выйдет)\s*(?:в|от|порядка)?\s*[\d\s]{3,}\s*(?:тыс|млн|руб|₽| тысяч| миллионов| рублей)", re.I)


def detect_truncation(text: str) -> List[str]:
    """Детектор обрыва статьи: последний содержательный раздел кончается посреди
    предложения или без завершения мысли. Кейс: «Шаг пятый — если решение делегировано
    аутсорс» (2026-09-27) — hard-cut обрезал текст, потом приклеился блок Источники."""
    issues = []
    if not text or len(text) < 200:
        return issues
    # Найти последний содержательный блок перед Источниками/Примечаниями
    body = text
    for tail_marker in ["\n## Источники", "\n## Примечания", "\n## Ссылки", "\n---"]:
        idx = body.rfind(tail_marker)
        if idx > len(body) * 0.5:
            body = body[:idx]
            break
    body = body.rstrip()
    if not body:
        return issues
    # Последние 100 символов тела
    tail = body[-100:]
    # Признаки обрыва:
    # 1. Нет завершающего знака препинания
    last_char = body[-1] if body else ''
    if last_char not in '.!?…»"\'\n)':
        issues.append(f"🔴 ОБРЫВ ТЕКСТА: статья кончается на «…{tail[-60:]}» — "
                      f"нет завершения мысли (последний символ: «{last_char}»)")
    # 2. Слова-маркеры незавершённости в последних 80 символах
    truncation_words = ['аутсорс', 'далее', 'продолжение', 'шаг', 'во-первых']
    for w in truncation_words:
        if w in tail.lower() and not tail.rstrip().endswith(('.', '!', '?', '…')):
            issues.append(f"🔴 ОБРЫВ НА «{w}»: текст обрезан в незавершённом блоке")
            break
    # 3. Последний H2-раздел слишком короткий (< 50 символов тела)
    import re as _re
    sections = _re.split(r'^##\s', body, flags=_re.MULTILINE)
    if len(sections) > 1:
        last_sec = sections[-1]
        # Убираем заголовок
        lines = last_sec.split('\n', 1)
        sec_body = lines[1].strip() if len(lines) > 1 else ''
        if sec_body and len(sec_body) < 50:
            issues.append(f"🔴 ПОСЛЕДНИЙ РАЗДЕЛ ПУСТ: «{lines[0][:50]}» — "
                          f"только {len(sec_body)} символов тела")
    return issues


def lint_legal(text: str) -> List[str]:
    """Оборванные ссылки, абсолюты без источника, псевдо-практика, формы без оговорки,
    заглушки статей, псевдо-экспертность, неподтверждённые цены."""
    issues = []
    for pat, desc in _BROKEN_CITATION_PATTERNS:
        for m in list(pat.finditer(text))[:3]:
            frag = text[max(0, m.start() - 20):m.end() + 25].replace("\n", " ")
            issues.append(f"🔴 БИТАЯ ССЫЛКА ({desc}): …{frag}…")
    # Заглушки статей (кейс: «статья от КоАП РФ» без номера — 2026-09-27)
    for pat, desc in _PLACEHOLDER_PATTERNS:
        for m in list(pat.finditer(text))[:3]:
            issues.append(f"🔴 ЗАГЛУШКА СТАТЬИ ({desc}): «{m.group(0)}» — номер обязателен")
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
    # Псевдо-экспертность (кейс: «на моей практике как IT-юриста» — 2026-09-27)
    for phrase in _PSEUDO_EXPERT:
        if phrase in low:
            issues.append(f"🔴 ПСЕВДО-ЭКСПЕРТ: «{phrase}» — ИИ не может выдумывать "
                          f"личный профессиональный опыт. Замените на нейтральную формулировку.")
    # Неподтверждённые цены миграции/работ
    for m in list(_UNVERIFIED_COST.finditer(text))[:2]:
        frag = text[max(0, m.start() - 30):m.end() + 20].replace("\n", " ")
        issues.append(f"🟡 ЦЕНА БЕЗ ИСТОЧНИКА: «{frag}» — рыночная оценка без источника; "
                      f"уберите цифру или маркируйте как условную")
    # Неподтверждённая статистика (кейс: «80% всех ограничений» — 2026-09-27)
    for m in list(_STAT_WITHOUT_SRC.finditer(text))[:2]:
        frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
        issues.append(f"🔴 СТАТИСТИКА БЕЗ ИСТОЧНИКА: «{frag}» — процент без названия "
                      f"исследования/ведомства. Уберите цифру или дайте источник.")
    # Неполные ссылки на нормативные акты (кейс: «Приказ №23» без ведомства — 2026-09-27)
    for m in list(_INCOMPLETE_CITATION.finditer(text))[:2]:
        frag = text[max(0, m.start() - 10):m.end() + 30].replace("\n", " ")
        issues.append(f"🟡 НЕПОЛНАЯ ССЫЛКА: «{frag}» — укажите ведомство, дату и название акта")
    for m in list(_FORM_RE.finditer(text))[:3]:
        window = text[max(0, m.start() - 80):m.end() + 80].lower()
        if "актуальн" not in window and "проверьт" not in window:
            issues.append(f"🟡 НОМЕР ФОРМЫ без оговорки актуальности: «{m.group(0)}» — "
                          f"добавь «по актуальной форме ФНС (проверьте в личном кабинете)»")
    return issues
