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
    (re.compile(r"утрат\w*[^.]{0,60}(?:с\s+)?начал[ао]\s+квартала", re.I),
     "«с начала квартала» — неверно: утрата права на УСН наступает с 1-го числа месяца "
     "превышения (ст. 346.13 НК РФ)"),
    (re.compile(r"переход\w*\s+на\s+ОСНО[^.]{0,60}(?:с\s+)?начал[ао]\s+квартала", re.I),
     "«переход на ОСНО с начала квартала» — неверно: с 1-го числа месяца превышения"),
]


def lint_year_rules(text: str, year: int, params_field) -> List[str]:
    """Проверка годовой согласованности паспорта + квартал-vs-месяц + проектные дефляторы."""
    issues = []
    params, _ = _unwrap(params_field)
    if year in KNOWN_FUTURE_RULES:
        for rule_key, rule in KNOWN_FUTURE_RULES[year].items():
            # «20 000 000» в тексте может быть «20 млн» — нормализуем обе формы
            cur_num = re.search(r"(\d[\d\s]*\d|\d)", rule["current_2026"])
            if not cur_num:
                continue
            n = cur_num.group(1).replace(" ", "")
            # Число в тексте: полная форма (20000000) ИЛИ сокращённая (20 млн)
            full_form = n in text.replace(" ", "")
            m = re.search(r"(\d+)\s*млн", text, re.I)
            short_form = bool(m and n.rstrip("0") and m.group(1).lstrip("0") == n[:len(m.group(1).lstrip("0"))][:3].lstrip("0")
                              and (len(n.rstrip("0")) <= 3 or n.rstrip("0")[:1] == m.group(1)))
            short_match = bool(re.search(rf"{int(n[:len(n)-6] or n)}\s*млн", text, re.I)) if len(n) >= 7 else False
            if full_form or short_match or short_form:
                all_param_values = " ".join(p.get("value", "") for p in params).replace(" ", "")
                if rule["expected_2027"][:3].replace(" ", "") not in all_param_values:
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


def enforce_params(text: str, params_field, article_year: int = 0) -> tuple:
    """
    Принудительная петля: детерминированная замена устаревших чисел на действующие.
    Работает ВСЕГДА, даже без паспорта (кейс: RAG-чанк с «60 млн» в статье 2026 —
    старые пороги НДС-освобождения заменяются по KNOWN_FUTURE_RULES/LEGACY_VALUES).
    Возвращает (исправленный_текст, список_замен).
    """
    if not text:
        return text, []
    fixes = []
    params, _ = _unwrap(params_field)
    fixes = []
    for p in params:
        cur_nums = {n for n, _ in _num_unit_pairs(p.get("value", ""))}
        for num, unit in _num_unit_pairs(p.get("old_value", "")):
            if num in cur_nums:
                continue
            cur_val = p.get("value", "")
            # Кейс 23: value = «до 240 часов (…)», split()[0] = «до» → все «120»
            # в тексте превращались в «до» («сверх до часов»). Замена обязана
            # начинаться с цифры; иначе подставляем число из value, а не слово.
            import re as _re_v
            _num_in_val = _re_v.search(r"\d[\d\s.,/]*", cur_val)
            if _num_in_val:
                cur_first = _num_in_val.group(0).strip()
            elif cur_val:
                cur_first = cur_val  # числового значения нет — нечем подменять число
            else:
                continue
            if unit:
                # Число с единицей («20%», «60 млн») — специфичная игла, точная замена.
                # ФИКС (аудит 07-🔴1, 2026-10-02): замена обязана сохранять единицу
                # из value: «20% → 22» и «60 млн → 20» порождали неверные числа.
                # Берём пару число+единица из value; если в value единицы нет —
                # наследуем единицу old_value.
                val_pairs = _num_unit_pairs(cur_val)
                if val_pairs:
                    cur_num_v, cur_unit_v = val_pairs[0]
                    _sp = "" if cur_unit_v in ("%", "₽") else " "
                    replacement = f"{cur_num_v}{_sp}{cur_unit_v}" if cur_unit_v \
                        else cur_num_v
                else:
                    _sp = "" if unit in ("%", "₽") else " "
                    replacement = f"{cur_first}{_sp}{unit}"
                for needle in (f"{num}{unit}", f"{num} {unit}", f"{num}\u00a0{unit}"):
                    if needle in text:
                        text = text.replace(needle, replacement)
                        fixes.append(f"{needle} → {replacement} ({p.get('name','?')})")
            else:
                # ГОЛОЕ число — только как самостоятельный токен. Без границ
                # калечатся годы и даты: «2026» → «22/12226» (кейс статьи 15, 2026-09-29)
                import re as _re
                pattern = _re.compile(rf"(?<![\d.,/\-]){re.escape(num)}(?![\d.,/])")
                new_text, n = pattern.subn(cur_first, text)
                if n:
                    text = new_text
                    fixes.append(f"{num} → {cur_first} ({p.get('name','?')}) [{n}×]")
    # Детерминированное удаление псевдо-экспертности (Heart не убирает из промпта).
    # Работает ВСЕГДА, без паспорта — это глобальный стоп-паттерн.
    _PSEUDO = [
        ("на моей практике как", "при аудите подобных систем"),
        ("на моей практике", "в практике отрасли"),
        ("в моей практике", "в практике отрасли"),
        ("я сопровождал", "специалисты сопровождали"),
        ("мы защищали", "специалисты защищали"),
        ("налоговый консультант,", "специалисты,"),
        ("как налоговый консультант", ""),
        ("как юрист", ""),
        ("как IT-юриста", ""),
        ("как бухгалтер", ""),
    ]
    for old, new in _PSEUDO:
        if old in text.lower():
            import re as _re
            text = _re.sub(_re.escape(old), new, text, flags=_re.IGNORECASE)
            fixes.append(f"псевдо-эксперт: «{old}» → «{new}»")
    # ── LEGACY VALUES: замена известных устаревших порогов даже БЕЗ паспорта.
    # Кейс: «60 млн» (порог 2024-2025) просачивается из RAG-чанков старых статей.
    for legacy in _LEGACY_VALUES:
        if article_year and article_year >= legacy["min_year"]:
            current = legacy["current"]
            name = legacy["name"]
            for old_num in legacy["old_numbers"]:
                if old_num in current["numbers"]:
                    continue
                # Замена полного вида «60 млн» и «60 000 000»
                for form in legacy["old_numbers_forms"]:
                    if form in text:
                        text = text.replace(form, current["replacement"])
                        fixes.append(f"legacy: {form} → {current['replacement']} ({name})")
    return text, fixes


# Известные устаревшие значения (обновляется при изменении норм).
# Числа сравниваются в двух формах: «60 млн» и «60 000 000».
_LEGACY_VALUES = [
    {
        "name": "порог освобождения от НДС при УСН",
        "old_numbers": ["60"],
        "old_numbers_forms": ["60 млн", "60 000 000", "60 млн ₽", "60 млн руб"],
        "current": {"numbers": ["20"], "replacement": "20 млн"},
        "min_year": 2026,
    },
    {
        "name": "основная ставка НДС",
        "old_numbers": ["20"],
        "old_numbers_forms": ["20%", "20 %"],
        "current": {"numbers": ["22"], "replacement": "22%"},
        "min_year": 2026,
    },
    {
        "name": "лимит доходов УСН",
        "old_numbers": ["450"],
        "old_numbers_forms": ["450 млн", "450 000 000"],
        "current": {"numbers": ["490"], "replacement": "490,5 млн"},
        "min_year": 2026,
    },
]


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
    if '26.2-1' in low and ('нпд' in low or 'самозанят' in low):
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
    (re.compile(r"стать[уяеюы]\s+от\s+(?:(?:КоАП|НК|ГК|ТК)|$|\.)", re.I), "заглушка «статья от»"),
    (re.compile(r"ст\.\s*от\s+", re.I), "заглушка «ст. от»"),
    (re.compile(r"по\s+статье\s+от\s", re.I), "заглушка «по статье от»"),
    (re.compile(r"стать[уяеюы]\s+от\s*[^0-9]", re.I), "заглушка: после «от» нет номера"),
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


_UNVERIFIED_HEURISTIC = re.compile(
    r"(?:если|когда|при)\s+дол[яеи]\s+[^.]{0,30}(?:меньше|менее|ниже|выше|более)\s*(\d{2,3})\s*%"
    r"|(?:если|когда|при)\s+дол[яеи]\s+клиент\w*\s+[^.]{0,30}(?:выше|более)\s*(\d{2,3})\s*%", re.I)


# Числовые артефакты: процент рядом с валютой («113% рублей», «272,13% рублей»)
_PCT_CURRENCY = re.compile(
    r"(\d+[.,]?\d*)\s*%\s*(?:руб|₽|млн|тыс|тысяч|миллионов)", re.I)
# Контекстные запрещённые термины (роль термина ≠ контекст)
_TERM_ROLE_RULES = [
    (re.compile(r"профессион\w+\s+доход", re.I),
     "«профессиональный доход» — термин НПД. Для ИП на ОСНО используйте "
     "«предпринимательский доход» или «доход от предпринимательской деятельности»"),
    (re.compile(r"(?:переход\w+\s+на\s+ОСНО|с\s+УСН)[^.]{0,80}(?:восстанов\w+|восстанавлив\w*)\s+НДС", re.I),
     "«восстановление НДС» при переходе на ОСНО — неверное направление. "
     "При переходе с УСН на ОСНО можно ЗАЯВИТЬ ВЫЧЕТ входного НДС по остаткам "
     "(при соблюдении условий), а не восстановить налог"),
    (re.compile(r"(?:восстанов\w+|восстанавлив\w*)\s+НДС[^.]{0,80}(?:переход\w+\s+на\s+ОСНО|с\s+УСН)", re.I),
     "«восстановление НДС» при переходе на ОСНО — см. выше (два направления фразы)"),
]
# Расширенный псевдо-эксперт
_PSEUDO_EXPERT_PATTERNS = [
    "на моей практике", "в моей практике", "я сопровождал", "мы сопровождали",
    "мы защищали", "я не раз видел", "моим клиентам", "как налоговый консультант",
    "как юрист", "как IT-юриста", "как бухгалтер", "налоговый консультант с практикой",
    "при аудите подобных систем налоговый консультант",
    "в практике налоговых консультантов", "наши клиенты",
]


# Бенчмарки без источника (кейс: CPL 1500-3000 ₽ как «средний по рынку»)
_BENCHMARK_NO_SRC = re.compile(
    r"(?:CPL|CAC|CPC|конверсия|ставка|стоимость)\s*[^.]{0,30}(\d[\d\s,–-]{2,10})\s*(?:₽|руб|тыс|млн)", re.I)
# Доменная контаминация: налоговые термины в НЕ-налоговой статье
_TAX_CONTAMINATION = re.compile(
    r"(?:рекламн\w+\s+сбор|нормируем\w+\s+расход|сверхнормативн\w+\s+расход|"
    r"налогов\w+\s+учет\s+реклам|ст\.\s*264\s*НК|ст\.\s*270\s*НК|"
    r"лимит\s*1\s*%\s*выручк|нормируем\w+\s+реклам)", re.I)


# Рыночные цифры без методологии («65,8 трлн ₽» без источника)
_MARKET_SIZE_NO_SRC = re.compile(
    r"(?:объ[её]м|рынок|размер|оборот)\s*[^.]{0,20}(\d+[.,]?\d*)\s*(?:трлн|триллион)", re.I)
# Коммерческие абсолюты в заголовках и тексте
_COMMERCIAL_ABSOLUTE = re.compile(
    r"(?:конкурент\w+[^.]{0,20}не\s*занял|свободн\w+\s*ниш|"
    r"без\s+конкурентов|гарантированн\w+\s*(?:результат|рост|прибыль)|"
    r"стопроцентн\w+\s*окупаемость)", re.I)


# Прогнозные цифры для будущих лет без маркировки
_FORECAST_NO_MARK = re.compile(
    r"(?:CPL|CAC|CPC|конверсия|бюджет|рост|стоимость)\s*[^.]{0,30}\d[\d\s,–-]{2,10}\s*(?:₽|руб|%)\s*"
    r"(?!.*(?:прогноз|сценарий|ориентир|зависит|условный))", re.I)
# Драматизация в деловом тексте
_DRAMATIC = re.compile(
    r"(?:единственн\w+\s+окно|обратного\s+хода\s+нет|последний\s+шанс|"
    r"год\s+слепых\s+расходов|точка\s+невозврата|"
    r"обруш\w+|взлет\w+\s+до|шторм\w+|цунами)", re.I)


# ── P0 (редактура L1-маркетинг, 2026-09-29): прогноз 2027 как факт ──
# «к 2027 году средний чек составит 5600 ₽» без маркировки прогноза
_FORECAST_YEAR_FACT = re.compile(
    r"(?:к|в)\s*20(2[7-9]|3\d)\s*году?[^.]{0,90}?"
    r"(?:составит|достигнет|вырастет|увеличится|упадёт|упадет|снизится|"
    r"подорожают|поднимутся|будут стоить|будет стоить)[^.\d]{0,25}\d", re.I)
_FORECAST_MARKERS = re.compile(
    r"прогноз|сценари|оценк|исследован|по данным|ожидает|модельн|условн|"
    r"ориентир|консенсус|аналитик|допущени", re.I)

# Модельный сценарий: цифры детерминированного калькулятора с явными
# допущениями — не «рыночная оценка» (статья 25, P&L-таблица).
_MODEL_MARKER = re.compile(
    r"допущени\w+|условн\w+|модельн\w+|модель\s+со\s+следующими|иллюстративн\w+|требует\s+уточнения\s+по\s+дате|ориентировочн\w+|расч[её]тн\w+|предварительн\w+\s+расч[её]т\w*|в\s+данном\s+сценари|оценочн\w+|примерн\w+", re.I)

# P0: условный кейс читается как реальный (имя + цифры без маркировки)
_CASE_NAMES = re.compile(
    r"(?:Игор[ьяе]\w*|Анн[аыуе]\w*|Марин[аыуе]\w*|Серге[йяе]\w*|Ольг[аиуе]\w*|"
    r"Дмитри[йяе]\w*|Елен[аыуе]\w*|Алексе[йяе]\w*|Наталь[яиЕ]\w*|Иван[ауе]\w*|"
    r"Марии\b|Мария|Андре[йяе]\w*|Виктор[ауеи]\w*|Ксени[иая]\w*|Пётр\w*|Петр\w*|"
    r"ООО\s*[«\"']?[А-ЯЁ][а-яё]+|ИП\s+[А-ЯЁ][а-яё]+)")
_CASE_MARKERS = re.compile(
    r"условн|пример|модель|гипотет|допустим|представ(им|ьте|им себе)|сценари|"
    r"виртуальн|вымышленн|допустимой|расчётн|расчетн", re.I)
_CASE_DIGITS = re.compile(r"\d[\d\s,.\u2013-]*\s*(?:₽|руб|тыс|млн|млрд|%|раза?)", re.I)

# P1: ссылка на статью без акта («ст. 13.11» — КоАП или 152-ФЗ?)
_BARE_ARTICLE_REF = re.compile(r"(?:ст\.?\s*|стать[ьяе]\s*)(?:№\s*)?\d+(?:\.\d+)?")
# Кейс статьи 24v2: «ст. 105 Закона № 44-ФЗ» фальшиво ловилась — русский
# стиль «№ 44-ФЗ» ставит ФЗ ПОСЛЕ номера; добавлены «NNN-ФЗ» и «закона»
_ACT_QUALIFIERS = re.compile(
    r"152-ФЗ|КоАП|ГК\s*РФ|НК\s*РФ|ТК\s*РФ|кодекс|приказ|постановлен|"
    r"распоряжен|ФЗ\s*[-№]|\d+\s*[-–—]\s*ФЗ\b|закон[ауеи]\b|№\s*\d+", re.I)

# P1: вердикт каналу без контекста («Telegram — лучший канал»)
_CHANNEL_VERDICT = re.compile(
    r"(?:Telegram|Телеграм|Директ|Яндекс\s*Директ|таргет\w*|VK|ВК|email|SEO|"
    r"контекстн\w+|рассылк\w+)[^.\n]{0,60}?"
    r"(?:плохой|плохо\s+подходит|не\s+работает|худший|лучший|самый\s+эффективный|"
    r"идеальн\w+|бесполезн\w+|мёртв\w+|мертв\w+)", re.I)

# P1: термин без расшифровки при первом использовании
_TERM_DEFS = {
    "CPL": ("стоимость заявки", "цена заявки", "cost per lead"),
    "CAC": ("стоимость привлечения клиента", "цена привлечения клиента",
            "customer acquisition"),
    "LTV": ("пожизненная ценность", "lifetime value", "ценность клиента"),
    "CTR": ("кликабельность", "click-through", "кликабельн"),
    "ROMI": ("окупаемость маркетинг", "return on marketing"),
    "ДРР": ("доля рекламных расходов",),
    "CAC-доходность": (),
    "PFL": ("pay for lead",),
}


# ── P0 (ревью статьи 15, 2026-09-29): подпись работника при КЭДО ──
# Инвариант: если в предложении утверждается подпись РАБОТНИКА квалифицированной
# ЭП/УКЭП — в том же окне обязаны быть альтернативы (УНЭП/простая/ст. 22.3).
# «работник — аналогично» после УКЭП работодателя — всегда ошибка.
_WORKER_SIGN_CLAIM = re.compile(
    r"работник[а-яё]*[^.]{0,30}?подпис\w+[^.]{0,120}?"
    r"(?:квалифицированн\w+|УКЭП)|"
    r"работник[а-яё]*\s*[—–-]\s*аналогично", re.I)
_WORKER_SIGN_ALTS = re.compile(
    r"УНЭП|неквалифицированн\w+|прост\w+\s+(?:электронн\w+\s+)?подпис|22\.3", re.I)
# Отрицание альтернатив — само по себе ошибка, окно альтернатив не спасает
_SIGN_ALT_DENIED = re.compile(
    r"(?:прост\w+|неквалифицированн\w+)\s+(?:электронн\w+\s+)?подпис\w+"
    r"[^.\n]{0,100}?(?:не\s+предусмотрен\w*|не\s+допускает\w*|нет\s+места|"
    r"невозможн\w+|запрещ\w+)", re.I)
# Работодатель «только УКЭП» как универсальное правило — неверно
# (УКЭП работодателя — для критичных документов ст. 22.3, не для всех)
_EMPLOYER_UKEP_ONLY = re.compile(
    r"работодатель\w*[^.]{0,60}(?:подпис\w+|применя\w+|использ\w+)[^.]{0,100}?"
    r"(?:только|исключительн\w+)\s+(?:усиленн\w+\s+квалифицированн\w+|УКЭП)", re.I)

# Неподтверждённые планы регулятора («итоги эксперимента к 2027»)
_REGULATOR_PLAN_NO_SRC = re.compile(
    r"(?:регулятор\w*|министерство|минтруд\w*)[^.\n]{0,80}?"
    r"(?:планирует|подвед[её]т|обещает)|"
    r"(?:итог\w+|результат\w+)\s+(?:эксперимент\w+|пилот\w+)[^.\n]{0,60}?к\s+20\d{2}", re.I)

# Миф об общем дедлайне КЭДО/ЭТК
_KEDO_MANDATORY_MYTH = re.compile(
    r"(?:КЭДО|электронн\w+\s+трудов\w+\s+книжк\w+)[^.\n]{0,120}?"
    r"(?:обязательн\w+|станут\s+обязательн\w+)[^.\n]{0,60}?"
    r"(?:для\s+большинств\w+|для\s+всех)", re.I)

# Категоричные исходы проверки ГИТ — обобщено до всех органов ниже
# Процессуальный линтер: субъект власти + категоричный глагол без процедуры
# (ревью 17-18: «РКН докажет», «банк ограничит за сутки», «ФНС переквалифицирует»)
_AUTHORITY = (r"(?:ФНС|ГИТ|инспекци\w*|Роскомнадзор\w*|РКН|суд[ауе]?|банк\w*|"
              r"СФР|прокуратур\w*|налогов\w+|регулятор\w*|казначейств\w*|"
              r"ФАС|заказчик\w*|оператор\w*\s+ЭТП|площадк\w*)")
_CAT_VERB = (r"(?:обяжет|обяжутся|признает|признаёт|доначислит|заблокирует|"
             r"потребует|оштрафует|переквалифицирует|откажет|аннулирует|"
             r"докажет|вынесет\s+предписание\s+восстановить|ограничит|"
             r"закроет\s+сч[её]т|остановит\s+процедур\w+|"
             r"вынесет\s+предписание|обязан\s+(?:закрыть|заблокировать|"
             r"отказать|остановить)|обязаны\s+(?:закрыть|заблокировать|"
             r"отказать|остановить))")
_PROCEDURAL_CATEGORICAL = re.compile(
    _AUTHORITY + r"[^.\n]{0,80}?" + _CAT_VERB, re.I)
# PROCESS_AUTOMATION_OVERCLAIM (рекомендации агента 2026-10-02):
# автоматизм без условия («жалоба сразу блокирует контракт»)
_PROC_AUTOMATION = re.compile(
    r"(?:автоматически|сразу|непременно|неминуемо|обязательно)"
    r"[^.\n]{0,60}?(?:приостанавливает|блокирует|останавливает|"
    r"приводит\s+к|приведёт\s+к|ведёт\s+к\s+суд|закрывает)", re.I)

# Субъектный скоуп: обязанность перенесена на неверное лицо
# (115-ФЗ — банки, а не платформы; мониторинг операций — кредитные организации)
_SUBJECT_SCOPE = [
    (re.compile(r"(?:платформ\w+|площадк\w+|маркетплейс\w*)[^.\n]{0,80}?"
                r"(?:обязан\w+|должн\w+)[^.\n]{0,80}?115-ФЗ", re.I),
     "115-ФЗ обязывает банки и кредитные организации, а не платформы"),
    (re.compile(r"кажд\w+\s+(?:работодатель|компания|организация)[^.\n]{0,100}?"
                r"(?:обязан\w+|должн\w+)[^.\n]{0,60}?"
                r"(?:подтверждать\s+реальность|мониторинг\w+\s+операци\w+)", re.I),
     "обязанность мониторинга операций — у кредитных организаций"),
]

# Ложная обязательность: предположение маскируется под закон/практику
_FALSE_OBLIGATION = re.compile(
    r"(?:практика\s+показывает|инспекторы\s+всегда|налоговая\s+видит|"
    r"обычно\s+штрафуют|суд\s+призна[её]т\s+без|автоматически\s+вед[её]т\s+к\s+"
    r"(?:штраф|доначисл|блокировк)|рынок\s+требует\s+от\s+всех)", re.I)

# mode_split: смешение несовместимых режимов без явной развилки
_MODE_MIX_RULES = [
    (re.compile(r"УСН"), re.compile(r"ОСНО|общий\s+режим"),
     "УСН и ОСНО требуют раздельных сценариев (ставки, базы, вычеты различаются)"),
    (re.compile(r"КЭДО"), re.compile(r"трудов\w+\s+книжк"),
     "трудовые книжки исключены из КЭДО (ст. 22.1 ТК) — не смешивайте режимы"),
]
_SPLIT_MARKERS = re.compile(
    r"сценари|отдельн\w+|различи\w+|для\s+каждого|раздел\w+|в\s+зависимости\s+от\s+режима|"
    r"исключен\w+|не\s+входит|не\s+относится|разведени\w+", re.I)

# Ревью 20: три контура банковских проверок — смешение = фактическая ошибка
_CIRCUIT_MARKERS = [
    re.compile(r"агент\w*\s+валютного\s+контрол|173-ФЗ|валютного\s+контрол\w+\s*\(173"),
    re.compile(r"115-ФЗ|противодейст\w+\s+легализац|делов\w+\s+цели\s+операции"),
    re.compile(r"161-ФЗ|национальн\w+\s+платёжн\w+\s+систем|антифрод"),
]
# прямое смешение: агент валютного контроля «на основании 161-ФЗ»
_VC_161_MIX = re.compile(
    r"(?:агент\w*\s+валютного\s+контрол\w*|валютного\s+контрол\w*)[^.\n]{0,120}?161-ФЗ|"
    r"161-ФЗ[^.\n]{0,120}?(?:агент\w*\s+валютного\s+контрол)", re.I)

# Ревью 20: неверный штрафной диапазон 15.25 КоАП (юридически 20–40%)
_FINE_1525_WRONG = re.compile(
    r"(?:15\.25|валютн\w+)[^.\n]{0,120}?20\s*(?:[—–-]|до)\s*30\s*%", re.I)

# Широкие формулы о снятии требований — без указания применимости
_BROAD_RELIEF = re.compile(
    r"(?:снят\w+|отменен\w+|упраздн\w+)[^.\n]{0,60}?требован\w+"
    r"(?:\s+об\s+)?(?:обязательн\w+\s+)?(?:репатр|валютн)", re.I)

# Псевдо-практика: «суды в таких спорах оценивают…» без дел
_COURT_PSEUDO = re.compile(
    r"суды?\s+(?:в\s+таких\s+спорах|обычно|как\s+правило|в\s+подобных\s+делах)"
    r"[^.\n]{0,120}?оценива", re.I)

# ── Валютный домен: 4 правила (предложение агента по ревью 20) ──
# 1. Право vs обязанность банка (обязан заблокировать/расторгнуть/отказать)
_BANK_DUTY_OVERREACH = re.compile(
    r"(?:банк|уполномоченн\w+\s+банк)[^.\n]{0,30}?обязан\w*[^.\n]{0,80}?"
    r"(?:заблокир|расторгн\w+|отказ\w+|приостанов\w+)", re.I)

# 2. Санкция ≠ запрос документов: пугать 15.25 при простом запросе — манипуляция
_FINE_SCARE_NO_GUILT = re.compile(
    r"запрос\w*\s+документ\w+", re.I)
_FINE_SCARE = re.compile(r"15\.25|штраф", re.I)
_GUILT_MARKER = re.compile(
    r"признана?\s+незаконн|доказан\w+|установлен\w+\s+нарушен|состав\w+\s+нарушен", re.I)

# 3. Обжалование без причины/маршрута
_APPEAL_NO_ROUTE = re.compile(r"обжалова\w*", re.I)
_APPEAL_ROUTE_MARKERS = re.compile(
    r"основан\w+|причин\w+|зависит|маршрут|комисси\w+|в\s+зависимости|115-ФЗ|161-ФЗ|"
    r"валютного\s+контрол", re.I)

# 4. Обещанный исход в условном кейсе
_CASE_OUTCOME_PROMISE = re.compile(
    r"(?:платёж\s+прошёл|банк\s+принял|суд\s+восстановил|счёт\s+разблокировал)", re.I)
_CASE_CONTEXT = re.compile(r"кейс|сценари|пример|компани\w+\s+из\s+разбор", re.I)
_CASE_SOFTENERS = re.compile(
    r"условн\w+|модельн\w+|могл\w+|в\s+идеальн\w+|при\s+благоприятн\w+", re.I)

# ── Ревью 21: технические процедуры без первичного источника ──
# Универсальные коди СПД/видов операций («импорт = 24_4») без классификатора
_SPD_CODE_PATTERN = re.compile(
    r"(?:код\w*\s+(?:СПД|вида\s+подтверждающ\w+\s+документ\w*|валютн\w+\s+операции|вида\s+операции)"
    r"|\b2\d_\d\b)[^.\n]{0,120}?(?:импорт\w*|экспорт\w*)|"
    r"(?:импорт\w*|экспорт\w*)[^.\n]{0,40}?\b2\d_\d\b", re.I)
_SPD_CODE_SOURCE = re.compile(r"классификатор|приложени\w+\s+к\s+Инструкции|181-И", re.I)

# TECHNICAL_CODE_REQUIRES_PRIMARY_SOURCE, закупочный контур (2026-10-02):
# ОКПД2/КТРУ/номера форм/XML/реестровые записи — не выдумываются LLM
_TECH_CODE_PROC = re.compile(
    r"ОКПД2?\s*\d{2}\.\d{2}|\bКТРУ\s*[\d.]{6,}|форма\s+\d{7,8}|"
    r"XML-схем\w*|реестров\w+\s+запис[ьи]\s*№?\s*\d{5,}", re.I)
_TECH_CODE_PROC_SRC = re.compile(
    r"классификатор\w*|КТРУ[^.\n]{0,40}каталог|ЕИС|ГИСП|приказ|"
    r"перечень|справочник|минпромторг|проверьте\s+код", re.I)

# «поставить на учёт немедленно» без проверки порога
_ACCOUNT_IMMEDIATE = re.compile(
    r"(?:немедленно|сразу\s+же)[^.\n]{0,100}?(?:уч[её]т|постановк\w+)|"
    r"(?:постановк\w+|на\s+уч[её]т)[^.\n]{0,100}?(?:немедленно|сразу\s+же)", re.I)
_THRESHOLD_MARKERS = re.compile(r"3\s*млн|10\s*млн|порог|должен\s+ли|проверь\w+.{0,40}порог", re.I)

# СПД «15 рабочих дней» как универсальная обязанность после любого импорта
_SPD_UNIVERSAL = re.compile(
    r"СПД[^.\n]{0,160}?15\s*рабочих\s*дн\w+|15\s*рабочих\s*дн\w+[^.\n]{0,120}?СПД", re.I)
_SPD_CONDITION = re.compile(r"УНК|ФТС|если\s+требуется|когда\s+требуется|таможен\w+ деклар", re.I)

# Запрос в банк-корреспондент
_CORRESPONDENT_BANK = re.compile(
    r"(?:пиш\w+|направля\w+|отправля\w+|запрос\w*)[^.\n]{0,60}?банк\w*-?\s*корреспондент", re.I)

# Репатриация как автоматическое нарушение без Указа 529
_REPATRIATION_AUTO_VIOLATION = re.compile(
    r"(?:задержк\w+|невозврат\w+|непоступл\w+)[^.\n]{0,100}?(?:нарушен\w+|состав\w+|ответствен\w+)"
    r"[^.\n]{0,80}?репатр|"
    r"нарушен\w+[^.\n]{0,60}репатр", re.I)
_UKAZ_529_MARKER = re.compile(r"Указ\w*\s*№?\s*529|временно\s+не\s+применя", re.I)

# ── Ревью 22: объект нормы вместо числа ──
# LIMIT_SCOPE_DRIFT: «4 часа в день» вместо «за два дня подряд»
_LIMIT_4H_DAY = re.compile(
    r"4\s*час\w*[^.\n]{0,60}(?:в\s+(?:один\s+)?день|за\s+день|ежедневно)|"
    r"(?:в\s+(?:один\s+)?день|ежедневно)[^.\n]{0,40}?4\s*час\w*", re.I)
_LIMIT_4H_CORRECT = re.compile(r"двух\s+дн\w+\s+подряд|2\s+дн\w+\s+подряд", re.I)

# REQUIRED_CONDITION_MISSING: 240 часов без согласия/колдоговора
_OVERTIME_240_CLAIM = re.compile(
    r"(?:увелич\w+|поднят\w+|лимит\w*|допуска\w+|можн\w+|до)\s*[^.\n]{0,60}?240\s*час\w+|"
    r"240\s*час\w+[^.\n]{0,80}(?:лимит|предел|год)", re.I)
_OVERTIME_CONDITIONS = [
    (re.compile(r"письменн\w+\s+согласи\w+|согласи\w+\s+работника", re.I), "письменное согласие работника"),
    (re.compile(r"коллективн\w+\s+договор\w*|отраслев\w+\s+(?:межотраслев\w+\s+)?соглашен", re.I),
     "коллективный договор/отраслевое соглашение"),
]
_OVERTIME_GUIDE_ANGLE = re.compile(
    r"как\s+оформить|условия|инструкци|пошагов|чек-лист|что\s+изменилось|полное\s+руководство", re.I)
_MEDICAL_DAY = re.compile(r"диспансеризац", re.I)

# SANCTION_OVERCLAIM: категоричная привязка к части КоАП с суммой
_SANCTION_OVERCLAIM = re.compile(
    r"(?:штраф\s+составит|оштрафуют\s+на|цена\s+ошибки|наказыва\w+\s+по\s+ч\.|"
    r"инспектор\w*\s+оштрафует)", re.I)
_SANCTION_CONDITIONS = re.compile(
    r"в\s+зависимости\s+от|зависят\s+от|состава|повторн\w+|субъект\w+|может\s+повлечь", re.I)

# UNSUPPORTED_LEGAL_BRANCH: статья сама признаёт непроверенность
_SELF_UNVERIFIED = re.compile(
    r"(?:до\s+сверки|до\s+проверки|нельзя\s+переписывать|не\s+проверено|"
    r"требует\s+уточнения\s+по\s+первоисточн\w+|до\s+подтверждения\s+первичн\w+)"
    r"[^.\n]{0,80}", re.I)

# MISSING_PROMISED_SECTION: заголовок секции без содержимого
_PROMISED_SECTION = re.compile(
    r"^#{2,3}\s*(?:Быстрая самопроверка|Чек-лист\w*|FAQ|Итоги\w*|Что\s+сделать)", re.I | re.M)

# ── Ревью 23: артефакты замены и различение согласия ──
# UNRESOLVED_TEMPLATE_VARIABLE: «до часов», «с 3-го по до-й», {{var}}, [placeholder]
_UNRESOLVED_TEMPLATE = re.compile(
    r"(?:сверх\s+до\s|по\s+до-й|до-й\s+час|от\s+до\s+до|^\s*до\s+часов|"
    r"\{\{[^}]+\}\}|\[[а-яё\s]{2,20}\](?!\()|<[а-яёa-z\s_]{2,20}>|"
    r"\bдо\s+(?:час|млн|руб|тыс|дн)\b(?!,\sчто))", re.I | re.M)

# CONSENT_NOT_EQUAL_ACKNOWLEDGMENT: «ознакомить под подпись» при требовании согласия
_ACK_NOT_CONSENT = re.compile(
    r"ознаком\w+[^.\n]{0,80}(?:под\s+подпись|под\s+роспись)", re.I)
_CONSENT_MARKER = re.compile(r"письменн\w+\s+согласи\w+", re.I)
# темы, где персональное согласие обязательно
_CONSENT_TOPICS = re.compile(
    r"сверхурочн|свыше\s+120|сверх\s+120|переработк|240\s+час|"
    r"перевод\w+\s+на\s+другую|обработк\w+\s+персональн", re.I)

# MANDATORY_EMPLOYEE_GUARANTEE (расширение): медконтур + отдых-альтернатива
_MED_CONTEXT = re.compile(r"медицинск\w+\s+(?:осмотр|заключен|противопоказан)|диспансеризац", re.I)
_REST_ALTERNATIVE = re.compile(r"дополнительт\w+\s+врем\w+\s+отдых\w*|время\s+отдыха\s+вместо", re.I)

# ── Маркетплейс-домен ──
_MPLACE_CTX = re.compile(r"маркетплейс|селлер|WB|Ozon|Озон|Wildberries|вайлдберриз|FBO|FBS", re.I)
# GMV назван деньгами/выручкой селлера
_GMV_AS_MONEY = re.compile(
    r"(?:GMV|оборот\s+товаров)[^.\n]{0,120}?(?:ваши\w*\s+деньги|деньги\s+селлера|"
    r"выручк\w+\s+селлера|прибыль\s+селлера|это\s+ваши\s+продажи)", re.I)
# Конкретный тариф без даты снимка
_TARIFF_NO_DATE = re.compile(
    r"(?:комисси\w+|тариф\w+|хранение\s+стоит|логистик\w+\s+стоит)[^.\n]{0,60}?"
    r"\d+[.,]?\d*\s*%", re.I)
_SNAPSHOT_DATE = re.compile(
    r"снимок|на\s+дату|по\s+состоянию\s+на|актуальн\w+\s+на|202[5-7]", re.I)
# FBO/FBS в одном совете без разводки
_FBO_FBS_MIX = re.compile(r"FBO", re.I)
_FBS_RX = re.compile(r"FBS", re.I)

# ── Закупочный домен (44/223-ФЗ) — снимок норм 2026-09-29 ──
_PROC_CTX = re.compile(
    r"закуп|тендер|госконтракт|44-ФЗ|223-ФЗ|ФАС|НМЦК|антидемпинг|"
    r"единственн\w+\s+поставщик|участник\w+\s+закупки|извещени\w+\s+об\s+осуществлении|"
    r"обеспечение\s+(?:заявки|исполнения)|обеспечени\w+\s+исполнения\s+контракт|"
    r"7\.30\.[1-4]|КоАП\s+РФ\s+в\s+(?:сфер|контракт)|контрактн\w+\s+систем", re.I)
# Оба закона в одном тексте без разводки (кейс-прецедент: контуры 115/161-ФЗ)
_FZ44_RX = re.compile(r"44-ФЗ", re.I)
_FZ223_RX = re.compile(r"223-ФЗ", re.I)
_FZ_SPLIT = re.compile(
    r"различ|отлича|в\s+отличие|не\s+путайте?|не\s+смешива|справочн\w+\s+врезк|"
    r" отдельн\w+\s+(?:статья|материал|раздел)|по\s+223-ФЗ\s+(?:друг|иначе)|"
    r"устроен\w*\s+(?:иначе|по-другому)|по\s+223-ФЗ[^.]{0,60}?(?:друг|иначе|отлич)|"
    r"не\s+по\s+223|не\s+по\s+44-ФЗ|а\s+не\s+по\s+223", re.I)
# Старый срок обжалования: «жалоба … десять дней» (действует 5 дней, 624-ФЗ)
_APPEAL_10D = re.compile(
    r"(?:жалоб\w+|обжалова\w+)[^.\n]{0,60}?(?:десят\w+|10)\s+(?:календарн\w+\s+)?дн\w+"
    r"|(?:десят\w+|10)\s+(?:календарн\w+\s+)?дн\w+[^.\n]{0,60}?(?:жалоб\w+|обжалова\w+)", re.I)
# Малая закупка 600 тыс без годового лимита (п. 4 ч. 1 ст. 93)
_SMALL_600K = re.compile(r"600\s*(?:тыс|000)|шестисот\s+тысяч", re.I)
_SMALL_LIMIT = re.compile(
    r"2\s*млн|два\s+миллиона|10\s*%|СГОЗ|совокупн\w+ годов|"
    r"годов\w+\s+(?:объ[её]м|лимит|объём)", re.I)
# Полуторное антидемпинговое обеспечение без порога НМЦК 15 млн (ст. 37)
_ANTIDUMP_HALF = re.compile(r"(?:полтора|1,5|1\.5)\s*(?:раза|кратн\w+)", re.I)
_ANTIDUMP_15M = re.compile(r"15\s*млн|пятнадцат\w+\s+миллион", re.I)
# Гарантированный исход жалобы/ФАС (overclaim)
_FAS_WIN = re.compile(
    r"(?:гарантированн\w+|стопроцентн\w+|на\s+100\s*%)[^.\n]{0,70}?"
    r"(?:выигр\w+|отмен\w+|обяж\w+|возврат\w+)|(?:жалоб\w+|ФАС)[^.\n]{0,40}?"
    r"(?:гарантир|обеспеч\w+\s+победу)", re.I)

# ── Правила из расширенной системы (PROCUREMENT_KB_OPERATING_RULES) ──
# PRACTICE_AS_RULE: один кейс ФАС/суда выдан как общее правило
_PRACTICE_AS_RULE = re.compile(
    r"(?:ФАС|суд(?:ы|ами|е)|арбитражн\w+ суд\w*|Верховн\w+ Суд\w*)[^.\n]{0,60}?"
    r"(?:всегда|неизменно|во\s+всех\s+случаях|повсеместно)"
    r"|(?:суды|судьи|ФАС)\s+признают[^.\n]{0,60}?(?:люб\w+|все[ехм]?\s+)"
    r"|участник\w+\s+обязаны\s+допустить", re.I)
# DEADLINE_CONTEXT: срок без события начала отсчёта
_DEADLINE_NAKED = re.compile(
    r"в\s+течени[еи]\s+\d+\s*(?:\(?[а-яё]+\)?\s*)?дн\w*", re.I)
_DEADLINE_EVENT = re.compile(
    r"после|со\s+дня|с\s+дня|с\s+даты|с\s+момента|следующего\s+за|"
    r"истечени\w+|последнего\s+дня|размещени\w+|подписани\w+|получени\w+|"
    r"уведомлени\w+|направлени\w+|вступлени\w+", re.I)
# 223-ФЗ: категоричность без оговорки о положении о закупке
_FZ223_CATEGORIC = re.compile(
    r"\b(?:всегда|обязательн\w+|единственн\w+\s+порядок|срок\s+составляет|"
    r"одинаков\w+|для\s+всех\s+заказчиков)", re.I)
_FZ223_REGULATION = re.compile(r"положени\w+\s+о\s+закупке", re.I)
# Будущая норма подана как действующая: «действует … с ДАТА(будущая)»
# (два порядка слов: «действует с ДАТА» и «С ДАТА действует»)
_FUTURE_AS_CURRENT = re.compile(
    r"(?:действ\w+|применяется?|работает)[^.\n]{0,60}?"
    r"с\s+(\d{1,2})[./](\d{1,2})[./](\d{4})"
    r"|с\s+(\d{1,2})[./](\d{1,2})[./](\d{4})[^.\n]{0,60}?"
    r"(?:действ\w+|применяется?|работает)", re.I)
# Срок жалобы искажён: 3 месяца — судебный срок обжалования РЕШЕНИЯ ФАС
# (ч. 9 ст. 105), не срок подачи жалобы в контрольный орган (5 дней,
# ч. 2 ст. 105). Кейс статьи 24 (2026-10-01): конкатенация из блога.
_APPEAL_3M = re.compile(
    r"(?:жалоб\w+|обжалова\w+)[^.\n]{0,80}?(?:тр[её]х|3)\s*месяцев"
    r"|(?:тр[её]х|3)\s*месяцев[^.\n]{0,80}?(?:жалоб\w+|обжалова\w+)", re.I)
_APPEAL_3M_OK = re.compile(
    r"суд\w+|решени\w+\s+ФАС|по\s+существу|кассаци\w+|исков", re.I)
# Действующая норма подана как гипотетическая («могут вступить с ДАТА» /
# «с ДАТА могут вступить»)
_MAYBE_FUTURE = re.compile(
    r"(?:мог[уе]т|мож[ео]т)\s+(?:вступить|заработать|действовать|появиться)"
    r"[^.\n]{0,60}?с\s+(\d{1,2})[./](\d{1,2})[./](\d{4})"
    r"|с\s+(\d{1,2})[./](\d{1,2})[./](\d{4})[^.\n]{0,60}?"
    r"(?:мог[уе]т|мож[ео]т)\s+(?:вступить|заработать|действовать|появиться)",
    re.I)

# ── Ревью статьи 24 (2026-10-02, 5/10) ──
# UNSUPPORTED_FUTURE_PROCUREMENT_CHANGE: ожидаемые изменения закупочного
# законодательства без нормативного акта («Ожидается… порог может
# составить порядка 30 млн» — и ни одного ФЗ/постановления рядом)
_FUT_CHANGE = re.compile(
    r"(?:ожидается|ожидаем\w+|планируется|предполагается|"
    r"мог[уе]т\s+быть\s+увеличен\w*|может\s+составить|будут\s+применяться)"
    r"[^.\n]{0,160}?(\d+(?:[.,]\d+)?)\s*(?:млн|млрд|тыс|%|руб)", re.I)
_ACT_REF = re.compile(r"ФЗ\b|постановлен|приказ|распоряжен|№\s*\d+", re.I)
# HEDGED_LEGAL: «вероятно, неактуальны» — неуверенность в правовой норме
_HEDGED_LEGAL = re.compile(
    r"(?:вероятн\w+|по-видимому|скорее\s+всего|как\s+кажется|"
    r"видимо,?|по\s+имеющимся\s+данным|обычно\s+опира\w+|"
    r"в\s+большинстве\s+случаев|"
    r"насколько\s+известно)[^.\n]{0,80}?"
    r"(?:неактуальн\w+|утрат\w+|отмен\w+|недействит\w+|опира\w+|"
    r"основыва\w+|полагает\w+)"
    r"|(?:стат\w+|норм\w+|ст\.?\s*\d+)[^.\n]{0,80}?(?:вероятн\w+|скорее\s+всего)"
    r"|(?:в\s+большинстве\s+случаев|по\s+имеющимся\s+данным)[^.\n]{0,60}?"
    r"(?:ставк\w*|норм\w*|размер\w*|правил\w*|срок\w*)",
    re.I)
# CALENDAR_TYPE / закон у сроков (для PROCUREMENT_DEADLINE_NEEDS_CONTEXT)
_CAL_TYPE = re.compile(r"календарн\w+|рабоч\w+\s+дн\w+|рабочих\s+дней", re.I)
_DEADLINE_LAW = re.compile(r"44-ФЗ|223-ФЗ|ст\.?\s*\d+|закон[ау]?", re.I)

# ── Ревью статьи 24 v3 (2026-10-02, 7.5/10) ──
# EIS_SIGNATURE_CONTRADICTION: «КЭП обязательна» и «КЭП не требуется» в одном тексте
_KEP_MANDATORY = re.compile(
    r"КЭП|квалифицированн\w+\s+(?:электронн\w+\s+)?подпис\w*[^.\n]{0,60}?"
    r"(?:обязательн|необходим|нужна)", re.I)
_KEP_NOT_NEEDED = re.compile(
    r"(?:КЭП|квалифицированн\w+\s+(?:электронн\w+\s+)?подпис\w*|подпис\w+)\s+"
    r"(?:не\s+(?:требуется|нужна|обязательн)|заверять\s+[^.\n]{0,40}не\s+требуется)", re.I)
# COMPLAINT_SUSPENSION_OVERCLAIM: «жалоба приостанавливает закупку» автоматом
_SUSPEND_OVERCLAIM = re.compile(
    r"(?:жалоб\w+|подача\s+жалобы|обжалование)[^.\n]{0,60}?"
    r"(?:приостанавливает|останавливает|блокирует|замораживает)[^.\n]{0,40}?"
    r"(?:закупк|процедур|контракт|определение\s+поставщика)", re.I)
# COURT_ROUTE_OVERCLAIM: «спор переходит в арбитражный суд» без предмета
_COURT_AUTO = re.compile(
    r"(?:спор|дело|жалоба|разногласия)[^.\n]{0,50}?"
    r"(?:переходит?\s+в|идт\w+\s+в|подадим?\s+в)\s+(?:арбитражн\w+\s+)?суд"
    r"|(?:остаётся|останется)\s+только\s+судебн\w+\s+трек", re.I)
_COURT_OK = re.compile(
    r"ч\.?\s*9\s*ст\.?\s*105|решени\w+\s+ФАС|АПК|КАС|процессуальн", re.I)
# SPECIAL_COMPLAINT_JURISDICTION: категоричный запрет жалобы по оценке заявок
_SPECIAL_JURISDICTION = re.compile(
    r"(?:жалоб\w+\s+на\s+оценку|оценку\s+заявки)[\s\S]{0,120}?"
    r"не\s+пода[её]тся[\s\S]{0,120}?(?:только\s+суд|проверяет\s+только)", re.I)

# ── Ревью статьи 25 (2026-10-02, 6/10) ──
# UNSUPPORTED_CAUSALITY: «закон изменил тарифы/маржу» — эффект не следует из нормы
_CAUSALITY_LAW_EFFECT = re.compile(
    r"(?:289-ФЗ|закон[ауеи]?\s+о\s+платформенн\w+|768-ФЗ|ПП\s*№?\s*768)"
    r"[^.\n]{0,160}?(?:изменил\w*|перестроил\w*|повлиял\w*\s+на|поменял\w*)"
    r"[^.\n]{0,80}?(?:тариф\w*|комисси\w*|маржу|юнит|экономик\w*|логистик\w*|"
    r"расход\w+\s+продавца|расчёт)", re.I)
_CAUSALITY_OK = re.compile(
    r"могут\s+пересматривать|не\s+автоматически|сам\w+\s+по\s+себе\s+не\s+меня|"
    r"проверьте\s+фактические\s+тарифы|регулярно\s+обновля|обновляйте\s+юнит", re.I)
# MARKETPLACE_TAX_MODE_SPLIT: несколько режимов + юнит-таблица без развилки
_TAX_MODES_RX = re.compile(r"УСН\s*6|НДС\s*5\s*%|НДС\s*22|ОСНО", re.I)
_UNIT_TABLE_RX = re.compile(
    r"юнит|SKU|на\s+единицу\s+товара|марж\w+\s+карточк|P&L", re.I)
# ORDER_VS_BUYOUT: ДРР на заказы vs маржа на выкуп — знаменатели разные
_ORDER_BUYOUT_RX = re.compile(
    r"(?:реклам\w+|ДРР)[^.\n]{0,80}?(?:заказ\w+|на\s+заказ)"
    r"|(?:выкуп\w*)[^.\n]{0,80}(?:маржа|прибыль|экономик)", re.I)
# SKU_VS_PORTFOLIO: общие расходы в юнит-формуле без маркировки типа затрат
_PORTFOLIO_COST_RX = re.compile(
    r"(?:аренд\w+|зарплат\w+\s+офис\w*|подписк\w+|общ\w+\s+аналитик)"
    r"[^.\n]{0,80}(?:юнит|SKU|единиц\w+|карточк)", re.I)

# ── Ревью 28 (2026-10-03, 6.5/10): нацрежим, усечение, право vs автоматизм ──
# TRUNCATED_CONTENT: абзац оборван на союзе/союзном слове
_TRUNCATED_ENDING = re.compile(
    r"(?:однако|вторая\s+ошибка\s*[—–—])\s*$", re.I | re.M)
# DISCRETION_VS_AUTOMATIC: право ≠ автоматическое применение
_DISCRETION_AUTO = re.compile(
    r"(?:порог[-\s]*исключение[^.]{0,60}действует|"
    r"запрет\s+не\s+применяется\s+автоматически|"
    r"исключение\s+действует\s+если|"
    r"запрет\s+отменяется)", re.I)
# «российский товар» без упоминания ЕАЭС в контексте нацрежима
_NATREGIME_CTX = re.compile(r"1875|национальн\w+\s+режим|нацрежим", re.I)
_RUSSIAN_ONLY = re.compile(
    r"(?:российск\w+\s+товар\w*|товар\w+\s+российск\w+)", re.I)
_EAEU_MENTION = re.compile(r"ЕАЭС|евразийск", re.I)
# PRODUCT_CLASSIFICATION_MULTIFACTOR: ОКПД2 совпал → мера применяется
_OKPD2_AUTO = re.compile(
    r"(?:если\s+)?(?:код\s+)?ОКПД2?\s*(?:в\s+перечне|совпадает|"
    r"входит\s+в\s+перечень)[^.\n]{0,40}(?:мера\s+применяется|"
    r"применя\w+\s+запрет|применя\w+\s+ограничени)", re.I)

# ── Ревью 28 (2026-10-03, 8/10): нацрежим, мех-мы, прил.3, ст.14 vs 33 ──
# MECHANISM_DESCRIPTION_CHECK: «ограничение = понижение при оценке» —
# это преимущество, а не ограничение
_MECH_DESC_WRONG = re.compile(
    r"(?:ограничени\w+|запрет)[^.\n]{0,80}"
    r"(?:понижени\w+\s+при\s+оценк|получают\s+понижени|"
    r"расчётн\w+\s+снижени\w+\s+ценов\w+)", re.I)
# APPENDIX_3_NOT_ADMISSION: прил.3 назван «мерой допуска»
_APPENDIX3_ADMISSION = re.compile(
    r"(?:мера\s+допуска|мер[аы]\s+допуск\w*)[^.\n]{0,40}"
    r"(?:приложени\w+\s*№?\s*3|прил\.?\s*№?\s*3)", re.I)
# MEDICAL_DEVICES_UNIVERSAL_RULE: «для медизделий используется СТ-1»
_MED_DEVICES_ST1 = re.compile(
    r"[Дд]ля\s+медицинских\s+изделий\s+используется\s+[^.]{0,60}документ", re.I)
# ARTICLE_14_VS_33: ст.14 для описания объекта закупки
_ART14_DESCRIPTION = re.compile(
    r"[Сс]татья\s+14[^.\n]{0,80}(?:предписывает|указывать|"
    r"характеристик)", re.I)
# PRIMARY_SOURCE_ONLY: поисковые редиректы в источниках
_SEARCH_REDIRECT = re.compile(
    r"vertexaisearch|google\.com/url\?|yandex\.ru/click|"
    r"redirect\.|search\?q=", re.I)
# LEGAL_CERTAINTY_GATE: хеджирующие слова рядом с нормой права
_HEDGE_NEAR_NORM = re.compile(
    r"(?:вероятно|обычно|по\s+вторичн\w+\s+источник|"
    r"требует\s+юрпроверки|по\s+имеющимся\s+данным|"
    r"в\s+открыт\w+\s+источник\w+\s+различа\w*)[^.\n]{0,100}?"
    r"(?:ст\.?\s*\d|штраф|санкци|статья\s+\d|срок|давност)", re.I)
# обратный порядок: норма, потом хедж в том же предложении
_HEDGE_AFTER_NORM = re.compile(
    r"(?:ст\.?\s*\d+\.\d+|штраф\s+\d|ч\.\s*\d+\s+ст)[^.\n]{0,80}?"
    r"(?:вероятно|обычно|по\s+вторичн\w+|требует\s+юрпроверки)", re.I)
# ARTICLE_RANGE_SANITY: невозможные диапазоны нумерации
_BAD_ARTICLE_RANGE = re.compile(
    r"стать\w+\s+1\s*[—–-]\s*7\.3\d|стать\w+\s+\d+\s*[—–-]\s*7\.32\.5"
    r"|стать\w+\s+44\s*[—–-]\s*105|глав\w+\s+1\s*[—–-]\s*10", re.I)
# STAGE_ARTICLE_MISMATCH: выбор способа поставщика ≠ 7.30.2
_STAGE_ARTICLE_MISMATCH = re.compile(
    r"(?:выбор\s+способ\w+|способ\s+определени\w+\s+поставщик)"
    r"[\s\S]{0,120}?ст\.?\s*7\.30\.2|"
    r"ст\.?\s*7\.30\.2[\s\S]{0,120}?"
    r"(?:выбор\s+способ\w+|способ\s+определени\w+\s+поставщик)", re.I)
# PART_TOPIC_MISMATCH: неверная привязка части к предмету (ревью 27v3)
# Инварианты сверенны с текстом КоАП РФ (снимок 2026-10-03)
# ч.1 ст.7.30.1 = планирование, НЕ НМЦК; ч.2 = нормирование, НЕ НМЦК
# ч.11 = СМП/СОНКО, НЕ планирование; ч.10 ст.7.30.2 = дисквалификация
_PART1_WRONG_TOPIC = re.compile(
    r"ч\.?\s*1[^.\n]{0,60}(?:НМЦК|обосновани\w+\s+НМЦК)", re.I)
_PART2_WRONG_TOPIC = re.compile(
    r"ч\.?\s*2[^.\n]{0,60}(?:НМЦК|обосновани\w+\s+НМЦК)", re.I)
_PART4_WRONG_TOPIC = re.compile(
    r"ч\.?\s*4[^.\n]{0,60}НМЦК", re.I)
_PART11_WRONG_TOPIC = re.compile(
    r"ч\.?\s*11[^.\n]{0,60}(?:планировани|иные\s+нарушени)", re.I)
_PART10_PENALTY = re.compile(
    r"ч\.?\s*10[^.\n]{0,40}ст\.?\s*7\.30\.2[^.\n]{0,80}(?:штраф|1%\s*НМЦК)"
    r"|ч\.?\s*10[^.\n]{0,40}1%\s*НМЦК", re.I)
_SEARCH_REDIRECT = re.compile(
    r"vertexaisearch|google\.com/url\?|yandex\.ru/click|"
    r"redirect\.|search\?q=", re.I)
# LAW_REGIME_CONSISTENCY: 223-ФЗ в лиде при 44-ФЗ в теле
_LEAD_223 = re.compile(
    r"(?:закупк\w+|ошибк\w+|нарушени\w+)[^.\n]{0,80}по\s*223-ФЗ", re.I)
# SANCTION_TABLE_NO_PLACEHOLDERS: «уточняется» в таблице штрафов
_SANCTION_PLACEHOLDER = re.compile(
    r"(?:\|\s*)?уточняется(?:\s*\|)|"
    r"(?:\|\s*)?по\s+соответствующ\w+\s+част\w+\s*(?:\|)", re.I)
# ARTICLE_PART_MERGE_GUARD: несколько частей с одной санкцией
_MULTI_PART_SANCTION = re.compile(
    r"ч\.?\s*\d+\s*,\s*ч\.?\s*\d+[^.\n]{0,60}(?:штраф|санкци|составля\w+)", re.I)
# PROCEDURAL_ROUTE: широкое «остальные — ФАС»
_BROAD_APPEAL = re.compile(
    r"(?:остальн\w+|все\s+остальн\w*)[^.\n]{0,40}(?:ФАС|рассматривают)",
    re.I)

# ── Ревью 27 (2026-10-03, 6/10): санкции без сумм, заголовок-обещание ──
# NO_APPROXIMATE_SANCTIONS: приблизительные штрафы в юридической статье
_APPROX_SANCTION = re.compile(
    r"(?:по\s+предварительным\s+оценк\w+|точная\s+сумма\s+требует\s+"
    r"проверки|может\s+грозить\s+(?:от|до)\s*\d+[^\n]{0,40}(?:руб|₽)"
    r"|штраф\s*,\s*точная)", re.I)
_SANCTION_OK = re.compile(
    r"ст\.?\s*\d+\s*ч\.?\s*\d+|ч\.?\s*\d+\s*ст\.?|"
    r"предупреждение\s+или\s+штраф|от\s+\d[\d\s.,]*\s+до\s+"
    r"\d[\d\s.,]*\s*(?:руб|₽)", re.I)
# TITLE_SCOPE_CHECK: заголовок обещает два субъекта
_DUAL_SUBJECT_RX = re.compile(
    r"[Дд]ля\s+(?:заказчика\s+и\s+поставщика|поставщика\s+и\s+заказчика|"
    r"работника\s+и\s+работодателя|работодателя\s+и\s+работника|"
    r"ИП\s+и\s+ООО|ООО\s+и\s+ИП)", re.I)

# ── Ревью 25v2 (2026-10-02, 8/10) ──
# ОСНО-строка в юнит-статье маркетплейса: необъяснимый расчёт, юрриск
# (налог в P&L SKU — параметр модели, сравнение режимов — отдельная тема)
# НДС-сравнение 5%/22% без статуса цены (включает/сверх/расчётная ставка)
_VAT_COMPARE = re.compile(
    r"НДС\s*5\s*%[^|\n]{0,200}НДС\s*22|НДС\s*22[^|\n]{0,200}НДС\s*5\s*%", re.I)
_PRICE_VAT_STATUS = re.compile(
    r"включа\w+\s+НДС|сверх\s+цены|цена\s+без\s+НДС|цена\s+с\s+НДС|"
    r"выдел\w+\s+НДС|расч[её]тн\w+\s+ставк|22/122", re.I)
# ── Ревью 24v5 (2026-10-02, 7/10): контаминация 59-ФЗ → срок жалобы ──
# «не позднее 30 календарных дней ... размещения протокола» — 30 дней это
# срок обращения гражданина по 59-ФЗ, не жалобы по ч. 2 ст. 105 (5 дней)
_APPEAL_30D = re.compile(
    r"не\s+позднее\s+30\s*(?:календарн\w+\s*)?дн\w*|"
    r"в\s+течени[еи]\s+30\s*(?:календарн\w+\s*)?дн\w+", re.I)
_APPEAL_30D_CTX = re.compile(r"протокол|итог\w*|жалоб\w+|обжалова\w+", re.I)
_APPEAL_30D_OK = re.compile(
    r"59-ФЗ|обращени\w+\s+граждан\w*|не\s+явля[eю]тся\s+участником\w*|"
    r"рассматрива\w+\s+30|заявлени\w+\s+граждан", re.I)
# ── Ревью 26 (2026-10-02, 6.5/10) ──
# LEGAL_NUMERIC_TOKEN: повреждённый числовой токен — «22 НМЦК» вместо
# «20% НМЦК». Голое число перед НМЦК недопустимо (валидно: «10% НМЦК»,
# «20% НМЦК», «от НМЦК»)
_DAMAGED_NUM_TOKEN = re.compile(r"(?<![%\d])\b\d{1,3}\s+НМЦК", re.I)
# SPECIAL_REGIME_BRANCH: спецрежим упомянут + антидемпинг, а ветки нет
_SPECIAL_REGIME = re.compile(
    r"СМП|СОНКО|МСП|УИС|организац\w+\s+инвалид\w*|аванс\w*|"
    r"неопредел[её]нн\w+\s+объ[её]м\w*", re.I)
_SPECIAL_REGIME_OK = re.compile(
    r"отдельн\w+\s+(?:правил|ветк|случа|расчёт|таблица)|специальн\w+\s+"
    r"правил|для\s+закупок\s+у\s+МСП|п\.?\s*1\s*ч\.?\s*1\s*ст\.?\s*30|"
    r"проверьте\s+(?:вид|условия|статус)", re.I)
# Выдуманный универсальный порог («от 30 дней — достаточно» / оба порядка)
_FAIR_THRESHOLD = re.compile(
    r"(?:от|через|после)\s+\d+\s+дн\w+[^.\n]{0,60}?"
    r"(?:достаточн|устойчив|над[её]жн|коррект)"
    r"|(?:достаточн|устойчив|над[её]жн|коррект)[^.\n]{0,60}?"
    r"(?:от|через|после)\s+\d+\s+дн\w+", re.I)

# P1: внутренний контроль подан как норма («проверяем не менее 10% документов»)
_INTERNAL_CTRL_AS_LAW = re.compile(
    r"(?:проверяем|проверк[ау]|контролируем|выборочн\w+)[^.\n]{0,40}?не\s+менее\s+\d+\s*%", re.I)

# P1: советы в стиле маскировки трудовых отношений при ГПХ
_GPH_DISGUISE = re.compile(
    r"(?:убираем|убрать|уберём|уберем|скрываем|изымаем|отбираем|аннулируем)\s+"
    r"[^.\n]{0,50}?(?:корпоративн\w+\s+почт|пропуск\w*|"
    r"закрепл[её]нн\w+\s+рабочее\s+место|рабочее\s+место|ноутбук\w*|"
    r"служебн\w+\s+техник\w+|технику)", re.I)


# Конструкция «X, а не X» — одинаковые значения в противопоставлении
_SAME_VALUE_CONTRAST = re.compile(
    r"(\d+[.,]?\d*\s*(?:%|млн|руб|₽)?)\s*(?:,\s*а\s+не|—\s*не)\s+(\d+[.,]?\d*\s*(?:%|млн|руб|₽)?)", re.I)


def _check_same_contrast(text: str) -> Optional[str]:
    """Найти «X, а не X» — когда модель вставила одинаковые значения."""
    for m in _SAME_VALUE_CONTRAST.finditer(text):
        left = m.group(1).strip().replace(" ", "").replace(",", ".")
        right = m.group(2).strip().replace(" ", "").replace(",", ".")
        if left == right:
            return m.group(0)
    return None


# Неподтверждённая статистика ведомства («по данным ФНС — главная причина»)
_OFFICIAL_STAT_NO_SRC = re.compile(
    r"(?:по\s+данн\w+\s+(?:ФНС|Минфин|Росстат|обзор\w+)|главная\s+причин[аы]|"
    r"самая\s+частая\s+причин[аы]|чаще\s+всего\s+(?:возникает|всплывает|причиняет))", re.I)


# Промпт-лик: инструкции агентов попали в текст статьи
_PROMPT_LEAK = re.compile(
    # ФИКС (аудит 🟡1): однословные «исправь/перепиши/сгенерируй» ловили
    # легитимные советы («исправьте декларацию») — оставлены только
    # безошибочно-промптовые конструкции
    r"(?:Вот\s+несколько\s+вариантов|предлагаю\s+следующ\w+|варианты\s+подзаголовк|"
    r"вот\s+черновик|перепиши\s+(?:этот|стать|текст)|редактура:|SEO-оптимизац|"
    r"цитата\s+клиента:|Citation\s+Bait|LSI-ключ|наживка|"
    r"сгенерируй\s+(?:мне|стать|текст|вариант)|напиши\s+(?:мне\s+)?(?:текст|статью)\s+"
    r"про|создай\s+статью)", re.I)


def fix_table_of_contents(text: str) -> tuple:
    """
    Исправить блок «Содержание»: Heart иногда оставляет пустой заголовок
    без списка, или генерирует H2 вместо маркированного списка.
    Автоматически генерирует содержание из реальных H2-заголовков.
    Возвращает (исправленный_текст, было_исправлено).
    """
    if not text or "## Содержание" not in text:
        return text, False

    lines = text.split('\n')
    toc_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '## Содержание':
            toc_idx = i
            break

    if toc_idx is None:
        return text, False

    # Найти конец блока Содержание (следующий H2)
    next_h2 = None
    for i in range(toc_idx + 1, len(lines)):
        if lines[i].strip().startswith('## ') and 'Содержание' not in lines[i]:
            next_h2 = i
            break

    # Собрать все H2-заголовки (кроме «Содержание» и «Источники»)
    h2_titles = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('## ') and 'Содержание' not in stripped and 'Источник' not in stripped:
            title = stripped.lstrip('# ').strip()
            if title:
                h2_titles.append(title)

    if not h2_titles:
        return text, False

    # Проверить: есть ли маркированный список после «Содержание»
    has_list = False
    if next_h2:
        for i in range(toc_idx + 1, next_h2):
            if lines[i].strip().startswith('- ') or lines[i].strip().startswith('* '):
                has_list = True
                break

    if has_list:
        return text, False  # Список уже есть

    # Сгенерировать маркированный список
    toc_lines = ['## Содержание', '']
    for title in h2_titles:
        toc_lines.append(f'- {title}')
    toc_lines.append('')

    # Заменить блок от «## Содержание» до следующего H2
    end = next_h2 if next_h2 else toc_idx + 1
    new_lines = lines[:toc_idx] + toc_lines + lines[end:]
    return '\n'.join(new_lines), True


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
        wide = text[max(0, m.start() - 1000):m.end() + 1000]
        if _MODEL_MARKER.search(wide):
            continue
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
    # Доменная контаминация: налоговое в маркетинговой статье (кейс 2026-09-28)
    for m in list(_TAX_CONTAMINATION.finditer(text))[:2]:
        frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
        issues.append(f"🔴 НАЛОГОВАЯ ВРЕДИНА: «{frag}» в маркетинговой статье — "
                      f"налоговая классификация требует профильной проверки. "
                      f"Уберите или замените: «налоговый учёт рекламных расходов "
                      f"требует отдельной консультации»")
    # GREY_ZONE_ALERT: схема похожа на уклонение (аудит: категория «как законно сэкономить»)
    for m in list(re.finditer(
            r"(?:дроблени\w+\s+(?:бизнес\w*|компани\w*|выручк\w*)|"
            r"фиктивн\w+\s+(?:договор|сделк|документ)|"
            r"вывод\w+\s+(?:актив\w+|доход\w+)\s+(?:в|на)\s+(?:офшор|зарубежн)|"
            r"перевод\w+\s+сотрудник\w+\s+на\s+договор\w+\s+ГПХ\s+"
            r"без\s+(?:реальн\w+|факт\w+))", text, re.I))[:2]:
        frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
        issues.append(f"🔴 GREY ZONE: «{frag}» — схема имеет признаки уклонения "
                      f"от налогов. Обязательны: предупреждение о рисках "
                      f"(доначисления, штрафы, дробление), ссылка на судебную "
                      f"практику, формулировка «требует индивидуальной правовой "
                      f"оценки». Без этого — не публиковать")
    # RISK_DISCLOSURE: обещание экономии без предупреждения о рисках
    for m in list(re.finditer(
            r"(?:законно\s+снизить|законн\w+\s+(?:сэкономить|оптимизироват\w+)|"
            r"снизить\s+налог[аи]|экономия\s+(?:на\s+)?налог)", text, re.I))[:2]:
        window = text[max(0, m.start() - 300):m.end() + 500]
        if not re.search(r"риск|предупреждени|внимани|осторожн|не\s+"
                         r"гарантир|консультац\w+\s+(?:юрист|налогов)", window, re.I):
            frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
            issues.append(f"🟡 RISK DISCLOSURE: «{frag}» — обещание налоговой "
                          f"экономии без предупреждения о рисках. Добавьте "
                          f"оговорку о необходимости консультации и рисках")
            break
    # CASE_NOT_GUARANTEE: чужой результат кейса подан как обещание читателю
    for m in list(re.finditer(
            r"(?:получит\w+|выраст\w+|увеличит\w+|подним\w+|даст|дадут|"
            r"обеспечит\w*)\s+(?:на|до)\s+\d+(?:[.,]\d+)?\s*%", text, re.I))[:2]:
        frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
        issues.append(f"🔴 КЕЙС НЕ ГАРАНТИЯ: «{frag}» — результат чужого кейса "
                      f"нельзя обещать читателю. Замените на «в кейсе N "
                      f"получили X при таких-то условиях» или уберите обещание")
    # CASE_CONTEXT_REQUIRED: кейс-цифра без периода/условий в окне
    for m in list(re.finditer(r"кейс\w*[^.!?]{0,220}?\+\d+(?:[.,]\d+)?\s*%", text, re.I))[:2]:
        window = text[max(0, m.start() - 200):m.end() + 200]
        if not re.search(r"20\d\d|период|за\s+(?:год|месяц|квартал)|"
                         r"канал|сегмент|ограничени|методик", window, re.I):
            frag = text[max(0, m.start() - 10):m.end() + 30].replace("\n", " ")
            issues.append(f"🟡 КОНТЕКСТ КЕЙСА: «{frag}» — цифра кейса без периода, "
                          f"канала или механики. Добавьте: компания, период, канал, "
                          f"методику и ограничения (чужой результат не прогноз)")
    # EMAIL_CONSENT_REQUIRED: материал про рассылки без согласия/отписки
    if len(re.findall(r"рассылк\w+|email-цепочк\w+|e-mail-цепочк\w+|"
                      r"email-рассылк\w+", text, re.I)) >= 3:
        if not re.search(r"согласи\w+|отписк\w+|спам|персональн\w+ данны\w+|"
                         r"152-ФЗ", text, re.I):
            issues.append("🟡 СОГЛАСИЕ В РАССЫЛКАХ: материал про email-цепочки "
                          "без упоминания согласия, отписки и законности базы. "
                          "Добавьте блок о правовых основаниях рассылок")
    # OPEN_RATE_NOT_PRIMARY: open rate объявлен главной метрикой (оба порядка слов)
    for m in list(re.finditer(
            r"(?:(?:главн\w+|основн\w+|ключев\w+)\s+(?:метрик\w+|показател\w+)\s*"
            r"(?:\w+\s+){0,3}?(?:—|-|это)?\s*(?:open\s*rate|открываемост\w+)|"
            r"(?:open\s*rate|открываемост\w+)\s*(?:—|-|это)\s*"
            r"(?:главн\w+|основн\w+|ключев\w+))", text, re.I))[:1]:
        frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
        issues.append(f"🟡 OPEN RATE НЕ ГЛАВНЫЙ: «{frag}» — открываемость искажается "
                      f"почтовыми клиентами. Главная метрика — квалифицированные "
                      f"лиды и экономика, open rate — диагностика доставки")
    # AI_JARGON: технический жаргон в бизнес-статье (словарь ai_for_business)
    for m in list(re.finditer(
            r"\bRAG\b|embedding\w*|chunking|\bMCP\b|\bLLM\b|"
            r"промпт[- ]инжинир\w*|prompt[- ]engineering|"
            r"векторн\w+\s+(?:баз|поиск|хранилищ)|reranking|реранкинг|"
            r"fine-?tuning|\bHITL\b", text, re.I))[:2]:
        window = text[max(0, m.start() - 150):m.end() + 200]
        if not re.search(r"то\s+есть|проще\s+говоря|иными\s+словами|"
                         r"это\s+значит", window, re.I):
            frag = text[max(0, m.start() - 15):m.end() + 25].replace("\n", " ")
            issues.append(f"🟡 AI-ЖАРГОН: «{frag}» — переведи в бизнес-язык "
                          f"(словарь ai_for_business) или объясни одним "
                          f"предложением и свяжи с эффектом")
    # SEGMENT_BEFORE_SEQUENCE: универсальная цепочка без сегмента
    for m in list(re.finditer(
            r"(?:универсальн\w+|един\w+|общ\w+)\s+(?:цепочк\w+|сери\w+\s+писем|"
            r"воронк\w+)\s+(?:для\s+всех|без\s+сегментац\w+)", text, re.I))[:1]:
        frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
        issues.append(f"🟡 СЕГМЕНТАЦИЯ ПЕРВИЧНА: «{frag}» — цепочка задаётся "
                      f"сегментом, целью и стоп-условиями. Универсальные серии "
                      f"писем «для всех» не описываем")
    # MARKETING_ABSOLUTE_RULE: эвристика подачи как универсальный закон (ревью 29)
    for m in list(re.finditer(
            r"(?:B2B|b2b)[^.!?]{0,60}(?:не\s+покупа\w+|покупа\w+\s+не)[^.!?]{0,30}"
            r"(?:эмоци|настроени|чувств)|"
            r"(?:на\s+первом\s+экране|на\s+странице)[^.!?]{0,50}"
            r"(?:обязательн\w+|должн\w+)\s+(?:быть\s+)?(?:цен\w+|вилк\w+|прайс)|"
            r"(?:обязательн\w+|всегда)\s+(?:нужн\w+|должн\w+|публикуйте|"
            r"показывайте|ставьте|оставляйте)[^.!?]{0,30}(?:цен|прайс|один\s+CTA|"
            r"одна\s+кнопка)|"
            r"форма\s+из\s+\d+\s+полей\s+(?:снижает|убивает|уменьшает)|"
            r"один\s+CTA\s+(?:всегда|обязательн\w+)", text, re.I))[:2]:
        frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
        issues.append(f"🟡 КАТЕГОРИЧНОЕ ПРАВИЛО: «{frag}» — маркетинговая "
                      f"эвристика подана как универсальный закон. Смягчите: "
                      f"«часто», «в ряде B2B-сценариев», «если услуга "
                      f"стандартизирована», «это гипотеза для проверки»")
    # FUNNEL_STAGE_SEPARATION: выводы через смешение уровней воронки (ревью 29)
    for m in list(re.finditer(
            r"(?:нет|мало)\s+(?:прода\w*|сделок|оплат)[^.!?]{0,50}значит"
            r"[^.!?]{0,40}(?:лендинг|сайт|реклама|форма)|"
            r"есть\s+клик\w*[^.!?]{0,40}значит[^.!?]{0,30}(?:есть\s+)?спрос|"
            r"много\s+заявок[^.!?]{0,40}значит[^.!?]{0,40}реклама\s+эффективна|"
            r"нет\s+заявок[^.!?]{0,40}значит[^.!?]{0,40}форма\s+"
            r"(?:сломана|не\s+работает)", text, re.I))[:2]:
        frag = text[max(0, m.start() - 20):m.end() + 30].replace("\n", " ")
        issues.append(f"🟡 СМЕШЕНИЕ УРОВНЕЙ ВОРОНКИ: «{frag}» — трафик, "
                      f"поведение, заявка, квалифицированный лид и сделка "
                      f"разные уровни: у наблюдения перечислите альтернативные "
                      f"причины, прежде чем делать вывод")
    # QUALITATIVE_SIGNALS_NOT_PROOF: вебвизор/карты как «доказательство» (ревью 29)
    for m in list(re.finditer(
            r"точк\w+\s+отвала|точную\s+причину\s+(?:отказа|ухода)", text, re.I))[:2]:
        window = text[max(0, m.start() - 350):m.end() + 250]
        if re.search(r"карт[аы]\s+(?:кликов|скроллинга|поведения)|вебвизор|"
                     r"теплов\w+\s+карт\w+|UX-аудит", window, re.I) and not re.search(
                r"гипотез|предположен|не\s+доказыва|причин\w+\s+несколько|"
                r"провер\w+\s+(?:A/B|тест|на\s+данных)", window, re.I):
            frag = text[max(0, m.start() - 25):m.end() + 30].replace("\n", " ")
            issues.append(f"🟡 ДИАГНОСТИКА НЕ ДОКАЗАТЕЛЬСТВО: «{frag}» — "
                          f"вебвизор и карты дают гипотезы о причине отвала, "
                          f"а не доказательство; назовите альтернативные "
                          f"причины и проверьте гипотезу A/B-тестом")
    # TITLE_PROMISE_CHECK: заголовок обещает N пунктов/чек-лист (ревью 29)
    _h1 = re.search(r"^#\s+(.+)$", text, re.M)
    if _h1:
        _n = re.search(r"\b(\d+)\s+(?:причин|ошибки|ошибок|способов|шагов|"
                       r"советов|приёмов|правил)", _h1.group(1), re.I)
        if _n:
            _h2_titles = re.findall(r"^##\s+([^\n]+)", text, re.M)
            # служебные секции (чек-лист, выводы, источники) не считаются пунктами
            _h2_count = sum(
                1 for h in _h2_titles if not re.search(
                    r"чек-?лист|самопроверк|послеслови|источник|вывод|итог|FAQ",
                    h, re.I))
            if int(_n.group(1)) != _h2_count:
                issues.append(f"🟡 ОБЕЩАНИЕ ЗАГОЛОВКА: заголовок обещает "
                              f"{_n.group(1)} пунктов, в тексте {_h2_count} "
                              f"секций-пунктов — сделайте явную нумерацию и "
                              f"равное количество")
        if re.search(r"чек-?лист", _h1.group(1), re.I) and not re.search(
                r"^##[^#\n]*чек-?лист", text, re.I | re.M) and not re.search(
                r"^\s*[-*]\s+\[", text, re.M):
            issues.append("🟡 ОБЕЩАНИЕ ЗАГОЛОВКА: заголовок обещает чек-лист, "
                          "но явного чек-листа (секция или чекбоксы) в тексте "
                          "нет — добавьте финальный чек-лист")
    # LEGAL_SCOPE_AND_EXCEPTIONS_CHECK: норма без условий применимости (ревью 30)
    _LEGAL_MARK = re.compile(
        r"закон\w*|ФЗ|КоАП|ст\.\s*\d|штраф|ЕРИР|erid|ФАС|Роскомнадзор", re.I)
    for m in list(re.finditer(
            r"(?:вс\w+|люб\w+|кажд\w+)\s+(?:реклам\w+|иностранн\w+\s+слов\w*|"
            r"публикаци\w+|пост\w+)[^.!?]{0,60}(?:должн\w+|запрещен\w*|"
            r"требу\w+|обязательн\w+|подлежит\w*|риск)|"
            r"кажд\w+\s+иностранн\w+\s+слов\w+[^.!?]{0,30}риск|"
            r"переходн\w+\s+период[^.!?]{0,80}(?:не\s+влеч\w+\s+штраф\w*|"
            r"без\s+штраф\w*|разрешен\w+)|"
            r"(?:Telegram|YouTube|WhatsApp|VPN)\w*[^.!?]{0,90}"
            r"не\s+влеч\w+\s+штраф\w*", text, re.I))[:3]:
        window = text[max(0, m.start() - 300):m.end() + 300]
        if _LEGAL_MARK.search(window) and not re.search(
                r"исключени\w+|товарн\w+ знак\w*|потребительск\w+|"
                r"к\s+кому\s+применяется|при\s+каких\s+услови", window, re.I):
            frag = text[max(0, m.start() - 25):m.end() + 35].replace("\n", " ")
            issues.append(f"🔴 LEGAL SCOPE: «{frag}» — норма подана как "
                          f"универсальная, без условий применимости и "
                          f"исключений. Добавьте: к кому применяется, при "
                          f"каких условиях, исключения (товарные знаки, "
                          f"потребительская информация, статус площадки), "
                          f"источник и дату проверки")
    # ADMIN_LIABILITY_MATRIX_REQUIRED: штрафы без состава/субъекта/органа (ревью 30)
    for m in list(re.finditer(
            r"(?:её|его|их)?\s*(?:постановление|решение)\s+(?:ФАС|"
            r"Роскомнадзор\w*)?\s*запускает\s+финансов\w*|"
            r"(?:ФАС|Роскомнадзор\w*)\s+запускает\s+финансов\w*", text, re.I))[:2]:
        frag = text[max(0, m.start() - 25):m.end() + 40].replace("\n", " ")
        issues.append(f"🔴 МАТРИЦА ОТВЕТСТВЕННОСТИ: «{frag}» — смешение "
                      f"состава, органа и процедуры. Либо заполните матрицу "
                      f"(состав, часть КоАП, субъект, кто составляет протокол "
                      f"и рассматривает, санкция, дата проверки, источник), "
                      f"либо уберите связь «орган → деньги»")
    if re.search(r"штраф", text, re.I) and _LEGAL_MARK.search(text):
        has_part = re.search(r"ч\.\s*\d+\s*ст\.\s*\d+|ст\.\s*\d+(?:\.\d+)?\s*"
                             r"ч\.\s*\d+", text)
        has_source = re.search(r"^#{1,3}\s*источник", text, re.I | re.M) or \
            re.search(r"дата\s+проверки|проверено\s+на\s+дату", text, re.I)
        if not has_part or not has_source:
            issues.append("🟡 МАТРИЦА ОТВЕТСТВЕННОСТИ: текст обещает штрафы, "
                          "но без привязки к конкретной части статьи КоАП и "
                          "даты проверки источника. Либо точная матрица "
                          "(состав, субъект, орган, санкция, дата, источник), "
                          "либо уберите суммы и штрафные обещания из "
                          "заголовка и текста")
    # LEGAL_TITLE_SCOPE_CHECK: юр-обещание заголовка vs доказательная база (ревью 30)
    if _h1 and re.search(
            r"штраф|запрет|как\s+не\s+получить|пошагов\w+\s+аудит|"
            r"юридическ\w+\s+чек|все\s+правила", _h1.group(1), re.I):
        has_part = re.search(r"ч\.\s*\d+\s*ст\.\s*\d+|ст\.\s*\d+(?:\.\d+)?\s*"
                             r"ч\.\s*\d+", text)
        if not has_part:
            issues.append("🔴 ЮРИДИЧЕСКОЕ ОБЕЩАНИЕ ЗАГОЛОВКА: заголовок "
                          "гарантирует штрафы/запреты/аудит, но в тексте нет "
                          "ни одной конкретной части статьи КоАП. Сузьте "
                          "заголовок («предварительная проверка…») либо "
                          "дайте точную матрицу составов с датой проверки")
    # Бенчмарки без источника
    for m in list(_BENCHMARK_NO_SRC.finditer(text))[:3]:
        wide = text[max(0, m.start() - 1000):m.end() + 1000]
        if _MODEL_MARKER.search(wide):
            continue
        frag = text[max(0, m.start() - 10):m.end() + 20].replace("\n", " ")
        issues.append(f"🟡 БЕНЧМАРК БЕЗ ИСТОЧНИКА: «{frag}» — рыночная цифра "
                      f"без исследования. Уберите или маркируйте как "
                      f"«ориентир для тестового планирования»")
    # Противопоставление с одинаковыми значениями («22%, а не 22%»)
    same = _check_same_contrast(text)
    if same:
        issues.append(f"🔴 ОДИНАКОВЫЕ ЗНАЧЕНИЯ В ПРОТИВОПОСТАВЛЕНИИ: «{same}» — "
                      f"проверьте цифры. Например «22%, а не 22%» должно быть «22%, а не 20%»")
    # Промпт-лик (кейс: «Вот несколько вариантов подзаголовка...» — 2026-09-29)
    for m in list(_PROMPT_LEAK.finditer(text))[:2]:
        frag = text[max(0, m.start() - 10):m.end() + 30].replace("\n", " ")
        issues.append(f"🔴 ПРОМПТ-ЛИК: «{frag}» — инструкция агента попала в текст. "
                      f"Удалите этот блок полностью.")
    # Рыночные цифры без источника (кейс: «65,8 трлн ₽» — 2026-09-28)
    for m in list(_MARKET_SIZE_NO_SRC.finditer(text))[:2]:
        frag = text[max(0, m.start() - 15):m.end() + 20].replace("\n", " ")
        issues.append(f"🔴 РЫНОЧНАЯ ЦИФРА БЕЗ ИСТОЧНИКА: «{frag}» — трлн без "
                      f"методологии/определения рынка/периода. Уберите или дайте источник.")
    # Коммерческие абсолюты
    for m in list(_COMMERCIAL_ABSOLUTE.finditer(text))[:2]:
        issues.append(f"🟡 КОММЕРЧЕСКИЙ АБСОЛЮТ: «{m.group(0)}» — слишком сильное "
                      f"обещание. Замените на: «в отдельных вертикалях есть пространство»")
    # Прогнозные цифры без маркировки (кейс: «CPC 75-120 ₽» для 2027 — 2026-09-28)
    for m in list(_FORECAST_NO_MARK.finditer(text))[:3]:
        wide = text[max(0, m.start() - 1000):m.end() + 1000]
        if _MODEL_MARKER.search(wide):
            continue
        frag = text[max(0, m.start() - 10):m.end() + 20].replace("\n", " ")
        issues.append(f"🟡 ПРОГНОЗ БЕЗ МАРКИРОВКИ: «{frag}» — рыночная цифра для "
                      f"будущего периода. Добавьте: «условный сценарий», "
                      f"«ориентир для планирования» или источник")
    # Драматизация
    for m in list(_DRAMATIC.finditer(text))[:2]:
        frag = text[max(0, m.start() - 15):m.end() + 20].replace("\n", " ")
        issues.append(f"🟡 ДРАМАТИЗАЦИЯ: «{frag}» — экспертная статья не должна "
                      f"использовать маркетинговые метафоры как факт")
    # P0: прогноз будущего года как факт (редактура L1, 2026-09-29)
    for m in list(_FORECAST_YEAR_FACT.finditer(text))[:3]:
        sent = text[max(0, m.start() - 60):m.end() + 80]
        if not _FORECAST_MARKERS.search(sent):
            frag = m.group(0)[:70].replace("\n", " ")
            issues.append(f"🔴 ПРОГНОЗ КАК ФАКТ: «{frag}…» — утверждение о будущих "
                          f"цифрах без маркировки. Добавьте «по прогнозу/оценке» "
                          f"и источник, либо уберите цифру")
    # P0: условный кейс читается как реальный (имя + цифры без маркировки)
    for m in list(_CASE_NAMES.finditer(text))[:4]:
        window = text[max(0, m.start() - 150):m.end() + 250]
        if _CASE_DIGITS.search(window) and not _CASE_MARKERS.search(window):
            issues.append(f"🔴 КЕЙС БЕЗ МАРКИРОВКИ: «{m.group(0)}» — имя с цифрами, "
                          f"похоже на реальный случай. Введите маркировку: "
                          f"«Условный пример» / «Модельный сценарий»")
    # P1: ссылка на статью без указания акта
    for m in list(_BARE_ARTICLE_REF.finditer(text))[:4]:
        window = text[max(0, m.start() - 120):m.end() + 120]
        if not _ACT_QUALIFIERS.search(window):
            frag = text[max(0, m.start() - 15):m.end() + 15].replace("\n", " ")
            issues.append(f"🟡 СТАТЬЯ БЕЗ АКТА: «{frag}» — номер статьи без акта "
                          f"(152-ФЗ? КоАП? НК?). Укажите полный реквизит")
    # P1: вердикт каналу без контекста
    for m in list(_CHANNEL_VERDICT.finditer(text))[:2]:
        issues.append(f"🟡 ВЕРДИКТ КАНАЛУ БЕЗ КОНТЕКСТА: «{m.group(0)}» — "
                      f"оценка канала без условий (ниша, чек, цикл сделки, бюджет). "
                      f"Переформулируйте: «подходит при X, не подходит при Y»")
    # P1: термин без расшифровки при первом использовании
    low_terms = re.sub(r"[^a-zA-Zа-яА-ЯЁё0-9\s]", " ", text)
    for term, expansions in _TERM_DEFS.items():
        if not expansions:
            continue
        if re.search(rf"\b{re.escape(term)}\b", text) and not any(
                exp.lower() in low_terms.lower() for exp in expansions):
            issues.append(f"🟡 ТЕРМИН БЕЗ РАСШИФРОВКИ: «{term}» — при первом "
                          f"употреблении дайте расшифровку "
                          f"({expansions[0]})")
    # P0: подпись работника при КЭДО (ревью статьи 15, 2026-09-29)
    for m in list(_SIGN_ALT_DENIED.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 ПОДПИСЬ РАБОТНИКА: «{frag}» — по ч. 4 ст. 22.3 ТК "
                      f"УНЭП и простая ЭП работника допускаются для кадровых "
                      f"документов. Отрицание альтернатив неверно")
    # P0: работодатель «только УКЭП» — не универсальное правило (ревью 16)
    for m in list(_EMPLOYER_UKEP_ONLY.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 ПОДПИСЬ РАБОТОДАТЕЛЯ: «{frag}» — УКЭП работодателя "
                      f"обязательна для критичных документов ст. 22.3 ТК, "
                      f"не для всех. Исправить по матрице подписей")
    for m in list(_WORKER_SIGN_CLAIM.finditer(text))[:3]:
        window = text[max(0, m.start() - 100):m.end() + 150]
        if _WORKER_SIGN_ALTS.search(window):
            continue  # альтернативы подписи названы — корректная матрица
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 ПОДПИСЬ РАБОТНИКА: «{frag}» — по ч. 4 ст. 22.3 ТК "
                      f"работник подписывает УКЭП, УНЭП или простую ЭП "
                      f"(по виду документа); только УКЭП — работодатель. "
                      f"Уточните по подписной матрице")
    # P0: ст. 312.2 подменяет общий КЭДО
    if re.search(r"КЭДО|кадров\w+\s+электронн\w+\s+документооборот",
                 text, re.I) \
            and re.search(r"312\.[123]", text) and not re.search(r"22\.[123]", text):
        issues.append("🟡 ОБЛАСТЬ НОРМЫ: статья ссылается только на ст. 312.x "
                      "(дистанционная работа), тогда как общий КЭДО — "
                      "ст. 22.1–22.3 ТК РФ. Разведите общую и специальную нормы")
    # P1: внутренний контроль как норма закона
    for m in list(_INTERNAL_CTRL_AS_LAW.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🟡 ВНУТРЕННИЙ КОНТРОЛЬ КАК НОРМА: «{frag}» — процент "
                      f"выборки не установлен законом. Маркируйте: "
                      f"«внутренний стандарт компании»")
    # P1: маскировка трудовых отношений при ГПХ
    for m in list(_GPH_DISGUISE.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 МАСКИРОВКА ТРУДОВЫХ ОТНОШЕНИЙ: «{frag}» — совет "
                      f"убрать признаки вместо управления риском. Корректно: "
                      f"закрепить предоставление доступа/техники в договоре "
                      f"рамками задачи")
    # P1: неподтверждённые планы регулятора (ревью 16: «итоги пилота к 2027»)
    for m in list(_REGULATOR_PLAN_NO_SRC.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🟡 ПЛАН РЕГУЛЯТОРА БЕЗ ИСТОЧНИКА: «{frag}» — нет "
                      f"действующего акта/проекта. Удалите или дайте ссылку")
    # P0: миф об общем дедлайне КЭДО/ЭТК (ревью 16)
    for m in list(_KEDO_MANDATORY_MYTH.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 МИФ ОБ ОБЯЗАТЕЛЬНОСТИ КЭДО: «{frag}» — общего "
                      f"федерального дедлайна нет (КЭДО — право, ст. 22.1 ТК). "
                      f"Если это разоблачение мифа — маркируйте явно")
    # P1: категоричные исходы — процессуальный линтер на все органы
    for m in list(_PROCEDURAL_CATEGORICAL.finditer(text))[:3]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🟡 КАТЕГОРИЧНЫЙ ПРОЦЕССУАЛЬНЫЙ ИСХОД: «{frag}» — исход зависит "
                      f"от процедуры и обстоятельств. Смягчите: «может стать "
                      f"основанием для спора и мер реагирования»")
    # PROCESS_AUTOMATION_OVERCLAIM: автоматизм без условий
    for m in list(_PROC_AUTOMATION.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🟡 АВТОМАТИЗМ БЕЗ УСЛОВИЙ: «{frag}» — последствие не "
                      f"наступает само собой; укажите условие, норму и "
                      f"орган, принимающий решение")
    # техкоды закупочного контура без первоисточника
    for m in list(_TECH_CODE_PROC.finditer(text))[:2]:
        window = text[max(0, m.start() - 200):m.end() + 200]
        if not _TECH_CODE_PROC_SRC.search(window):
            frag = m.group(0)[:60].replace("\n", " ")
            issues.append(f"🔴 ТЕХКОД БЕЗ ИСТОЧНИКА: «{frag}» — коды/формы/"
                          f"номера реестров не выдумываются; проверьте в ЕИС, "
                          f"ГИСП, каталоге КТРУ или классификаторе и дайте "
                          f"ссылку")
    # P0: субъектный скоуп обязанностей
    for pat, msg in _SUBJECT_SCOPE:
        for m in list(pat.finditer(text))[:2]:
            frag = m.group(0)[:70].replace("\n", " ")
            issues.append(f"🔴 СУБЪЕКТ ОБЯЗАННОСТИ: «{frag}» — {msg}")
    # P1: ложная обязательность
    for m in list(_FALSE_OBLIGATION.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🟡 ЛОЖНАЯ ОБЯЗАТЕЛЬНОСТЬ: «{frag}» — предположение подано "
                      f"как закон/практика. Нужен источник или мягкая формулировка")
    # P0: mode_split — смешение режимов без развилки
    for rx_a, rx_b, msg in _MODE_MIX_RULES:
        if rx_a.search(text) and rx_b.search(text) \
                and not _SPLIT_MARKERS.search(text):
            issues.append(f"🔴 СМЕШЕНИЕ РЕЖИМОВ: {msg}. Разбейте на отдельные "
                          f"сценарии или добавьте явное разведение")
    # P0: смешение трёх контуров банковских проверок (ревью 20)
    for m in list(_VC_161_MIX.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 СМЕШЕНИЕ КОНТУРОВ: «{frag}» — агент валютного "
                      f"контроля действует по 173-ФЗ; отказ в операции — "
                      f"115-ФЗ; антифрод-приостановление — 161-ФЗ. "
                      f"Не приписывайте одному закону полномочия другого")
    _circuit_hits = sum(1 for rx in _CIRCUIT_MARKERS if rx.search(text))
    if _circuit_hits >= 2 and not _SPLIT_MARKERS.search(text):
        issues.append("🟡 ТРИ КОНТУРА БЕЗ РАЗВЕДЕНИЯ: статья упоминает "
                      "несколько контуров проверки (валютный контроль / "
                      "115-ФЗ / 161-ФЗ) без явного разделения — у них разные "
                      "основания, полномочия и способы защиты")
    # P0: неверный штрафной диапазон по 15.25 КоАП
    for m in list(_FINE_1525_WRONG.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 ШТРАФ 15.25: «{frag}» — для юрлиц за незаконную "
                      f"валютную операцию штраф 20–40% суммы, не 20–30%. "
                      f"И не применяйте состав к факту «банк запросил "
                      f"документы» — нужна доказанность нарушения")
    # P1: широкие формулы о снятии требований
    for m in list(_BROAD_RELIEF.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🟡 ШИРОКАЯ ФОРМУЛА: «{frag}» — без указания "
                      f"применимости (резиденты/операции/период). "
                      f"Уточните или уберите блок")
    # P1: псевдо-практика судов
    for m in list(_COURT_PSEUDO.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🟡 ПСЕВДО-ПРАКТИКА СУДОВ: «{frag}» — афоризм без "
                      f"конкретных дел. Суд оценивает совокупность "
                      f"доказательств; не выдавайте формулу за практику")
    # ── Валютный домен: 4 правила ──
    for m in list(_BANK_DUTY_OVERREACH.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 ПРАВО vs ОБЯЗАННОСТЬ: «{frag}» — отказ/расторжение/"
                      f"приостановление это ПРАВО банка в предусмотренных "
                      f"случаях, не обязанность. Переформулируйте: «вправе»")
    # санкция ≠ запрос документов (по абзацам)
    for para in text.split("\n\n"):
        if _FINE_SCARE_NO_GUILT.search(para) and _FINE_SCARE.search(para) \
                and not _GUILT_MARKER.search(para):
            issues.append("🟡 САНКЦИЯ ≠ ЗАПРОС: в одном абзаце запрос банком "
                          "документов и штраф по 15.25 без условия «если "
                          "операция признана незаконной» — запугивание вместо "
                          "права. Разведите или добавьте условие состава")
            break
    if _APPEAL_NO_ROUTE.search(text) and not _APPEAL_ROUTE_MARKERS.search(text):
        issues.append("🟡 ОБЖАЛОВАНИЕ БЕЗ МАРШРУТА: способ защиты зависит от "
                      "основания (115-ФЗ → комиссия ЦБ; валютный контроль → "
                      "документы; 161-ФЗ → антифрод-процедура). Укажите "
                      "развилку, а не общее «обжаловать в суде»")
    for m in list(_CASE_OUTCOME_PROMISE.finditer(text))[:2]:
        window = text[max(0, m.start() - 400):m.end() + 100]
        if _CASE_CONTEXT.search(window) and not _CASE_SOFTENERS.search(window):
            frag = m.group(0)[:70].replace("\n", " ")
            issues.append(f"🟡 ОБЕЩАННЫЙ ИСХОД В КЕЙСЕ: «{frag}» в условном "
                          f"примере как нормальная механика. Смягчите: "
                          f"«в модельном сценарии компания могла бы…»")
    # ── Ревью 21: техпроцедуры без первичного источника ──
    for m in list(_SPD_CODE_PATTERN.finditer(text))[:2]:
        window = text[max(0, m.start() - 150):m.end() + 150]
        if not _SPD_CODE_SOURCE.search(window):
            frag = m.group(0)[:70].replace("\n", " ")
            issues.append(f"🔴 КОД СПД БЕЗ ИСТОЧНИКА: «{frag}» — код вида "
                          f"документа зависит от вида документа/операции/"
                          f"основания, а не от «импорт/экспорт». Нужна ссылка "
                          f"на классификатор или «согласуйте с банком»")
    for m in list(_ACCOUNT_IMMEDIATE.finditer(text))[:2]:
        window = text[max(0, m.start() - 300):m.end() + 300]
        if not _THRESHOLD_MARKERS.search(window):
            issues.append("🟡 УЧ[ЁЕ]Т КОНТРАКТА БЕЗ ПОРОГА: «немедленно на "
                          "учёт» без проверки порога (импорт/смешанные — "
                          "от 3 млн ₽, экспорт — от 10 млн ₽, 181-И). "
                          "Сначала проверьте обязанность, потом действуйте")
    for m in list(_SPD_UNIVERSAL.finditer(text))[:2]:
        window = text[max(0, m.start() - 300):m.end() + 300]
        if not _SPD_CONDITION.search(window):
            issues.append("🟡 СПД КАК УНИВЕРСАЛЬНАЯ ОБЯЗАННОСТЬ: 15 рабочих "
                          "дней — только когда СПД требуется по 181-И; при "
                          "таможенном декларировании с УНК сведения идут из "
                          "ФТС. Добавьте условие применимости")
    for m in list(_CORRESPONDENT_BANK.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🟡 БАНК-КОРРЕСПОНДЕНТ: «{frag}» — у клиента нет "
                      f"прямого договорного контакта с ним. Запрос — в "
                      f"обслуживающий банк")
    for m in list(_REPATRIATION_AUTO_VIOLATION.finditer(text))[:2]:
        window = text[max(0, m.start() - 400):m.end() + 200]
        if not _UKAZ_529_MARKER.search(window):
            frag = m.group(0)[:70].replace("\n", " ")
            issues.append(f"🔴 РЕПАТРИАЦИЯ КАК АВТОНАРУШЕНИЕ: «{frag}» — по "
                          f"Указу № 529 ч. 1, 2 ст. 19 173-ФЗ временно не "
                          f"применяются для внешнеторговых сделок юрлиц/ИП. "
                          f"Обязанности по документам остаются, но «задержка "
                          f"= нарушение» — неверно")
    # ── Ревью 22: объект нормы вместо числа ──
    if _LIMIT_4H_DAY.search(text) and not _LIMIT_4H_CORRECT.search(text):
        frag = _LIMIT_4H_DAY.search(text).group(0)[:60].replace("\n", " ")
        issues.append(f"🔴 LIMIT SCOPE DRIFT: «{frag}» — лимит сверхурочной "
                      f"работы: не более 4 часов в течение ДВУХ ДНЕЙ ПОДРЯД, "
                      f"не «в день». Цифра верна, период — нет")
    if _OVERTIME_240_CLAIM.search(text):
        for cond_rx, cond_name in _OVERTIME_CONDITIONS:
            if not cond_rx.search(text):
                issues.append(f"🔴 REQUIRED CONDITION MISSING: статья "
                              f"заявляет лимит 240 часов без обязательного "
                              f"условия «{cond_name}» (ст. 99 ТК в ред. "
                              f"144-ФЗ). Добавьте условие в алгоритм")
        if _OVERTIME_GUIDE_ANGLE.search(text[:500]) and not _MEDICAL_DAY.search(text):
            issues.append("🟡 НЕТ ГАРАНТИИ: инструкция о 240 часах без дня "
                          "диспансеризации (сохранение среднего заработка) — "
                          "новая гарантия 144-ФЗ, обязательна для гайда")
    for m in list(_SANCTION_OVERCLAIM.finditer(text))[:2]:
        window = text[max(0, m.start() - 200):m.end() + 200]
        if not _SANCTION_CONDITIONS.search(window):
            frag = m.group(0)[:60].replace("\n", " ")
            issues.append(f"🟡 SANCTION OVERCLAIM: «{frag}» — конкретная часть "
                          f"КоАП и сумма без условий (состав/субъект/"
                          f"повторность). Смягчите: «может повлечь… в "
                          f"зависимости от обстоятельств»")
    for m in list(_SELF_UNVERIFIED.finditer(text))[:2]:
        frag = m.group(0)[:70].replace("\n", " ")
        issues.append(f"🔴 НЕПОДТВЕРЖДЁННАЯ ВЕТКА: «{frag}» — статья сама "
                      f"признаёт непроверенность утверждения. Уберите ветку "
                      f"или проверьте по первоисточнику; оговорка не "
                      f"заменяет верификацию")
    # обещанные секции без содержимого
    for m in list(_PROMISED_SECTION.finditer(text))[:3]:
        tail = text[m.end():m.end() + 2000]
        nxt = re.search(r"^#{2,3}\s", tail, re.M)
        section = tail[:nxt.start()] if nxt else tail
        bullets = len(re.findall(r"^\s*(?:-|\*|\d+[.)]|\[ \])", section, re.M))
        # чек-лист/самопроверка — порог 5 пунктов (рекомендация агента),
        # остальные обещанные секции — 3
        head = m.group(0).lower()
        need = 5 if ("чек-лист" in head or "самопроверк" in head) else 3
        if bullets < need and "<table" not in section and "|" not in section:
            title = text[m.start():m.end()].lstrip("# \n")[:40]
            issues.append(f"🔴 ПУСТАЯ ОБЕЩАННАЯ СЕКЦИЯ: «{title}» — заголовок "
                          f"есть, содержимого (≥{need} пунктов/таблицы) нет")
    # ── Ревью 23 ──
    for m in list(_UNRESOLVED_TEMPLATE.finditer(text))[:3]:
        frag = m.group(0)[:50].replace("\n", " ")
        issues.append(f"🔴 НЕЗАПОЛНЕННЫЙ ШАБЛОН: «{frag}» — переменная/число "
                      f"не подставились. Блок публикации: читатель не поймёт, "
                      f"о каком значении речь")
    if _CONSENT_TOPICS.search(text):
        for m in list(_ACK_NOT_CONSENT.finditer(text))[:3]:
            window = text[max(0, m.start() - 300):m.end() + 300]
            if not _CONSENT_MARKER.search(window):
                frag = m.group(0)[:60].replace("\n", " ")
                issues.append(f"🟡 ОЗНАКОМЛЕНИЕ ≠ СОГЛАСИЕ: «{frag}» — "
                              f"ознакомление под подпись не заменяет "
                              f"персонального письменного согласия. "
                              f"Добавьте различие явно")
    if _OVERTIME_240_CLAIM.search(text):
        if not _MED_CONTEXT.search(text):
            issues.append("🟡 НЕТ МЕДИЦИНСКОГО КОНТУРА: инструкция о 240 часах "
                          "без медосмотров/противопоказаний (пенсионеры, "
                          "предпенсионеры, вредные 3.1/3.2 — только при "
                          "письменном согласии и отсутствии противопоказаний)")
        if not _REST_ALTERNATIVE.search(text):
            issues.append("🟡 НЕТ АЛЬТЕРНАТИВЫ ОТДЫХА: по желанию работника "
                          "повышенная оплата заменяется дополнительным "
                          "временем отдыха — обязательное упоминание для "
                          "инструкции по сверхурочным")
    # ── Маркетплейс-домен ──
    if _MPLACE_CTX.search(text):
        for m in list(_GMV_AS_MONEY.finditer(text))[:2]:
            frag = m.group(0)[:70].replace("\n", " ")
            issues.append(f"🔴 GMV ≠ ДЕНЬГИ СЕЛЛЕРА: «{frag}» — GMV это метрика "
                          f"площадки; деньги селлера — выплаты минус удержания. "
                          f"Разведите понятия")
        for m in list(_TARIFF_NO_DATE.finditer(text))[:2]:
            wide = text[max(0, m.start() - 1000):m.end() + 1000]
            if _MODEL_MARKER.search(wide):
                continue
            window = text[max(0, m.start() - 250):m.end() + 250]
            if not _SNAPSHOT_DATE.search(window):
                frag = m.group(0)[:60].replace("\n", " ")
                issues.append(f"🟡 ТАРИФ БЕЗ ДАТЫ: «{frag}» — тарифы площадок "
                              f"меняются часто; без даты снимка/источника "
                              f"цифра вводит в заблуждение")
        if _FBO_FBS_MIX.search(text) and _FBS_RX.search(text) \
                and not _SPLIT_MARKERS.search(text):
            issues.append("🟡 FBO/FBS БЕЗ РАЗВОДКИ: модели работы различаются "
                          "складом, скоростью, штрафами и юнит-экономикой — "
                          "не давайте один совет для обеих без развилки")
        # ── Ревью статьи 25 ──
        for m in list(_CAUSALITY_LAW_EFFECT.finditer(text))[:2]:
            window = text[max(0, m.start() - 200):m.end() + 200]
            if not _CAUSALITY_OK.search(window):
                issues.append("🔴 ПРИЧИННОСТЬ БЕЗ НОРМЫ: закон о платформенной "
                              "экономике регулирует отношения сторон, а не "
                              "тарифы — экономический эффект не следует из "
                              "нормы. Писать: «правила изменились, тарифы "
                              "могут пересматриваться — обновляйте модель "
                              "по фактическим данным площадки»")
                break
        modes = len(set(_TAX_MODES_RX.findall(text)))
        _mode_split_ok = re.search(r"для\s+каждого\s+режима|отдельн\w+\s+"
                                   r"(?:расч[её]т|таблица|сценарий)|зависит\s+от\s+"
                                   r"режима|теги\s+затрат|direct_variable", text, re.I)
        if modes >= 3 and _UNIT_TABLE_RX.search(text) and not _mode_split_ok:
            issues.append("🟡 НАЛОГОВЫЕ РЕЖИМЫ БЕЗ СЦЕНАРИЕВ: в материале "
                          "несколько режимов налогообложения при юнит-модели — "
                          "для каждого режима отдельный расчёт (УСН без НДС / "
                          "УСН + НДС 5% / УСН + НДС 22% / ОСНО), общая таблица "
                          "недопустима")
        if _ORDER_BUYOUT_RX.search(text) and re.search(r"выкуп", text, re.I)                 and not re.search(r"не\s+равен\s+выкупу|на\s+выкупленн\w+|"
                                  r"дол[ея]\s+выкупа|коэффициент\s+выкупа|"
                                  r"CPO[^.]{0,40}дол\w+\s+выкупа", text, re.I):
            issues.append("🟡 ЗАКАЗ ≠ ВЫКУП: реклама и маржа с разными "
                          "знаменателями искажают модель — приводите расходы "
                          "на рекламу к выкупленной единице либо явно "
                          "включайте коэффициент выкупа")
        if _PORTFOLIO_COST_RX.search(text):
            issues.append("🟡 ТИП ЗАТРАТ НЕ МАРКИРОВАН: аренда/офис/подписки — "
                          "общие расходы; в юнит-модели маркируйте "
                          "(direct_variable / shared_variable / "
                          "allocated_fixed / portfolio_cost)")
        # ── Ревью 25v2 ──
        if re.search(r"ОСНО", text) and _UNIT_TABLE_RX.search(text) \
                and not re.search(r"выбор режима|сравнени\w+ режим|ИП\s+или\s+ООО", text, re.I):
            issues.append("🔴 ОСНО В ЮНИТ-СТАТЬЕ: сравнение режимов в P&L SKU "
                          "требует баз, статуса цены и юрформы — удалить "
                          "ОСНО-строку; налог здесь параметр модели "
                          "(УСН-строка), сравнение режимов — отдельная статья")
        if _VAT_COMPARE.search(text) and not _PRICE_VAT_STATUS.search(text):
            issues.append("🟡 НДС-СРАВНЕНИЕ БЕЗ СТАТУСА ЦЕНЫ: 5% и 22% "
                          "несравнимы без указания, включает ли цена НДС "
                          "или он сверх цены (и расчётной ставки 22/122); "
                          "раскройте базы или уберите таблицу")
        if _FAIR_THRESHOLD.search(text):
            issues.append("🟡 ВЫДУМАННЫЙ ПОРОГ ДОСТАТОЧНОСТИ: «от N дней — "
                          "достаточно» без источника; пишите правило по "
                          "фактическому числу заказов и циклу возвратов")
    # ── Закупочный домен (44/223-ФЗ) ──
    if _PROC_CTX.search(text):
        if _FZ44_RX.search(text) and _FZ223_RX.search(text) \
                and not _FZ_SPLIT.search(text):
            issues.append("🟡 44/223 СМЕШЕНИЕ: правила законов различаются "
                          "(пороги, сроки, обжалование, положение о закупке) — "
                          "разведите или пометьте вторую систему справочной "
                          "врезкой")
        for m in list(_APPEAL_10D.finditer(text))[:2]:
            frag = m.group(0)
            window = text[max(0, m.start() - 130):m.end() + 130]
            low = window.lower()
            # «10 дней на подписание контракта», «10 дней на возврат
            # обеспечения» (кейс статьи 24v2) — другие нормы, не срок жалобы
            if any(w in low for w in ("подпис", "возврат", "обеспечени",
                                      "таможен", "транзит", "исполнено")):
                continue
            issues.append("🔴 СТАРЫЙ СРОК ЖАЛОБЫ: «десять дней» — в действующей "
                          "редакции ч. 2 ст. 105 44-ФЗ (624-ФЗ, с 25.03.2024) "
                          "жалоба подаётся не позднее 5 дней после протокола "
                          "итогов")
            break
        # Ревью v5: контаминация 59-ФЗ (30 дней обращения гражданина)
        # со сроком жалобы (5 дней) — дословно формулировку закона
        for m in list(_APPEAL_30D.finditer(text))[:2]:
            window = text[max(0, m.start() - 200):m.end() + 200]
            if _APPEAL_30D_OK.search(window):
                continue
            if _APPEAL_30D_CTX.search(window):
                issues.append("🔴 ЖАЛОБА 30 ДНЕЙ: 30 дней — срок обращения "
                              "гражданина по 59-ФЗ, НЕ срок жалобы по ч. 2 "
                              "ст. 105 44-ФЗ («не позднее пяти дней» со дня, "
                              "следующего за размещением итогового протокола). "
                              "Формулировку закона воспроизводить дословно")
                break
        for m in list(_SMALL_600K.finditer(text))[:2]:
            window = text[max(0, m.start() - 250):m.end() + 250]
            if not _SMALL_LIMIT.search(window):
                issues.append("🟡 МАЛАЯ ЗАКУПКА БЕЗ ЛИМИТА: 600 тыс ₽ (п. 4 "
                              "ч. 1 ст. 93) без годового лимита — 2 млн или "
                              "10% СГОЗ и не более 50 млн; норма неполная")
                break
        # Ревью 26: повреждённый числовой токен («22 НМЦК» вместо «20% НМЦК»)
        for m in list(_DAMAGED_NUM_TOKEN.finditer(text))[:2]:
            frag = m.group(0)
            issues.append(f"🔴 ПОВРЕЖДЁННЫЙ ЧИСЛОВОЙ ТОКЕН: «{frag}» — голое "
                          f"число перед НМЦК; валидные формы «10% НМЦК» / "
                          f"«20% НМЦК». Блокировка публикации")
            break
        # ── Ревью 27v4 (2026-10-03, 7/10): режим, уточняется, merge ──
        # LAW_REGIME_CONSISTENCY: 223-ФЗ в лиде при 44-ФЗ в теле
        if _LEAD_223.search(text[:800]) and re.search(r"44-ФЗ", text[800:]):
            issues.append("🔴 LAW REGIME MISMATCH: лид упоминает 223-ФЗ, "
                          "тело — 44-ФЗ. Один закон на одну статью; вторую "
                          "систему не упоминать")
        # SANCTION_TABLE_NO_PLACEHOLDERS: «уточняется» в таблице
        for m in list(_SANCTION_PLACEHOLDER.finditer(text))[:2]:
            issues.append("🔴 ЗАПОЛНИТЕЛЬ В ТАБЛИЦЕ САНКЦИЙ: «уточняется» — "
                          "указать точную часть статьи и сумму либо убрать "
                          "строку из таблицы")
            break
        # ARTICLE_PART_MERGE_GUARD: ч.1+ч.2 одной санкцией
        for m in list(_MULTI_PART_SANCTION.finditer(text))[:2]:
            issues.append("🟡 ОБЪЕДИНЕНИЕ ЧАСТЕЙ: разные части статьи не "
                          "могут иметь одну санкцию без подтверждения — "
                          "разделить на отдельные строки")
            break
        # PROCEDURAL_ROUTE: широкое «остальные — ФАС»
        if _BROAD_APPEAL.search(text):
            issues.append("🟡 МАРШРУТ БЕЗ КОНТЕКСТА: орган рассмотрения "
                          "зависит от части КоАП и подведомственности — "
                          "проверьте или уберите категоричное «остальные — ФАС»")
        # ── Ревью 28: нацрежим, обрыв, право vs автоматизм ──
        # TRUNCATED_CONTENT
        if _TRUNCATED_ENDING.search(text):
            issues.append("🔴 ТЕКСТ ОБОРВАН: абзац заканчивается на союзе — "
                          "статья неполна, блокировка публикации")
        # DISCRETION_VS_AUTOMATIC
        if _DISCRETION_AUTO.search(text):
            issues.append("🟡 ПРАВО ≠ АВТОМАТИЗМ: право заказчика не устанавливать "
                          "меру ≠ автоматическое отключение — укажите «вправе» "
                          "и условия применения")
        # PRODUCT_CLASSIFICATION_MULTIFACTOR
        if _OKPD2_AUTO.search(text):
            issues.append("🟡 ОКПД2 НЕДОСТАТОЧНО: мера нацрежима определяется "
                          "по коду + наименованию + описанию + примечаниям — "
                          "не только по совпадению кода")
        # NATREGIME_EAEU: «российский товар» без ЕАЭС в нацрежиме
        if _NATREGIME_CTX.search(text) and _RUSSIAN_ONLY.search(text) \
                and not _EAEU_MENTION.search(text):
            issues.append("🟡 НАЦРЕЖИМ: «российский товар» без упоминания "
                          "ЕАЭС — меры применяются и к товарам из Беларуси, "
                          "Казахстана, Армении, Кыргызстана; укажите «российского "
                          "или евразийского происхождения»")
        # Ревью 26: спецрежим + антидемпинг без отдельной ветки
        if _SPECIAL_REGIME.search(text) and re.search(r"антидемпинг", text, re.I) \
                and not _SPECIAL_REGIME_OK.search(text):
            issues.append("🟡 СПЕЦРЕЖИМ БЕЗ ВЕТКИ: упомянуты СМП/УИС/аванс/"
                          "неизвестный объём вместе с антидемпингом — дайте "
                          "отдельную ветку условий или проверочную формулировку "
                          "(«проверьте вид закупки, статус, аванс»)")
        # ── Ревью 27: санкции и заголовок ──
        for m in list(_APPROX_SANCTION.finditer(text))[:2]:
            window = text[max(0, m.start() - 200):m.end() + 200]
            if not _SANCTION_OK.search(window):
                issues.append("🔴 ПРИБЛИЗИТЕЛЬНАЯ САНКЦИЯ: штраф без точной "
                              "части статьи и суммы — указать «ч. N ст. 7.30.X "
                              "КоАП РФ, от X до Y ₽» или убрать сумму")
                break
        if _DUAL_SUBJECT_RX.search(text):
            h1 = re.search(r"^#\s+(.+)$", text, re.M)
            h1_text = h1.group(1) if h1 else text[:200]
            if _DUAL_SUBJECT_RX.search(h1_text):
                issues.append("🟡 ЗАГОЛОВОК ОБЕЩАЕТ ДВА СУБЪЕКТА: «для заказчика "
                              "и поставщика» — проверьте, что матрица для "
                              "каждого субъекта полноценна (норма + санкция + "
                              "действие), либо сузьте заголовок")
        # ── Ревью 27v2: хедж у нормы, диапазоны, stage mismatch ──
        for m in list(_HEDGE_NEAR_NORM.finditer(text))[:2]:
            issues.append("🟡 ХЕДЖ У НОРМЫ: неопределённость рядом с правовой "
                          "нормой — для red-domain «либо проверенный текст, "
                          "либо не публикуем»")
            break
        for m in list(_HEDGE_AFTER_NORM.finditer(text))[:2]:
            issues.append("🟡 ХЕДЖ ПОСЛЕ НОРМЫ: номер статьи + «вероятно» — "
                          "недопустимо; норма либо действует, либо нет")
            break
        if _BAD_ARTICLE_RANGE.search(text):
            issues.append("🔴 НЕВОЗМОЖНЫЙ ДИАПАЗОН СТАТЕЙ: «статьи 1–7.32.5» — "
                          "перечислять конкретные отменённые статьи (7.29, "
                          "7.30, 7.31, 7.32...), а не диапазон")
        if _STAGE_ARTICLE_MISMATCH.search(text):
            issues.append("🔴 STAGE ARTICLE MISMATCH: выбор способа "
                          "определения поставщика — это предконтрактная "
                          "стадия, относится к 7.30.1, не к 7.30.2")
        for m in list(_ANTIDUMP_HALF.finditer(text))[:2]:
            window = text[max(0, m.start() - 300):m.end() + 300]
            if "обеспечен" in window.lower() and not _ANTIDUMP_15M.search(window):
                issues.append("🟡 АНТИДЕМПИНГ БЕЗ ПОРОГА: полуторное "
                              "обеспечение без порога НМЦК 15 млн ₽ — при "
                              "меньшей НМЦК есть альтернатива (добросовестность: "
                              "3 контракта за 3 года без неустоек, один ≥ 20% "
                              "НМЦК, ч. 2–3 ст. 37)")
                break
        if _FAS_WIN.search(text):
            issues.append("🔴 ГАРАНТИРОВАННЫЙ ИСХОД ЖАЛОБЫ: решение ФАС зависит "
                          "от обстоятельств дела — обещание исхода недопустимо")
        # ── Правила расширенной системы (2026-10-02) ──
        if _PRACTICE_AS_RULE.search(text):
            issues.append("🟡 PRACTICE AS RULE: практика ФАС/судов выдана как "
                          "универсальная норма — смягчить («в подобных делах», "
                          "«как правило») или опереться на несколько однотипных "
                          "решений")
        for m in list(_DEADLINE_NAKED.finditer(text))[:3]:
            window = text[max(0, m.start() - 250):m.end() + 250]
            if not _DEADLINE_EVENT.search(window):
                frag = m.group(0)[:40]
                issues.append(f"🟡 СРОК БЕЗ СОБЫТИЯ ОТСЧЁТА: «{frag}» — "
                              f"укажите, от какого события считаются дни "
                              f"(размещение протокола, подписание, получение)")
                break
        if _FZ223_RX.search(text) and _FZ223_CATEGORIC.search(text) \
                and not _FZ223_REGULATION.search(text):
            issues.append("🟡 223-ФЗ БЕЗ ЛОКАЛЬНОГО ПРАВИЛА: по 223-ФЗ рамку "
                          "задаёт положение о закупке конкретного заказчика — "
                          "категоричность («всегда», «срок составляет») "
                          "допустима только с оговоркой")
        # Ревью 24: будущие изменения закупок без акта
        # (модельные расчёты «ожидаемая маржа… в данном сценарии» — не
        # законодательные изменения, пропуск через _MODEL_MARKER)
        for m in list(_FUT_CHANGE.finditer(text))[:2]:
            window = text[max(0, m.start() - 200):m.end() + 200]
            if _MODEL_MARKER.search(window):
                continue
            if not _ACT_REF.search(window):
                frag = text[m.start():m.start() + 60].replace("\n", " ")
                issues.append(f"🔴 БУДУЩЕЕ ИЗМЕНЕНИЕ БЕЗ АКТА: «{frag}…» — "
                              f"ожидаемые цифры без нормативного акта, статуса "
                              f"и даты вступления; убрать или дать реквизиты "
                              f"(ФЗ/постановление + дата)")
                break
        # Ревью 24: неуверенность в правовой норме
        if _HEDGED_LEGAL.search(text):
            issues.append("🟡 НЕУВЕРЕННОСТЬ В НОРМЕ: «вероятно/ожидается» при "
                          "правовом утверждении — сверить с первоисточником и "
                          "писать утвердительно, либо убрать блок вовсе")
        # Ревью 24: срок без полного контекста (событие + календарный тип + акт)
        for m in list(_DEADLINE_NAKED.finditer(text))[:3]:
            window = text[max(0, m.start() - 250):m.end() + 250]
            if not _DEADLINE_EVENT.search(window):
                issues.append("🟡 СРОК БЕЗ СОБЫТИЯ ОТСЧЁТА: укажите, от какого "
                              "события считаются дни (размещение протокола, "
                              "подписание, получение)")
                break
            if _PROC_CTX.search(window) and not _CAL_TYPE.search(window):
                issues.append("🟡 СРОК БЕЗ ТИПА ДНЕЙ: календарные или рабочие? "
                              "Для 44-ФЗ указывать явно — сроки различаются")
                break

        # Ревью v3: противоречие по КЭП в одном тексте — блокер
        if _KEP_MANDATORY.search(text) and _KEP_NOT_NEEDED.search(text):
            issues.append("🔴 КЭП-ПРОТИВОРЕЧИЕ: в статье и «КЭП обязательна», "
                          "и «не требуется» — сверить регламент ЕИС и писать "
                          "безопасно («проверьте требования ЕИС на дату подачи»)")
        for _m in list(_SUSPEND_OVERCLAIM.finditer(text))[:1]:
            issues.append("🟡 ПРИОСТАНОВЛЕНИЕ КАК АВТОМАТ: подача жалобы не "
                          "останавливает закупку сама по себе — указать норму, "
                          "условия и момент наступления эффекта")
            break
        for _m in list(_COURT_AUTO.finditer(text))[:2]:
            _w = text[max(0, _m.start() - 150):_m.end() + 150]
            if not _COURT_OK.search(_w):
                issues.append("🟡 СУД БЕЗ МАРШРУТА: «спор переходит в суд» — "
                              "указать предмет спора, тип акта, заявителя и "
                              "процессуальную норму (АПК/КАС)")
                break
        if _SPECIAL_JURISDICTION.search(text):
            issues.append("🔴 КОМПЕТЕНЦИЯ ЖАЛОБЫ НЕ ПОДТВЕРЖДЕНА: категоричный "
                          "запрет жалобы без части ст. 105 — дать норму или "
                          "смягчить («проверьте, относится ли спор к "
                          "компетенции ФАС»)")
    # Будущая норма подана как действующая (глобально, юридический контекст)
    if _PROC_CTX.search(text) or re.search(r"ФЗ|стать[ея]|закон", text, re.I):
        import datetime as _dt
        today = _dt.date.today()

        def _date_from(m):
            parts = [g for g in m.groups() if g]
            if len(parts) >= 3:
                try:
                    return _dt.date(int(parts[-1]), int(parts[-2]), int(parts[-3]))
                except ValueError:
                    return None
            return None

        for m in list(_FUTURE_AS_CURRENT.finditer(text))[:2]:
            d = _date_from(m)
            if d is not None and d > today:
                issues.append(f"🔴 БУДУЩАЯ НОРМА КАК ДЕЙСТВУЮЩАЯ: «…с "
                              f"{d.day}.{d.month}.{d.year}» — дата "
                              f"ещё не наступила; писать «вступит в силу с»")
                break
        for m in list(_MAYBE_FUTURE.finditer(text))[:2]:
            d = _date_from(m)
            if d is not None and d <= today:
                issues.append(f"🟡 НОРМА УЖЕ ДЕЙСТВУЕТ: «могут … с "
                              f"{d.day}.{d.month}.{d.year}» — дата "
                              f"настала/прошла; гипотеза звучит устаревшей, "
                              f"писать в утвердительном порядке")
                break
    # Срок жалобы искажён (кейс статьи 24): судебный срок 3 месяца
    # приписан административной жалобе
    for m in list(_APPEAL_3M.finditer(text))[:2]:
        window = text[max(0, m.start() - 120):m.end() + 120]
        if not _APPEAL_3M_OK.search(window):
            issues.append("🔴 СРОК ЖАЛОБЫ ИСКАЖЁН: 3 месяца — это судебный срок "
                          "обжалования РЕШЕНИЯ ФАС по существу (ч. 9 ст. 105 "
                          "44-ФЗ), а не срок подачи жалобы в контрольный "
                          "орган. Подача жалобы: не позднее 5 дней после "
                          "размещения протокола итогов (ч. 2 ст. 105)")
            break
    # Неподтверждённая статистика ведомства
    for m in list(_OFFICIAL_STAT_NO_SRC.finditer(text))[:2]:
        frag = text[max(0, m.start() - 10):m.end() + 30].replace("\n", " ")
        issues.append(f"🟡 ВЕДОМСТВЕННАЯ СТАТИСТИКА БЕЗ ИСТОЧНИКА: «{frag}» — "
                      f"укажите конкретный обзор/исследование или уберите утверждение")
    # Числовые артефакты: «113% рублей» (кейс статьи 13 — 2026-09-28)
    for m in list(_PCT_CURRENCY.finditer(text))[:3]:
        frag = text[max(0, m.start() - 20):m.end() + 15].replace("\n", " ")
        issues.append(f"🔴 ЧИСЛОВОЙ АРТЕФАКТ: «{frag}» — процент не может быть "
                      f"валютой. Проверьте значение и единицу измерения.")
    # Контекстные запрещённые термины
    for pat, msg in _TERM_ROLE_RULES:
        if pat.search(text):
            issues.append(f"🔴 ТЕРМИН В НЕВЕРНОМ КОНТЕКСТЕ: {msg}")
    # Псевдо-эксперт (расширенный список)
    for phrase in _PSEUDO_EXPERT_PATTERNS:
        if phrase in low:
            issues.append(f"🔴 ПСЕВДО-ЭКСПЕРТ: «{phrase}» — без верифицированного автора запрещено")
            break  # один раз достаточно, не спамим
    # Неподтверждённые эвристики («если доля клиентов выше 70%, то…»)
    for m in list(_UNVERIFIED_HEURISTIC.finditer(text))[:2]:
        frag = text[max(0, m.start() - 15):m.end() + 40].replace("\n", " ")
        issues.append(f"🟡 ЭВРИСТИКА БЕЗ МЕТОДИКИ: «{frag}» — пороговые проценты для "
                      f"бизнес-решений без исследования. Замените качественной матрицей "
                      f"(низкая/средняя/высокая доля) или укажите источник.")
    for m in list(_FORM_RE.finditer(text))[:3]:
        window = text[max(0, m.start() - 80):m.end() + 80].lower()
        if "актуальн" not in window and "проверьт" not in window:
            issues.append(f"🟡 НОМЕР ФОРМЫ без оговорки актуальности: «{m.group(0)}» — "
                          f"добавь «по актуальной форме ФНС (проверьте в личном кабинете)»")
    return issues
