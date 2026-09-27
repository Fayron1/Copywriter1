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
    "Верни строго JSON: {\"params\": [{\"name\": \"порог освобождения от НДС\", "
    "\"value\": \"20 млн руб.\", \"old_value\": \"60 млн руб. (до 2026)\", "
    "\"source\": \"ФНС / НК РФ ст. 145\", \"url\": \"...\"}]}. "
    "Пустой old_value, если параметр не менялся. Только параметры, критичные для темы."
)


def get_norm_params(topic: str, year: int) -> List[Dict[str, Any]]:
    """Извлечь действующие на {year} ключевые параметры темы. Сбой -> []."""
    try:
        from .vane import research as vane_research
        vr = vane_research(
            f"Официальные разъяснения ФНС и НК РФ, ДЕЙСТВУЮЩИЕ на {year} год: {topic}. "
            f"Только актуальные на {year} год значения: пороги, ставки, лимиты, даты вступления, "
            f"порядки перехода. Значения прошлых лет помечай как устаревшие. "
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
    """LLM-структуризация брифинга в таблицу параметров (DeepSeek flash)."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            timeout=90.0,
        )
        user = (
            f"ГОД СТАТЬИ: {year} (сегодняшние правила = правила {year} года).\n\n"
            f"БРИФИНГ:\n{briefing[:6000]}\n\n"
            f"ИСТОЧНИКИ:\n{src_line[:800]}\n\n"
            f"Извлеки параметры, действующие на {year} год."
        )
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_DEEPSEEK_FLASH", "deepseek-v4-flash"),
            messages=[{"role": "system", "content": EXTRACT_SYSTEM},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=90.0,
        )
        data = json.loads(resp.choices[0].message.content)
        params = data.get("params") or []
        clean = []
        for p in params:
            if isinstance(p, dict) and p.get("name") and p.get("value"):
                clean.append({
                    "name": str(p["name"])[:120],
                    "value": str(p["value"])[:120],
                    "old_value": str(p.get("old_value", "") or "")[:120],
                    "source": str(p.get("source", ""))[:160],
                    "url": str(p.get("url", ""))[:300],
                })
        logger.info(f"norm_params: извлечено {len(clean)} параметров на {year} год")
        return clean[:8]
    except Exception as e:
        logger.warning(f"norm_params: извлечение сбой: {e}")
        return []


def format_params_block(params: List[Dict[str, Any]], year: int) -> str:
    """Блок для промпта Heart: ЗАФИКСИРОВАННЫЕ ПАРАМЕТРЫ."""
    if not params:
        return ""
    lines = [f"=== ЗАФИКСИРОВАННЫЕ ПАРАМЕТРЫ НА {year} ГОД (проверены по официальным источникам) ===",
             "ЭТИ И ТОЛЬКО ЭТИ значения используй в статье. Отклонение = критический брак.", ""]
    for i, p in enumerate(params, 1):
        lines.append(f"{i}. {p.get('name','')}: {p.get('value','')} "
                     f"(источник: {p.get('source') or 'брифинг'})")
        if p.get("old_value"):
            lines.append(f"   ❌ УСТАРЕЛО (запрещено в статье): {p['old_value']}")
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


def check_draft(draft: str, params: List[Dict[str, Any]]) -> List[str]:
    """Детерминированная сверка: устаревшие значения в тексте = критический конфликт."""
    issues = []
    if not draft or not params:
        return issues
    text = draft.replace("\u00a0", " ")
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
    return issues
