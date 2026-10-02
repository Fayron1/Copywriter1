"""Математический валидатор финального текста.

Детерминированно извлекает арифметические выражения вида
«X × Y% = Z», «X% от Y = Z», «X − Y = Z», «X + Y = Z»
(с русскими форматами чисел и единиц) и пересчитывает их Python'ом.

Расхождение → 🔴 MATH ERROR (блокирует публикацию через валидацию).
Кейс: «2 млрд × 3% = 20 млн» (ожидалось 60 млн) — статья 18.
"""
from __future__ import annotations

import re
from typing import List

# Русские числительные суффиксы → множители
_UNIT_MULT = {
    "трлн": 1e12, "триллион": 1e12, "триллиона": 1e12,
    "млрд": 1e9, "миллиард": 1e9, "миллиарда": 1e9,
    "млн": 1e6, "миллион": 1e6, "миллиона": 1e6,
    "тыс": 1e3, "тысяча": 1e3, "тысячи": 1e3, "к": 1e3,
}

_NUM = r"\d[\d\s\u00a0]*(?:[.,]\d+)?"
_UNIT = r"(?:\s*(?:трлн|млрд|млн|тыс\.?))?"


def _parse_number(raw: str) -> float:
    """'2 млрд', '57 390', '1,5 млн', '22%' → float."""
    s = raw.replace("\u00a0", " ").replace(" ", "").strip()
    unit_mult = 1.0
    m = re.match(r"^([\d.,]+)(.*)$", s)
    if not m:
        return float("nan")
    num_part, rest = m.group(1), m.group(2).strip().rstrip("%")
    for u, mult in _UNIT_MULT.items():
        if rest.startswith(u):
            unit_mult = mult
            break
    num = float(num_part.replace(",", "."))
    return num * unit_mult


# ── Шаблоны выражений: группы (a, b, result) ──
_EXPR_PATTERNS = [
    # «X × Y% = Z», «X * Y% = Z», «X умножить на Y% = Z»
    re.compile(
        rf"({_NUM}{_UNIT})\s*(?:[×*]|умножить\s+на)\s*({_NUM})\s*%\s*=\s*({_NUM}{_UNIT})",
        re.I),
    # «Y% от X = Z», «Y% с X = Z»
    re.compile(
        rf"({_NUM})\s*%\s*(?:от|с)\s+({_NUM}{_UNIT})\s*=\s*({_NUM}{_UNIT})",
        re.I),
    # «X × Y = Z» (без процентов — только с денежными единицами у результата)
    re.compile(
        rf"({_NUM}{_UNIT})\s*[×*]\s*({_NUM}{_UNIT})\s*=\s*({_NUM}{_UNIT})\s*(?:₽|руб)",
        re.I),
    # «X − Y = Z», «X – Y = Z»
    re.compile(
        rf"({_NUM}{_UNIT})\s*[−–-]\s*({_NUM}{_UNIT})\s*=\s*({_NUM}{_UNIT})",
        re.I),
    # «X + Y = Z»
    re.compile(
        rf"({_NUM}{_UNIT})\s*\+\s*({_NUM}{_UNIT})\s*=\s*({_NUM}{_UNIT})",
        re.I),
]


def extract_math_expressions(text: str) -> List[dict]:
    """Найти все арифметические утверждения с результатом."""
    found = []
    for pat in _EXPR_PATTERNS:
        for m in pat.finditer(text):
            a, b, r = (_parse_number(m.group(i)) for i in (1, 2, 3))
            if any(v != v for v in (a, b, r)):  # NaN
                continue
            found.append({"match": m.group(0).strip(), "a": a, "b": b, "result": r,
                          "op": "%" if "%" in m.group(0) and "от" not in m.group(0).lower()
                          else ("%" if "от" in m.group(0).lower() or "%" in m.group(0) else None)})
    return found


def validate_math(text: str, tolerance: float = 0.02) -> List[str]:
    """Пересчитать все выражения. Расхождение > tolerance → 🔴."""
    issues = []
    for pat in _EXPR_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0)
            a, b, r = (_parse_number(m.group(i)) for i in (1, 2, 3))
            if any(v != v for v in (a, b, r)):
                continue
            if "%" in raw and ("×" in raw or "*" in raw or "умножить" in raw.lower()):
                expected = a * b / 100.0
                op = f"{a:g} × {b:g}% "
            elif "%" in raw and ("от" in raw.lower() or " с " in raw.lower()):
                expected = a / 100.0 * b
                op = f"{a:g}% от {b:g} "
            elif "×" in raw or "*" in raw:
                expected = a * b
                op = f"{a:g} × {b:g} "
            elif "+" in raw:
                expected = a + b
                op = f"{a:g} + {b:g} "
            else:
                expected = a - b
                op = f"{a:g} − {b:g} "
            if expected == 0 and r == 0:
                continue
            if abs(expected - r) > max(abs(expected) * tolerance, 1.0):
                issues.append(
                    f"🔴 МАТЕМАТИЧЕСКАЯ ОШИБКА: «{raw.strip()}» — пересчёт: "
                    f"{op}= {expected:g}, в тексте {r:g}. "
                    f"Блокировка публикации: числа в статье не сходятся")
    issues.extend(check_usn_reduction(text))
    return issues


# ── Глобальный инвариант УСН-уменьшения (кейс статьи 25: «6 820 792 ₽»
# выдавал наш же калькулятор с забытой строкой взносов за работников).
# Правило п. 3.1 ст. 346.21 НК РФ: tax_before = доход × ставка;
# с работниками уменьшение ≤ 50% налога; tax_after = tax_before −
# min(взносы, 50% × tax_before).
_USN_REV_RX = re.compile(r"выручк\w+[^.\n]{0,30}?(\d[\d\s.,]*)\s*млн", re.I)
_USN_AFTER_RX = re.compile(
    r"УСН[\s\S]{0,60}?после\s+уменьшения[\s\S]{0,20}?"
    r"(\d[\d\s.,]{3,})\s*(?:₽|рубл)", re.I)
_USN_CONTRIB_RX = re.compile(r"взнос\w+[^.\n]{0,30}?(\d+[.,]\d+)\s*млн", re.I)


def _num(raw: str) -> float:
    return float(re.sub(r"[^\d.,]", "", raw).replace(" ", "").replace(",", "."))


def check_usn_reduction(text: str) -> List[str]:
    """Пересчёт УСН «после уменьшения» по инварианту п. 3.1 ст. 346.21."""
    issues = []
    rev_m = _USN_REV_RX.search(text)
    if not rev_m:
        return issues
    revenue = _num(rev_m.group(1)) * 1_000_000
    tax_before = revenue * 0.06
    contrib_m = _USN_CONTRIB_RX.search(text)
    contributions = _num(contrib_m.group(1)) * 1_000_000 if contrib_m else 0.0
    cap = tax_before * 0.5
    expected = tax_before - min(contributions, cap) if contributions else None
    claimed_values = [(_num(m.group(1)), m.group(0)[:90].replace("\n", " "))
                      for m in _USN_AFTER_RX.finditer(text)]
    # Таблица: заголовок с колонкой «после уменьшения» → значение из той
    # же колонки строк данных (кейс статьи 25: «| УСН/приб. после
    # уменьшения | ... |» и значения в строках ниже).
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if "после уменьшения" in ln and ln.count("|") >= 3:
            cells = [c.strip() for c in ln.split("|")]
            col = next((j for j, c in enumerate(cells)
                        if "после уменьшения" in c), None)
            if col is None:
                continue
            for ln2 in lines[i + 2:i + 6]:
                if "УСН" not in ln2 or ln2.count("|") < 3:
                    continue
                cells2 = [c.strip() for c in ln2.split("|")]
                if col < len(cells2):
                    mnum = re.match(r"(\d[\d\s.,]{3,})\s*(?:₽|рубл)", cells2[col])
                    if mnum:
                        claimed_values.append(
                            (_num(mnum.group(1)),
                             f"таблица «{cells2[1][:40]}» → {cells2[col][:40]}"))
            break
    for claimed, frag in claimed_values:
        if not claimed or claimed > tax_before * 1.2:
            continue
        if expected is None:
            continue
        if abs(claimed - expected) > max(expected * 0.01, 1.0):
            issues.append(
                f"🔴 УСН-ИНВАРИАНТ НАРУШЕН: «{frag}» — пересчёт по п. 3.1 "
                f"ст. 346.21: налог до уменьшения {tax_before:,.0f} ₽, "
                f"уменьшение = min(взносы {contributions:,.0f}; 50% налога "
                f"{cap:,.0f}) = {min(contributions, cap):,.0f} ₽, после "
                f"уменьшения {expected:,.0f} ₽, в тексте {claimed:,.0f} ₽. "
                f"Блокировка публикации")
    return issues
