"""Обязательные сущности по доменам + HTML-валидатор.

required_entities: статья может быть без прямой лжи, но опасной из-за
ОТСУТСТВИЯ критического условия (кейс: оборотные штрафы без 24/72 часов).
html_validator: структура таблиц, якоря содержания, LaTeX-артефакты.
"""
from __future__ import annotations

import re
from typing import List

# ── Обязательные сущности ──
REQUIRED_ENTITIES = {
    # утечки ПДн: без сроков реагирования инструкция опасна
    "breach": {
        "when": lambda t, d: d == "защита данных" and
        re.search(r"утечк|инцидент|оборотн\w+\s+штраф", t, re.I),
        "required_any_groups": [
            [r"24\s*час", r"72\s*час", r"ст\.\s*21\s*152-ФЗ"],
        ],
        "missing": "🟡 НЕТ ОБЯЗАТЕЛЬНОГО КОНТЕКСТА: статья про утечки/инциденты "
                   "без сроков реагирования (24/72 часа, ст. 21 152-ФЗ) — "
                   "инструкция без них опасна",
    },
    # НДС на УСН: порог и дата возникновения
    "usn_vat": {
        "when": lambda t, d: d == "налоги" and
        re.search(r"НДС", t) and re.search(r"УСН", t),
        "required_any_groups": [
            [r"20\s*млн", r"освобождени\w+\s+от\s+НДС"],
            [r"первого\s+числа|1-го\s+числа|с\s+1-го\s+числа\s+месяца,\s+следующего"],
        ],
        "missing": "🟡 НЕТ ОБЯЗАТЕЛЬНОГО КОНТЕКСТА: НДС на УСН без порога "
                   "освобождения или правила «с 1-го числа следующего месяца» — "
                   "описание неполное",
    },
    # взносы ИП: разведение за себя / за работников
    "ip_contrib": {
        "when": lambda t, d: d == "налоги" and
        re.search(r"взнос\w+", t) and re.search(r"\bИП\b", t),
        "required_any_groups": [
            [r"за\s+себя", r"за\s+работник|за\s+сотрудник"],
        ],
        "missing": "🟡 НЕТ ОБЯЗАТЕЛЬНОГО КОНТЕКСТА: взносы ИП без разведения "
                   "«за себя» / «за работников» — читатель смешает режимы",
    },
}


def check_required_entities(text: str, direction: str) -> List[str]:
    issues = []
    if not text:
        return issues
    for rule in REQUIRED_ENTITIES.values():
        try:
            if not rule["when"](text, direction):
                continue
            for group in rule["required_any_groups"]:
                if not any(re.search(p, text, re.I) for p in group):
                    issues.append(rule["missing"])
                    break
        except Exception:
            continue
    return issues


# ── HTML-валидатор (MVP: регексы; bs4 — если доступен) ──
_LATEX_ARTIFACTS = re.compile(
    r"\\\\times|\\times|\bdtimes\b|\\\\\(|\\\\\)|\d\\,\d|\\\\frac|"
    r">\s*times\s*<|120\\,000", re.I)
_ANCHOR = re.compile(r'href="#([^"]+)"')
_ID = re.compile(r'id="([^"]+)"')


def validate_html(html: str) -> List[str]:
    issues = []
    if not html:
        return issues
    # LaTeX-артефакты
    for m in list(_LATEX_ARTIFACTS.finditer(html))[:3]:
        frag = html[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")
        issues.append(f"🟡 LaTeX-АРТЕФАКТ в HTML: «{frag}» — формула не отрендерилась")
    # таблицы: строки с разным числом ячеек
    for tbl in re.findall(r"<table.*?</table>", html, re.S)[:5]:
        rows = re.findall(r"<tr.*?</tr>", tbl, re.S)
        counts = [len(re.findall(r"<t[dh]", r)) for r in rows]
        if counts and len(set(counts)) > 1:
            issues.append(f"🟡 КРИВАЯ ТАБЛИЦА в HTML: строки с разным числом "
                          f"ячеек ({counts[:6]}) — вёрстка поедет")
        if len(rows) >= 2 and not re.search(r"<th", tbl) and counts and counts[0] == 1:
            issues.append("🟡 СКЛЕЕННАЯ ТАБЛИЦА: первая строка из одной ячейки — "
                          "заголовок не распознан, таблица превращается в текст")
    # якоря содержания
    hrefs = set(_ANCHOR.findall(html))
    ids = set(_ID.findall(html))
    broken = [h for h in hrefs if h not in ids]
    if broken:
        issues.append(f"🟡 БИТЫЕ ЯКОРЯ: содержание ссылается на несуществующие "
                      f"id ({', '.join(sorted(broken)[:4])})")
    return issues
