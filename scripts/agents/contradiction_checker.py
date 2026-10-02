"""Чекер противоречий: H1/лид vs тело, конфликты значений параметров.

Детерминированный MVP:
1. Числа из H1 и лида должны встречаться в теле (или в claim pack).
2. Одноимённые параметры не должны иметь разные значения без маркера
   сравнения («было», «прежде», «устарел», «с 2025»).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_H1_RE = re.compile(r"^#\s+(.+)$", re.M)

# Значимые числа (суммы, ставки, пороги) — отсекаем порядковые маркеры списков
_SIGNIFICANT_NUM = re.compile(r"\d[\d\u00a0 ]{2,12}(?:[.,]\d+)?|\d+[.,]\d+|\d+%")
# Маленькие «номерные» числа (1., 2., 10 ошибок) не считаем утверждениями
_TRIVIAL = re.compile(r"^\d{1,2}$")
# Год в заголовке — контекст (статья года Y), не фактическое утверждение
_YEAR = re.compile(r"^(19|20)\d{2}$")
# Число + сроковая единица = дедлайн, требующий подтверждения («за 10 дней»)
_NUM_WITH_TERM = re.compile(
    r"\d{1,3}\s*(?:дн\w*|час\w*|недел\w*|месяц\w*|рабочих\s+дн\w*)", re.I)


def _numbers(s: str) -> set:
    out = set()
    for m in _SIGNIFICANT_NUM.finditer(s):
        raw = m.group(0).replace("\u00a0", " ").strip()
        if _TRIVIAL.match(raw):
            continue
        # нормализуем «57 390» → «57390» для сопоставимости
        norm = raw.replace(" ", "")
        out.add(norm)
    return out


def _deadline_numbers(s: str) -> set:
    """«за 10 дней», «24 часа» — числа-дедлайны (в т.ч. маленькие)."""
    return {m.group(0).replace("\u00a0", " ").strip()
            for m in _NUM_WITH_TERM.finditer(s)}


def check_h1_support(article: str, claim_pack_numbers: Optional[set] = None) -> List[str]:
    """Числа в H1 и первом абзаце должны подтверждаться телом статьи."""
    if not article:
        return []
    issues = []
    m = _H1_RE.search(article)
    head = m.group(1) if m else ""
    # лид: первый абзац после H1 (до первого H2)
    body_start = article.find("\n\n", m.end()) if m else 0
    next_h2 = article.find("\n## ", m.end() if m else 0)
    lead = article[body_start:next_h2 if next_h2 > 0 else body_start + 1500]
    body = article[next_h2:] if next_h2 > 0 else ""

    body_nums = _numbers(body)
    allowed = body_nums | (claim_pack_numbers or set())
    for part_name, part in (("H1", head), ("лид", lead)):
        part_nums = _numbers(part)
        # дедлайны «за N дней/часов» — подтверждение обязательнее обычных чисел
        for d in _deadline_numbers(part):
            if d not in body and d not in (claim_pack_numbers or set()):
                frag = next((l for l in part.splitlines() if d in l), part)[:60]
                issues.append(
                    f"🔴 ДЕДЛАЙН В {part_name.upper()} БЕЗ ПОДТВЕРЖДЕНИЯ: «{d}» "
                    f"({frag.strip()}…) — в теле и claim pack нет ни этого срока, "
                    f"ни его нормативного основания. Выдуманный дедлайн = брак")
        for n in part_nums:
            if _YEAR.match(n):
                continue  # год публикации/темы — не значение, а контекст
            if n not in allowed:
                frag = next((l for l in part.splitlines() if n in l), part)[:60]
                issues.append(
                    f"🔴 ЧИСЛО В {part_name.upper()} БЕЗ ПОДТВЕРЖДЕНИЯ: «{n}» "
                    f"({frag.strip()}…) — в теле статьи и claim pack его нет. "
                    f"Заголовок обещает то, чего статья не подтверждает")
    return issues


# Конфликт значений: одноимённая сущность с разными числами без контекста сравнения
_COMPARE_MARKERS = re.compile(
    r"было|прежде|ранее|устарел|до\s+20\d\d|с\s+20\d\d|предыдущ\w+|вместо|был\s+порог|"
    r"не\s+путать|для\s+сравнения|разниц|изменени", re.I)

# Пары «маркер сущности → что считать конфликтом»: вхождения числа рядом с маркером
_CONFLICT_ENTITIES = {
    "usn_vat_limit": (r"(?:порог|лимит|освобождени\w+)[^.\n]{0,40}НДС", None),
    "ip_fixed_contrib": (r"фиксированн\w+\s+взнос\w+", None),
}


def check_value_conflicts(text: str) -> List[str]:
    """Пример: «порог НДС 20 млн» и «порог НДС 60 млн» в одном тексте
    без маркеров сравнения (было/стало) → противоречие."""
    issues = []
    for name, (pat, _) in _CONFLICT_ENTITIES.items():
        rx = re.compile(pat, re.I)
        vals = set()
        for m in rx.finditer(text):
            window = text[m.end():m.end() + 120]
            for n in _SIGNIFICANT_NUM.finditer(window):
                vals.add(n.group(0).replace("\u00a0", " ").strip())
        if len(vals) > 1:
            # есть ли рядом маркеры сравнения?
            ctx = text[max(0, text.lower().find(pat.split("[")[0][:12])):][:2000]
            if not _COMPARE_MARKERS.search(text):
                issues.append(
                    f"🔴 КОНФЛИКТ ЗНАЧЕНИЙ ({name}): найдено несколько чисел у одной "
                    f"сущности — {', '.join(sorted(vals)[:4])} — без маркеров "
                    f"сравнения («было/стало», «с 2026»). Проверьте, какое значение "
                    f"действующее")
    return issues


# ── NORM_REFERENCE_MISMATCH: номер закона без опоры в источниках ──
_FZ_REF = re.compile(r"№\s*(\d{1,3})\s*[-–]\s*ФЗ", re.I)
_FZ_IN_TEXT = re.compile(r"от\s+(\d{2})\.(\d{2})\.(\d{4})\s+№\s*(\d{1,3})\s*[-–]\s*ФЗ", re.I)


def check_norm_references(text: str, supporting_text: str) -> List[str]:
    """ФЗ-номера в статье должны встречаться в поддерживающем наборе
    (claim pack + источники + паспорт). Иначе — реквизит из вторички.

    Кейс 22: источники дали № 144-ФЗ, тело статьи — № 91-ФЗ."""
    if not text:
        return []
    support_nums = {m.group(1) for m in _FZ_REF.finditer(supporting_text or "")}
    if not support_nums:
        return []  # опоры нет — нечем сверять, молчим
    issues = []
    for m in _FZ_IN_TEXT.finditer(text):
        num = m.group(4)
        if num not in support_nums:
            frag = text[max(0, m.start() - 30):m.end() + 20].replace("\n", " ")
            issues.append(
                f"🟡 NORM REFERENCE: «{frag.strip()}» — номер закона "
                f"№ {num}-ФЗ отсутствует в claim pack/источниках (опорные: "
                f"№ {'-, № '.join(sorted(support_nums)[:4])}-ФЗ). Реквизит из "
                f"вторичного источника; сверьте с первоисточником")
    return issues
