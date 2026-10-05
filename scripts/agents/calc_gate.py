# -*- coding: utf-8 -*-
"""CALCULATION_TRACE_REQUIRED (план фиксировщика, P0).

Детерминированный гейт расчётов: таблицы с налоговой/финансовой нагрузкой
в черновике сверяются с одобренным блоком калькуляторов (state.approved_calc_block).
Число в нагрузочной таблице, которого нет в одобренных расчётах, = красный флаг:
Heart не имеет права дорисовывать цифры, которых детерминированный калькулятор
не выдавал (кейс статьи 36: «ОСНО 4,68 млн» при УСН-only калькуляторе).
"""
import re
from typing import List, Tuple

# Таблица считается «нагрузочной», только если это таблица РЕЗУЛЬТАТОВ
# (что к уплате / нагрузка / санкции), а не справочник порогов и ставок
_TABLE_CONTEXT = re.compile(
    r"НДС\s+к\s+уплат\w*|налог\w*\s+к\s+уплат\w*|нагрузк\w*|штраф\w*\s+"
    r"(?:состав\w+|по)|пени|сэкономит|экономи\w*\s+состав\w*|"
    r"к\s+уплате\s+(?:итого|всего)", re.I)

# Допущения, которые обязаны быть раскрыты рядом с нагрузочной таблицей
_ASSUMPTION_MARKERS = re.compile(
    r"допущени|входной\s+НДС|расход\w*\s*(?:составля|приня)|"
    r"цена\s+(?:с\s+НДС|без\s+НДС|включает|начисляется)|формул\w*|"
    r"условн\w+\s+пример|предположим|исходн\w+\s+данн\w*|"
    r"по\s+этим\s+(?:вводным|данным)|при\s+этих\s+услови|"
    r"выручка\s+(?:без\s+НДС)?[:\s]*\d|объект\s+(?:УСН|налогообложени)", re.I)

_MONEY_RE = re.compile(
    r"(\d[\d\s.,]{0,22}?)\s*(млрд|млн|тыс\.?)?\s*(?:₽|рубл?\w*)", re.I)

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _parse_money(raw: str, unit: str) -> float:
    r"""«3,6 млн» -> 3_600_000.0; «6,000,000 ₽» -> 6_000_000.0 (тысячные
    запятые разграничиваются формой \d{1,3}(,\d{3})+); «400000» -> 400000.0."""
    num = raw.replace(" ", "").replace("\u00a0", "")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", num):
        num = num.replace(",", "")          # разделитель тысяч
    else:
        num = num.replace(",", ".")         # десятичная запятая
    try:
        val = float(num)
    except ValueError:
        return 0.0
    u = (unit or "").lower()
    if u.startswith("млрд"):
        return val * 1_000_000_000
    if u.startswith("млн"):
        return val * 1_000_000
    if u.startswith("тыс"):
        return val * 1_000
    return val


def _extract_tables(text: str) -> List[Tuple[str, str]]:
    """Разбить markdown на таблицы. Возвращает [(таблица, контекст ±300 зн.)]."""
    tables: List[Tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            start = i
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            table = "\n".join(lines[start:i])
            before = "\n".join(lines[max(0, start - 6):start])
            after = "\n".join(lines[i:i + 6])
            tables.append((table, before + "\n" + after))
        else:
            i += 1
    return tables


def _table_numbers(table: str) -> List[float]:
    """Денежные числа таблицы в каноническом виде (рубли)."""
    out = []
    for raw, unit in _MONEY_RE.findall(table):
        v = _parse_money(raw, unit)
        if v > 0:
            out.append(v)
    return out


def check_calc_trace(text: str, approved_block: str) -> List[str]:
    """Проверить нагрузочные таблицы черновика против одобренных расчётов.

    Возвращает список проблем:
    - 🔴 CALC_TRACE_REQUIRED — нагрузочная таблица без раскрытых допущений;
    - 🔴 CALC_NUMBER_UNAPPROVED — число из таблицы отсутствует в одобренном
      блоке калькулятора (допуск 0.5% на округление).
    """
    issues: List[str] = []
    approved_numbers = set(_table_numbers(approved_block or ""))

    for table, ctx in _extract_tables(text):
        if not _TABLE_CONTEXT.search(table):
            continue
        window = ctx + "\n" + table
        if not _ASSUMPTION_MARKERS.search(window):
            head = table.splitlines()[0][:80]
            issues.append(
                f"🔴 CALC_TRACE_REQUIRED: нагрузочная таблица «{head}…» без раскрытых "
                f"допущений. Обязательны: выручка (с НДС/без), расходы, входной НДС, "
                f"объект УСН, формулы — либо таблица удаляется")
            continue

        for v in _table_numbers(table):
            if v >= 100_000 and not any(
                    abs(v - a) <= max(a, v) * 0.005 for a in approved_numbers):
                shown = f"{v / 1_000_000:.2f} млн" if v >= 1_000_000 else f"{v:,.0f}"
                issues.append(
                    f"🔴 CALC_NUMBER_UNAPPROVED: {shown} ₽ в таблице отсутствует в "
                    f"одобренном блоке калькулятора — дорисовывать цифры запрещено")
                break  # одна пометка на таблицу
    return issues


_TOTAL_HEADER = re.compile(r"итого|всего|нагрузк", re.I)
_PCT_CELL = re.compile(r"^\s*\d+(?:[.,]\d+)?\s*%\s*$")


def _row_money(cell: str):
    """Деньги из ячейки или None (проценты/текст пропускаются)."""
    c = cell.strip()
    if not c or _PCT_CELL.match(c):
        return None
    m = _MONEY_RE.search(c)
    if not m:
        return None
    v = _parse_money(m.group(1), m.group(2))
    return v if v > 0 else None


def check_table_reconciliation(text: str) -> List[str]:
    """TABLE_RECONCILIATION_GATE (ревью 37, BLOCKER без LLM): строка «Итого /
    Всего нагрузка» обязана равняться сумме раскрытых компонентов строки.
    Кейс статьи 37: 6 000 000 + 3 600 000 = 9 600 000, а «Итого» 13 579 208 —
    читатель не может воспроизвести модель."""
    issues: List[str] = []
    for table, ctx in _extract_tables(text):
        if not _TABLE_CONTEXT.search(table):
            continue
        rows = [l for l in table.splitlines() if l.lstrip().startswith("|")]
        if len(rows) < 3:
            continue
        header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
        total_idx = next((i for i, h in enumerate(header)
                          if _TOTAL_HEADER.search(h)), None)
        if total_idx is not None:
            for row in rows[2:]:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                if len(cells) <= total_idx:
                    continue
                total = _row_money(cells[total_idx])
                if total is None:
                    continue
                s = 0.0
                for i, c in enumerate(cells):
                    if i in (0, total_idx):
                        continue
                    v = _row_money(c)
                    if v is not None:
                        s += v
                if s > 0 and abs(s - total) > total * 0.01:
                    label = cells[0][:40] if cells else "?"
                    issues.append(
                        f"🔴 TABLE_RECONCILIATION_FAIL: «{label}…» — Итого {total:,.0f} ₽ "
                        f"не равна сумме раскрытых компонентов {s:,.0f} ₽. Либо считайте "
                        f"только налоги (НДС + УСН), либо включайте остальные элементы "
                        f"(взносы, ФОТ) отдельной строкой с формулой и пояснением")
                    break  # одна пометка на таблицу

        # ТРАНСПОНИРОВАННЫЙ вид: итог — СТРОКА («Общая нагрузка»), компоненты —
        # строки выше (кейс статьи 37: нагрузка по сценариям в колонках).
        total_row_i = next((i for i, r in enumerate(rows[2:])
                            if _TOTAL_HEADER.search(r)), None)
        if total_idx is None and total_row_i is not None:
            data_rows = rows[2:]
            total_cells = [c.strip() for c in
                           data_rows[total_row_i].strip().strip("|").split("|")]
            for ci in range(1, len(total_cells)):
                total = _row_money(total_cells[ci])
                if total is None:
                    continue
                s = 0.0
                for i2, r2 in enumerate(data_rows):
                    if i2 == total_row_i:
                        continue
                    cells2 = [c.strip() for c in r2.strip().strip("|").split("|")]
                    if len(cells2) <= ci:
                        continue
                    v = _row_money(cells2[ci])
                    if v is not None:
                        s += v
                if s > 0 and abs(s - total) > total * 0.01:
                    label = total_cells[0][:40] if total_cells else "?"
                    issues.append(
                        f"🔴 TABLE_RECONCILIATION_FAIL: колонка «{label}…» — итог "
                        f"{total:,.0f} ₽ не равен сумме компонентов {s:,.0f} ₽. "
                        f"Либо считайте только налоги (НДС + УСН), либо показывайте "
                        f"остальные элементы отдельной строкой с формулой")
                    break  # одна пометка на таблицу
    return issues


def check_calc_claims(text: str) -> List[str]:
    """Запрет оценочных расчётных выводов без раскрытых допущений."""
    issues: List[str] = []
    pat = re.compile(
        r"оптимальн\w+\s+ставк\w*|минимальн\w+\s+(?:налогов\w+\s+)?нагрузк\w*|"
        r"выгоднее\s+применять|компания\s+сэкономит|экономия\s+составит|"
        r"минимальная\s+нагрузка\s+достигается", re.I)
    ok = re.compile(
        r"допущени|при\s+условии|для\s+этого\s+сценария|по\s+этим\s+(?:вводным|данным)|"
        r"при\s+высок\w+\s+доле\s+входного|зависит\s+от\s+(?:дол|структур)|"
        r"расчёт\s+по\s+своим\s+данным", re.I)
    for m in list(pat.finditer(text))[:3]:
        window = text[max(0, m.start() - 200):m.end() + 200]
        if not ok.search(window):
            issues.append(
                f"🟡 CALC_CLAIM_UNPROVEN: «{m.group(0)}» без явного набора допущений — "
                f"при высокой доле входного НДС вывод может развернуться на 180°. "
                f"Добавьте условия («при доле вычетов до X%») или уберите категоричность")
    return issues
