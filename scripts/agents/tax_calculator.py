"""
Налоговый калькулятор — детерминированный модуль расчётов для статей.

Проблема (разбор «УСН или ОСНО» 4.5/10): LLM не может надёжно считать налоги
в тексте. Она путает «22% без вычетов» (неверно для общей ставки на УСН),
не различает «цена с НДС / без НДС», и строит таблицы, которые невозможно
воспроизвести из входных данных.

Решение: все расчёты делает ЭТОТ модуль (чистый Python, ноль LLM).
Heart получает готовые числа + формулы + пояснения и ОБЯЗАН вывести их
дословно. Любая цифра в статье, не совпадающая с результатом калькулятора,
ловится детерминированной сверкой (enforce_params).

Покрываемые сценарии (2026 год):
  1. УСН + НДС 5% без вычетов
  2. УСН + НДС 7% без вычетов
  3. УСН + общая ставка НДС 22% с вычетами
  4. ОСНО (НДС 22% + налог на прибыль/НДФЛ + УСН-налога нет)

Вход: выручка (с/без НДС), закупки (с/без НДС), наценка, форма (ИП/ООО),
доля B2B (для контекста), зарплаты.
Выход: начисленный НДС, входной НДС, НДС к уплате, УСН-налог,
прибыльный налог (ОСНО), совокупная нагрузка, дельты между сценариями.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================================
# Параметры 2026 года (источник: norm_params / паспорт)
# ============================================================

PARAMS_2026 = {
    "vat_standard": 0.22,
    "vat_usn_special_1": 0.05,      # до порога
    "vat_usn_special_2": 0.07,      # после превышения порога спецставки
    "vat_threshold_special_1": 272_500_000,   # порог спецставки 5%
    "usn_income_rate": 0.06,
    "usn_income_expense_rate": 0.15,
    "profit_tax_rate": 0.25,        # налог на прибыль ООО на ОСНО (2026)
    "ndfl_ip_rate_base": 0.13,      # упрощённо: базовая НДФЛ для ИП на ОСНО
    "insurance_fixed": 57_390,
    "insurance_1pct_threshold": 300_000,
    "insurance_1pct_max": 321_818,
    "usn_revenue_limit": 490_500_000,
    "vat_vat_exemption_threshold": 20_000_000,
}


@dataclass
class ScenarioInput:
    """Входные данные одного расчёта."""
    name: str                       # человекочитаемое имя сценария
    legal_form: str                 # "IP" | "OOO"
    regime: str                     # "USN" | "OSNO"
    vat_method: str                 # "special_5" | "special_7" | "standard_22_with_deductions" | "none"
    revenue_ex_vat: float           # выручка БЕЗ НДС (или с НДС — см. includes_vat)
    purchases_ex_vat: float         # закупки БЕЗ НДС
    includes_vat: bool = False      # True, если revenue/purchases указаны С НДС
    usn_object: str = "income"      # "income" (6%) | "income_minus_expenses" (15%)
    usn_rate_override: Optional[float] = None
    employees: bool = True
    payroll: float = 0.0
    year: int = 2026
    b2b_share: float = 0.5          # доля клиентов-плательщиков НДС (для контекста)


@dataclass
class ScenarioResult:
    """Результат одного расчёта — трассируемая таблица."""
    name: str
    regime: str
    vat_method: str
    # НДС
    revenue_ex_vat: float = 0.0
    output_vat: float = 0.0
    purchases_ex_vat: float = 0.0
    input_vat: float = 0.0
    vat_payable: float = 0.0
    vat_deduction_allowed: bool = False
    vat_rate: float = 0.0
    # Налог режима
    usn_tax: float = 0.0
    usn_after_reduction: float = 0.0
    profit_or_ndfl_tax: float = 0.0
    profit_base: float = 0.0
    insurance: float = 0.0
    # Трассировка (для полной воспроизводимости)
    payroll: float = 0.0
    insurance_employer: float = 0.0
    other_expenses: float = 0.0
    # Итог
    total_burden: float = 0.0
    effective_rate: float = 0.0
    # Пояснения для Heart
    formulas: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    traceability: List[str] = field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "regime": self.regime, "vat_method": self.vat_method,
            "revenue_ex_vat": self.revenue_ex_vat, "output_vat": self.output_vat,
            "purchases_ex_vat": self.purchases_ex_vat, "input_vat": self.input_vat,
            "vat_payable": self.vat_payable, "vat_deduction_allowed": self.vat_deduction_allowed,
            "vat_rate": self.vat_rate, "usn_tax": self.usn_tax,
            "profit_or_ndfl_tax": self.profit_or_ndfl_tax, "insurance": self.insurance,
            "total_burden": self.total_burden, "effective_rate": self.effective_rate,
            "formulas": self.formulas, "notes": self.notes, "ok": self.ok,
        }


def _to_ex_vat(amount: float, vat_rate: float, includes_vat: bool) -> float:
    """Привести сумму к виду «без НДС». Если цена С НДС: x / (1 + rate)."""
    if not includes_vat:
        return amount
    return amount / (1 + vat_rate)


# ============================================================
# Ядро расчёта
# ============================================================

def calc_scenario(inp: ScenarioInput) -> ScenarioResult:
    """Детерминированный расчёт одного сценария. Ноль LLM."""
    P = PARAMS_2026
    r = ScenarioResult(name=inp.name, regime=inp.regime, vat_method=inp.vat_method)

    # ── 1. Приводим к «без НДС» ──
    if inp.regime == "OSNO" or inp.vat_method == "standard_22_with_deductions":
        r.vat_rate = P["vat_standard"]
        # Для ОСНО / общей ставки на УСН: выручка С НДС делится на 1.22
        r.revenue_ex_vat = _to_ex_vat(inp.revenue_ex_vat, r.vat_rate, inp.includes_vat)
        r.purchases_ex_vat = _to_ex_vat(inp.purchases_ex_vat, r.vat_rate, inp.includes_vat)
    else:
        # Спецставки 5/7: начисляются на выручку (включая НДС по версии НК),
        # но для сравнения с ОСНО приводим к «без НДС» по 22%
        r.revenue_ex_vat = _to_ex_vat(inp.revenue_ex_vat, P["vat_standard"], inp.includes_vat)
        r.purchases_ex_vat = _to_ex_vat(inp.purchases_ex_vat, P["vat_standard"], inp.includes_vat)
        r.vat_rate = (P["vat_usn_special_1"] if inp.vat_method == "special_5"
                      else P["vat_usn_special_2"] if inp.vat_method == "special_7" else 0.0)

    # ── 2. НДС ──
    if inp.vat_method == "special_5":
        r.output_vat = r.revenue_ex_vat * r.vat_rate
        r.input_vat = 0.0
        r.vat_deduction_allowed = False
        r.vat_payable = r.output_vat
        r.formulas.append(f"НДС к уплате = Выручка (без НДС по 22%) × 5% = {r.revenue_ex_vat:,.0f} × 0.05 = {r.vat_payable:,.0f} ₽")
        r.notes.append("Спецставка 5%: входной НДС НЕ принимается к вычету.")
    elif inp.vat_method == "special_7":
        r.output_vat = r.revenue_ex_vat * r.vat_rate
        r.input_vat = 0.0
        r.vat_deduction_allowed = False
        r.vat_payable = r.output_vat
        r.formulas.append(f"НДС к уплате = Выручка × 7% = {r.vat_payable:,.0f} ₽")
        r.notes.append("Спецставка 7%: входной НДС НЕ принимается к вычету.")
    elif inp.vat_method == "standard_22_with_deductions":
        r.output_vat = r.revenue_ex_vat * r.vat_rate
        r.input_vat = r.purchases_ex_vat * r.vat_rate
        r.vat_deduction_allowed = True
        r.vat_payable = max(0, r.output_vat - r.input_vat)
        r.formulas.append(f"Исходящий НДС = {r.revenue_ex_vat:,.0f} × 22% = {r.output_vat:,.0f} ₽")
        r.formulas.append(f"Входной НДС = {r.purchases_ex_vat:,.0f} × 22% = {r.input_vat:,.0f} ₽")
        r.formulas.append(f"НДС к уплате = {r.output_vat:,.0f} − {r.input_vat:,.0f} = {r.vat_payable:,.0f} ₽")
        r.notes.append("Общая ставка 22% на УСН: входной НДС принимается к вычету на общих условиях.")
    elif inp.vat_method == "none":
        if inp.regime == "USN":
            r.notes.append("УСН без НДС: доход до 20 млн ₽ (освобождение от НДС).")
        else:
            r.notes.append("ОСНО всегда с НДС — метод 'none' неприменим.")
            r.ok, r.error = False, "ОСНО не бывает без НДС"
            return r
    else:
        r.ok, r.error = False, f"Неизвестный vat_method: {inp.vat_method}"
        return r

    # ── 3. Налог режима ──
    # Трассировка: все входные параметры фиксируются для воспроизводимости
    r.payroll = inp.payroll
    # Взносы работодателя (~30% от ФОТ, упрощённо)
    r.insurance_employer = inp.payroll * 0.30 if inp.employees else 0.0

    if inp.regime == "USN":
        if inp.usn_object == "income":
            rate = inp.usn_rate_override or P["usn_income_rate"]
            r.usn_tax = r.revenue_ex_vat * rate
            r.formulas.append(f"УСН 6% = {r.revenue_ex_vat:,.0f} × 0.06 = {r.usn_tax:,.0f} ₽ (до уменьшения на взносы)")
            # Уменьшение на страховые взносы: без работников — до 100%,
            # с работниками — не более 50% (п. 3.1 ст. 346.21 НК РФ).
            # ФИКС (аудит 07-🔴1, 2026-10-02): с работниками в уменьшение
            # входят взносы ЗА СЕБЯ (фикс + 1%) И ЗА РАБОТНИКОВ — раньше
            # считались только «за себя», занижая уменьшение на миллионы.
            ins = P["insurance_fixed"]
            if r.revenue_ex_vat > P["insurance_1pct_threshold"]:
                ins += min((r.revenue_ex_vat - P["insurance_1pct_threshold"]) * 0.01, P["insurance_1pct_max"])
            # r.insurance = только «за себя» (семантика едина с ОСНО-веткой);
            # работодателя учитываем отдельно в reduction и total_burden.
            r.insurance = ins
            if inp.employees:
                ins_full = ins + r.insurance_employer
            else:
                ins_full = ins
            if inp.employees:
                reduction = min(ins_full, r.usn_tax * 0.5)
                r.usn_after_reduction = r.usn_tax - reduction
                r.formulas.append(f"Уменьшение на взносы (за себя {P['insurance_fixed']:,.0f} + 1% + за работников {r.insurance_employer:,.0f}; max 50% налога) = min({ins_full:,.0f}; {r.usn_tax:,.0f} × 50%) = {reduction:,.0f} ₽")
                r.formulas.append(f"УСН к уплате = {r.usn_tax:,.0f} − {reduction:,.0f} = {r.usn_after_reduction:,.0f} ₽")
                r.notes.append("С работниками УСН уменьшается на взносы за себя И за работников НЕ БОЛЕЕ чем на 50% (п. 3.1 ст. 346.21 НК РФ).")
            else:
                reduction = min(ins, r.usn_tax)
                r.usn_after_reduction = r.usn_tax - reduction
                r.formulas.append(f"Уменьшение на взносы (без работников, до 100%) = min({ins:,.0f}; {r.usn_tax:,.0f}) = {reduction:,.0f} ₽")
                r.formulas.append(f"УСН к уплате = {r.usn_tax:,.0f} − {reduction:,.0f} = {r.usn_after_reduction:,.0f} ₽")
                r.notes.append("Без работников УСН уменьшается на взносы до 100%.")
        else:
            rate = inp.usn_rate_override or P["usn_income_expense_rate"]
            base = max(0, r.revenue_ex_vat - r.purchases_ex_vat)
            r.usn_tax = base * rate
            r.profit_base = base
            r.formulas.append(f"УСН 15% = ({r.revenue_ex_vat:,.0f} − {r.purchases_ex_vat:,.0f}) × 0.15 = {r.usn_tax:,.0f} ₽")
            ins = P["insurance_fixed"]
            if r.revenue_ex_vat > P["insurance_1pct_threshold"]:
                ins += min((r.revenue_ex_vat - P["insurance_1pct_threshold"]) * 0.01, P["insurance_1pct_max"])
            r.insurance = ins
            r.usn_after_reduction = r.usn_tax  # на «доходы-расходы» взносы идут в расходы, не уменьшают налог
            r.notes.append("УСН «доходы минус расходы»: взносы учитываются в расходах, налог напрямую не уменьшают.")
    elif inp.regime == "OSNO":
        # Трассировка базы прибыли: выручка − закупки − ФОТ − взносы работодателя
        r.other_expenses = 0.0
        if inp.legal_form == "OOO":
            base = max(0, r.revenue_ex_vat - r.purchases_ex_vat - inp.payroll - r.insurance_employer)
            r.profit_or_ndfl_tax = base * P["profit_tax_rate"]
            r.profit_base = base
            r.formulas.append(f"Налог на прибыль = ({r.revenue_ex_vat:,.0f} − {r.purchases_ex_vat:,.0f} "
                              f"− {inp.payroll:,.0f} (ФОТ) − {r.insurance_employer:,.0f} (взносы раб.)) "
                              f"× 25% = {r.profit_or_ndfl_tax:,.0f} ₽")
            r.notes.append("ОСНО ООО: налог на прибыль 25% (2026). НДС по общей механике с вычетами.")
            r.insurance = 0.0  # взносы уже в payroll
            r.traceability = [
                f"Выручка (без НДС):     {r.revenue_ex_vat:>14,.0f} ₽",
                f"Закупки (без НДС):     {r.purchases_ex_vat:>14,.0f} ₽",
                f"ФОТ:                   {inp.payroll:>14,.0f} ₽",
                f"Взносы работодателя:   {r.insurance_employer:>14,.0f} ₽ (30% от ФОТ)",
                f"Налоговая прибыль:     {base:>14,.0f} ₽",
                f"Налог на прибыль 25%:  {r.profit_or_ndfl_tax:>14,.0f} ₽",
            ]
        else:  # IP на ОСНО
            base = max(0, r.revenue_ex_vat - r.purchases_ex_vat - inp.payroll - r.insurance_employer)
            # Прогрессивная шкала НДФЛ — по СТУПЕНЯМ, не плоской ставкой
            brackets = [
                (2_400_000, 0.13),
                (5_000_000, 0.15),
                (20_000_000, 0.18),
                (50_000_000, 0.20),
                (float('inf'), 0.22),
            ]
            ndfl = 0.0
            remaining = base
            prev_threshold = 0.0
            bracket_details = []
            for threshold, rate in brackets:
                if remaining <= 0:
                    break
                taxable_in_bracket = min(remaining, threshold - prev_threshold)
                tax_in_bracket = taxable_in_bracket * rate
                ndfl += tax_in_bracket
                if taxable_in_bracket > 0:
                    bracket_details.append(f"{taxable_in_bracket:,.0f} × {rate*100:.0f}% = {tax_in_bracket:,.0f} ₽")
                remaining -= taxable_in_bracket
                prev_threshold = threshold
            r.profit_or_ndfl_tax = ndfl
            r.profit_base = base
            r.formulas.append(f"НДФЛ ИП (прогрессивная шкала, по ступеням):")
            for bd in bracket_details:
                r.formulas.append(f"  {bd}")
            r.formulas.append(f"  Итого НДФЛ = {ndfl:,.0f} ₽")
            ins = P["insurance_fixed"]
            if r.revenue_ex_vat > P["insurance_1pct_threshold"]:
                ins += min((r.revenue_ex_vat - P["insurance_1pct_threshold"]) * 0.01, P["insurance_1pct_max"])
            r.insurance = ins
            r.traceability = [
                f"Выручка (без НДС):     {r.revenue_ex_vat:>14,.0f} ₽",
                f"Закупки (без НДС):     {r.purchases_ex_vat:>14,.0f} ₽",
                f"ФОТ:                   {inp.payroll:>14,.0f} ₽",
                f"Взносы работодателя:   {r.insurance_employer:>14,.0f} ₽",
                f"Предприним. доход:     {base:>14,.0f} ₽",
                f"НДФЛ (по ступеням):    {ndfl:>14,.0f} ₽",
            ]

    # ── 4. Совокупная нагрузка (УСН после уменьшения на взносы) ──
    # ФИКС (аудит 05-§2.1, 🔴5): в «Всего нагрузка» не входили страховые
    # взносы (фикс ИП + 1% + работодателя) — сравнение режимов смещалось
    # в пользу УСН на десятки процентов. Взносы платятся в любом режиме.
    usn_final = getattr(r, "usn_after_reduction", None)
    if usn_final is None:
        usn_final = r.usn_tax
    # r.insurance теперь всегда только «за себя»; работодатель — отдельным полем
    _insurance_total = getattr(r, "insurance", 0.0) + getattr(r, "insurance_employer", 0.0)
    r.total_burden = r.vat_payable + usn_final + r.profit_or_ndfl_tax + _insurance_total
    r.notes.append(f"Всего нагрузка включает страховые взносы: "
                   f"{_insurance_total:,.0f} ₽ (фикс/1% + работодателя).")
    if r.revenue_ex_vat > 0:
        r.effective_rate = r.total_burden / r.revenue_ex_vat
    return r


def calc_comparison(
    revenue: float,
    purchases: float,
    includes_vat: bool = True,
    legal_form: str = "IP",
    usn_object: str = "income",
    payroll: float = 12_000_000,
    year: int = 2026,
) -> List[ScenarioResult]:
    """
    Полное сравнение сценариев. УСН 7% включается ТОЛЬКО при доходе > 272,5 млн
    (иначе для компании с 120 млн это неприменимый выбор — кейс редактора).
    """
    P = PARAMS_2026
    results = []

    # УСН 5% — применимо при доходе 20 млн — 272,5 млн
    results.append(calc_scenario(ScenarioInput(
        name="УСН + НДС 5% (без вычетов)", legal_form=legal_form, regime="USN",
        vat_method="special_5", revenue_ex_vat=revenue, purchases_ex_vat=purchases,
        includes_vat=includes_vat, usn_object=usn_object, payroll=payroll, year=year)))

    # УСН 7% — только если выручка > порога спецставки 5% (272,5 млн)
    if revenue > P["vat_threshold_special_1"]:
        results.append(calc_scenario(ScenarioInput(
            name="УСН + НДС 7% (без вычетов)", legal_form=legal_form, regime="USN",
            vat_method="special_7", revenue_ex_vat=revenue, purchases_ex_vat=purchases,
            includes_vat=includes_vat, usn_object=usn_object, payroll=payroll, year=year)))
    else:
        # Помечаем как неприменимый для этого уровня дохода
        r7 = ScenarioResult(name="УСН + НДС 7%", regime="USN", vat_method="special_7")
        r7.ok = False
        r7.error = (f"неприменимо при доходе {revenue:,.0f} ₽ (ставка 7% действует "
                    f"после превышения {P['vat_threshold_special_1']:,.0f} ₽)")
        results.append(r7)

    # УСН + общая ставка с вычетами
    results.append(calc_scenario(ScenarioInput(
        name="УСН + НДС 22% с вычетами", legal_form=legal_form, regime="USN",
        vat_method="standard_22_with_deductions", revenue_ex_vat=revenue,
        purchases_ex_vat=purchases, includes_vat=includes_vat,
        usn_object=usn_object, payroll=payroll, year=year)))
    # ОСНО
    results.append(calc_scenario(ScenarioInput(
        name="ОСНО (НДС 22% с вычетами)", legal_form=legal_form, regime="OSNO",
        vat_method="standard_22_with_deductions", revenue_ex_vat=revenue,
        purchases_ex_vat=purchases, includes_vat=includes_vat,
        usn_object=usn_object, payroll=payroll, year=year)))

    return results


# ============================================================
# Форматирование для Heart
# ============================================================

def format_for_heart(results: List[ScenarioResult]) -> str:
    """Готовая таблица сравнения с формулами для вставки в статью."""
    lines = ["=== РАСЧЁТ НАЛОГОВОЙ НАГРУЗКИ (детерминированный калькулятор, 2026) ===",
             "ЭТИ ЦИФРЫ — единственный источник расчётов в статье. Используй их дословно,",
             "включая формулы. Пересчитывать самостоятельно ЗАПРЕЩЕНО.", ""]
    lines.append("| Сценарий | НДС к уплате | УСН/приб. после уменьшения | Всего нагрузка | Эффективная ставка |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        if not r.ok:
            lines.append(f"| {r.name} | ошибка: {r.error} | | | |")
            continue
        usn_final = getattr(r, "usn_after_reduction", None)
        usn_final = usn_final if usn_final is not None else r.usn_tax
        tax = (f"УСН {usn_final:,.0f} ₽" if r.regime == "USN"
               else f"приб./НДФЛ {r.profit_or_ndfl_tax:,.0f} ₽")
        lines.append(f"| {r.name} | {r.vat_payable:,.0f} ₽ | {tax} | {r.total_burden:,.0f} ₽ | {r.effective_rate*100:.1f}% |")

    lines.append("")
    for r in results:
        if r.ok:
            lines.append(f"**{r.name}:**")
            for f in r.formulas:
                lines.append(f"  • {f}")
            if r.traceability:
                lines.append("  📊 Трассировка расчёта:")
                for tr in r.traceability:
                    lines.append(f"     {tr}")
            for n in r.notes:
                lines.append(f"  ℹ️ {n}")
            lines.append("")

    # Вывод: какой сценарий дешевле
    ok_results = [r for r in results if r.ok]
    if ok_results:
        best = min(ok_results, key=lambda x: x.total_burden)
        lines.append(f"ИТОГ: минимальная нагрузка — {best.name} ({best.total_burden:,.0f} ₽, "
                     f"{best.effective_rate*100:.1f}% от выручки). Но решение зависит от доли "
                     f"B2B-клиентов (им важен входной НДС) и структуры закупок.")
    # Помечаем неприменимые сценарии
    for r in results:
        if not r.ok and r.error and "неприменимо" in r.error:
            lines.append(f"\n⚠️ {r.name}: {r.error}. Не включай в основное сравнение для этого кейса — "
                         f"вынеси в отдельный блок «что изменится при росте выручки».")
    lines.append("=== КОНЕЦ РАСЧЁТА ===")
    return "\n".join(lines)


# ============================================================
# CLI для проверки
# ============================================================

if __name__ == "__main__":
    # Кейс из статьи редактора: оборот 120 млн, наценка 25% (закупки 78 млн)
    results = calc_comparison(
        revenue=120_000_000, purchases=78_000_000,
        includes_vat=False, legal_form="IP", usn_object="income")
    print(format_for_heart(results))
