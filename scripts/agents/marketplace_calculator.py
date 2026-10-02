"""Детерминированный P&L-калькулятор юнит-экономики маркетплейса.

Тот же паттерн, что tax_calculator: расчёт кодом, не LLM.
Heart получает готовый объект с трассировкой и объясняет его словами.

Запуск в pipeline: для тем про маркетплейсы/юнит-экономику перед Heart.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger("agents.marketplace_calc")

TOPIC_MARKERS = (
    "маркетплейс", "юнит-экономик", "селлер", "fbo", "fbs", "dbs",
    "wb", "ozon", "Wildberries", "озон", "вайлдберриз",
)


def is_marketplace_topic(topic: str, description: str = "") -> bool:
    t = (topic or "") + " " + (description or "")
    return any(m.lower() in t.lower() for m in TOPIC_MARKERS)


@dataclass
class UnitEconomics:
    """Юнит-экономика товара на маркетплейсе."""

    price: float = 0.0                # цена на витрине
    commission_pct: float = 0.0       # комиссия площадки, % от цены
    logistics: float = 0.0            # логистика на единицу
    storage: float = 0.0              # хранение на единицу
    acquiring_pct: float = 0.0        # эквайринг, % от цены
    ad_spend_per_unit: float = 0.0    # реклама на единицу
    return_rate_pct: float = 0.0      # доля возвратов, %
    return_cost: float = 0.0          # стоимость одного возврата (обр. логистика + уценка)
    unit_cost: float = 0.0            # себестоимость (закупка + доставка + упаковка)
    tax_regime: str = "usn6"          # usn6 | usn15 | osno
    tax_rate: float = 0.06            # эффективная ставка для расчёта


def calc_unit_economics(inp: UnitEconomics) -> Dict:
    """Посчитать юнит-экономику с полной трассировкой."""
    steps: List[Dict] = []

    def step(name, formula, value):
        steps.append({"step": name, "formula": formula, "value": round(value, 2)})

    step("Цена на витрине", "price", inp.price)
    commission = inp.price * inp.commission_pct / 100
    step("Комиссия площадки", f"{inp.price:g} × {inp.commission_pct:g}%", -commission)
    acquiring = inp.price * inp.acquiring_pct / 100
    step("Эквайринг", f"{inp.price:g} × {inp.acquiring_pct:g}%", -acquiring)
    step("Логистика", "logistics", -inp.logistics)
    step("Хранение", "storage", -inp.storage)
    step("Реклама на единицу", "ad_spend_per_unit", -inp.ad_spend_per_unit)

    # Возвраты: доля проданных единиц, вернувшихся назад, размазанная
    # на каждую проданную единицу
    return_share = inp.return_rate_pct / 100
    returns_cost = return_share * inp.return_cost
    step("Возвраты (на единицу)", f"{inp.return_rate_pct:g}% × {inp.return_cost:g}", -returns_cost)

    payout_per_unit = (inp.price - commission - acquiring - inp.logistics
                       - inp.storage - inp.ad_spend_per_unit - returns_cost)
    step("Выплата площадки на единицу", "цена − удержания", payout_per_unit)

    step("Себестоимость", "unit_cost", -inp.unit_cost)
    margin_before_tax = payout_per_unit - inp.unit_cost
    step("Маржа до налога", "выплата − себестоимость", margin_before_tax)

    if inp.tax_regime == "usn6":
        tax = inp.price * inp.tax_rate
        step("Налог УСН 6%", f"{inp.price:g} × {inp.tax_rate:g}", -tax)
    elif inp.tax_regime == "usn15":
        tax = max(0.0, margin_before_tax) * inp.tax_rate
        step("Налог УСН 15%", f"max(0, {margin_before_tax:g}) × {inp.tax_rate:g}", -tax)
    else:  # osno — грубая оценка, детальный НДС-контур в tax_calculator
        tax = max(0.0, margin_before_tax) * 0.20
        step("Налог на прибыль (оценка)", f"max(0, {margin_before_tax:g}) × 20%", -tax)

    net = margin_before_tax - tax
    step("Чистая юнит-маржа", "маржа − налог", net)
    margin_pct = (net / inp.price * 100) if inp.price else 0.0
    step("Маржа, % от цены", f"{net:g} / {inp.price:g}", round(margin_pct, 2))

    return {
        "model": "marketplace_unit_economics_v1",
        "inputs": asdict(inp),
        "steps": steps,
        "result": {
            "payout_per_unit": round(payout_per_unit, 2),
            "margin_before_tax": round(margin_before_tax, 2),
            "net_margin": round(net, 2),
            "net_margin_pct": round(margin_pct, 2),
        },
        "assumptions_note": (
            "Все ставки — допущения модели; комиссии и тарифы площадок "
            "меняются, в статье обязательно указывать дату снимка тарифов "
            "и источник долей возвратов/рекламы"
        ),
    }


def build_heart_block(result: Dict, max_chars: int = 2600) -> str:
    """Форматировать объект расчёта для вставки в промпт Heart."""
    r = result.get("result", {})
    lines = [
        "=== РАСЧЁТ ЮНИТ-ЭКОНОМИКИ (детерминированный, не пересчитывай — объясняй) ===",
        f"Выплата площадки на единицу: {r.get('payout_per_unit', '?')} ₽",
        f"Маржа до налога: {r.get('margin_before_tax', '?')} ₽",
        f"Чистая юнит-маржа: {r.get('net_margin', '?')} ₽ "
        f"({r.get('net_margin_pct', '?')}% от цены)",
        "",
        "Трассировка:",
    ]
    for s in result.get("steps", []):
        lines.append(f"- {s['step']}: {s['formula']} = {s['value']}")
    lines.append("")
    lines.append(result.get("assumptions_note", ""))
    block = "\n".join(lines)
    return block[:max_chars]


# Демонстрационные допущения для тестов и первичного прогона
DEMO = UnitEconomics(
    price=2000.0, commission_pct=15.0, logistics=90.0, storage=15.0,
    acquiring_pct=1.5, ad_spend_per_unit=150.0, return_rate_pct=5.0,
    return_cost=200.0, unit_cost=900.0, tax_regime="usn6", tax_rate=0.06,
)


if __name__ == "__main__":
    import json
    print(json.dumps(calc_unit_economics(DEMO), ensure_ascii=False, indent=1))
