"""Детерминированный калькулятор обеспечений по 44-ФЗ.

Паттерн tax_calculator / marketplace_calculator: расчёт кодом, не LLM.
Считает обеспечение заявки (ст. 44), обеспечение исполнения контракта
(ст. 96) и антидемпинговое повышение (ст. 37, редакция 484-ФЗ с 01.01.2026).
Heart получает готовый объект с трассировкой шагов и ОБЯЗАН использовать
его дословно — LLM не считает деньги.

Запуск в pipeline: для тем по госзакупкам перед Heart.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger("agents.procurement_calc")

TOPIC_MARKERS = (
    "закупк", "тендер", "госконтракт", "44-фз", "223-фз", "фас",
    "обеспечение заявки", "обеспечени", "антидемпинг", "нмцк",
    "единственный поставщик", "госзакуп", "участник закупки",
    "сгоз", "жалоба", "контрактная система", "контракт", "оик",
)

# Нормы 44-ФЗ, снимок 2026-09-29 (см. KB: 44-ФЗ_снимок_...)
BID_SEC_MAX_20M = (0.5, 1.0)    # ч. 2 ст. 44: НМЦК ≤ 20 млн → 0,5–1%
BID_SEC_OVER_20M = (0.5, 5.0)   # ч. 2 ст. 44: НМЦК > 20 млн → 0,5–5%
OIK_MIN_PCT, OIK_MAX_PCT = 0.5, 30.0   # ст. 96: ОИК 0,5–30% НМЦК
ANTIDUMP_DROP_PCT = 25.0        # ст. 37: снижение на 25% и более
ANTIDUMP_NMC_THRESHOLD = 15_000_000.0  # ст. 37: порог 15 млн
ANTIDUMP_MULT = 1.5             # ст. 37 ч. 1: превышение в полтора раза
ANTIDUMP_FLOOR_PCT = 10.0       # ст. 37 ч. 1: но не менее 10% НМЦК


def _fmt(x: float) -> str:
    """Без научной нотации: 18500000 → «18 500 000»."""
    if float(x) == int(x) and abs(x) >= 1000:
        return f"{int(x):,}".replace(",", " ")
    return f"{x:g}".replace(".", ",")


def is_procurement_topic(topic: str, description: str = "") -> bool:
    t = (topic or "") + " " + (description or "")
    tl = t.lower()
    # «жалоба»/«обеспечение» слишком широкие — требуют закупочного соседства
    weak = ("жалоба", "обеспечени", "сгоз")
    hits = [m for m in TOPIC_MARKERS if m in tl and m not in weak]
    weak_hits = [m for m in weak if m in tl]
    return bool(hits) or (bool(weak_hits) and (
        "закупк" in tl or "тендер" in tl or "44" in tl or "223" in tl
        or "фас" in tl or "госконтракт" in tl))


@dataclass
class SecurityInput:
    """Входные данные расчёта обеспечений."""

    nmc: float = 18_500_000.0        # НМЦК, руб.
    bid_sec_pct: float = 1.0         # обеспечение заявки, % НМЦК (устанавливает заказчик)
    oik_pct: float = 5.0             # обеспечение исполнения контракта, % НМЦК
    advance_pct: float = 0.0         # аванс, % НМЦК
    final_price: float = 13_000_000.0  # цена предложения участника, руб.


def calc_security(inp: SecurityInput) -> Dict:
    """Посчитать обеспечения с трассировкой по статьям 44-ФЗ."""
    steps: List[Dict] = []

    def step(name, formula, value, ref):
        steps.append({"step": name, "formula": formula,
                      "value": value, "норма": ref})

    # ── Обеспечение заявки (ст. 44 ч. 2) ──
    lo, hi = BID_SEC_MAX_20M if inp.nmc <= 20_000_000 else BID_SEC_OVER_20M
    in_range = lo <= inp.bid_sec_pct <= hi
    bid_sec = inp.nmc * inp.bid_sec_pct / 100
    step("Обеспечение заявки",
         f"{_fmt(inp.nmc)} × {_fmt(inp.bid_sec_pct)}%", bid_sec,
         f"ст. 44 ч. 2 (диапазон {lo:g}–{hi:g}% НМЦК"
         f"{' — ВНЕ ДИАПАЗОНА' if not in_range else ''})")

    # ── Обеспечение исполнения контракта (ст. 96) ──
    oik = inp.nmc * inp.oik_pct / 100
    advance = inp.nmc * inp.advance_pct / 100
    oik_effective = oik
    advance_note = ""
    if advance > oik:
        oik_effective = advance
        advance_note = " — поднято до аванса"
    step("Обеспечение исполнения контракта",
         f"{_fmt(inp.nmc)} × {_fmt(inp.oik_pct)}%", oik_effective,
         f"ст. 96 (0,5–30% НМЦК{advance_note}; при авансе ОИК ≥ аванса)")

    # ── Антидемпинг (ст. 37, ред. 484-ФЗ с 01.01.2026) ──
    drop_pct = (1 - inp.final_price / inp.nmc) * 100 if inp.nmc else 0.0
    antidump: Dict = {
        "снижение_%": round(drop_pct, 1),
        "порог_снижения_%": ANTIDUMP_DROP_PCT,
        "применяется": drop_pct >= ANTIDUMP_DROP_PCT,
    }
    if antidump["применяется"]:
        over_15m = inp.nmc > ANTIDUMP_NMC_THRESHOLD
        raised = oik_effective * ANTIDUMP_MULT
        floor = inp.nmc * ANTIDUMP_FLOOR_PCT / 100
        final_oik = max(raised, floor, advance)
        antidump.update({
            "НМЦК_больше_15_млн": over_15m,
            "повышенное_ОИК_×1,5": round(raised, 2),
            "пол_10%_НМЦК": round(floor, 2),
            "аванс": round(advance, 2),
            "итоговое_ОИК": round(final_oik, 2),
            "правило": (
                "НМЦК > 15 млн: ОИК × 1,5, но не менее 10% НМЦК и не менее "
                "аванса (ч. 1 ст. 37)" if over_15m else
                "НМЦК ≤ 15 млн: повышенное ОИК ИЛИ добросовестность "
                "(3 контракта за 3 года без неустоек, один ≥ 20% НМЦК) "
                "+ ОИК в обычном размере (ч. 2 ст. 37)"),
        })
        step("Антидемпинговое ОИК",
             f"max({_fmt(oik_effective)} × 1,5; {_fmt(inp.nmc)} × 10%; аванс {_fmt(advance)})",
             final_oik, "ст. 37 ч. 1 (ред. 484-ФЗ)")
    return {
        "вход": {
            "НМЦК": inp.nmc, "цена_предложения": inp.final_price,
            "обеспечение_заявки_%": inp.bid_sec_pct, "ОИК_%": inp.oik_pct,
            "аванс_%": inp.advance_pct,
        },
        "шаги": steps,
        "итог": {
            "обеспечение_заявки": round(bid_sec, 2),
            "ОИК": round(oik_effective, 2),
        },
        "антидемпинг": antidump,
    }


DEMO = SecurityInput()  # НМЦК 18,5 млн; предложение 13 млн (−29,7%)


# ── ПЕНИ И ШТРАФЫ: ПП № 1042 (действующая ред. = 1042 + ПП № 1011 от
#    02.08.2019; подтверждено 2026-10-02, более поздних изменений нет)
#    + ст. 34 ч. 5/ч. 7 44-ФЗ (ред. 71-ФЗ с 12.05.2019) ──

# п. 3 Правил № 1042 — общая сетка штрафа поставщика (цена контракта/этапа)
SUPPLIER_FINE_TIERS = [
    (3_000_000, 10.0),        # ≤ 3 млн → 10%
    (50_000_000, 5.0),        # 3–50 млн → 5%
    (100_000_000, 1.0),       # 50–100 млн → 1%
    (500_000_000, 0.5),       # 100–500 млн → 0,5%
    (1_000_000_000, 0.4),     # 0,5–1 млрд → 0,4%
    (2_000_000_000, 0.3),     # 1–2 млрд → 0,3%
    (5_000_000_000, 0.25),    # 2–5 млрд → 0,25%
    (10_000_000_000, 0.2),    # 5–10 млрд → 0,2%
    (float("inf"), 0.1),      # > 10 млрд → 0,1%
]
# п. 4 — контракт с МСП (п. 1 ч. 1 ст. 30 44-ФЗ)
SME_FINE_TIERS = [
    (3_000_000, 3.0),       # ≤ 3 млн → 3%
    (10_000_000, 2.0),      # 3–10 млн → 2%
    (20_000_000, 1.0),      # 10–20 млн → 1%
]
# п. 9 — штраф заказчика (фиксированные суммы, ₽)
CUSTOMER_FINE_TIERS = [
    (3_000_000, 1_000),
    (50_000_000, 5_000),
    (100_000_000, 10_000),
    (float("inf"), 100_000),
]


def _tier_pct(price: float, tiers) -> float:
    for cap, pct in tiers:
        if price <= cap:
            return pct
    return tiers[-1][1]


def _tier_fixed(price: float, tiers) -> float:
    for cap, amount in tiers:
        if price <= cap:
            return amount
    return tiers[-1][1]


@dataclass
class PenaltyInput:
    """Входные данные расчёта неустойки по госконтракту."""

    price: float = 9_500_000.0      # цена контракта (этапа), руб.
    executed: float = 4_000_000.0   # фактически исполнено (для пени поставщика)
    delay_days: int = 30            # дни просрочки
    key_rate: float = 9.5           # ключевая ставка ЦБ на дату уплаты, %
    is_sme: bool = False            # контракт с МСП по п. 1 ч. 1 ст. 30


def calc_penalties(inp: PenaltyInput) -> Dict:
    """Пени и штрафы по ПП № 1042 + ст. 34 44-ФЗ, с трассировкой."""
    steps: List[Dict] = []

    def step(name, formula, value, ref):
        steps.append({"step": name, "formula": formula,
                      "value": round(value, 2), "норма": ref})

    # ── Пени поставщика (ч. 7 ст. 34 + п. 10 ПП 1042) ──
    base = max(0.0, inp.price - inp.executed)
    daily = inp.key_rate / 100 / 300
    peni_supplier = base * daily * inp.delay_days
    step("База пени (цена − исполненное)",
         f"{_fmt(inp.price)} − {_fmt(inp.executed)}", base,
         "ч. 7 ст. 34 44-ФЗ, п. 10 ПП 1042")
    step("Пени поставщика за просрочку",
         f"{_fmt(base)} × ({_fmt(inp.key_rate)}% / 300) × {inp.delay_days} дн.",
         peni_supplier, "1/300 ключевой ставки на дату уплаты")

    # ── Штраф поставщика за ненадлежащее исполнение (п. 3 или п. 4) ──
    tiers = SME_FINE_TIERS if inp.is_sme else SUPPLIER_FINE_TIERS
    pct = _tier_pct(inp.price, tiers)
    fine_supplier = inp.price * pct / 100
    step("Штраф поставщика (ненадлежащее исполнение)",
         f"{_fmt(inp.price)} × {_fmt(pct)}%", fine_supplier,
         f"п. {'4' if inp.is_sme else '3'} ПП 1042"
         f"{' (сетка МСП)' if inp.is_sme else ''}")

    # ── Штраф заказчика (п. 9, фикс. сумма) ──
    fine_customer = _tier_fixed(inp.price, CUSTOMER_FINE_TIERS)
    step("Штраф заказчика (неисполнение, кроме просрочки)",
         "фикс. сумма по цене контракта", fine_customer, "п. 9 ПП 1042")

    # ── Потолок неустойки (п. 11–12) ──
    total_supplier = min(peni_supplier + fine_supplier, inp.price)
    capped = peni_supplier + fine_supplier > inp.price
    return {
        "вход": {"цена_контракта": inp.price, "исполнено": inp.executed,
                 "дней_просрочки": inp.delay_days,
                 "ключевая_ставка_%": inp.key_rate, "МСП": inp.is_sme},
        "шаги": steps,
        "итог": {
            "пени_поставщика": round(peni_supplier, 2),
            "штраф_поставщика": round(fine_supplier, 2),
            "штраф_заказчика": fine_customer,
            "итого_поставщик": round(total_supplier, 2),
            "потолок_применён": capped,
        },
        "правила": [
            "Пени заказчика за просрочку оплаты считаются так же — 1/300 "
            "ключевой ставки, но от НЕ УПЛАЧЕННОЙ В СРОК суммы (ч. 5 ст. 34)",
            "Общая неустойка поставщика не может превышать цену контракта "
            "(п. 11 ПП 1042)",
            "Для контрактов с МСП сетка штрафа другая — 3/2/1% (п. 4), "
            "указывать применимую сетку обязательно",
            "Ставка берётся на ДАТУ УПЛАТЫ пени, не на дату просрочки",
        ],
    }


DEMO_PEN = PenaltyInput()  # 9,5 млн; исполнено 4 млн; 30 дней; ставка 9,5%


def build_penalty_block(res: Dict) -> str:
    """Блок для Heart: неустойки по госконтракту — дословно."""
    lines = [
        "🧮 РАСЧЁТ НЕУСТОЙКИ ПО ГОСКОНТРАКТУ (детерминированный калькулятор,",
        "ПП № 1042 в действующей редакции + ст. 34 44-ФЗ, снимок 2026-10-02).",
        "Использовать ЭТИ цифры дословно; считать вручную запрещено.",
        "",
    ]
    for s in res["шаги"]:
        lines.append(f"— {s['step']}: {s['formula']} = {_fmt(s['value'])} ₽ "
                     f"[{s['норма']}]")
    it = res["итог"]
    lines.append("")
    lines.append(f"ИТОГО с поставщика: {_fmt(it['итого_поставщик'])} ₽ "
                 f"(пени {_fmt(it['пени_поставщика'])} + штраф "
                 f"{_fmt(it['штраф_поставщика'])}"
                 f"{'; потолок цены контракта применён' if it['потолок_применён'] else ''}).")
    for r in res["правила"]:
        lines.append(f"⚠️ {r}")
    return "\n".join(lines)



def build_heart_block(res: Dict) -> str:
    """Блок для Heart: использует дословно, своими словами не пересчитывает."""
    lines = [
        "🧮 РАСЧЁТ ОБЕСПЕЧЕНИЙ ПО 44-ФЗ (детерминированный калькулятор,",
        "снимок норм 2026-09-29, ред. 484-ФЗ с 01.01.2026). Используйте",
        "ЭТИ цифры дословно; считать вручную запрещено. Пороги и правила",
        "цитировать с номером статьи. НМЦК {nmc} ₽, цена предложения",
        "{price} ₽ (снижение {drop}%).",
        "",
    ]
    nmc = res["вход"]["НМЦК"]
    price = res["вход"]["цена_предложения"]
    drop = res["антидемпинг"]["снижение_%"]
    head = "\n".join(lines).format(nmc=_fmt(nmc), price=_fmt(price), drop=drop)
    body = [head]
    for s in res["шаги"]:
        body.append(f"— {s['step']}: {s['formula']} = {s['value']:,.0f} ₽ "
                    f"[{s['норма']}]")
    ad = res["антидемпинг"]
    if ad["применяется"]:
        body.append("")
        body.append(f"АНТИДЕМПИНГ: снижение {ad['снижение_%']}% ≥ 25% — "
                    f"применяется. {ad['правило']}. Итоговое ОИК: "
                    f"{ad['итоговое_ОИК']:,.0f} ₽.")
        body.append("В статье обязательны: порог 15 млн ₽, альтернатива "
                    "добросовестности при НМЦК ≤ 15 млн (3 контракта за 3 "
                    "года без неустоек, один ≥ 20% НМЦК).")
    return "\n".join(body)


if __name__ == "__main__":
    import json
    print(json.dumps(calc_security(DEMO), ensure_ascii=False, indent=2))
    print()
    print(build_heart_block(calc_security(DEMO)))
    print()
    print(build_penalty_block(calc_penalties(DEMO_PEN)))
