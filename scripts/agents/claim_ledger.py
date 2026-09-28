"""
Claim Ledger — ядро evidence-first архитектуры.

Ключевая идея: Heart имеет право свободно генерировать ЯЗЫК (переходы,
объяснения, аналогии, структуру абзацев). Heart НЕ имеет права свободно
генерировать ФАКТЫ (цифры, даты, ставки, нормы, кейсы, цитаты, бенчмарки).

Три компонента:
1. extract_claims() — извлечь атомарные утверждения из фактов Fact-Finder
2. verify_claims() — проверить каждое утверждение (supported / insufficient)
3. build_claim_pack() — собрать компактный пакет для Heart (approved only)

Формат claim (адаптирован из рекомендаций редактора):
{
  "claim_id": "ndsn_014",
  "text": "Спецставки 5% и 7% не дают права на вычет входного НДС.",
  "claim_type": "legal_rule",           # legal_rule | market_fact | calculation | example | recommendation
  "risk": "high",                        # high | medium | low
  "status": "supported",                # supported | supported_with_limitations | insufficient
  "source": "НК РФ ст. 346.3",
  "allowed_wording": ["не дают права на вычет", "входной НДС не принимается"],
  "forbidden_wording": ["всегда невыгодно", "нельзя использовать"],
  "conditions": ["при соблюдении общих условий"],
}
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agents.claim_ledger")

# Типы утверждений и их риск по умолчанию
CLAIM_TYPES = {
    "legal_rule": "high",         # норма права (ставка, срок, лимит, процедура)
    "legal_conclusion": "high",   # вывод из нормы (что произойдёт при...)
    "market_fact": "high",        # рыночная цифра (объём, рост, доля)
    "market_observation": "medium", # качественное наблюдение (тренд, паттерн)
    "calculation": "medium",      # расчёт из калькулятора
    "example": "low",             # условный пример
    "recommendation": "low",      # совет
    "definition": "low",          # определение термина
    "process_step": "low",        # шаг процесса
}

# Паттерны для Claim Extractor: что считать проверяемым утверждением
_VERIFIABLE_PATTERNS = [
    (re.compile(r"\d+[.,]?\d*\s*(?:млн|млрд|трлн|тыс|руб|₽|%)\b", re.I), "numeric"),
    (re.compile(r"(?:ст|статья|п|пункт|ч|часть)\.?\s*\d+", re.I), "legal_reference"),
    (re.compile(r"(?:с\s+|до\s+|по\s+|начиная\s+с\s+)?\d{1,2}[./]\d{1,2}[./]\d{4}", re.I), "date"),
    (re.compile(r"(?:ФЗ|закон|приказ|постановление|письмо)\s*[№N]?\s*\d+", re.I), "legal_act"),
    (re.compile(r"(?:НДС|НДФЛ|УСН|ОСНО|ПСН|НПД|ЕНС|ЕСХН|АУСН)\b", re.I), "tax_term"),
    (re.compile(r"(?:всегда|никогда|обязательно|гарантированно|автоматически|"
                r"все|весь|каждый|любой|только)", re.I), "absolute"),
    (re.compile(r"(?:кейс|на практике|в практике|реальн\w+ пример|истори\w+ компани)", re.I), "case_claim"),
]

# Запрещённые паттерны для Heart
_HEART_FORBIDDEN = [
    "на моей практике", "в моей практике", "как юрист", "как налоговый консультант",
    "наши клиенты", "мы сопровождали", "я сопровождал",
]


def extract_claims(facts: Dict[str, Any], topic: str = "") -> List[Dict[str, Any]]:
    """
    Извлечь атомарные проверяемые утверждения из фактов Fact-Finder.

    Вход: state.facts (JSON от Fact-Finder: facts[], missing_data[])
    Выход: список claim-объектов для верификации.
    """
    claims = []
    items = facts.get("facts", []) if isinstance(facts, dict) else []

    for i, fact in enumerate(items):
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("claim", ""))

        # Определяем тип и риск по содержанию
        claim_type = _classify_claim(text)
        risk = CLAIM_TYPES.get(claim_type, "medium")

        # Проверяем паттерны
        matched = _match_verifiable(text)

        claim = {
            "claim_id": f"cl_{i:03d}",
            "text": text,
            "claim_type": claim_type,
            "risk": risk,
            "status": "pending",  # pending → supported / insufficient / refuted
            "source": fact.get("source", ""),
            "source_class": fact.get("source_class", ""),
            "reliability": fact.get("reliability", 0.5),
            "norm_quote": fact.get("norm_quote", ""),
            "effective_date": fact.get("effective_date", ""),
            "verifiable": bool(matched or claim_type in ("legal_rule", "market_fact")),
            "matched_patterns": matched,
            "allowed_wording": [],
            "forbidden_wording": [],
        }

        # Низкая надёжность → insufficient
        if claim["reliability"] < 0.5 and claim["risk"] == "high":
            claim["status"] = "insufficient"
            claim["forbidden_wording"] = ["использовать как подтверждённый факт"]

        claims.append(claim)

    logger.info(f"Claim Extractor: {len(claims)} утверждений "
                f"({sum(1 for c in claims if c['risk'] == 'high')} high-risk)")
    return claims


def _classify_claim(text: str) -> str:
    """Классифицировать тип утверждения по содержанию."""
    low = text.lower()
    if any(w in low for w in ["статья", "ст.", "п.", "ч.", "фз", "кодекс", "закон", "нк рф", "тк рф"]):
        return "legal_rule"
    if any(w in low for w in ["трлн", "млрд", "млн", "рынок", "объём", "рост", "доля"]):
        return "market_fact"
    if any(w in low for w in ["пример", "кейс", "компания", "история"]):
        return "example"
    if any(w in low for w in ["рекомендуем", "следует", "нужно", "важно"]):
        return "recommendation"
    if any(w in low for w in ["это", "называется", "определяется как"]):
        return "definition"
    if any(w in low for w in ["шаг", "этап", "сначала", "затем", "после"]):
        return "process_step"
    if any(w in low for w in ["тренд", "наблюдение", "практика показывает"]):
        return "market_observation"
    return "market_observation"  # default


def _match_verifiable(text: str) -> List[str]:
    """Найти проверяемые паттерны в тексте."""
    matched = []
    for pat, name in _VERIFIABLE_PATTERNS:
        if pat.search(text):
            matched.append(name)
    return matched


def build_claim_pack(claims: List[Dict], calculations: Optional[List] = None) -> str:
    """
    Собрать компактный пакет для Heart: только approved утверждения + расчёты.
    Формат: текстовый блок для вставки в промпт (заменяет 50 запретов).
    """
    # Для MVP: pending + надёжный источник → supported
    # (в полной версии это сделает Claim Verifier через Vane)
    approved = []
    for c in claims:
        if c["status"] in ("supported", "supported_with_limitations"):
            approved.append(c)
        elif c["status"] == "pending":
            if c.get("reliability", 0) >= 0.6 and c.get("source_class") == "primary":
                c["status"] = "supported"
                approved.append(c)
            elif c.get("reliability", 0) >= 0.5 and c.get("source"):
                c["status"] = "supported_with_limitations"
                c.setdefault("conditions", []).append("по данным вторичного источника")
                approved.append(c)

    if not approved:
        return ""

    lines = ["=== УТВЕРЖДЕННЫЕ ФАКТЫ (единственный источник для статьи) ===",
             "Используй ТОЛЬКО эти утверждения и цифры. Не добавляй новые факты,",
             "ставки, сроки, кейсы, цитаты или рыночные цифры.", ""]

    for c in approved:
        lines.append(f"[{c['claim_id']}] {c['text']}")
        if c.get("conditions"):
            lines.append(f"  Условия: {'; '.join(c['conditions'])}")
        if c.get("allowed_wording"):
            lines.append(f"  Разрешённые формулировки: {'; '.join(c['allowed_wording'])}")
        if c.get("forbidden_wording"):
            lines.append(f"  ЗАПРЕЩЕНО: {'; '.join(c['forbidden_wording'])}")

    # Расчёты из калькулятора
    if calculations:
        lines.append("")
        lines.append("=== УТВЕРЖДЁННЫЕ РАСЧЁТЫ (не пересчитывай самостоятельно) ===")
        for calc in calculations:
            if isinstance(calc, dict):
                lines.append(f"[{calc.get('name', 'calc')}]")
                for f in calc.get("formulas", []):
                    lines.append(f"  {f}")

    lines.append("=== КОНЕЦ УТВЕРЖДЁННЫХ ФАКТОВ ===")
    lines.append("")
    lines.append("ЗАПРЕЩЕНО (глобально):")
    lines.append("- Добавлять цифры, ставки, сроки, нормы, не указанные выше")
    lines.append("- Создавать кейсы с именами, компаниями, цитатами")
    lines.append("- Использовать «на моей практике», «как юрист», «наши клиенты»")
    lines.append("- Делать юридические выводы вне перечисленных")
    lines.append("- Утверждать рыночные бенчмарки без источника")
    lines.append("")
    lines.append("РАЗРЕШЕНО (свободно):")
    lines.append("- Переходы, объяснения, аналогии, примеры-иллюстрации без цифр")
    lines.append("- Структура абзацев, ритм, стиль, метафоры")
    lines.append("- Вопросы к читателю, подводки, выводы")

    return "\n".join(lines)


def audit_coverage(text: str, claims: List[Dict]) -> List[str]:
    """
    Coverage Auditor: проверить, не добавил ли Heart новые факты.
    Извлекает числа/даты/проценты из текста и сверяет с claim pack.
    """
    issues = []
    if not text or not claims:
        return issues

    # Все числа из утверждений (что разрешено)
    approved_numbers = set()
    for c in claims:
        if c["status"] in ("supported", "supported_with_limitations"):
            for m in re.finditer(r"\d+[.,]?\d*", c["text"]):
                approved_numbers.add(m.group(0).replace(",", "."))

    # Числа из текста
    text_numbers = set()
    for m in re.finditer(r"\d+[.,]?\d*", text):
        text_numbers.add(m.group(0).replace(",", "."))

    # Новые числа (не в approved) — значимые: больше 10 или содержат точку
    new_numbers = text_numbers - approved_numbers - {str(i) for i in range(1, 11)}
    significant_new = [n for n in new_numbers
                       if float(n) > 10 or "." in n or len(n) >= 2]

    if significant_new:
        issues.append(f"🟡 НОВЫЕ ЧИСЛА ({len(significant_new)}): "
                      f"{', '.join(sorted(significant_new)[:10])} — "
                      f"не из утверждённого claim pack. Проверь источник.")

    # Псевдо-эксперт
    for phrase in _HEART_FORBIDDEN:
        if phrase in text.lower():
            issues.append(f"🔴 ПСЕВДО-ЭКСПЕРТ: «{phrase}»")
            break

    return issues


def insufficient_report(claims: List[Dict]) -> str:
    """Отчёт о неподтверждённых утверждениях (для паспорта)."""
    insufficient = [c for c in claims if c["status"] == "insufficient"]
    if not insufficient:
        return ""
    lines = [f"⚠️ Недостаточно доказательств: {len(insufficient)} утверждений"]
    for c in insufficient[:5]:
        lines.append(f"  • {c['text'][:80]}... (источник: {c.get('source', 'нет')})")
    return "\n".join(lines)
