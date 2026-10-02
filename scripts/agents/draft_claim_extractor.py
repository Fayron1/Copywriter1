"""Детерминированное извлечение проверяемых утверждений из черновика.

Замена сломанного KIE-экстрактора (статьи 17–18: «проверяемых утверждений
не найдено» при живом тексте). Регексами: числа с единицами, ссылки на
статьи законов, даты-дедлайны. Ноль токенов, всегда работает.

Используется в _step_claim_check как фаза 1 (извлечение) с последующей
детерминированной сверкой с claim pack / паспортом параметров.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Число + денежная единица или процент («57 390 ₽», «20 млн», «22%», «1–3%»)
_NUM_VALUE = re.compile(
    r"\d[\d\u00a0 ]{1,12}(?:[.,]\d+)?\s*(?:млрд|млн|тыс\.?)?\s*(?:₽|руб\w*|%|"
    r"млрд\s*₽|млн\s*₽)|\d+[.,]\d+\s*%|\d+\s*%\s*[—–-]\s*\d+\s*%", re.I)

# Ссылка на статью/ФЗ («ст. 22.3 ТК РФ», «152-ФЗ», «статьей 145 НК»)
_LAW_REF = re.compile(
    r"(?:ст\.?\s*|стать\w+\s*)\d+(?:\.\d+)?\s*(?:ТК|НК|ГК|КоАП|УК|АПК)?\s*(?:РФ)?|"
    r"(?:ФЗ|законом?)\s*[№\-]?\s*\d+(?:[-–]\w+)?|"
    r"\d+(?:[-–]\w+)?-ФЗ", re.I)

# Дата/срок («до 28 декабря», «в течение 24 часов», «не позднее 5 дней»)
_DEADLINE = re.compile(
    r"(?:до|не\s+позднее|в\s+течение|срок\w*)\s+\d{1,2}[.\s]\w+|"
    r"(?:до|не\s+позднее|в\s+течение)\s+\d+\s*(?:час\w+|дн\w+|день|рабочих\s+дн\w+|"
    r"календарн\w+)", re.I)


def extract_claims_deterministic(text: str) -> List[Dict[str, str]]:
    """Извлечь проверяемые утверждения. Формат совместим с factcheck:
    [{text, type, value}] — value это спанный маркер."""
    if not text:
        return []
    claims: List[Dict[str, str]] = []
    seen = set()
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    for sent in sentences:
        s = sent.strip()
        if len(s) < 10 or len(s) > 600:
            continue
        for rx, ctype in ((_NUM_VALUE, "numeric"),
                          (_LAW_REF, "legal_ref"),
                          (_DEADLINE, "deadline")):
            m = rx.search(s)
            if m:
                val = m.group(0).strip()
                key = (val, s[:80])
                if key in seen:
                    continue
                seen.add(key)
                claims.append({"text": s[:300], "type": ctype, "value": val})
                break  # одно утверждение на предложение — не дублируем
    return claims


def match_to_sources(claims: List[Dict[str, str]],
                     claim_pack_text: str,
                     param_values: List[str]) -> List[Dict[str, str]]:
    """Разметить: поддержано claim pack / паспортом / нет.

    claim_pack_text — текст claim pack (строка);
    param_values — список строковых значений параметров паспорта («22%», «20 млн»…)
    """
    sources = (claim_pack_text or "") + "\n" + "\n".join(param_values or [])
    for c in claims:
        val = re.sub(r"[\s\u00a0]", "", c["value"])
        src_norm = re.sub(r"[\s\u00a0]", "", sources)
        # проценты/пороги сравниваем без пробелов: «22%» в «ставка 22% (2026)»
        c["supported"] = val in src_norm
    return claims
