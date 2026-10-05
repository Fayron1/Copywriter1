# -*- coding: utf-8 -*-
"""P2 план фиксировщика: детерминированный линтер шаблонности.

Структурная сигнатура статьи (H2-последовательность, счётчики блоков) и
n-gram повторы против последних статей. Пороги из плана фиксировщика:
- 8+ слов подряд, совпавших с прошлой статьёй → предупреждение;
- 15+ слов подряд → блок (кроме цитат законов и обязательных дисклеймеров);
- одинаковая H2-последовательность в 3 из последних 5 статей → другой шаблон.
Без LLM; семантическое сравнение — отдельное решение.
"""
import json
import re
from pathlib import Path
from typing import Dict, List

_PROJECT = Path(__file__).resolve().parent.parent.parent
_OUTPUT = _PROJECT / "output"

_WORD_RE = re.compile(r"[а-яёa-z0-9]+")
# n-граммы законов/дисклеймеров не считаются шаблонностью
_LEGAL_GRAM = re.compile(
    r"\bст\.?\s|\bп\.?\s*\d|\bнк\s+рф|\bфз|\bкоап|\bтк\s+рф|дата\s+проверки|"
    r"условн\w+\s+пример", re.I)
_CTA_LINE = re.compile(
    r"обсудить|контент-пакет|white-label|посмотреть\s+формат", re.I)


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(text: str, n: int) -> set:
    w = _words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _strip_boilerplate(md: str) -> str:
    """Убрать обязательный CTA-футер и фронтматтер — они совпадают по дизайну."""
    lines = []
    in_front = False
    for l in md.splitlines():
        if l.strip() == "---":
            in_front = not in_front
            continue
        if in_front:
            continue
        if _CTA_LINE.search(l):
            continue
        lines.append(l)
    return "\n".join(lines)


def extract_structure(md: str) -> Dict:
    """Структурная сигнатура статьи."""
    body = _strip_boilerplate(md)
    h2 = [re.sub(r"^\s*\d+[.)]?\s*", "", h).strip().lower()
          for h in re.findall(r"^##\s+(.+)$", body, re.M)]
    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip() and not p.strip().startswith("|") and not p.strip().startswith("#")]
    para_words = sorted(len(_words(p)) for p in paras)
    median_pw = para_words[len(para_words) // 2] if para_words else 0
    return {
        "h2_sequence": h2,
        "h2_norm": " > ".join(h2[:12]),
        "table_count": len(re.findall(r"^\|.*\|$", body, re.M)) and
                       len([1 for grp in re.split(r"\n(?=\|)", body) if grp.startswith("|")]) or 0,
        "checklist_items": len(re.findall(r"^- \[ \]", body, re.M)),
        "error_blocks": len(re.findall(r"^Ошибка\s*[—:]", body, re.M)),
        "faq": bool(re.search(r"##\s*FAQ|част[оы]\s+задаваем", body, re.I)),
        "ending": h2[-1] if h2 else "",
        "median_para_words": median_pw,
    }


def corpus_articles(limit: int = 20, exclude: Path = None) -> List[Dict]:
    """Последние статьи из output/ (по mtime), без текущей."""
    files = list(_OUTPUT.glob("*/20*/article.md"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        if exclude and p.resolve() == exclude.resolve():
            continue
        try:
            md = p.read_text(encoding="utf-8")
        except Exception:
            continue
        out.append({"path": p, "md": md})
        if len(out) >= limit:
            break
    return out


def check_template_repetition(md: str, corpus: List[Dict]) -> List[str]:
    """N-gram повторы против корпуса. Возвращает предупреждения/блокеры."""
    issues: List[str] = []
    if not corpus:
        return issues
    grams8 = _ngrams(_strip_boilerplate(md), 8)
    grams15 = _ngrams(_strip_boilerplate(md), 15)

    common8_total = 0
    worst8_corpus = ""
    worst8_count = 0
    block15 = None
    for c in corpus:
        c8 = _ngrams(_strip_boilerplate(c["md"]), 8)
        common = {g for g in grams8 & c8 if not _LEGAL_GRAM.search(g)}
        if len(common) > worst8_count:
            worst8_count = len(common)
            worst8_corpus = c["path"].parent.name[:40]
        common8_total += len(common)
        if block15 is None:
            c15 = _ngrams(_strip_boilerplate(c["md"]), 15)
            bad15 = {g for g in grams15 & c15 if not _LEGAL_GRAM.search(g)}
            if bad15:
                block15 = (c["path"].parent.name[:40], sorted(bad15)[0])

    if block15:
        issues.append(
            f"🔴 TEMPLATE_NGRAM_BLOCK: 15+ слов подряд совпадают со статьёй "
            f"«{block15[0]}»: «{block15[1]}…». Дословный перенос блока из прошлой "
            f"статьи — переформулировать или удалить")
    elif worst8_count >= 1:
        issues.append(
            f"🟡 TEMPLATE_NGRAM_WARN: {worst8_count} совпадени(й/я) по 8+ слов подряд "
            f"со статьёй «{worst8_corpus}» — переиспользование готовых абзацев "
            f"(порог плана фиксировщика: 8 слов → предупреждение)")

    # H2-последовательность: одинаковая структура в ≥3 из 5 последних
    new_h2 = extract_structure(md)["h2_norm"]
    same = sum(1 for c in corpus[:5]
               if extract_structure(c["md"])["h2_norm"] == new_h2)
    if new_h2 and same >= 3:
        issues.append(
            f"🟡 TEMPLATE_H2_SAME: идентичная H2-структура в {same} из 5 последних "
            f"статей — конвейер. Сменить editorial template (мини-сценарий, таблица "
            f"выбора, comparison-формат, decision tree)")
    return issues


def save_signature(md: str, out_dir: Path) -> None:
    """Сохранить сигнатуру рядом со статьёй для будущих сравнений."""
    try:
        sig = extract_structure(md)
        (out_dir / "structure_signature.json").write_text(
            json.dumps(sig, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
