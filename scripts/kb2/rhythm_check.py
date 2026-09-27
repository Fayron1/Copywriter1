"""
Сверка ритма статьи с эталонами.

Считает метрики живого голоса (снятые с эталонных статей копирайтеров):
- медиана длины предложения (слов)
- доля коротких предложений (<=7 слов) — «рубленые удары»
- доля длинных (>=25 слов)
- количество H2 и коэффициент асимметрии их размеров (CV = std/mean)
- медиана абзаца (слов)

Использование:
  python scripts/kb2/rhythm_check.py file1.md [file2.md ...]
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path


def split_sentences(text: str):
    text = re.sub(r"\|.*?\|", " ", text)  # таблицы не считаем прозой
    text = re.sub(r"^#{1,6}\s.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[-*]\s+\[[x ]\]", " ", text)  # чекбоксы
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def analyze(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    body = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("!["))
    sentences = split_sentences(body)
    lengths = [len(s.split()) for s in sentences]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    para_lens = [len(re.findall(r"\S+", p)) for p in paragraphs if not p.startswith("#")]

    h2_sizes = []
    sections = re.split(r"^##\s", body, flags=re.MULTILINE)[1:]
    for s in sections:
        h2_sizes.append(len(re.findall(r"\S+", s)))

    cv = (statistics.pstdev(h2_sizes) / statistics.mean(h2_sizes)) if len(h2_sizes) > 1 else 0
    return {
        "file": Path(path).name[:44],
        "sentences": len(lengths),
        "median_sent": statistics.median(lengths) if lengths else 0,
        "short_pct": round(100 * sum(1 for x in lengths if x <= 7) / max(1, len(lengths))),
        "long_pct": round(100 * sum(1 for x in lengths if x >= 25) / max(1, len(lengths))),
        "median_para": statistics.median(para_lens) if para_lens else 0,
        "h2_count": len(h2_sizes),
        "h2_cv": round(cv, 2),
    }


def main():
    rows = [analyze(p) for p in sys.argv[1:]]
    hdr = f"{'файл':46s} {'предл':>5s} {'мед.слов':>8s} {'корот%':>7s} {'длинн%':>7s} {'мед.абз':>8s} {'H2':>3s} {'асимм':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['file']:46s} {r['sentences']:5d} {r['median_sent']:8.1f} "
              f"{r['short_pct']:6d}% {r['long_pct']:6d}% {r['median_para']:8.0f} "
              f"{r['h2_count']:3d} {r['h2_cv']:6.2f}")


if __name__ == "__main__":
    main()
