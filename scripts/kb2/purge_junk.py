#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Purge мусора из уже залитой коллекции БЕЗ переэмбеддинга.

Логика:
1. Scroll всех точек (payload: text, source_file, domain).
2. Локальная эвристика отбирает кандидатов в мусор (доля нав-строк,
   повторы строк, пустышки).
3. LLM (gpt-5.4-mini) верифицирует только кандидатов — это дёшево.
4. Подтверждённый мусор удаляется по ID.

Запуск: QDRANT_HOST=127.0.0.1 QDRANT_PORT=16333 \
  python scripts/kb2/purge_junk.py --domains craft/writing,craft/editorial
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from distiller import nav_junk_ratio, distill_chunks  # noqa: E402
from loader import COLLECTION, get_client  # noqa: E402

from qdrant_client.models import FieldCondition, Filter, MatchAny  # noqa: E402

SCROLL_BATCH = 256
HEURISTIC_NAV = 0.45      # доля нав-строк в чанке
HEURISTIC_MIN_TEXT = 120  # слишком короткий без структуры


def is_candidate(text: str) -> bool:
    if not text or len(text.strip()) < HEURISTIC_MIN_TEXT:
        return True
    if nav_junk_ratio(text) >= HEURISTIC_NAV:
        return True
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.5:
        return True  # полчанка — повторяющиеся строки
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default=None,
                    help="через запятую; по умолчанию все домены коллекции")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = get_client()
    domains = [d.strip() for d in args.domains.split(",")] if args.domains else None
    flt = Filter(must=[FieldCondition(key="domain", match=MatchAny(any=domains))]) \
        if domains else None

    scanned = candidates = 0
    cand_ids: list[str] = []
    cand_texts: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, scroll_filter=flt, limit=SCROLL_BATCH,
            offset=offset, with_payload=["text", "source_file"], with_vectors=False)
        if not points:
            break
        for p in points:
            scanned += 1
            text = (p.payload or {}).get("text", "")
            if is_candidate(text):
                candidates += 1
                cand_ids.append(p.id)
                cand_texts.append(text)
        print(f"  …просмотрено {scanned}, кандидатов {candidates}", flush=True)
        if offset is None:
            break

    print(f"Итого: просмотрено {scanned}, кандидатов в мусор {candidates}")
    if not cand_ids or args.dry_run:
        return 0

    confirmed = []
    B = 16  # чанков на LLM-батч (~16 × 800 симв ≈ 3.5k токенов)
    for i in range(0, len(cand_texts), B):
        batch = cand_texts[i:i + B]
        marks = distill_chunks(batch)
        for pid, m in zip(cand_ids[i:i + B], marks):
            if m.get("is_junk"):
                confirmed.append(pid)
        done = min(i + B, len(cand_texts))
        print(f"  …LLM проверил {done}/{len(cand_texts)}, мусор {len(confirmed)}", flush=True)

    print(f"LLM подтвердил мусор: {len(confirmed)} из {len(cand_texts)} кандидатов")
    if confirmed:
        for i in range(0, len(confirmed), 512):
            client.delete(collection_name=COLLECTION, points_selector=confirmed[i:i + 512])
        print(f"🗑️ удалено точек: {len(confirmed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
