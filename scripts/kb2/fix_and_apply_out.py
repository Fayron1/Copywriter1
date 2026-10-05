# -*- coding: utf-8 -*-
"""Дочинка хэшей в *_out.json по 8-значным префиксам + применение к кэшу.
Использование: python fix_and_apply_out.py <имя_файла_out.json> <имя_файла_in.json>
Если in-файл не указан, берется соответствующий *_in.json по имени out-файла.
"""
import json
import sys
from pathlib import Path

QDIR = Path(__file__).parent / ".distill_queue"
CACHE = Path(__file__).parent / ".distill_cache.json"


def main():
    out_name = sys.argv[1]
    out_path = QDIR / out_name
    in_name = sys.argv[2] if len(sys.argv) > 2 else out_name.replace("_out.json", "_in.json")
    in_path = QDIR / in_name

    out = json.loads(out_path.read_text(encoding="utf-8"))
    inp = json.loads(in_path.read_text(encoding="utf-8"))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    real_missing = [c["hash"] for c in inp if c["hash"] not in cache]
    # хэши, которые уже покрыты (дубли текста), тоже пригодятся для префикс-матча
    real_all = [c["hash"] for c in inp]
    real_set = set(real_all)

    unmatched = []
    fixed = 0
    for it in out:
        h = it["hash"]
        if h in real_set:
            continue  # уже верный
        # хэш неверный (придуманный хвост) — матч по 8-символьному префиксу
        candidates = [r for r in real_all if r.startswith(h[:8])]
        if len(candidates) == 1:
            it["hash"] = candidates[0]
            fixed += 1
        else:
            unmatched.append(h[:8])

    if unmatched:
        print(f"ВНИМАНИЕ: не удалось сопоставить префиксы: {unmatched}")
        sys.exit(1)

    # контроль: нет двух записей с одним хэшем
    hashes = [it["hash"] for it in out]
    if len(hashes) != len(set(hashes)):
        dupes = {h for h in hashes if hashes.count(h) > 1}
        print(f"ВНИМАНИЕ: дубли хэшей внутри out: {dupes}")
        sys.exit(1)

    json.dump(out, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"хэшей исправлено: {fixed} из {len(out)}")


if __name__ == "__main__":
    main()
