"""Миграция kb_v2 → гибридную коллекцию (named dense + sparse) с алиасом.

Схема (2026-10-02, гибридный поиск):
1. Создать kb_v2_h1: vectors={"dense": 768 Cosine}, sparse_vectors_config.
2. Скроллить старый kb_v2 (unnamed dense, with_vectors) батчами,
   пересобирать в named + sparse (sparse из payload.text).
3. Сверить счётчики.
4. Алиас: снапшот старого → delete kb_v2 → alias kb_v2 → kb_v2_h1.
   (вручную, отдельным шагом — скрипт печатает команды)

Запуск (VPS): venv/bin/python scripts/kb2/migrate_sparse.py --batch 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kb.migrate_sparse")

from kb2.sparse_utils import sparse_vector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-copy", action="store_true",
                    help="только создать коллекцию и сверить счётчики")
    args = ap.parse_args()

    import os
    from qdrant_client import QdrantClient, models

    key = os.getenv("QDRANT_API_KEY") or [
        l.split("=", 1)[1].strip() for l in open(".env", encoding="utf-8")
        if l.startswith("QDRANT_API_KEY=")
    ][0]
    qc = QdrantClient(url="http://localhost:6333", api_key=key, timeout=120)

    SRC, DST = "kb_v2", "kb_v2_h1"

    info_src = qc.get_collection(SRC)
    dim = info_src.config.params.vectors.size
    logger.info(f"источник {SRC}: {info_src.points_count} точек, dim={dim}")

    # 1) целевая коллекция
    try:
        qc.delete_collection(DST)
        logger.info(f"старая {DST} удалена (догоны прошлой попытки)")
    except Exception:
        pass
    qc.create_collection(
        collection_name=DST,
        vectors_config={"dense": models.VectorParams(size=dim,
                                                     distance=models.Distance.COSINE)},
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False))},
    )
    logger.info(f"создана {DST} (named dense {dim} + sparse)")

    if args.skip_copy:
        return 0

    # 2) копия с вычислением sparse
    offset = None
    copied = 0
    t0 = time.time()
    while True:
        pts, offset = qc.scroll(SRC, limit=args.batch, with_vectors=True,
                                with_payload=True, offset=offset)
        if not pts:
            break
        upserts = []
        for p in pts:
            vec = p.vector if isinstance(p.vector, list) else p.vector.get("")
            text = (p.payload or {}).get("text", "")
            upserts.append(models.PointStruct(
                id=p.id,
                vector={"dense": list(vec),
                        "sparse": models.SparseVector(**sparse_vector(text))},
                payload=p.payload))
        qc.upsert(DST, points=upserts)
        copied += len(pts)
        if copied % 5000 < args.batch:
            rate = copied / max(time.time() - t0, 1)
            logger.info(f"скопировано {copied}/{info_src.points_count} "
                        f"({rate:.0f} т/с)")
        time.sleep(0.2)  # щадим прод

    # 3) сверка
    n_dst = qc.count(DST, exact=True).count
    logger.info(f"итог: {SRC}={info_src.points_count}, {DST}={n_dst}")
    if n_dst != info_src.points_count:
        logger.error("СЧЁТЧИКИ НЕ СХОДЯТСЯ — алиас НЕ переключать!")
        return 1
    logger.info("Миграция полная. Далее вручную: снапшот kb_v2 → delete "
                "kb_v2 → алиас kb_v2 → kb_v2_h1 (см. команды в логе ниже)")
    print(f'АЛИАС: POST /collections/aliases {{"actions":[{{"delete_collection":'
          f'{{"collection_name":"{SRC}"}}}},{{"create_aliases":{{"alias_name":'
          f'"{SRC}","collection_name":"{DST}"}}}}]}}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
