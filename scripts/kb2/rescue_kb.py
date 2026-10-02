"""Спасательный перелив kb_v2: сломанный Qdrant (6333, читается) →
здоровый (6340, named dense + sparse). Один проход, 95k точек.

Запуск (VPS): venv/bin/python scripts/kb2/rescue_kb.py --batch 500
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
logger = logging.getLogger("kb.rescue")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    import os
    from qdrant_client import QdrantClient, models
    from kb2.sparse_utils import sparse_vector

    key = os.getenv("QDRANT_API_KEY")
    q_src = QdrantClient(url="http://localhost:6333", api_key=key, timeout=120)
    q_dst = QdrantClient(url="http://localhost:6340", api_key=key, timeout=120)

    DST = "kb_v2_h1"
    info_src = q_src.get_collection("kb_v2")
    dim = info_src.config.params.vectors.size
    logger.info(f"источник: {info_src.points_count} точек, dim={dim}")

    try:
        qc_count = q_dst.get_collection(DST).points_count
        logger.info(f"приёмник {DST} уже содержит {qc_count} — докачка")
    except Exception:
        q_dst.create_collection(
            collection_name=DST,
            vectors_config={"dense": models.VectorParams(
                size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False))},
        )
        logger.info(f"создан приёмник {DST} (named dense + sparse)")

    offset = None
    copied = 0
    t0 = time.time()
    while True:
        pts, offset = q_src.scroll("kb_v2", limit=args.batch, with_vectors=True,
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
        q_dst.upsert(DST, points=upserts)
        copied += len(pts)
        if copied % 5000 < args.batch:
            rate = copied / max(time.time() - t0, 1)
            logger.info(f"спасено {copied}/{info_src.points_count} ({rate:.0f} т/с)")
        time.sleep(0.15)

    n_dst = q_dst.count(DST, exact=True).count
    logger.info(f"итог: источник {info_src.points_count}, спасено {n_dst}")
    if n_dst < info_src.points_count:
        logger.error("НЕДОБОР — повторить запуск (докачка по offset)")
        return 1
    logger.info("✅ ДАННЫЕ СПАСЕНЫ ПОЛНОСТЬЮ. Далее: алиас kb_v2 → kb_v2_h1 "
                "на 6340-инстансе, перевод порта 6333")
    return 0


if __name__ == "__main__":
    sys.exit(main())
