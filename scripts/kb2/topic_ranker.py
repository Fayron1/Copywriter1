"""
Topic Ranker — фильтр 210 тем из topics_full.json по трём сигналам:

  1. KB-опора (вес 0.5): семантический поиск топика в Qdrant.
     strong = чанки с cosine ≥ 0.62, weak = 0.50–0.62.
     Тема без strong-чанков — Fact-Finder пойдёт в веб вслепую → риск факто-дыр.
  2. Виральность (вес 0.3): детерминированный скоринг по 6 B2B-критериям
     (боль/штраф, миф, вопрос-хук, цифра-конкретика, дедлайн, трансформация).
  3. Спрос-эвристика (вес 0.2): формула темы + маркеры поискового интента
     (год, «как», «что делать», сроки, проценты/лимиты в формулировке).

Выход:
  - topics_ranked.json — все темы с баллами и вердиктом (ready / needs_kb / weak)
  - topics_shortlist.md — топ-30 для очереди генерации

Запуск (на VPS, где Qdrant и e5):
    cd /root/Copywriter1 && python3 scripts/kb2/topic_ranker.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("topic_ranker")

TOPICS_FILE = PROJECT / "topics_full.json"
OUT_JSON = PROJECT / "topics_ranked.json"
OUT_MD = PROJECT / "topics_shortlist.md"

# Пороги e5-cosine (подобраны по релевантным выдачам RAG-агентов)
STRONG_SIM = 0.62
WEAK_SIM = 0.50
TOP_K = 10

# ── Виральность: детерминированные паттерны по 6 критериям ──
_VIRAL_PATTERNS = [
    (re.compile(r"штраф|блокировк|заблок|риск|ошибк|наруша|пропустил|потеря", re.I), 2),      # фрустрация
    (re.compile(r"миф|заблуждени|думаете|на самом деле|развенчиваем|неправда", re.I), 2),    # миф
    (re.compile(r"что делать|как быть|или\s|когда\b", re.I), 1),                              # вопрос-хук
    (re.compile(r"\d", re.I), 1),                                                             # цифра-конкретика
    (re.compile(r"дедлайн|срок|до\s+\d+|успеть|не\s+пропуст", re.I), 2),                      # дедлайн
    (re.compile(r"было\s*\{|станет|сократит|уменьшит|с\s+\d+\s+до\s+\d+|переход\s+от", re.I), 1),  # трансформация
]

# ── Спрос-эвристика: формула темы + интент-маркеры ──
_FORMULA_WEIGHTS = {
    "change": 1.0,       # изменения — высокий поисковый спрос
    "deadline": 1.0,     # дедлайны — сезонный пик
    "mistakes": 0.9,     # ошибки — устойчивый спрос
    "algorithm": 0.8,    # как сделать
    "compare": 0.7,      # vs-запросы
    "case": 0.5,         # кейсы — слабый поисковый интент
    "table": 0.5,        # справки — справочный трафик
}
_DEMAND_MARKERS = re.compile(
    r"202[5-7]|как\s|что\s+делать|чек-лист|пошагов|поря?док|лимит|порог|ставк|"
    r"процент|договор|штраф|дедлайн|срок", re.I)


def viral_score(topic_text: str) -> int:
    score = 0
    for pat, weight in _VIRAL_PATTERNS:
        if pat.search(topic_text):
            score += weight
    return score


def demand_score(topic: dict) -> float:
    text = (topic.get("title") or topic.get("topic") or "") + " " + (topic.get("seo_query") or "")
    formula = (topic.get("formula") or "").lower()
    base = _FORMULA_WEIGHTS.get(formula, 0.6)
    markers = len(set(m.group(0).lower() for m in _DEMAND_MARKERS.finditer(text)))
    return base * 0.7 + min(markers, 3) * 0.1


def kb_support(query_text: str, embed_fn, qdrant, collection: str) -> dict:
    """Топ-K чанков по топику → {strong, weak, top_score}."""
    try:
        vec = embed_fn([query_text], input_type="query")[0]
        if vec is None:
            return {"strong": 0, "weak": 0, "top_score": 0.0}
        res = qdrant.query_points(collection_name=collection, query=vec, limit=TOP_K)
        scores = [p.score for p in res.points if p.score is not None]
        strong = sum(1 for s in scores if s >= STRONG_SIM)
        weak = sum(1 for s in scores if WEAK_SIM <= s < STRONG_SIM)
        return {"strong": strong, "weak": weak, "top_score": max(scores, default=0.0)}
    except Exception as e:
        logger.warning(f"Qdrant query failed for «{query_text[:40]}»: {e}")
        return {"strong": 0, "weak": 0, "top_score": 0.0}


def main():
    if not TOPICS_FILE.exists():
        raise SystemExit(f"Файл не найден: {TOPICS_FILE}")

    topics = json.loads(TOPICS_FILE.read_text(encoding="utf-8"))
    if isinstance(topics, dict) and "topics" in topics:
        topics = topics["topics"]
    logger.info(f"Загружено тем: {len(topics)}")

    # Qdrant + e5 (только на VPS; локально без туннеля — KB-скор = 0)
    embed_fn, qdrant, collection = None, None, None
    try:
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT / ".env")
        except ImportError:
            pass
        from copywriter_kb.loader import get_embeddings_batch
        from copywriter_kb.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
        from qdrant_client import QdrantClient
        embed_fn = get_embeddings_batch
        qdrant = QdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            timeout=20)
        qdrant.get_collections()  # проверка связи
        # kb_update.py и registry используют KB_COLLECTION (default kb_v2);
        # config.COLLECTION_NAME устарел и указывает на несуществующую коллекцию
        import os as _os
        collection = _os.getenv("KB_COLLECTION", "kb_v2")
        logger.info(f"Qdrant: {collection} — OK")
    except Exception as e:
        logger.warning(f"Qdrant недоступен ({e}) — KB-скор будет 0, "
                       f"ранжирование только по виральности и спросу")

    raw = []
    for i, t in enumerate(topics):
        title = t.get("title") or t.get("topic") or ""
        kb = (kb_support(title, embed_fn, qdrant, collection)
              if qdrant else {"strong": 0, "weak": 0, "top_score": 0.0})
        raw.append({
            **t,
            "kb_strong": kb["strong"], "kb_weak": kb["weak"],
            "kb_top": round(kb["top_score"], 4),
            "viral": round(viral_score(title) / 6.0, 2),
            "demand": round(demand_score(t), 2),
        })
        if (i + 1) % 30 == 0:
            logger.info(f"  …{i + 1}/{len(topics)}")

    # e5-cosine по готовым темам идёт плотной полосой (например 0.82–0.89):
    # абсолютные пороги не дискриминируют. kb-скор — min-max нормализация kb_top.
    tops = [r["kb_top"] for r in raw]
    tmin, tmax = min(tops), max(tops)
    span = (tmax - tmin) or 1.0
    for r in raw:
        kb_norm = (r["kb_top"] - tmin) / span
        r["total"] = round(0.5 * kb_norm + 0.3 * min(r["viral"], 1.0) + 0.2 * r["demand"], 3)

    ranked = sorted(raw, key=lambda r: r["total"], reverse=True)
    # Вердикты по перцентилю итогового счёта: низ = слабая опора/виральность
    p50 = ranked[len(ranked) // 2]["total"]
    p25 = ranked[len(ranked) // 4]["total"]
    for r in ranked:
        r["verdict"] = "ready" if r["total"] >= p50 else (
            "review" if r["total"] >= p25 else "weak")

    OUT_JSON.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")

    ready = [r for r in ranked if r["verdict"] == "ready"]
    review = [r for r in ranked if r["verdict"] == "review"]
    weak = [r for r in ranked if r["verdict"] == "weak"]

    lines = [
        "# Шорт-лист тем (топ-30: KB-опора + виральность + спрос)",
        "",
        f"Всего: {len(ranked)} | ready: {len(ready)} | review: {len(review)} | weak: {len(weak)}",
        f"KB-скор: min-max нормализация kb_top (полоса {tmin:.3f}–{tmax:.3f})",
        "",
        "| # | Тема | Напр. | KB top | Вир. | Спрос | Итог |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(ranked[:30], 1):
        title = (r.get("title") or r.get("topic") or "")[:70]
        lines.append(
            f"| {i} | {title} | {r.get('sphere', r.get('direction', ''))} | "
            f"{r['kb_top']:.3f} | {r['viral']} | {r['demand']} | {r['total']} |")
    lines += ["", "## weak (хвост — не ставить в очередь без усиления БЗ)", ""]
    for r in weak[:10]:
        title = r.get("title") or r.get("topic") or ""
        lines.append(f"- [{r.get('sphere', r.get('direction', ''))}] {title} (KB top: {r['kb_top']:.3f})")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"\nГотово: {OUT_JSON.name}")
    logger.info(f"  ready: {len(ready)} | review: {len(review)} | weak: {len(weak)}")
    logger.info(f"  Шорт-лист: {OUT_MD.name}")


if __name__ == "__main__":
    main()
