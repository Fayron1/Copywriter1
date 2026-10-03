"""
KB v2 — загрузчик новой базы знаний в Qdrant.

Ключевые отличия от copywriter_kb (закрывают баги ревью):
- ТОЛЬКО локальные эмбеддинги (multilingual-e5, 768d). OpenAI-пути нет.
- Идемпотентные ID: uuid5(collection|source|chunk_index|content_hash).
  Повторный запуск перезаписывает те же точки: чекпойнты и scroll-дедупликация
  не нужны в принципе, --fresh = пересоздание коллекции.
- Законодательство: чанк = статья целиком; номер статьи сохраняется в
  под-чанках; короткие статьи («утратила силу») не выбрасываются.
- Книги: чанки по предложениям, перекрытие по границе слова (не посреди слова).
- Отчёты/справочники: год из имени файла -> published_year + valid_until
  (фильтр актуальности в agents/rag.py начнёт работать по-настоящему).
- agent_target — список: один чанк доступен нескольким агентам
  (MatchValue в Qdrant матчится по любому элементу массива).

Запуск (локально, туннель до VPS Qdrant поднят отдельно):
  python scripts/kb2/loader.py --list                     # что видит загрузчик
  python scripts/kb2/loader.py --folder style_client --dry-run
  python scripts/kb2/loader.py --folder style_client      # залить
  python scripts/kb2/loader.py --folder all --fresh       # полная пересборка
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from copywriter_kb.parsers import extract_text  # noqa: E402
from embedding_system.local_embeddings import embed, get_dim  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kb2.loader")

# Гибридный поиск: sparse-ветка (BM25-стиль) рядом с dense e5
from kb2.sparse_utils import sparse_vector as _sparse_of



def _load_env() -> None:
    """Автономный запуск: подтянуть .env проекта (dotenv или ручной парсер)."""
    env_file = PROJECT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_env()

KB_ROOT = Path(os.getenv("KB_ROOT", PROJECT / "knowledge_base"))
COLLECTION = os.getenv("KB_COLLECTION", "kb_v2")
EMBED_BATCH = int(os.getenv("KB_EMBED_BATCH", "64"))
QDRANT_BATCH = int(os.getenv("KB_QDRANT_BATCH", "100"))
QDRANT_RETRIES = 3

SUPPORTED = {".pdf", ".fb2", ".txt", ".docx", ".odt", ".epub"}

# ============================================================
# Конфигурация папок: подпапка knowledge_base/ -> стратегия
# agents: список agent_id (payload.agent_target)
# chunk: law | book | guide | reference | style
# report: True -> год из имени файла -> published_year/valid_until
# ============================================================
FOLDERS: Dict[str, Dict[str, Any]] = {
    "legislation": {
        "agents": ["fact_finder"], "chunk": "law", "source_type": "law",
        "label": "⚖️ Законодательство",
    },
    "craft/writing": {
        "agents": ["heart", "engineer"], "chunk": "book", "source_type": "book",
        "label": "✍️ Письмо (Писатель)",
    },
    "craft/editorial": {
        "agents": ["sheriff"], "chunk": "book", "source_type": "reference",
        "label": "🔫 Редактура (Редактор)",
    },
    "craft/structure": {
        "agents": ["engineer"], "chunk": "book", "source_type": "book",
        "label": "🏗️ Структура (Структурировщик)",
    },
    "craft/seo": {
        "agents": ["booster"], "chunk": "guide", "source_type": "guide",
        "label": "🚀 SEO/GEO",
    },
    "craft/visual": {
        "agents": ["artist"], "chunk": "reference", "source_type": "reference",
        "label": "🎨 Визуал",
    },
    "business/marketing": {
        "agents": ["booster", "engineer"], "chunk": "book", "source_type": "book",
        "label": "📈 Маркетинг",
    },
    "business/marketplaces": {
        # Тарифные снимки, оферты и регламенты WB/Ozon: точечные факты для
        # fact_finder (комиссии, КВВ, даты вступления) и engineer (P&L-модели).
        # Отдельная папка, а не business/marketing: loader не пропускает уже
        # залитые файлы, а в маркетинге лежат тяжёлые книги — пересчёт дорог.
        "agents": ["fact_finder", "engineer"], "chunk": "guide", "source_type": "reference",
        "label": "🛒 Маркетплейсы (тарифы и оферты)",
    },
    "business/methodology": {
        "agents": ["engineer", "heart"], "chunk": "book", "source_type": "book",
        "label": "🧭 Методологии",
    },
    "business/unit_economics": {
        "agents": ["fact_finder", "engineer"], "chunk": "guide", "source_type": "guide",
        "label": "🧮 Юнит-экономика",
    },
    "business/finance": {
        "agents": ["fact_finder"], "chunk": "book", "source_type": "book",
        "label": "💰 Финансы",
    },
    "business/reports": {
        "agents": ["fact_finder"], "chunk": "guide", "source_type": "report",
        "report": True, "label": "📊 Отчёты (датированные)",
    },
    "business/reference": {
        "agents": ["fact_finder"], "chunk": "reference", "source_type": "reference",
        "report": True, "label": "📋 Справочники (датированные)",
    },
    "style_client": {
        "agents": ["heart", "booster"], "chunk": "style", "source_type": "style_example",
        "label": "📝 Стиль клиента",
    },
    # Эталонные статьи по типам (живые тексты копирайтеров — few-shot якоря Heart)
    "style_client/чек_листы": {
        "agents": ["heart", "engineer"], "chunk": "style", "source_type": "style_example",
        "label": "📝 Эталон: чек-листы",
    },
    "style_client/ситуации": {
        "agents": ["heart", "engineer"], "chunk": "style", "source_type": "style_example",
        "label": "📝 Эталон: ситуации (кейсы)",
    },
    "style_client/разбор_законов": {
        "agents": ["heart", "engineer"], "chunk": "style", "source_type": "style_example",
        "label": "📝 Эталон: разбор законов",
    },
    "style_client/полезное": {
        "agents": ["heart", "engineer"], "chunk": "style", "source_type": "style_example",
        "label": "📝 Эталон: справочники",
    },
    "style_client/актуальные_проблемы": {
        "agents": ["heart", "engineer"], "chunk": "style", "source_type": "style_example",
        "label": "📝 Эталон: лонгриды",
    },
}

# Размеры чанков по стратегиям (символы)
CHUNK_SIZES = {"law": 1500, "book": 900, "guide": 800, "reference": 700, "style": 1000}
OVERLAP = 100

ARTICLE_RE = re.compile(
    r'(?=(?:Статья|СТАТЬЯ|Ст\.|Глава|ГЛАВА|Раздел|РАЗДЕЛ|§)\s*[\dIVXЛ]+)'
)
ARTICLE_NUM_RE = re.compile(
    r'^(?:Статья|СТАТЬЯ|Ст\.|Глава|ГЛАВА|Раздел|РАЗДЕЛ|§)\s*([\dIVXЛ.]+)'
)
YEAR_RE = re.compile(r'(20\d{2})')


# ============================================================
# Чанкинг v2
# ============================================================

def chunk_law(text: str, size: int) -> List[Dict[str, Any]]:
    """Статья целиком; длинные режем по предложениям с сохранением заголовка."""
    segments = [s.strip() for s in ARTICLE_RE.split(text) if s.strip()]
    chunks: List[Dict[str, Any]] = []

    def article_number(head: str) -> str:
        m = ARTICLE_NUM_RE.match(head)
        return m.group(1).rstrip(".") if m else ""

    for seg in segments:
        art = article_number(seg)
        if len(seg) <= size:
            chunks.append({"text": seg, "article_number": art, "part": 0})
            continue
        # Длинная статья/глава: режем предложениями, префиксуем номером статьи
        head = seg.split("\n", 1)[0][:120]
        pieces = _split_sentences(seg, size, OVERLAP)
        total = len(pieces)
        for i, piece in enumerate(pieces):
            prefix = f"{head} (часть {i + 1}/{total}): " if i > 0 else ""
            chunks.append({
                "text": prefix + piece,
                "article_number": art,
                "part": i,
                "total_parts": total,
            })
    return chunks


def chunk_generic(text: str, size: int, kind: str) -> List[Dict[str, Any]]:
    """Книги/гайды/справочники/стиль: по предложениям, перекрытие по границе слова."""
    if kind in ("guide", "reference"):
        # Пытаемся резать по markdown-заголовкам, фолбэк — предложения
        segments = [s.strip() for s in re.split(r'(?=\n#{1,3}\s)', text) if s.strip()]
        if len(segments) >= 3:
            out: List[Dict[str, Any]] = []
            for seg in segments:
                if len(seg) <= size:
                    if len(seg) > 40:
                        out.append({"text": seg, "part": 0})
                else:
                    out.extend({"text": p, "part": i} for i, p in enumerate(_split_sentences(seg, size, OVERLAP)))
            return out
    pieces = _split_sentences(text, size, OVERLAP)
    return [{"text": p, "part": i} for i, p in enumerate(pieces) if len(p) > 40]


def _split_sentences(text: str, size: int, overlap: int) -> List[str]:
    """Разбиение по предложениям; перекрытие — хвост по границе слова."""
    sentences = re.split(r'(?<=[.!?…])\s+', text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            # Сверхдлинное предложение — по словам
            words, buf = sentence.split(), ""
            for w in words:
                if len(buf) + len(w) + 1 <= size:
                    buf = f"{buf} {w}".strip()
                else:
                    chunks.append(buf)
                    buf = w
            if buf:
                chunks.append(buf)
            continue

        if len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            tail = current[-overlap:] if overlap and current else ""
            cut = tail.find(" ")
            if cut > 0:
                tail = tail[cut + 1:]
            current = (tail + " " + sentence).strip() if tail else sentence
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_text(text: str, kind: str, size: int) -> List[Dict[str, Any]]:
    if kind == "law":
        chunks = chunk_law(text, size)
    else:
        chunks = chunk_generic(text, size, kind)
    # Короткий мусор убираем — но не для законов (короткая статья = ценность)
    if kind != "law":
        chunks = [c for c in chunks if len(c["text"].strip()) > 40]
    return chunks


# ============================================================
# Метаданные
# ============================================================

def content_hash(text: str) -> str:
    normalized = re.sub(r'[\s\W]+', '', text.lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def year_meta(filename: str) -> Dict[str, Any]:
    """Год из имени файла -> published_year, valid_until (конец следующего года)."""
    m = YEAR_RE.search(filename)
    if not m:
        return {}
    year = int(m.group(1))
    return {"published_year": year, "valid_until": f"{year + 1}-12-31"}


def point_id(source_rel: str, chunk_index: int, chash: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{COLLECTION}|{source_rel}|{chunk_index}|{chash}"))


# ============================================================
# Qdrant
# ============================================================

def get_client():
    from qdrant_client import QdrantClient
    host = os.getenv("QDRANT_HOST", "127.0.0.1")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    key = os.getenv("QDRANT_API_KEY", "")
    return QdrantClient(url=f"http://{host}:{port}", api_key=key or None, timeout=60.0, https=False)


def ensure_collection(client, recreate: bool = False) -> None:
    from qdrant_client.models import VectorParams, Distance
    if recreate:
        # Автоснапшот перед сносом: --fresh необратим, а инцидент 2026-09-29
        # стоил 80k точек. Снапшот остаётся на сервере Qdrant — восстановление
        # через POST /collections/{name}/snapshots/upload.
        try:
            snap = client.create_snapshot(collection_name=COLLECTION)
            logger.info(f"📸 Автоснапшот перед --fresh: {getattr(snap, 'name', snap)}")
        except Exception as e:
            logger.warning(f"⚠️ Автоснапшот не создан (продолжаем): {e}")
        try:
            client.delete_collection(COLLECTION)
            logger.info(f"🗑️ Коллекция {COLLECTION} удалена (--fresh)")
        except Exception:
            pass
    try:
        info = client.get_collection(COLLECTION)
        logger.info(f"✅ Коллекция {COLLECTION} существует: {info.points_count} точек")
        return
    except Exception:
        pass
    from qdrant_client.models import SparseVectorParams, SparseIndexParams
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=get_dim(),
                                              distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False))},
    )
    logger.info(f"📦 Создана коллекция {COLLECTION} "
                f"(dense dim={get_dim()} + sparse гибрид, локальные e5)")


def upload(client, points: List[Dict[str, Any]]) -> int:
    from qdrant_client.models import PointStruct
    uploaded = 0
    for i in range(0, len(points), QDRANT_BATCH):
        batch = points[i:i + QDRANT_BATCH]
        for attempt in range(QDRANT_RETRIES):
            try:
                from qdrant_client.models import SparseVector
                client.upsert(collection_name=COLLECTION, points=[
                    PointStruct(
                        id=p["id"],
                        vector={
                            "dense": p["vector"],
                            "sparse": SparseVector(**_sparse_of(
                                p.get("text") or p["payload"].get("text", ""))),
                        },
                        payload=p["payload"])
                    for p in batch
                ])
                uploaded += len(batch)
                break
            except Exception as e:
                if attempt == QDRANT_RETRIES - 1:
                    logger.error(f"❌ Upsert не прошёл (батч {i // QDRANT_BATCH}): {e}")
                else:
                    time.sleep(2 * (attempt + 1))
    return uploaded


def prune_folder(client, folder_key: str, live_sources: List[str]) -> int:
    """
    Удалить точки папки, чьи source_file больше не существуют на диске
    (заменили файл свежей редакцией / убрали из подборки).
    Идемпотентные ID не знают об удалениях — этот проход закрывает дыру.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, FilterSelector
    flt = Filter(
        must=[FieldCondition(key="domain", match=MatchValue(value=folder_key))],
        must_not=[FieldCondition(key="source_file", match=MatchAny(any=live_sources))],
    )
    try:
        before = client.count(collection_name=COLLECTION,
                              count_filter=flt, exact=True).count
        if before == 0:
            return 0
        client.delete(collection_name=COLLECTION, points_selector=FilterSelector(filter=flt))
        logger.info(f"   🧹 prune {folder_key}: удалено {before} осиротевших точек "
                    f"(файлов на диске: {len(live_sources)})")
        return before
    except Exception as e:
        logger.warning(f"   ⚠️ prune {folder_key} не удался: {e}")
        return 0


# ============================================================
# Обработка
# ============================================================

def process_folder(folder_key: str, dry_run: bool = False, limit: Optional[int] = None,
                   client=None, files_filter: Optional[str] = None) -> Dict[str, int]:
    cfg = FOLDERS[folder_key]
    folder = KB_ROOT / folder_key
    stats = {"files": 0, "chunks": 0, "uploaded": 0, "skipped": 0, "chars": 0}

    logger.info(f"\n{'=' * 60}\n{cfg['label']} — {folder}")
    if not folder.exists():
        logger.warning("   папка не найдена")
        return stats

    files = sorted(f for f in folder.iterdir()
                   if f.is_file() and f.suffix.lower() in SUPPORTED)
    if files_filter:
        # Точечная дозагрузка: glob-паттерн по имени файла (например
        # «Минфин_письмо*»), чтобы не пересчитывать всю папку ради
        # нескольких новых файлов. Prune при этом НЕ запускается.
        import fnmatch
        before = len(files)
        files = [f for f in files if fnmatch.fnmatch(f.name, files_filter)]
        logger.info(f"   фильтр --files '{files_filter}': {before} → {len(files)}")
    if limit:
        files = files[:limit]
    logger.info(f"   файлов: {len(files)}, стратегия чанков: {cfg['chunk']}")

    pending_points: List[Dict[str, Any]] = []

    for f in files:
        text = extract_text(f, source_type="law" if cfg["chunk"] == "law" else None)
        if not text:
            logger.warning(f"   ⚠️ пусто после парсинга: {f.name}")
            stats["skipped"] += 1
            continue
        chunks = chunk_text(text, cfg["chunk"], CHUNK_SIZES[cfg["chunk"]])
        if not chunks:
            logger.warning(f"   ⚠️ нет чанков: {f.name}")
            stats["skipped"] += 1
            continue

        source_rel = f"{folder_key}/{f.name}"
        meta_extra = year_meta(f.name) if cfg.get("report") else {}

        for idx, ch in enumerate(chunks):
            payload: Dict[str, Any] = {
                "text": ch["text"],
                "source_file": source_rel,
                "domain": folder_key,
                "agent_target": list(cfg["agents"]),
                "source_type": cfg["source_type"],
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "content_hash": content_hash(ch["text"]),
                **meta_extra,
            }
            if "article_number" in ch and ch["article_number"]:
                payload["article_number"] = ch["article_number"]
                payload["codex"] = _guess_codex(f.name)
            stats["chunks"] += 1
            stats["chars"] += len(ch["text"])
            if not dry_run:
                pending_points.append({
                    "id": point_id(source_rel, idx, payload["content_hash"]),
                    "payload": payload,
                    "text": ch["text"],
                })
        stats["files"] += 1
        logger.info(f"   📖 {f.name}: {len(chunks)} чанков, {len(text):,} символов")

    if dry_run or not pending_points:
        return stats

    live_sources = [f"{folder_key}/{f.name}" for f in files]

    # ФИКС (аудит 🟡18, 2026-10-02): uuid5 включает content_hash — обновлённый
    # файл получает новые ID, а старые чанки остаются висеть (prune смотрит
    # только на source_file, который не изменился). Удаляем точки файлов
    # этой папки ДО заливки: перезаливка файла = чистая замена.
    if files_filter is None and not dry_run:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny, FilterSelector
            stale_flt = Filter(must=[
                FieldCondition(key="domain", match=MatchValue(value=folder_key)),
                FieldCondition(key="source_file", match=MatchAny(any=live_sources)),
            ])
            stale = client.count(collection_name=COLLECTION,
                                 count_filter=stale_flt, exact=True).count
            if stale:
                client.delete(collection_name=COLLECTION,
                              points_selector=FilterSelector(filter=stale_flt))
                logger.info(f"   🧹 delete-by-source {folder_key}: "
                            f"удалено {stale} старых точек перед перезаливкой")
        except Exception as e:
            logger.warning(f"   ⚠️ delete-by-source не удался (продолжаем): {e}")

    # Эмбеддинги батчами (локальные, e5) и заливка
    for i in range(0, len(pending_points), EMBED_BATCH):
        batch = pending_points[i:i + EMBED_BATCH]
        try:
            vectors = embed([p["text"] for p in batch], input_type="passage")
        except Exception as e:
            logger.error(f"❌ Ошибка эмбеддинга батча {i // EMBED_BATCH}: {e} — батч пропущен")
            continue
        points = [
            {"id": p["id"], "vector": v, "payload": p["payload"]}
            for p, v in zip(batch, vectors)
        ]
        stats["uploaded"] += upload(client, points)
        if (i // EMBED_BATCH) % 10 == 0:
            logger.info(f"   ⏳ залито {stats['uploaded']}/{len(pending_points)}")

    # Чистка осиротевших точек (только полный прогон папки, без --limit;
    # при точечном --files prune ЗАПРЕЩЁН — live_sources строится из
    # отфильтрованного списка и prune удалил бы точки остальных файлов)
    if limit is None and files_filter is None:
        stats["pruned"] = prune_folder(client, folder_key, live_sources)

    return stats


def _guess_codex(filename: str) -> str:
    name = filename.lower()
    for key, label in [("тк", "ТК РФ"), ("гк", "ГК РФ"), ("нк", "НК РФ"), ("коап", "КоАП РФ")]:
        if key in name:
            return label
    if "115" in name:
        return "ФЗ-115"
    if "38" in name:
        return "ФЗ-38"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="KB v2 loader")
    ap.add_argument("--folder", default="all",
                    choices=list(FOLDERS.keys()) + ["all"])
    ap.add_argument("--fresh", action="store_true", help="пересоздать коллекцию")
    ap.add_argument("--dry-run", action="store_true", help="парсинг/чанкинг без заливки")
    ap.add_argument("--limit", type=int, default=None, help="максимум файлов на папку")
    ap.add_argument("--files", default=None,
                    help="точечная дозагрузка: glob-паттерн имён (например 'Минфин_письмо*')")
    ap.add_argument("--list", action="store_true", help="показать, что видит загрузчик")
    args = ap.parse_args()

    if args.list:
        for key, cfg in FOLDERS.items():
            folder = KB_ROOT / key
            files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED] if folder.exists() else []
            total = sum(f.stat().st_size for f in files) / 1048576
            print(f"{cfg['label']:32s} {key:24s} {len(files):3d} файлов {total:8.1f} МБ")
        return 0

    client = None if args.dry_run else get_client()
    if client:
        ensure_collection(client, recreate=args.fresh)

    targets = list(FOLDERS.keys()) if args.folder == "all" else [args.folder]
    totals = {"files": 0, "chunks": 0, "uploaded": 0, "skipped": 0}
    for key in targets:
        st = process_folder(key, dry_run=args.dry_run, limit=args.limit,
                            client=client, files_filter=args.files)
        for k in totals:
            totals[k] += st[k]

    logger.info("\n" + "=" * 60)
    logger.info(f"{'ЧЕРНОВЫЙ ПРОГОН (dry-run)' if args.dry_run else 'ГОТОВО'}: "
                f"файлов {totals['files']}, чанков {totals['chunks']}, "
                f"залито {totals['uploaded']}, пропущено {totals['skipped']}")
    if not args.dry_run:
        info = client.get_collection(COLLECTION)
        logger.info(f"Коллекция {COLLECTION}: {info.points_count} точек")
    return 0


if __name__ == "__main__":
    sys.exit(main())
