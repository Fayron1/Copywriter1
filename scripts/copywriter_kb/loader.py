"""
Загрузчик в Qdrant для Knowledge Base Pipeline.

- Создание коллекции `copywriter_kb` с text-embedding-3-large (3072d)
- Батчевый embedding через OpenAI
- Resume-поддержка (пропуск уже загруженных файлов + чанковый checkpoint)
- Content Hash дедупликация (MD5 fingerprint)
- Retry при ошибках embedding
- Статистика по агентам
"""
import uuid
import time
import json
import hashlib
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

from .config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
    COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DIM,
    OPENAI_API_KEY, EMBEDDING_BATCH_SIZE, QDRANT_BATCH_SIZE,
    API_DELAY, AGENT_MAP, MAX_RETRIES, RETRY_DELAYS,
    CHECKPOINT_DIR,
)
from .classifier import BudgetExhaustedError

logger = logging.getLogger("kb.loader")


# ============================================================
# OpenAI клиент (ленивая инициализация)
# ============================================================

_openai_client = None


def _get_openai():
    global _openai_client
    if not _openai_client:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=90.0)
    return _openai_client


# ============================================================
# Qdrant клиент
# ============================================================

def get_qdrant():
    """Получить клиент Qdrant"""
    from qdrant_client import QdrantClient
    return QdrantClient(
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout=60.0,
        https=False,
    )


def ensure_collection(client) -> None:
    """Создать коллекцию если не существует"""
    from qdrant_client.models import VectorParams, Distance, CollectionStatus

    try:
        info = client.get_collection(COLLECTION_NAME)
        if info.status == CollectionStatus.GREEN:
            logger.info(f"✅ Коллекция '{COLLECTION_NAME}' существует ({info.points_count} точек)")
            return
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"📦 Создана коллекция '{COLLECTION_NAME}' (dim={EMBEDDING_DIM}, model={EMBEDDING_MODEL})")


# ============================================================
# Embeddings (батчевый)
# ============================================================

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Получить embeddings для списка текстов (батчевый запрос).
    Включает retry при ошибках API.
    """
    trimmed = [t[:20000] for t in texts]

    for attempt in range(MAX_RETRIES):
        try:
            response = _get_openai().embeddings.create(
                model=EMBEDDING_MODEL,
                input=trimmed,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except Exception as e:
            error_str = str(e).lower()

            # Баланс исчерпан — немедленный выход
            if "insufficient" in error_str or "402" in error_str or "billing" in error_str:
                logger.error(f"💸 БАЛАНС OPENAI ИСЧЕРПАН (embedding): {e}")
                raise BudgetExhaustedError(f"Баланс OpenAI исчерпан: {e}")

            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"⚠️ Embedding error (попытка {attempt + 1}): {e}, жду {delay} сек...")
                time.sleep(delay)
            else:
                raise


# ============================================================
# Content Hash — дедупликация чанков
# ============================================================

def compute_content_hash(text: str) -> str:
    """
    Вычислить MD5-fingerprint нормализованного текста.
    Используется для дедупликации: один и тот же текст
    в разных книгах → один и тот же hash.
    """
    # Нормализация: lowercase, убираем пробелы/пунктуацию
    normalized = re.sub(r'[\s\W]+', '', text.lower())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def get_existing_hashes(client, agent_id: str = None) -> Set[str]:
    """
    Получить все content_hash из Qdrant для дедупликации.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    hashes = set()
    try:
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count == 0:
            return hashes

        scroll_filter = None
        if agent_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="agent_target", match=MatchValue(value=agent_id))]
            )

        offset = None
        while True:
            results = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                scroll_filter=scroll_filter,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            points, next_offset = results
            for point in points:
                h = point.payload.get("content_hash", "")
                if h:
                    hashes.add(h)
            if next_offset is None:
                break
            offset = next_offset

    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить хэши: {e}")

    return hashes


# ============================================================
# Checkpoint — resume внутри файла
# ============================================================

def save_checkpoint(agent_id: str, source_file: str, chunk_index: int) -> None:
    """Сохранить checkpoint: последний обработанный чанк внутри файла"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp_file = CHECKPOINT_DIR / f"{agent_id}_checkpoint.json"

    data = {}
    if cp_file.exists():
        try:
            data = json.loads(cp_file.read_text(encoding='utf-8'))
        except Exception:
            data = {}

    data[source_file] = {"last_chunk": chunk_index}
    cp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def load_checkpoint(agent_id: str, source_file: str) -> int:
    """
    Загрузить checkpoint: номер последнего обработанного чанка.
    Возвращает -1 если checkpoint не найден.
    """
    cp_file = CHECKPOINT_DIR / f"{agent_id}_checkpoint.json"
    if not cp_file.exists():
        return -1

    try:
        data = json.loads(cp_file.read_text(encoding='utf-8'))
        return data.get(source_file, {}).get("last_chunk", -1)
    except Exception:
        return -1


def clear_checkpoint(agent_id: str, source_file: str) -> None:
    """Удалить checkpoint после успешной полной загрузки файла"""
    cp_file = CHECKPOINT_DIR / f"{agent_id}_checkpoint.json"
    if not cp_file.exists():
        return

    try:
        data = json.loads(cp_file.read_text(encoding='utf-8'))
        data.pop(source_file, None)
        cp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


# ============================================================
# Загрузка точек в Qdrant
# ============================================================

def upload_chunks(
    client,
    chunks_data: List[Dict[str, Any]],
    agent_id: str,
) -> int:
    """
    Загрузить обработанные чанки в Qdrant.

    Args:
        client: QdrantClient
        chunks_data: Список словарей с text, метаданными
        agent_id: ID агента-владельца

    Returns:
        Количество загруженных точек
    """
    from qdrant_client.models import PointStruct

    if not chunks_data:
        return 0

    total_uploaded = 0

    # Обрабатываем батчами
    for batch_start in range(0, len(chunks_data), EMBEDDING_BATCH_SIZE):
        batch = chunks_data[batch_start:batch_start + EMBEDDING_BATCH_SIZE]

        # Тексты для embedding
        embed_texts = []
        for item in batch:
            # Если есть дистиллированный текст для embedding — используем его
            embed_text = item.get("embed_text", item.get("text", ""))
            embed_texts.append(embed_text)

        # Батчевый embedding
        try:
            embeddings = get_embeddings_batch(embed_texts)
        except Exception as e:
            logger.error(f"❌ Ошибка батчевого embedding: {e}")
            # Фолбэк: по одному
            embeddings = []
            for text in embed_texts:
                try:
                    emb = get_embeddings_batch([text])
                    embeddings.extend(emb)
                except Exception as e2:
                    logger.warning(f"⚠️ Пропуск чанка: {e2}")
                    embeddings.append([0.0] * EMBEDDING_DIM)
                time.sleep(API_DELAY)

        # Формируем точки
        points = []
        for item, embedding in zip(batch, embeddings):
            # Собираем payload (убираем embed_text — он не нужен в базе)
            payload = {k: v for k, v in item.items() if k != "embed_text"}
            payload["agent_target"] = [agent_id]
            # Content hash для дедупликации
            payload["content_hash"] = compute_content_hash(item.get("text", ""))

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload,
            ))

        # Загружаем в Qdrant
        for qdrant_start in range(0, len(points), QDRANT_BATCH_SIZE):
            qdrant_batch = points[qdrant_start:qdrant_start + QDRANT_BATCH_SIZE]
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=qdrant_batch)
                total_uploaded += len(qdrant_batch)
            except Exception as e:
                logger.error(f"❌ Qdrant upsert error: {e}")

        time.sleep(API_DELAY)

    return total_uploaded


# ============================================================
# Resume: определение уже загруженных файлов
# ============================================================

def get_processed_files(client, agent_id: str = None) -> Set[str]:
    """
    Получить список уже обработанных файлов из Qdrant.
    Если agent_id задан — только файлы этого агента.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    processed = set()
    try:
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count == 0:
            return processed

        scroll_filter = None
        if agent_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="agent_target", match=MatchValue(value=agent_id))]
            )

        offset = None
        while True:
            results = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                scroll_filter=scroll_filter,
                with_payload=["source_file"],
                with_vectors=False,
            )
            points, next_offset = results
            for point in points:
                source = point.payload.get("source_file", "")
                if source:
                    processed.add(source)
            if next_offset is None:
                break
            offset = next_offset

    except Exception as e:
        logger.warning(f"⚠️ Не удалось проверить Qdrant: {e}")

    return processed


# ============================================================
# Статистика
# ============================================================

def show_status() -> None:
    """Показать статус загруженных данных в Qdrant"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_qdrant()

    logger.info("\n" + "=" * 60)
    logger.info(f"📊 СТАТУС QDRANT — {COLLECTION_NAME}")
    logger.info("=" * 60)

    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"\n📦 {COLLECTION_NAME}: {info.points_count} точек (всего)")
    except Exception:
        logger.info(f"\n📦 {COLLECTION_NAME}: ❌ коллекция не существует")
        return

    # Статус по каждому агенту
    from .config import BOOKS_BASE, SUPPORTED_FORMATS

    for dir_name, agent_info in AGENT_MAP.items():
        agent_id = agent_info["agent_id"]
        label = agent_info["label"]
        agent_name = agent_info["agent_name"]
        book_dir = BOOKS_BASE / dir_name

        processed = get_processed_files(client, agent_id)

        # Считаем файлы в папке
        if book_dir.exists():
            all_files = [
                f.name for f in book_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_FORMATS
            ]
            remaining = [f for f in all_files if f not in processed]
        else:
            all_files = []
            remaining = []

        logger.info(f"\n{label} ({agent_name}, agent_id: {agent_id}):")
        logger.info(f"   Стратегия: {agent_info['strategy']}")
        logger.info(f"   В Qdrant: {len(processed)} файлов")

        if processed:
            for book in sorted(processed):
                logger.info(f"   ✅ {book}")

        if remaining:
            logger.info(f"   ⏳ Осталось: {len(remaining)} файлов")
            for f in sorted(remaining)[:10]:
                logger.info(f"      ⏳ {f}")
            if len(remaining) > 10:
                logger.info(f"      ... и ещё {len(remaining) - 10}")
        elif all_files:
            logger.info(f"   Все файлы обработаны! ✅")
        else:
            logger.info(f"   ⚠️ Папка {dir_name}/ не найдена или пуста")
