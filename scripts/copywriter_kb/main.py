"""
📚 Knowledge Base Pipeline — «Антигравити Копирайтер»

Главный скрипт загрузки базы знаний в Qdrant.
Обрабатывает книги из Books/ по папкам агентов, применяя
агенто-специфичные стратегии (RAW / дистилляция / гибрид).

Использование:
  python -m copywriter_kb.main --agent all            # Все агенты
  python -m copywriter_kb.main --agent fact_finder     # Только Исследователь
  python -m copywriter_kb.main --agent heart           # Только Писатель
  python -m copywriter_kb.main --agent sheriff         # Только Редактор
  python -m copywriter_kb.main --agent engineer        # Только Структурировщик
  python -m copywriter_kb.main --agent booster         # Только SEO-оптимизатор
  python -m copywriter_kb.main --agent artist          # Только Визуализатор
  python -m copywriter_kb.main --mode status           # Статистика
  python -m copywriter_kb.main --agent all --fresh     # Начать с нуля

Запуск с VPS:
  cd /путь/к/scripts && python -m copywriter_kb.main --agent all
"""
import argparse
import logging
import time
from pathlib import Path

from .config import (
    AGENT_MAP, BOOKS_BASE, SUPPORTED_FORMATS,
    OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT,
    COLLECTION_NAME, DISTILL_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM,
    OUTPUT_DIR,
)
from .parsers import extract_text
from .chunker import chunk_text
from .classifier import classify_chunk, BudgetExhaustedError
from .loader import (
    get_qdrant, ensure_collection, upload_chunks,
    get_processed_files, show_status,
    compute_content_hash, get_existing_hashes,
    save_checkpoint, load_checkpoint, clear_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("kb.main")


# ============================================================
# Обработка одного агента
# ============================================================

def process_agent(dir_name: str, agent_info: dict, fresh: bool = False) -> dict:
    """
    Обработать все файлы одного агента.

    Args:
        dir_name: Имя папки в Books/ (например, «Исследователь»)
        agent_info: Конфигурация агента из AGENT_MAP
        fresh: Начать с нуля (игнорировать resume)

    Returns:
        Статистика: {files_processed, chunks_created, points_uploaded, files_skipped}
    """
    agent_id = agent_info["agent_id"]
    label = agent_info["label"]
    strategy = agent_info["strategy"]
    chunk_size = agent_info["chunk_size"]
    default_source_type = agent_info["default_source_type"]

    book_dir = BOOKS_BASE / dir_name

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{label} ({agent_info['agent_name']})")
    logger.info(f"   Папка: {book_dir}")
    logger.info(f"   Стратегия: {strategy}")
    logger.info(f"   Размер чанка: {chunk_size}")
    logger.info(f"{'=' * 60}")

    if not book_dir.exists():
        logger.error(f"❌ Папка не найдена: {book_dir}")
        return {"files_processed": 0, "chunks_created": 0, "points_uploaded": 0, "files_skipped": 0}

    # Собираем файлы поддерживаемых форматов
    all_files = sorted([
        f for f in book_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    ])

    logger.info(f"   Найдено {len(all_files)} файлов")

    if not all_files:
        logger.warning(f"   ⚠️ Нет файлов в {book_dir}")
        return {"files_processed": 0, "chunks_created": 0, "points_uploaded": 0, "files_skipped": 0}

    # Qdrant
    qdrant = get_qdrant()
    ensure_collection(qdrant)

    # Resume: определяем уже загруженные файлы
    processed = set() if fresh else get_processed_files(qdrant, agent_id)
    if processed:
        logger.info(f"   ✅ Уже обработано {len(processed)} файлов, пропускаем")

    # Дедупликация: загружаем существующие хэши
    existing_hashes = set() if fresh else get_existing_hashes(qdrant, agent_id)
    if existing_hashes:
        logger.info(f"   🔑 Загружено {len(existing_hashes)} хэшей для дедупликации")

    stats = {"files_processed": 0, "chunks_created": 0, "points_uploaded": 0, "files_skipped": 0, "dedup_skipped": 0}
    budget_exhausted = False

    for file_path in all_files:
        if budget_exhausted:
            break

        # Пропускаем дубликаты (файлы с "(1)" в имени)
        if "(1)" in file_path.name:
            logger.info(f"   ⏭️ Пропуск дубликата: {file_path.name}")
            stats["files_skipped"] += 1
            continue

        # Пропускаем уже обработанные
        if file_path.name in processed:
            logger.info(f"   ⏭️ Уже в Qdrant: {file_path.name}")
            stats["files_skipped"] += 1
            continue

        logger.info(f"\n📖 {file_path.name} ({file_path.suffix})")

        # 1. Парсинг
        text = extract_text(file_path)
        if not text:
            logger.warning(f"   ⚠️ Не удалось извлечь текст")
            stats["files_skipped"] += 1
            continue

        logger.info(f"   Текст: {len(text):,} символов")

        # 2. Чанкинг
        chunks = chunk_text(text, chunk_size=chunk_size, source_type=default_source_type)
        logger.info(f"   Чанки: {len(chunks)}")

        if not chunks:
            logger.warning(f"   ⚠️ Нет чанков после разбиения")
            stats["files_skipped"] += 1
            continue

        # Checkpoint resume: пропускаем уже обработанные чанки внутри файла
        start_chunk = load_checkpoint(agent_id, file_path.name) + 1
        if start_chunk > 0:
            logger.info(f"   ♻️ Checkpoint: продолжаем с чанка {start_chunk}/{len(chunks)}")

        # 3. Классификация + метаданные
        classified_chunks = []
        skipped_chunks = 0

        try:
            for i, chunk in enumerate(chunks):
                if i < start_chunk:
                    continue  # Пропускаем уже обработанные чанки

                # Дедупликация: проверяем content hash
                chunk_hash = compute_content_hash(chunk)
                if chunk_hash in existing_hashes:
                    stats["dedup_skipped"] += 1
                    continue

                metadata = classify_chunk(
                    chunk=chunk, agent_id=agent_id, strategy=strategy,
                    source_file=file_path.name, file_format=file_path.suffix.lower(),
                )
                if metadata:
                    metadata["chunk_index"] = i
                    metadata["total_chunks"] = len(chunks)
                    classified_chunks.append(metadata)
                    existing_hashes.add(chunk_hash)  # Запоминаем для текущей сессии
                else:
                    skipped_chunks += 1

                # Задержка для стратегий с GPT
                if strategy in ("raw_classified", "distill", "hybrid"):
                    time.sleep(0.1)

                # Сохраняем checkpoint каждые 50 чанков
                if i > 0 and i % 50 == 0:
                    save_checkpoint(agent_id, file_path.name, i)

        except BudgetExhaustedError:
            logger.error(f"💸 БАЛАНС ИСЧЕРПАН на файле {file_path.name}, чанк {i}")
            save_checkpoint(agent_id, file_path.name, i)
            # Загружаем то, что успели обработать
            if classified_chunks:
                uploaded = upload_chunks(qdrant, classified_chunks, agent_id)
                stats["points_uploaded"] += uploaded
                logger.info(f"   💾 Сохранено {uploaded} точек до остановки")
            budget_exhausted = True
            continue

        logger.info(f"   Классифицировано: {len(classified_chunks)} (пропущено: {skipped_chunks}, дубли: {stats['dedup_skipped']})")

        # 4. Загрузка в Qdrant
        if classified_chunks:
            try:
                uploaded = upload_chunks(qdrant, classified_chunks, agent_id)
                stats["points_uploaded"] += uploaded
                logger.info(f"   📤 Загружено: {uploaded} точек")
                clear_checkpoint(agent_id, file_path.name)  # Файл обработан полностью
            except BudgetExhaustedError:
                save_checkpoint(agent_id, file_path.name, len(chunks) - 1)
                budget_exhausted = True
                continue
        else:
            logger.warning(f"   ⚠️ Нет чанков для загрузки")
            clear_checkpoint(agent_id, file_path.name)

        stats["files_processed"] += 1
        stats["chunks_created"] += len(classified_chunks)

    logger.info(f"\n{label} — итого:")
    logger.info(f"   Обработано файлов: {stats['files_processed']}")
    logger.info(f"   Создано чанков: {stats['chunks_created']}")
    logger.info(f"   Загружено точек: {stats['points_uploaded']}")
    logger.info(f"   Дедупликация: {stats['dedup_skipped']} дублей пропущено")
    logger.info(f"   Пропущено: {stats['files_skipped']}")
    if budget_exhausted:
        logger.error(f"   💸 ОСТАНОВЛЕНО: баланс OpenAI исчерпан. Перезапустите после пополнения.")

    return stats


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="📚 Knowledge Base Pipeline — Антигравити Копирайтер"
    )
    parser.add_argument(
        "--agent",
        choices=[info["agent_id"] for info in AGENT_MAP.values()] + ["all"],
        default="all",
        help="Какого агента обрабатывать (agent_id или 'all')",
    )
    parser.add_argument(
        "--mode",
        choices=["process", "status"],
        default="process",
        help="Режим: process (обработка) или status (статистика)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Начать с нуля, игнорируя уже обработанные файлы",
    )
    parser.add_argument(
        "--qdrant-host",
        default=None,
        help="Переопределить хост Qdrant (из .env)",
    )

    args = parser.parse_args()

    # Переопределяем хост если указан
    if args.qdrant_host:
        import copywriter_kb.config as cfg
        cfg.QDRANT_HOST = args.qdrant_host

    # Режим status
    if args.mode == "status":
        show_status()
        return

    # Проверка ключей
    if not OPENAI_API_KEY:
        logger.error("❌ Не задан OPENAI_API_KEY! Заполни .env")
        return

    # Шапка
    logger.info("=" * 60)
    logger.info("📚 Knowledge Base Pipeline — Антигравити Копирайтер")
    logger.info(f"   Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info(f"   Коллекция: {COLLECTION_NAME}")
    logger.info(f"   LLM: {DISTILL_MODEL}")
    logger.info(f"   Embeddings: {EMBEDDING_MODEL} ({EMBEDDING_DIM}d)")
    logger.info(f"   Агент: {args.agent}")
    logger.info(f"   Books: {BOOKS_BASE}")
    logger.info(f"   Resume: {'❌ начать с нуля' if args.fresh else '✅ продолжить'}")
    logger.info("=" * 60)

    # Определяем каких агентов обрабатывать
    if args.agent == "all":
        agents_to_process = list(AGENT_MAP.items())
    else:
        agents_to_process = [
            (dir_name, info) for dir_name, info in AGENT_MAP.items()
            if info["agent_id"] == args.agent
        ]

    if not agents_to_process:
        logger.error(f"❌ Агент '{args.agent}' не найден")
        return

    # Общая статистика
    total_stats = {"files_processed": 0, "chunks_created": 0, "points_uploaded": 0, "files_skipped": 0}
    start_time = time.time()

    # Обрабатываем
    budget_stopped = False
    for dir_name, agent_info in agents_to_process:
        if budget_stopped:
            logger.warning(f"⏭️ Пропуск {agent_info['label']} — баланс исчерпан")
            continue
        agent_stats = process_agent(dir_name, agent_info, fresh=args.fresh)
        for key in total_stats:
            total_stats[key] += agent_stats.get(key, 0)
        if agent_stats.get("budget_exhausted"):
            budget_stopped = True

    elapsed = time.time() - start_time

    # Финальный отчёт
    logger.info("\n" + "=" * 60)
    logger.info("✅ ЗАГРУЗКА ЗАВЕРШЕНА")
    logger.info(f"   Время: {elapsed:.1f} сек ({elapsed/60:.1f} мин)")
    logger.info(f"   Файлов обработано: {total_stats['files_processed']}")
    logger.info(f"   Чанков создано: {total_stats['chunks_created']}")
    logger.info(f"   Точек загружено: {total_stats['points_uploaded']}")
    logger.info(f"   Пропущено: {total_stats['files_skipped']}")
    logger.info("=" * 60)

    # Показываем финальный статус
    show_status()


if __name__ == "__main__":
    main()
