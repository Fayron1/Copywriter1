"""
Стратегический чанкер для Knowledge Base Pipeline.

Разные стратегии для разных типов контента:
  - Законы → по статьям/пунктам (сохранение юридических формулировок)
  - Книги → по абзацам с перекрытием
  - Справочники → по разделам
"""
import re
import logging
from typing import List

logger = logging.getLogger("kb.chunker")


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    source_type: str = "book",
) -> List[str]:
    """
    Разбить текст на чанки с учётом типа источника.

    Args:
        text: Чистый текст
        chunk_size: Максимальный размер чанка в символах
        overlap: Перекрытие между чанками
        source_type: Тип источника (law, book, guide, reference)

    Returns:
        Список чанков
    """
    if not text or len(text) < 50:
        return []

    # Выбираем стратегию
    if source_type == "law":
        chunks = _chunk_by_articles(text, chunk_size)
    elif source_type in ("guide", "reference"):
        chunks = _chunk_by_sections(text, chunk_size, overlap)
    else:
        chunks = _chunk_by_sentences(text, chunk_size, overlap)

    # Финальная фильтрация: убираем слишком короткие чанки
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    logger.debug(f"   Создано {len(chunks)} чанков (стратегия: {source_type}, размер: {chunk_size})")
    return chunks


# ============================================================
# Стратегия 1: По статьям/пунктам (для законов)
# ============================================================

def _chunk_by_articles(text: str, chunk_size: int) -> List[str]:
    """
    Разбиение юридических текстов по статьям.
    Паттерны: «Статья N.», «Ст. N», «Глава N», «Раздел N»
    """
    # Разбиваем по маркерам статей/глав
    pattern = r'(?=(?:Статья|Ст\.|Глава|Раздел|ГЛАВА|РАЗДЕЛ)\s+\d+)'
    segments = re.split(pattern, text)

    chunks = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Если сегмент вписывается в лимит — берём целиком
        if len(segment) <= chunk_size:
            chunks.append(segment)
        else:
            # Разбиваем длинную статью по пунктам или предложениям
            sub_chunks = _chunk_by_sentences(segment, chunk_size, overlap=50)
            chunks.extend(sub_chunks)

    return chunks


# ============================================================
# Стратегия 2: По разделам (для гайдов и справочников)
# ============================================================

def _chunk_by_sections(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Разбиение по заголовкам разделов (## / ### / H2 / H3).
    Если разделов нет — фолбэк на предложения.
    """
    # Паттерн: строки начинающиеся с ## или ### (Markdown-стиль)
    pattern = r'(?=\n#{1,3}\s+)'
    segments = re.split(pattern, text)

    # Если нашли мало разделов — фолбэк
    if len(segments) < 3:
        return _chunk_by_sentences(text, chunk_size, overlap)

    chunks = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        if len(segment) <= chunk_size:
            chunks.append(segment)
        else:
            sub_chunks = _chunk_by_sentences(segment, chunk_size, overlap)
            chunks.extend(sub_chunks)

    return chunks


# ============================================================
# Стратегия 3: По предложениям с перекрытием (для книг)
# ============================================================

def _chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Базовая стратегия: разбиение по предложениям с перекрытием.
    Сохраняет целостность предложений.
    """
    # Разбиваем на предложения (по . ! ? с учётом аббревиатур)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Если предложение само по себе больше chunk_size — разбиваем по словам
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Разбиваем длинное предложение по словам
            words = sentence.split()
            word_chunk = ""
            for word in words:
                if len(word_chunk) + len(word) + 1 <= chunk_size:
                    word_chunk += " " + word if word_chunk else word
                else:
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                    word_chunk = word
            if word_chunk:
                chunks.append(word_chunk.strip())
            continue

        # Стандартная логика: накапливаем предложения
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Перекрытие: берём хвост предыдущего чанка
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
