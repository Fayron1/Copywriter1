"""
RAG-модуль — запросы агентов к Qdrant.

Предоставляет единый интерфейс для семантического поиска
с учётом агенто-специфичных фильтров из registry.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("agents.rag")


def query_knowledge(
    query_text: str,
    agent_id: str,
    qdrant_client=None,
    extra_filters: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Семантический поиск в Qdrant для конкретного агента.

    Args:
        query_text: текст запроса для embedding
        agent_id: ID агента (определяет фильтры и top_k)
        qdrant_client: клиент Qdrant (если None — создаёт новый)
        extra_filters: дополнительные фильтры поверх агентских

    Returns:
        Список найденных чанков с payload
    """
    from .registry import get_agent
    agent = get_agent(agent_id)

    if not agent.rag.enabled:
        logger.debug(f"RAG отключен для {agent_id}")
        return []

    # Qdrant клиент
    if qdrant_client is None:
        from copywriter_kb.loader import get_qdrant
        qdrant_client = get_qdrant()

    # Embedding запроса
    from copywriter_kb.loader import get_embeddings_batch
    query_vector = get_embeddings_batch([query_text])[0]

    # Фильтры
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

    must_conditions = []

    # Фильтр по agent_target
    agent_filter = agent.rag.filters.get("agent_target")
    if agent_filter:
        must_conditions.append(
            FieldCondition(key="agent_target", match=MatchValue(value=agent_filter))
        )

    # Фильтр по source_type
    source_types = agent.rag.filters.get("source_type")
    if source_types and isinstance(source_types, list):
        must_conditions.append(
            FieldCondition(key="source_type", match=MatchAny(any=source_types))
        )

    # Extra фильтры
    if extra_filters:
        for key, value in extra_filters.items():
            if isinstance(value, list):
                must_conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )
            else:
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

    # Мягкий фильтр актуальности законов: valid_until >= today ИЛИ поле отсутствует
    if "valid_until" in agent.rag.payload_fields:
        from datetime import datetime
        from qdrant_client import models
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # В новых версиях qdrant-client для дат используется DatetimeRange, предотвращая предупреждения Pydantic
        try:
            range_val = models.DatetimeRange(gte=today_str)
        except AttributeError:
            # Резервный обход для старых версий qdrant-client
            range_val = models.Range(gte=0.0)
            range_val.gte = today_str
        
        must_conditions.append(
            Filter(should=[
                FieldCondition(
                    key="valid_until",
                    range=range_val,
                ),
                models.IsNullCondition(
                    is_null=models.PayloadField(key="valid_until"),
                ),
            ])
        )
        logger.info(f"RAG [{agent_id}]: фильтр актуальности (valid_until >= {today_str} OR null)")

    search_filter = Filter(must=must_conditions) if must_conditions else None

    # Поиск
    try:
        results = qdrant_client.query_points(
            collection_name=agent.rag.collection,
            query=query_vector,
            query_filter=search_filter,
            limit=agent.rag.top_k,
            score_threshold=agent.rag.score_threshold,
            with_payload=True,
        )

        chunks = []
        for hit in results.points:
            chunk = {"score": hit.score}
            for field in agent.rag.payload_fields:
                chunk[field] = hit.payload.get(field, None)
            # Всегда включаем text
            if "text" not in chunk:
                chunk["text"] = hit.payload.get("text", "")
            chunks.append(chunk)

        logger.info(f"RAG [{agent_id}]: найдено {len(chunks)} чанков (query: {query_text[:50]}...)")
        return chunks

    except Exception as e:
        logger.error(f"RAG [{agent_id}] ошибка: {e}")
        return []


def format_rag_context(chunks: List[Dict], max_chars: int = 8000) -> str:
    """
    Форматировать RAG-результаты в текстовый контекст для промпта.

    Args:
        chunks: результаты query_knowledge
        max_chars: максимальная длина контекста

    Returns:
        Отформатированный текст для вставки в промпт
    """
    if not chunks:
        return "[База знаний: релевантные данные не найдены]"

    lines = ["--- КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ---"]
    total = 0

    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        source = chunk.get("source_file", "неизвестно")
        stype = chunk.get("source_type", "")
        score = chunk.get("score", 0)

        entry = f"\n[{i}] (источник: {source}, тип: {stype}, релевантность: {score:.2f})\n{text}"

        if total + len(entry) > max_chars:
            lines.append(f"\n... (ещё {len(chunks) - i + 1} результатов обрезано)")
            break

        lines.append(entry)
        total += len(entry)

    lines.append("\n--- КОНЕЦ КОНТЕКСТА ---")
    return "\n".join(lines)
