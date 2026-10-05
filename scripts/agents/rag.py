"""
RAG-модуль — запросы агентов к Qdrant.

Предоставляет единый интерфейс для семантического поиска
с учётом агенто-специфичных фильтров из registry.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("agents.rag")

# Доменные ключи базы знаний: направление статьи → префиксы source_file.
# Используются только для буста ранжирования (+4%), НЕ для фильтра:
# смежные источники (ГК, общий КоАП, обзоры судов) остаются доступны.
DIRECTION_BOOST_PREFIXES: Dict[str, List[str]] = {
    "закупки": [
        "legislation/Закупки_", "legislation/44-ФЗ_снимок",
        "legislation/ФЗ-44_", "legislation/ФЗ-223_",
        "legislation/КоАП_закупочный", "legislation/Минфин_письмо",
    ],
    "вэд": [
        "legislation/Валютный", "legislation/Закупки_ФЗ_279",
    ],
    "налоги": [
        "legislation/НК_", "legislation/Налогов",
        "legislation/Закупки_ГК_РФ",
        "business/reference/УСН_ставки", "business/reference/Производственный",
    ],
    "трудовое": [
        "legislation/Изменения_трудового", "legislation/КЭДО",
        "legislation/ТК_РФ",
    ],
    "защита данных": [
        "legislation/152", "legislation/ФЗ-152",
    ],
    "маркетинг": [
        "business/marketplaces", "business/unit_economics",
        "business/marketing",
    ],
    "реклама": [
        "legislation/ФЗ-38", "legislation/КоАП",
        "business/marketing/paid_traffic_russia",
        "business/marketing/ai_for_business",
    ],
}

# Топик-буст: маркеры в тексте запроса → topic чанка. Дополняет
# DIRECTION_BOOST_PREFIXES: direction известен заранее (домен статьи),
# маркеры ловят тему внутри самого запроса агента («как писать заголовки»
# должно поднимать copywriting даже в статье про закупки). Тоже +4%,
# фильтром НЕ режем — смежные темы должны оставаться досягаемыми.
TOPIC_QUERY_MARKERS: Dict[str, List[str]] = {
    "psychology": [
        "психологи", "убежден", "влияни", "покупател", "поведен",
        "когнитив", "склонност", "мотивац", "дефицит", "триггер",
    ],
    "copywriting": [
        "заголовк", "копирайт", "редактур", "формулировк", "слог",
        "абзац", "канцеляр", "стилист", "переговор",
    ],
    "marketing": [
        "воронк", "лидо", "лидогенер", "трафик", "конверс", "оффер",
        "рассылк", "автоворонк", "лендинг", "crm", "email",
    ],
    "seo": ["seo", "гео-оптимизац", "поисков", "schema", "eeat", "e-e-a"],
}


def infer_query_topics(query_text: str) -> List[str]:
    """Грубое определение темы по маркерам запроса (см. TOPIC_QUERY_MARKERS)."""
    q = (query_text or "").lower()
    topics = []
    for topic, markers in TOPIC_QUERY_MARKERS.items():
        if any(m in q for m in markers):
            topics.append(topic)
    return topics


def apply_topic_boost(chunks: List[Dict], query_text: str) -> List[Dict]:
    """Чанки, чей topic совпал с темой запроса, получают +4% к скору.
    Тихо пропускает чанки без topic (старые точки до бэкфилла)."""
    topics = infer_query_topics(query_text)
    if not topics:
        return chunks
    for ch in chunks:
        if ch.get("topic") in topics:
            ch["score"] = ch.get("score", 0.0) * 1.04
    return chunks


def query_knowledge(
    query_text: str,
    agent_id: str,
    qdrant_client=None,
    extra_filters: Optional[Dict] = None,
    direction: str = "",
) -> List[Dict[str, Any]]:
    """
    Семантический поиск в Qdrant для конкретного агента.

    Args:
        query_text: текст запроса для embedding
        agent_id: ID агента (определяет фильтры и top_k)
        qdrant_client: клиент Qdrant (если None — создаёт новый)
        extra_filters: дополнительные фильтры поверх агентских
        direction: домен статьи (закупки/вэд/налоги/...) — включает
            доменный буст ранжирования (см. DIRECTION_BOOST_PREFIXES)

    Returns:
        Список найденных чанков с payload
    """
    from .registry import get_agent

    # Безопасное получение конфигурации агента
    try:
        agent = get_agent(agent_id)
    except Exception as e:
        logger.warning(f"RAG [{agent_id}]: агент не найден в реестре ({e}) — пропускаем RAG")
        return []

    # Уважаем отключение RAG на уровне агента
    if not agent.rag.enabled:
        logger.debug(f"RAG отключён для {agent_id}")
        return []

    # Клиент Qdrant не передан => RAG считается отключённым (Qdrant недоступен).
    # НЕ пересоздаём клиент здесь: generate.py намеренно передаёт None при
    # недоступном Qdrant ("RAG отключён"), и молчаливое пересоздание ломало бы это.
    if qdrant_client is None:
        logger.info(f"RAG [{agent_id}]: клиент Qdrant не передан — RAG отключён, пропускаем")
        return []

    # Любой сбой ниже (embedding, построение фильтров, запрос) не должен ронять
    # пайплайн: при ошибке тихо возвращаем пустой контекст.
    try:
        # Embedding запроса (input_type="query": префикс E5 для поисковых запросов)
        from copywriter_kb.loader import get_embeddings_batch
        embeddings = get_embeddings_batch([query_text], input_type="query")
        if not embeddings or embeddings[0] is None:
            logger.warning(f"RAG [{agent_id}]: не удалось получить embedding запроса — пропускаем RAG")
            return []
        query_vector = embeddings[0]

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

            # КРИТИЧНО: is_null НЕ матчит ПОЛНОСТЬЮ ОТСУТСТВУЮЩЕЕ поле, а у
            # законов valid_until нет вовсе — однажды is_null-вариант выкинул
            # из выдачи fact_finder все 31k законодательных чанков, оставив
            # ~1k отчётов (диагностика 2026-10-02). Поэтому тройное should:
            # актуальные ИЛИ пустые ИЛИ явные null.
            should_variants = [
                models.IsNullCondition(
                    is_null=models.PayloadField(key="valid_until"),
                ),
            ]
            try:
                should_variants.insert(0, models.IsEmptyCondition(
                    is_empty=models.PayloadField(key="valid_until"),
                ))
            except AttributeError:
                pass
            must_conditions.append(
                Filter(should=[
                    FieldCondition(
                        key="valid_until",
                        range=range_val,
                    ),
                    *should_variants,
                ])
            )
            logger.info(f"RAG [{agent_id}]: фильтр актуальности (valid_until >= {today_str} OR empty OR null)")

        search_filter = Filter(must=must_conditions) if must_conditions else None

        # Поиск
        # ГИБРИДНЫЙ ПОИСК (2026-10-02): dense e5 + sparse BM25-стиль, RRF.
        # Проблема: dense размывает точные лексемы — «ч. 2 ст. 105» уплывает
        # к «жалобам» вообще. Sparse-ветка ловит номера статей/законов
        # дословно. Fallback: нет sparse-конфига у коллекции → чистый dense
        # (старое поведение, прод не ломается).
        _sparse_ok = True
        try:
            from kb2.sparse_utils import query_sparse as _q_sparse
            from qdrant_client.models import FusionQuery, Prefetch, SparseVector
        except ImportError:
            _sparse_ok = False

        if _sparse_ok:
            try:
                results = qdrant_client.query_points(
                    collection_name=agent.rag.collection,
                    prefetch=[
                        Prefetch(query=query_vector, using="dense",
                                 limit=max(agent.rag.top_k * 3, 30),
                                 filter=search_filter,
                                 score_threshold=agent.rag.score_threshold),
                        Prefetch(
                            query=SparseVector(**_q_sparse(query_text)),
                            using="sparse",
                            limit=max(agent.rag.top_k * 3, 30),
                            filter=search_filter),
                    ],
                    query=FusionQuery(fusion="rrf"),
                    limit=agent.rag.top_k,
                    with_payload=True,
                )
            except Exception as _hyb_err:
                logger.warning(f"RAG [{agent_id}]: гибрид недоступен "
                               f"({_hyb_err}) — чистый dense")
                _sparse_ok = False

        if not _sparse_ok:
            # named-коллекция требует using="dense"; старые unnamed — без using
            try:
                results = qdrant_client.query_points(
                    collection_name=agent.rag.collection,
                    query=query_vector, using="dense",
                    query_filter=search_filter,
                    limit=agent.rag.top_k,
                    score_threshold=agent.rag.score_threshold,
                    with_payload=True,
                )
            except Exception:
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

        # ДОМЕННЫЙ БУСТ ранжирования (см. apply_direction_boost)
        chunks = apply_direction_boost(chunks, direction)
        # ТОПИК-БУСТ: маркеры запроса → topic (см. apply_topic_boost)
        chunks = apply_topic_boost(chunks, query_text)

        logger.info(f"RAG [{agent_id}]: найдено {len(chunks)} чанков (query: {query_text[:50]}...)")
        return chunks

    except Exception as e:
        logger.error(f"RAG [{agent_id}] ошибка — пропускаем RAG (возвращаем пустой контекст): {e}")
        return []


def apply_direction_boost(chunks: List[Dict], direction: str) -> List[Dict]:
    """Доменный буст ранжирования: чанки домена статьи получают +4% к
    скору. Фильтр не режет смежные источники (ГК/КоАП нужны закупкам),
    но «жалоба» из НК ст. 46 больше не обыгрывает жалобу в ФАС по 44-ФЗ
    (диагностика 2026-10-02: топ-1/2 занимали чужие домены)."""
    if not direction:
        return chunks
    prefixes = DIRECTION_BOOST_PREFIXES.get(direction)
    if not prefixes:
        return chunks
    for ch in chunks:
        src = str(ch.get("source_file") or "")
        if any(src.startswith(p) for p in prefixes):
            ch["score"] = ch.get("score", 0.0) * 1.04
    chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return chunks


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
