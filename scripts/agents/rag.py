"""
RAG-модуль — запросы агентов к Qdrant.

Предоставляет единый интерфейс для семантического поиска
с учётом агенто-специфичных фильтров из registry.
"""
import logging
import re as _re
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


# ────────────────────────────────────────────────────────────
# QUERY REWRITING (2026-10-05)
# ────────────────────────────────────────────────────────────

_LEGAL_REF_RE = _re.compile(
    r"ст\.?\s*\d|статья\s*\d|\d+\s*[-–]?\s*фз|фз\s*№|нк\s*рф|гк\s*рф|коап|тк\s*рф",
    _re.IGNORECASE,
)
_deepseek_rewrite_dead = False  # 401/402 — ключ/баланс, в этом процессе не дёргаем


def _is_exact_legal_ref(query_text: str) -> bool:
    """Точная юр. ссылка: sparse-ветка ловит её дословно, rewrite только размоет."""
    return bool(_LEGAL_REF_RE.search(query_text or ""))


def _rewrite_llm_call(prompt: str) -> str:
    """Один chat-вызов в OpenAI-совместимом API. Порядок: deepseek → kie.
    Возвращает текст ответа; любой сбой — исключение."""
    import os as _os
    import requests as _requests
    timeout = float(_os.getenv("KB_REWRITE_TIMEOUT", "15"))
    global _deepseek_rewrite_dead

    attempts = []
    ds_key = _os.getenv("DEEPSEEK_API_KEY")
    if ds_key and not _deepseek_rewrite_dead:
        attempts.append((
            f"{_os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1').rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {ds_key}"},
            _os.getenv("MODEL_DEEPSEEK_FLASH", "deepseek-flash"),
            True,
        ))
    kie_key = _os.getenv("KIE_API_KEY")
    if kie_key:
        # gemini-3-8-flash: у kie.ai отдаёт корректный OpenAI-формат;
        # claude-4.7 на этом шлюзе возвращает 200 с телом {code,msg,data}
        # без choices (проверено 2026-10-05). Модель нестабильна (~50%
        # internal error) — поэтому ниже один ретрай на провайдера.
        kie_model = _os.getenv("MODEL_REWRITE", "gemini-3-8-flash-openai")
        host = _os.getenv("KIE_API_HOST", "https://api.kie.ai").rstrip("/")
        attempts.append((f"{host}/{kie_model}/v1/chat/completions",
                         {"Authorization": f"Bearer {kie_key}"}, kie_model, False))

    last_err: Optional[Exception] = None
    for url, headers, model, is_ds in attempts:
        for _try in range(2):  # 1 ретрай: gemini на kie.ai периодически отдаёт ошибку
            try:
                r = _requests.post(
                    url,
                    headers={**headers, "Content-Type": "application/json"},
                    json={"model": model,
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0.3, "max_tokens": 300},
                    timeout=timeout,
                )
                if is_ds and r.status_code in (401, 402):
                    _deepseek_rewrite_dead = True
                r.raise_for_status()
                # валидация: kie.ai может вернуть 200 с телом без choices
                # (например claude-4.7: {"code":..,"msg":..,"data":..})
                data = r.json()
                content = ""
                try:
                    content = data["choices"][0]["message"].get("content") or ""
                except (KeyError, IndexError, TypeError, AttributeError):
                    pass
                if content:
                    return content
                last_err = RuntimeError(f"пустой/нестандартный ответ от {model}")
            except Exception as e:
                last_err = e
    raise RuntimeError(f"нет доступного LLM для rewrite: {last_err}")


_REWRITE_PROMPT = (
    "Разверни поисковый запрос в {n} самостоятельных формулировок для "
    "семантического поиска по базе знаний (маркетинг, копирайтинг, психология "
    "влияния, бизнес и право РФ).\n"
    "Правила:\n"
    "- каждая формулировка раскрывает отдельную грань запроса: синонимы, "
    "смежные термины, бытовая или бизнес-формулировка той же потребности;\n"
    "- НЕ отвечай на запрос и ничего не добавляй от себя — только переформулируй;\n"
    "- одна строка — одна формулировка, без нумерации, кавычек и пояснений.\n"
    "Запрос: {q}"
)

_LIST_PREFIX_RE = _re.compile(r"^\s*(?:[-•*]|\d+[.)])\s*")


def rewrite_query(query_text: str, num_variants: int = 2) -> List[str]:
    """LLM разворачивает поисковый запрос в самостоятельные формулировки.
    При любом сбое бросает исключение — вызывающий код ищет исходным запросом."""
    q = (query_text or "").strip()
    if len(q) < 12:
        return []
    raw = _rewrite_llm_call(_REWRITE_PROMPT.format(n=num_variants, q=q))
    out: List[str] = []
    for line in raw.splitlines():
        line = _LIST_PREFIX_RE.sub("", line.strip()).strip()
        if len(line) >= 12 and line.lower() != q.lower() and line not in out:
            out.append(line)
    return out[:num_variants]


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
        # QUERY REWRITING: LLM разворачивает запрос в 2-3 самостоятельных
        # формулировки, поиск идёт по каждой, выдачи сливаются RRF-ом.
        # Точные юр. ссылки («ст. 46 НК РФ») не переписываем — их ловит
        # sparse-ветка дословно. Выключается KB_QUERY_REWRITE=0.
        import os as _os
        variants = [query_text]
        if _os.getenv("KB_QUERY_REWRITE", "1") != "0" and not _is_exact_legal_ref(query_text):
            try:
                variants.extend(rewrite_query(query_text, num_variants=2)[:2])
                if len(variants) > 1:
                    logger.info(f"RAG [{agent_id}]: rewrite -> {len(variants)} формулировки")
            except Exception as _rw_err:
                logger.info(f"RAG [{agent_id}]: rewrite недоступен ({_rw_err}) — исходный запрос")

        # Embedding всех формулировок (input_type="query": префикс E5 для поиска)
        from copywriter_kb.loader import get_embeddings_batch
        embeddings = get_embeddings_batch(variants, input_type="query")
        variant_vectors = [(v, e) for v, e in zip(variants, embeddings or []) if e is not None]
        if not variant_vectors:
            logger.warning(f"RAG [{agent_id}]: не удалось получить embedding запроса — пропускаем RAG")
            return []

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
        try:
            from kb2.sparse_utils import query_sparse as _q_sparse
            from qdrant_client.models import FusionQuery, Prefetch, SparseVector
            _sparse_ok = True
        except ImportError:
            _sparse_ok = False

        pool_limit = max(agent.rag.top_k * 2, 20)  # пул на формулировку до слияния

        def _search_once(query_vector, q_text):
            """Один поиск: гибрид (dense+sparse → RRF) или dense-fallback."""
            if _sparse_ok:
                try:
                    res = qdrant_client.query_points(
                        collection_name=agent.rag.collection,
                        prefetch=[
                            Prefetch(query=query_vector, using="dense",
                                     limit=pool_limit,
                                     filter=search_filter,
                                     score_threshold=agent.rag.score_threshold),
                            Prefetch(
                                query=SparseVector(**_q_sparse(q_text)),
                                using="sparse",
                                limit=pool_limit,
                                filter=search_filter),
                        ],
                        query=FusionQuery(fusion="rrf"),
                        limit=pool_limit,
                        with_payload=True,
                    )
                    return res.points
                except Exception as _hyb_err:
                    logger.warning(f"RAG [{agent_id}]: гибрид недоступен "
                                   f"({_hyb_err}) — чистый dense")
            # named-коллекция требует using="dense"; старые unnamed — без using
            try:
                res = qdrant_client.query_points(
                    collection_name=agent.rag.collection,
                    query=query_vector, using="dense",
                    query_filter=search_filter,
                    limit=pool_limit,
                    score_threshold=agent.rag.score_threshold,
                    with_payload=True,
                )
            except Exception:
                res = qdrant_client.query_points(
                    collection_name=agent.rag.collection,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=pool_limit,
                    with_payload=True,
                )
            return res.points

        # Мультипоиск по формулировкам + RRF-слияние по point id (k=60):
        # чанк, найденный несколькими формулировками, поднимается выше —
        # в этом и смысл query rewriting.
        merged: Dict[Any, Dict[str, Any]] = {}
        for v_text, v_vec in variant_vectors:
            try:
                points = _search_once(v_vec, v_text)
            except Exception as _s_err:
                logger.warning(f"RAG [{agent_id}]: поиск не удался ({_s_err})")
                continue
            for rank, hit in enumerate(points):
                entry = merged.setdefault(hit.id, {"rrf": 0.0, "hit": hit})
                entry["rrf"] += 1.0 / (60 + rank)

        ranked = sorted(merged.values(),
                        key=lambda e: e["rrf"], reverse=True)[:agent.rag.top_k]

        chunks = []
        for entry in ranked:
            hit = entry["hit"]
            chunk = {"score": entry["rrf"]}
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

        logger.info(f"RAG [{agent_id}]: найдено {len(chunks)} чанков, "
                    f"формулировок {len(variant_vectors)} (query: {query_text[:50]}...)")
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
