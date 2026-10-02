"""
Локальная система генерации эмбеддингов (без OpenAI).

Модель: sentence-transformers (по умолчанию intfloat/multilingual-e5-base,
отлично работает с русским языком, 768 измерений, бежит на CPU).
Первый запуск скачивает модель с HuggingFace (~1.1 ГБ) и кладёт в кэш —
дальше полностью офлайн, ноль рублей за токены.

Правило семейства E5: тексты базы кодируются с префиксом "passage: ",
поисковые запросы — с префиксом "query: ". Префиксы ставятся автоматически
параметром input_type ("passage" при заливке, "query" при поиске).

Конфигурация (env):
  LOCAL_EMBEDDING_MODEL — имя модели HF (по умолчанию multilingual-e5-base).
    Альтернативы: intfloat/multilingual-e5-small (быстрее, 384d),
                  intfloat/multilingual-e5-large (точнее, 1024d, ~2.2 ГБ).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger("kb.local_embeddings")

DEFAULT_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-base").strip() \
    or "intfloat/multilingual-e5-base"

_model = None            # ленивый синглтон SentenceTransformer
_model_dim: Optional[int] = None
_lock = threading.Lock()

# Префиксы инструкции семейства E5 (для других семейств — пустые строки)
_PREFIXES = {
    "passage": "passage: ",
    "query": "query: ",
}


def _get_model():
    """Ленивая загрузка модели (один раз на процесс, потокобезопасно)."""
    global _model, _model_dim
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                logger.info(f"🧠 Загрузка локальной embedding-модели: {DEFAULT_MODEL} ...")
                _model = SentenceTransformer(DEFAULT_MODEL)  # CPU по умолчанию
                _model_dim = int(_model.get_sentence_embedding_dimension())
                logger.info(f"🧠 Модель готова: dim={_model_dim}")
    return _model


def get_dim() -> int:
    """Размерность вектора текущей модели (загружает модель при первом вызове)."""
    _get_model()
    return int(_model_dim or 0)


def embed(texts: List[str], input_type: str = "passage") -> List[List[float]]:
    """Прогнать тексты через локальную модель.

    input_type: "passage" — заливка документов в базу,
                "query"    — поисковый запрос (разные префиксы E5).
    Возвращает список векторов в порядке входа, нормализованных (для COSINE).
    """
    model = _get_model()
    prefix = _PREFIXES.get(input_type, "")
    prepared = [f"{prefix}{t}"[:20000] for t in texts]
    vectors = model.encode(
        prepared,
        batch_size=32,
        normalize_embeddings=True,   # косинусная близость в Qdrant
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]
