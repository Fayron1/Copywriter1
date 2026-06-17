"""
Конфигурация проекта «Антигравити Копирайтер» — Knowledge Base Pipeline

Все настройки: OpenAI, Qdrant, маппинг папок → агентов, метаданные.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка .env
for env_path in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent.parent / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

# ============================================================
# OpenAI
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DISTILL_MODEL = os.getenv("DISTILL_MODEL", "gpt-5.4-mini")  # Дистилляция KB
EMBEDDING_MODEL = "text-embedding-3-large"         # Embeddings (апгрейд с ada-002)
EMBEDDING_DIM = 3072                               # Размерность вектора

# ============================================================
# Qdrant
# ============================================================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "copywriter_kb"

# ============================================================
# Пути
# ============================================================
# Docker: /app/Books, локально: ../../Books (от scripts/copywriter_kb/)
BOOKS_BASE = Path(os.getenv("BOOKS_PATH", Path(__file__).parent.parent.parent / "Books"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_PATH", Path(__file__).parent.parent / "output"))

# ============================================================
# Чанкинг
# ============================================================
CHUNK_SIZE_DEFAULT = 800       # символов (для книг)
CHUNK_SIZE_LAW = 1200          # символов (для законов — чанки побольше для полноты)
CHUNK_SIZE_REFERENCE = 600     # символов (для справочных материалов)
CHUNK_OVERLAP = 100            # перекрытие

# ============================================================
# API
# ============================================================
API_DELAY = 0.3                # секунд между запросами OpenAI
EMBEDDING_BATCH_SIZE = 100     # чанков за один embedding-запрос (API limit: 2048)
QDRANT_BATCH_SIZE = 100        # точек за один upsert в Qdrant

# ============================================================
# Retry / Устойчивость к сбоям OpenAI
# ============================================================
MAX_RETRIES = 3                # макс. попыток при ошибке OpenAI
RETRY_DELAYS = [5, 15, 30]    # секунд между повторами (exponential backoff)
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_PATH", Path(__file__).parent.parent / "checkpoints"))

# ============================================================
# Маппинг: папка Books/{dir} → агент + стратегия
# ============================================================
AGENT_MAP = {
    "Исследователь": {
        "agent_id": "fact_finder",
        "agent_name": "The Fact-Finder",
        "label": "🔎 Исследователь",
        "strategy": "raw",              # Без дистилляции — точные формулировки
        "chunk_size": CHUNK_SIZE_LAW,
        "default_source_priority": 1,    # Законы — наивысший приоритет
        "default_source_type": "law",
    },
    "Писатель": {
        "agent_id": "heart",
        "agent_name": "The Heart",
        "label": "✍️ Писатель",
        "strategy": "raw_classified",    # RAW + лёгкая классификация через GPT
        "chunk_size": CHUNK_SIZE_DEFAULT,
        "default_source_priority": 2,
        "default_source_type": "book",
    },
    "Редактор": {
        "agent_id": "sheriff",
        "agent_name": "The Sheriff",
        "label": "🔫 Редактор",
        "strategy": "raw",               # Правила и справочники — точные
        "chunk_size": CHUNK_SIZE_REFERENCE,
        "default_source_priority": 2,
        "default_source_type": "reference",
    },
    "Структурировщик": {
        "agent_id": "engineer",
        "agent_name": "The Engineer",
        "label": "🏗️ Структурировщик",
        "strategy": "distill",           # Дистилляция — сжатие шаблонов в инструкции
        "chunk_size": CHUNK_SIZE_DEFAULT,
        "default_source_priority": 2,
        "default_source_type": "book",
    },
    "SEO-оптимизатор": {
        "agent_id": "booster",
        "agent_name": "The Booster",
        "label": "🚀 SEO-оптимизатор",
        "strategy": "hybrid",            # Гайды Google RAW, кейсы дистилляция
        "chunk_size": CHUNK_SIZE_DEFAULT,
        "default_source_priority": 2,
        "default_source_type": "guide",
    },
    "Визуализатор": {
        "agent_id": "artist",
        "agent_name": "The Artist",
        "label": "🎨 Визуализатор",
        "strategy": "raw",               # Брендбуки, промпт-словари — точные
        "chunk_size": CHUNK_SIZE_REFERENCE,
        "default_source_priority": 2,
        "default_source_type": "reference",
    },
}

# Style Fingerprinting: папка со статьями клиента → editorial_memory
STYLE_MAP = {
    "Стиль_клиента": {
        "agent_id": "style_fingerprint",
        "label": "📝 Стиль клиента",
        "strategy": "raw_classified",
        "chunk_size": CHUNK_SIZE_DEFAULT,
        "default_source_type": "style_example",
    },
}

# Коллекции Qdrant
COLLECTION_NAME_MEMORY = "editorial_memory"

# ============================================================
# Типы статей (из ТЗ клиента)
# ============================================================
ARTICLE_TYPES = {
    "checklist": {
        "label": "📋 10 пунктов О",
        "description": "Чек-лист по теме. Каждый пункт = блок текста + картинка",
        "structure": "block",           # Блочная структура
        "style_folder": "чек_листы",    # Папка в Books/Стиль_клиента/
    },
    "case_study": {
        "label": "🔍 Ситуации",
        "description": "Конкретная ситуация → проблема → анализ → выводы",
        "structure": "narrative",
        "style_folder": "ситуации",
    },
    "law_review": {
        "label": "⚖️ Разбор законодательства",
        "description": "НПА на человеческом языке: цитата закона + объяснение",
        "structure": "law_to_human",
        "style_folder": "разбор_законов",
    },
    "reference": {
        "label": "📊 Полезное",
        "description": "Таблица/памятка — справочная информация",
        "structure": "table",
        "style_folder": "полезное",
    },
    "analysis": {
        "label": "📰 Актуальные проблемы",
        "description": "Аналитика, обзоры, экспертное мнение",
        "structure": "analytical",
        "style_folder": "актуальные_проблемы",
    },
    "custom": {
        "label": "✏️ Свободный формат",
        "description": "Кастомный тип — ручные инструкции через админку",
        "structure": "custom",
        "style_folder": None,
    },
}

# Тематические направления
DIRECTIONS = ["налоги", "юридическое", "бизнес", "финансы", "экономика"]

# Поддерживаемые форматы файлов
SUPPORTED_FORMATS = {".pdf", ".fb2", ".txt", ".docx", ".odt"}
