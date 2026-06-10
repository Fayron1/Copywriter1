"""
Скрипт дистилляции PDF-книг → Qdrant (Бизнес-игра)

Извлекает текст из PDF, разбивает на чанки, дистиллирует
ключевые бизнес-концепты через OpenAI (gpt-4o-mini) и загружает в Qdrant.

Одна коллекция `business_game_kb` с богатыми метаданными:
  - category: marketing | finance | law_and_gov | production
  - topics: подтемы (список)
  - keywords: ключевые слова (список)

Поддерживает возобновление (resume) после обрыва.

Использование:
  python distill_books.py --mode all            # Обработать все категории (пропускает уже обработанные)
  python distill_books.py --mode marketing      # Только маркетинг
  python distill_books.py --mode finance        # Только финансы
  python distill_books.py --mode law            # Только законодательство
  python distill_books.py --mode production     # Только производство/методологии
  python distill_books.py --mode status         # Показать сколько уже загружено
  python distill_books.py --mode all --fresh    # Начать с нуля (игнорировать уже обработанные)

Требования:
  pip install pymupdf openai qdrant-client python-dotenv tqdm
"""
import os
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import uuid

# Загрузка .env
from dotenv import load_dotenv
for env_path in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

import fitz  # PyMuPDF
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    CollectionStatus, Filter, FieldCondition, MatchValue,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("distill")

# ============================================================
# Конфигурация
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6334"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Единая коллекция с метаданными категорий
COLLECTION_NAME = "business_game_kb"

# Пути к книгам (Docker: /app/books, локально: ../books)
BOOKS_BASE = Path(os.getenv("BOOKS_PATH", Path(__file__).parent.parent / "books"))

# Маппинг: режим CLI → папка с книгами → категория в Qdrant
CATEGORIES = {
    "marketing":  {"dir": "Markrting",    "category": "marketing",    "label": "📢 Маркетинг"},
    "finance":    {"dir": "Payback",      "category": "finance",      "label": "💰 Финансы"},
    "law":        {"dir": "Zakon",        "category": "law_and_gov",  "label": "⚖️ Законодательство"},
    "production": {"dir": "Methodology",  "category": "production",   "label": "🏭 Производство/Методологии"},
}

# Папка для результатов
OUTPUT_DIR = Path(os.getenv("OUTPUT_PATH", Path(__file__).parent))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Настройки чанкинга
CHUNK_SIZE = 800          # символов
CHUNK_OVERLAP = 100       # перекрытие
EMBEDDING_DIM = 1536      # text-embedding-ada-002

# Лимит API (предотвращение rate limit)
API_DELAY = 0.5           # секунд между запросами

# Модели OpenAI
DISTILL_MODEL = "gpt-4o-mini"           # для дистилляции
EMBEDDING_MODEL = "text-embedding-ada-002"  # для embeddings


# ============================================================
# OpenAI клиент
# ============================================================

openai_client = None


def get_openai():
    global openai_client
    if not openai_client:
        openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)
    return openai_client


# ============================================================
# Извлечение текста из PDF
# ============================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Извлечь весь текст из PDF через PyMuPDF"""
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        
        # Чистка
        text = re.sub(r'\n{3,}', '\n\n', text)            # Убираем лишние переносы
        text = re.sub(r'[ \t]{2,}', ' ', text)            # Убираем лишние пробелы
        text = re.sub(r'^\d+$\n?', '', text, flags=re.MULTILINE)  # Убираем номера страниц
        
        return text.strip()
    except Exception as e:
        logger.error(f"❌ Ошибка чтения {pdf_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Разбить текст на чанки с перекрытием"""
    if not text:
        return []
    
    # Разбиваем по предложениям (. ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Начинаем новый чанк с перекрытием
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Фильтруем слишком короткие чанки
    chunks = [c for c in chunks if len(c) > 50]
    
    return chunks


# ============================================================
# Промпты дистилляции по категориям
# ============================================================

DISTILL_PROMPTS = {
    "marketing": """Из следующего фрагмента текста из книги по маркетингу извлеки ключевой 
бизнес-концепт, полезный для предпринимателя в РФ 2026 года.

Фрагмент:
---
{chunk}
---

Ответь строго в JSON формате (без markdown!):
{{
  "concept": "Название концепта (2-5 слов)",
  "description": "Краткое описание (2-3 предложения): что это и как работает",
  "application": "Как предприниматель может применить это в России 2026 (1-2 предложения). Учитывай маркировку рекламы ОРД, стоимость трафика в Яндекс Директе, маркетплейсы.",
  "topics": ["подтема1", "подтема2"],
  "keywords": ["ключевое слово 1", "ключевое слово 2", "..."]
}}

Если фрагмент не содержит полезных концептов (оглавление, библиография, реклама и т.п.), 
верни: {{"skip": true}}""",

    "finance": """Из следующего фрагмента текста по финансам/экономике извлеки ключевой 
бизнес-концепт, полезный для предпринимателя в РФ 2026 года.

Фрагмент:
---
{chunk}
---

Ответь строго в JSON формате (без markdown!):
{{
  "concept": "Название концепта (2-5 слов)",
  "description": "Краткое описание (2-3 предложения): что это и как работает",
  "application": "Как предприниматель может применить это в России 2026 (1-2 предложения). Учитывай ставку ЦБ 16-20%, дорогие кредиты, юнит-экономику, ROI, кассовые разрывы.",
  "topics": ["подтема1", "подтема2"],
  "keywords": ["ключевое слово 1", "ключевое слово 2", "..."]
}}

Если фрагмент не содержит полезных концептов (оглавление, библиография, реклама и т.п.), 
верни: {{"skip": true}}""",

    "law_and_gov": """Из следующего фрагмента текста по законодательству/регулированию РФ извлеки 
ключевой юридический или нормативный концепт, важный для предпринимателя в РФ 2026 года.

Фрагмент:
---
{chunk}
---

Ответь строго в JSON формате (без markdown!):
{{
  "concept": "Название концепта (2-5 слов)",
  "description": "Краткое описание (2-3 предложения): суть нормы/закона и на кого распространяется",
  "application": "Какие риски и штрафы грозят предпринимателю при несоблюдении (1-2 предложения). Учитывай 152-ФЗ, Честный ЗНАК, ФАС, налоги, трудовой кодекс, ВЭД.",
  "topics": ["подтема1", "подтема2"],
  "keywords": ["ключевое слово 1", "ключевое слово 2", "..."]
}}

Если фрагмент не содержит полезных концептов (оглавление, библиография, реклама и т.п.), 
верни: {{"skip": true}}""",

    "production": """Из следующего фрагмента текста по бизнес-методологиям / производству / стратегии 
извлеки ключевой бизнес-концепт, полезный для предпринимателя в РФ 2026 года.

Фрагмент:
---
{chunk}
---

Ответь строго в JSON формате (без markdown!):
{{
  "concept": "Название концепта (2-5 слов)",
  "description": "Краткое описание (2-3 предложения): что это и как работает",
  "application": "Как предприниматель может применить это в России 2026 (1-2 предложения). Учитывай кадровый голод, дефицит инженеров, логистику из Азии, бережливое производство, санкции/ВЭД.",
  "topics": ["подтема1", "подтема2"],
  "keywords": ["ключевое слово 1", "ключевое слово 2", "..."]
}}

Если фрагмент не содержит полезных концептов (оглавление, библиография, реклама и т.п.), 
верни: {{"skip": true}}""",
}


# ============================================================
# Дистилляция и Embeddings через OpenAI
# ============================================================

def distill_chunk(chunk: str, category: str) -> Optional[Dict]:
    """
    Дистиллировать один чанк через OpenAI gpt-4o-mini.
    
    Returns:
        dict с концептом или None если skip
    """
    prompt = DISTILL_PROMPTS.get(category, DISTILL_PROMPTS["production"])
    
    try:
        response = get_openai().chat.completions.create(
            model=DISTILL_MODEL,
            messages=[
                {"role": "system", "content": "Ты — эксперт по извлечению бизнес-знаний для предпринимателей РФ. Отвечай ТОЛЬКО в JSON."},
                {"role": "user", "content": prompt.format(chunk=chunk[:2000])}
            ],
            temperature=0.3,
            max_tokens=600
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Убираем markdown обёртку если есть
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        
        data = json.loads(raw)
        
        if data.get("skip"):
            return None
        
        return data
        
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ JSON parse error: {e}")
        return None
    except Exception as e:
        logger.warning(f"⚠️ OpenAI distill error: {e}")
        return None


def get_embedding(text: str) -> List[float]:
    """Получить embedding через OpenAI"""
    response = get_openai().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000]
    )
    return response.data[0].embedding


# ============================================================
# Qdrant
# ============================================================

def get_qdrant() -> QdrantClient:
    """Получить клиент Qdrant"""
    return QdrantClient(
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
    )


def ensure_collection(client: QdrantClient):
    """Создать коллекцию если не существует"""
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
            distance=Distance.COSINE
        )
    )
    logger.info(f"📦 Создана коллекция '{COLLECTION_NAME}'")


def upload_points(client: QdrantClient, points: List[PointStruct]):
    """Загрузить точки в Qdrant батчами"""
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    logger.info(f"📤 Загружено {len(points)} точек в '{COLLECTION_NAME}'")


def get_processed_books(client: QdrantClient, category: str = None) -> Set[str]:
    """
    Получить список уже обработанных книг из Qdrant.
    Если category задана — только книги этой категории.
    """
    processed = set()
    try:
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count == 0:
            return processed
        
        # Фильтр по категории если указана
        scroll_filter = None
        if category:
            scroll_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )
        
        offset = None
        while True:
            results = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                scroll_filter=scroll_filter,
                with_payload=["source"],
                with_vectors=False
            )
            points, next_offset = results
            for point in points:
                source = point.payload.get("source", "")
                if source:
                    processed.add(source)
            if next_offset is None:
                break
            offset = next_offset
        
    except Exception as e:
        logger.warning(f"⚠️ Не удалось проверить Qdrant: {e}")
    
    return processed


def show_status():
    """Показать статус загруженных данных в Qdrant"""
    qdrant = get_qdrant()
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 СТАТУС QDRANT — business_game_kb")
    logger.info("=" * 60)
    
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"\n📦 {COLLECTION_NAME}: {info.points_count} точек (всего)")
    except Exception:
        logger.info(f"\n📦 {COLLECTION_NAME}: ❌ коллекция не существует")
        return
    
    # Статус по каждой категории
    for mode_key, cat_info in CATEGORIES.items():
        category = cat_info["category"]
        label = cat_info["label"]
        book_dir = BOOKS_BASE / cat_info["dir"]
        
        processed = get_processed_books(qdrant, category)
        
        # Считаем PDF в папке
        if book_dir.exists():
            pdf_files = [f.stem for f in book_dir.glob("*.pdf")]
            remaining = [f for f in pdf_files if f not in processed]
        else:
            pdf_files = []
            remaining = []
        
        logger.info(f"\n{label} (category: {category}):")
        logger.info(f"   В Qdrant: {len(processed)} книг")
        
        if processed:
            for book in sorted(processed):
                logger.info(f"   ✅ {book}")
        
        if remaining:
            logger.info(f"   ⏳ Осталось: {len(remaining)} книг")
            for book in remaining:
                logger.info(f"      ⏳ {book}")
        elif pdf_files:
            logger.info(f"   Все книги обработаны! ✅")
        else:
            logger.info(f"   ⚠️ Папка {cat_info['dir']}/ не найдена или пуста")


# ============================================================
# Основной процесс обработки
# ============================================================

def process_category(mode_key: str, fresh: bool = False):
    """Обработать книги одной категории"""
    cat_info = CATEGORIES[mode_key]
    category = cat_info["category"]
    label = cat_info["label"]
    book_dir = BOOKS_BASE / cat_info["dir"]
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{label} — обработка книг...")
    logger.info(f"   Папка: {book_dir}")
    logger.info(f"{'=' * 60}")
    
    if not book_dir.exists():
        logger.error(f"❌ Папка не найдена: {book_dir}")
        return []
    
    # Собираем только PDF файлы
    pdf_files = sorted(book_dir.glob("*.pdf"))
    logger.info(f"   Найдено {len(pdf_files)} PDF файлов")
    
    if not pdf_files:
        logger.warning(f"   ⚠️ Нет PDF файлов в {book_dir}")
        return []
    
    qdrant = get_qdrant()
    ensure_collection(qdrant)
    
    # Проверяем что уже обработано
    processed = set() if fresh else get_processed_books(qdrant, category)
    if processed:
        logger.info(f"   ✅ Уже обработано {len(processed)} книг, пропускаем их")
    
    all_concepts = []
    
    for pdf_path in pdf_files:
        # Пропускаем дубликаты (файлы с "(1)" в имени)
        if "(1)" in pdf_path.name:
            logger.info(f"   ⏭️ Пропуск дубликата: {pdf_path.name}")
            continue
        
        # Пропускаем уже обработанные
        if pdf_path.stem in processed:
            logger.info(f"   ⏭️ Уже в Qdrant: {pdf_path.stem}")
            continue
        
        logger.info(f"\n📖 {pdf_path.name}")
        
        # 1. Извлекаем текст
        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue
        
        logger.info(f"   Текст: {len(text)} символов")
        
        # 2. Чанкинг
        chunks = chunk_text(text)
        logger.info(f"   Чанки: {len(chunks)}")
        
        # 3. Дистилляция каждого чанка + embedding + загрузка
        book_concepts = []
        book_points = []
        
        for i, chunk in enumerate(tqdm(chunks, desc=f"   Дистилляция", unit="чанк")):
            concept = distill_chunk(chunk, category=category)
            
            if concept:
                concept["source_book"] = pdf_path.stem
                concept["chunk_index"] = i
                book_concepts.append(concept)
                
                # Формируем текст для embedding
                embed_text = (
                    f"{concept.get('concept', '')}. "
                    f"{concept.get('description', '')}. "
                    f"{concept.get('application', '')}"
                )
                
                try:
                    embedding = get_embedding(embed_text)
                    
                    book_points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": embed_text,
                            "concept": concept.get("concept", ""),
                            "description": concept.get("description", ""),
                            "application": concept.get("application", ""),
                            "topics": concept.get("topics", []),
                            "keywords": concept.get("keywords", []),
                            "source": pdf_path.stem,
                            "category": category,
                            "type": "business_knowledge"
                        }
                    ))
                except Exception as e:
                    logger.warning(f"   ⚠️ Embedding error: {e}")
            
            time.sleep(API_DELAY)
        
        # 4. Загружаем в Qdrant СРАЗУ ПОСЛЕ КАЖДОЙ КНИГИ (resume-friendly)
        if book_points:
            upload_points(qdrant, book_points)
        
        logger.info(f"   ✅ Извлечено {len(book_concepts)} концептов, загружено {len(book_points)} точек")
        all_concepts.extend(book_concepts)
    
    # 5. Сохраняем JSON для отладки
    output_path = OUTPUT_DIR / f"{category}_concepts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_concepts, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Сохранено {len(all_concepts)} концептов в {output_path}")
    
    return all_concepts


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="📚 Дистилляция PDF-книг → Qdrant (Бизнес-игра)")
    parser.add_argument(
        "--mode",
        choices=["marketing", "finance", "law", "production", "all", "status"],
        default="all",
        help="Что обрабатывать: marketing, finance, law, production, all или status"
    )
    parser.add_argument(
        "--qdrant-host",
        default=None,
        help="Хост Qdrant (переопределяет .env)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Начать с нуля, игнорируя уже обработанные книги"
    )
    
    args = parser.parse_args()
    
    # Переопределяем хост если указан
    global QDRANT_HOST
    if args.qdrant_host:
        QDRANT_HOST = args.qdrant_host
    
    # Режим status
    if args.mode == "status":
        show_status()
        return
    
    # Проверка ключей
    if not OPENAI_API_KEY:
        logger.error("❌ Не задан OPENAI_API_KEY! Заполни .env")
        return
    
    logger.info("=" * 60)
    logger.info("📚 Дистилляция PDF-книг → Qdrant (Бизнес-игра)")
    logger.info(f"   Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info(f"   Коллекция: {COLLECTION_NAME}")
    logger.info(f"   LLM: OpenAI {DISTILL_MODEL}")
    logger.info(f"   Embeddings: {EMBEDDING_MODEL}")
    logger.info(f"   Режим: {args.mode}")
    logger.info(f"   Resume: {'❌ начать с нуля' if args.fresh else '✅ продолжить'}")
    logger.info("=" * 60)
    
    # Определяем какие категории обрабатывать
    if args.mode == "all":
        modes_to_process = list(CATEGORIES.keys())
    else:
        modes_to_process = [args.mode]
    
    # Обрабатываем
    for mode_key in modes_to_process:
        process_category(mode_key, fresh=args.fresh)
    
    logger.info("\n✅ Дистилляция завершена!")
    show_status()


if __name__ == "__main__":
    main()
