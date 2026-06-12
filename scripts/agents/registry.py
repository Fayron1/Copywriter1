"""
Реестр агентов — параметры моделей, RAG-конфиги, I/O схемы.

Каждый агент описан как словарь с полями:
- id: уникальный идентификатор
- name: английское имя
- label: отображаемое имя с эмодзи
- model: модель GPT для вызова
- temperature: креативность (0.0 = строго, 1.0 = свободно)
- max_tokens: лимит ответа
- rag: конфигурация запросов к Qdrant
- input_from: от кого получает данные
- output_to: кому передаёт результат
- retry_on_fail: сколько раз повторить при неудаче
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================================
# Конфигурация RAG-запросов к Qdrant
# ============================================================

@dataclass
class RagConfig:
    """Настройки запросов агента к Qdrant."""
    collection: str = "copywriter_kb"
    top_k: int = 10
    score_threshold: float = 0.3
    filters: Dict[str, Any] = field(default_factory=dict)
    payload_fields: List[str] = field(default_factory=list)
    enabled: bool = True


# ============================================================
# Конфигурация агента
# ============================================================

@dataclass
class AgentConfig:
    """Полная конфигурация одного агента."""
    id: str
    name: str
    label: str
    model: str
    temperature: float
    max_tokens: int
    rag: RagConfig
    input_from: List[str]
    output_to: List[str]
    retry_on_fail: int = 2
    description: str = ""


# ============================================================
# Реестр: 9 агентов
# ============================================================

# ============================================================
# Единый реестр ID моделей — ЕДИНСТВЕННЫЙ источник правды.
# Значения переопределяются через окружение (.env) без правки кода.
# Дефолты = текущие значения (поведение не меняется); правильность
# проверяется preflight-ом на старте Pipeline.run().
# ============================================================

MODELS: Dict[str, str] = {
    # DeepSeek — основной провайдер генерации текста
    "deepseek_pro":   os.getenv("MODEL_DEEPSEEK_PRO",   "deepseek-v4-pro"),
    "deepseek_flash": os.getenv("MODEL_DEEPSEEK_FLASH", "deepseek-v4-flash"),
    # OpenAI — fallback при provider="openai"
    "openai_text":    os.getenv("MODEL_OPENAI_TEXT",    "gpt-4o"),
    # KIE — fallback при provider="kie"
    "kie_text":       os.getenv("MODEL_KIE_TEXT",       "claude-4.7"),
    # Генерация изображений (через OpenAI-совместимый эндпойнт)
    "openai_image_primary":  os.getenv("MODEL_OPENAI_IMAGE_PRIMARY",  "gpt-image-2"),
    "openai_image_fallback": os.getenv("MODEL_OPENAI_IMAGE_FALLBACK", "dall-e-3"),
}


def get_text_model_ids() -> set:
    """Уникальные ID текстовых моделей, реально используемых агентами."""
    return {a.model for a in AGENTS.values()}


AGENTS: Dict[str, AgentConfig] = {

    # ───────────────────────────────────────────────
    # 1. Оркестратор (The Brain)
    # ───────────────────────────────────────────────
    "brain": AgentConfig(
        id="brain",
        name="The Brain",
        label="🧠 Оркестратор",
        model=MODELS["deepseek_pro"],
        temperature=0.2,       # Строгая логика, минимум креатива
        max_tokens=4000,
        description="Главный планировщик. Декомпозиция задач, маршрутизация, контроль качества.",
        input_from=["admin"],
        output_to=["fact_finder", "scout", "engineer"],
        rag=RagConfig(
            collection="copywriter_kb",
            top_k=5,
            filters={"source_type": ["workflow", "reference"]},
            payload_fields=["text", "source_type", "chunk_type"],
        ),
    ),

    # ───────────────────────────────────────────────
    # 2. Исследователь (The Fact-Finder)
    # ───────────────────────────────────────────────
    "fact_finder": AgentConfig(
        id="fact_finder",
        name="The Fact-Finder",
        label="🔎 Исследователь",
        model=MODELS["deepseek_flash"],
        temperature=0.1,       # Максимальная точность
        max_tokens=6000,
        description="Хранитель истины. Семантический поиск фактов, законов, цитат.",
        input_from=["brain"],
        output_to=["engineer", "scout"],
        rag=RagConfig(
            collection="copywriter_kb",
            top_k=20,           # Много результатов — нужна полнота
            score_threshold=0.25,
            filters={"agent_target": "fact_finder"},
            payload_fields=[
                "text", "source_file", "source_type",
                "source_priority", "chunk_type",
                "effective_date", "valid_until",
            ],
        ),
    ),

    # ───────────────────────────────────────────────
    # 3. Разведчик (The Scout)
    # ───────────────────────────────────────────────
    "scout": AgentConfig(
        id="scout",
        name="The Scout",
        label="📡 Разведчик",
        model=MODELS["deepseek_flash"],
        temperature=0.4,       # Немного креатива для «угла подачи»
        max_tokens=3000,
        description="Разведчик повестки. Тренды, SERP, newsjacking, конкуренты.",
        input_from=["brain", "fact_finder"],
        output_to=["engineer", "booster"],
        rag=RagConfig(
            collection="copywriter_kb",
            top_k=5,
            filters={"agent_target": "booster"},
            payload_fields=["text", "source_type"],
            enabled=False,      # Основной источник — SearXNG, не Qdrant
        ),
    ),

    # ───────────────────────────────────────────────
    # 4. Структурировщик (The Engineer)
    # ───────────────────────────────────────────────
    "engineer": AgentConfig(
        id="engineer",
        name="The Engineer",
        label="🏗️ Структурировщик",
        model=MODELS["deepseek_pro"],
        temperature=0.2,
        max_tokens=5000,
        description="Архитектор логики. Каркас статьи, фреймворки, модульная структура.",
        input_from=["fact_finder", "scout", "brain"],
        output_to=["heart"],
        rag=RagConfig(
            collection="copywriter_kb",
            top_k=10,
            filters={"agent_target": "engineer"},
            payload_fields=[
                "text", "source_type", "chunk_type",
                "distilled_concept", "distilled_application",
            ],
        ),
    ),

    # ───────────────────────────────────────────────
    # 5. Писатель (The Heart)
    # ───────────────────────────────────────────────
    "heart": AgentConfig(
        id="heart",
        name="The Heart",
        label="✍️ Писатель",
        model=MODELS["deepseek_pro"],
        temperature=0.65,       # Стабильный ритм, достаточная вариативность
        max_tokens=16000,       # Длинные статьи (30k символов = ~16k токенов)
        description="Мастер слога. Текст, стиль, голос эксперта, ритм.",
        input_from=["engineer"],
        output_to=["sheriff"],
        retry_on_fail=3,        # 3 попытки по фидбеку Sheriff
        rag=RagConfig(
            collection="copywriter_kb",
            top_k=10,
            filters={"agent_target": "heart"},
            payload_fields=[
                "text", "source_type", "chunk_type",
                "rhythm_type",
            ],
        ),
    ),

    # ───────────────────────────────────────────────
    # 6. Редактор (The Sheriff)
    # ───────────────────────────────────────────────
    "sheriff": AgentConfig(
        id="sheriff",
        name="The Sheriff",
        label="🔫 Редактор",
        model=MODELS["deepseek_pro"],
        temperature=0.1,        # Минимум креатива — строгие правила
        max_tokens=6000,
        description="Безжалостный цензор. Факт-чекинг, стиль, anti-синтетика, Turing Score.",
        input_from=["heart", "fact_finder"],
        output_to=["heart", "mirror", "booster"],
        rag=RagConfig(
            collection="copywriter_kb",
            top_k=10,
            filters={"agent_target": "sheriff"},
            payload_fields=[
                "text", "source_type", "error_type",
                "chunk_type",
            ],
        ),
    ),

    # ───────────────────────────────────────────────
    # 7. Зеркало (The Mirror) — НОВЫЙ
    # ───────────────────────────────────────────────
    "mirror": AgentConfig(
        id="mirror",
        name="The Mirror",
        label="🪞 Зеркало",
        model=MODELS["deepseek_flash"],
        temperature=0.1,        # Строгий алгоритм
        max_tokens=4000,
        description="Anti-AI контроль. Perplexity, burstiness, humanization.",
        input_from=["sheriff"],
        output_to=["heart", "booster"],
        retry_on_fail=2,        # 2 итерации humanization
        rag=RagConfig(enabled=False),  # Не использует RAG
    ),

    # ───────────────────────────────────────────────
    # 8. SEO/GEO Специалист (The Booster)
    # ───────────────────────────────────────────────
    "booster": AgentConfig(
        id="booster",
        name="The Booster",
        label="🚀 SEO/GEO",
        model=MODELS["deepseek_flash"],
        temperature=0.3,
        max_tokens=8000,
        description="Хакер алгоритмов. Schema.org, E-E-A-T, Citation Bait, GEO.",
        input_from=["mirror", "scout"],
        output_to=["artist", "publisher"],
        rag=RagConfig(
            collection="copywriter_kb",
            top_k=10,
            filters={"agent_target": "booster"},
            payload_fields=[
                "text", "source_type", "chunk_type",
                "intent_type",
            ],
        ),
    ),

    # ───────────────────────────────────────────────
    # 9. Визуализатор (The Artist)
    # ───────────────────────────────────────────────
    "artist": AgentConfig(
        id="artist",
        name="The Artist",
        label="🎨 Визуализатор",
        model=MODELS["deepseek_flash"],
        temperature=0.5,
        max_tokens=3000,
        description="Арт-директор. Промпты для GPT Image 2.0, брендбук, инфографика.",
        input_from=["booster", "engineer"],
        output_to=["publisher"],
        rag=RagConfig(
            collection="copywriter_kb",
            top_k=5,
            filters={"agent_target": "artist"},
            payload_fields=[
                "text", "source_type", "image_style",
            ],
        ),
    ),
}


def get_agent(agent_id: str) -> AgentConfig:
    """Получить конфигурацию агента по ID."""
    if agent_id not in AGENTS:
        raise ValueError(f"Агент '{agent_id}' не найден. Доступные: {list(AGENTS.keys())}")
    return AGENTS[agent_id]
