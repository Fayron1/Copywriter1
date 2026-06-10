"""
Объединяющий модуль промптов — загружает все 9 системных промптов.
"""
from .prompts_part1 import PROMPTS_PART1
from .prompts_part2 import PROMPTS_PART2
from .prompts_part3 import PROMPTS_PART3


# Объединённый словарь всех промптов
_ALL_PROMPTS = {}
_ALL_PROMPTS.update(PROMPTS_PART1)
_ALL_PROMPTS.update(PROMPTS_PART2)
_ALL_PROMPTS.update(PROMPTS_PART3)


def get_system_prompt(agent_id: str) -> str:
    """
    Получить системный промпт агента по ID.

    Args:
        agent_id: ID агента (brain, fact_finder, scout, engineer,
                  heart, sheriff, mirror, booster, artist)

    Returns:
        Системный промпт (строка для system message)

    Raises:
        ValueError: если агент не найден
    """
    if agent_id not in _ALL_PROMPTS:
        available = ", ".join(sorted(_ALL_PROMPTS.keys()))
        raise ValueError(f"Промпт для '{agent_id}' не найден. Доступные: {available}")
    return _ALL_PROMPTS[agent_id]


def list_agents() -> list:
    """Список ID всех агентов с промптами."""
    return sorted(_ALL_PROMPTS.keys())
