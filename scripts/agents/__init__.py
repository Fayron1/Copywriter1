"""
Модуль агентов — ИИ-Редакция Копирайтер.

Содержит:
- registry.py — реестр агентов (параметры, RAG-конфиг, I/O)
- prompts.py — production-ready системные промпты
- rag.py — RAG-запросы к Qdrant
- pipeline.py — оркестратор мультиагентной генерации
"""
from .registry import AGENTS, get_agent
from .prompts import get_system_prompt, list_agents
from .pipeline import Pipeline, PipelineState
