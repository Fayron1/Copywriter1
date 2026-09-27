"""
Клиент Vane (Perplexica) — исследовательский поиск для агентов.

Vane — локальный движок в духе Perplexity: переформулирует запрос, ищет через
SearXNG (Яндекс + Википедия; западные движки капчуются с RU VPS IP),
реранжирует эмбеддингами и синтезирует ответ с цитатами [1][2][3].

Роль в пайплайне: Scout получает брифинг с источниками вместо сырых сниппетов;
майнер тем — обсуждения и форумы. Для строгой факт-верификации остаётся
Gemini через KIE grounding (см. searxng.py / factcheck.py).

Инфраструктура (VPS): контейнер copywriter-vane на 127.0.0.1:3006,
chat-модель DeepSeek V4 Flash через copywriter-ds-compat (json_schema→json_object).

Env: VANE_URL (по умолчанию http://127.0.0.1:3006).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("agents.vane")

DEFAULT_URL = "http://127.0.0.1:3006"
TIMEOUT = 240  # поиск + реранк + синтез — небыстро, но один вызов вместо пяти


def _base() -> str:
    return (os.getenv("VANE_URL") or DEFAULT_URL).rstrip("/")


# ============================================================
# Разрешение моделей (кэш на процесс): id провайдеров меняются
# при пересоздании config.json, имена стабильны.
# ============================================================

_models_cache: Optional[Dict[str, Any]] = None


def _resolve_models() -> Dict[str, Any]:
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    r = requests.get(f"{_base()}/api/providers", timeout=15)
    r.raise_for_status()
    providers = r.json().get("providers", [])

    chat_model, emb_model = None, None
    for p in providers:
        if p.get("name") == "DeepSeek":
            for m in p.get("chatModels", []):
                if "Flash" in m.get("name", ""):
                    chat_model = {"providerId": p["id"], "name": m["name"], "key": m.get("key", "")}
        for m in p.get("embeddingModels", []):
            if "MiniLM" in m.get("name", ""):
                emb_model = {"providerId": p["id"], "name": m["name"], "key": m.get("key", "")}

    if not chat_model or not emb_model:
        raise RuntimeError("Vane: не найдены DeepSeek chat / MiniLM embedding в реестре провайдеров")
    _models_cache = {"chat": chat_model, "embedding": emb_model}
    return _models_cache


# ============================================================
# Нормализация источников: Vane отдаёт документы в разных формах
# ============================================================

def _norm_source(s: Dict[str, Any]) -> Dict[str, str]:
    meta = s.get("metadata") or {}
    url = (s.get("url") or meta.get("url") or meta.get("source") or "").strip()
    title = (s.get("title") or meta.get("title") or "").strip()
    if not title and url:
        title = url.split("//")[-1].split("/")[0]
    snippet = (s.get("content") or s.get("pageContent") or meta.get("snippet") or "").strip()
    return {"title": title[:200], "url": url, "snippet": snippet[:400]}


def research(query: str, mode: str = "speed", timeout: int = TIMEOUT) -> Dict[str, Any]:
    """
    Исследовательский запрос: синтезированный ответ с цитатами + источники.

    Args:
        query: вопрос на естественном языке (не ключевые слова — Vane сам
               переформулирует).
        mode: 'speed' | 'balanced' | 'quality' (quality заметно дольше).

    Returns:
        {"message": str, "sources": [{"title", "url", "snippet"}]}
        При любом сбое — {"message": "", "sources": [], "error": str}.
    """
    try:
        models = _resolve_models()
        body = {
            "query": query,
            "optimizationMode": mode,
            "sources": ["web"],
            "chatModel": models["chat"],
            "embeddingModel": models["embedding"],
            "history": [],
            "stream": False,
        }
        r = requests.post(f"{_base()}/api/search", json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return {
            "message": data.get("message", ""),
            "sources": [_norm_source(s) for s in data.get("sources", []) if isinstance(s, dict)],
        }
    except Exception as e:
        logger.warning(f"Vane research сбой: {e}")
        return {"message": "", "sources": [], "error": str(e)}


def discussions(query: str, timeout: int = TIMEOUT) -> Dict[str, Any]:
    """Поиск по обсуждениям/форумам — для майнинга болей ЦА (критерии виральности)."""
    try:
        models = _resolve_models()
        body = {
            "query": query,
            "optimizationMode": "speed",
            "sources": ["reddit"],  # в Vane 'reddit' — источник обсуждений
            "chatModel": models["chat"],
            "embeddingModel": models["embedding"],
            "history": [],
            "stream": False,
        }
        r = requests.post(f"{_base()}/api/search", json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return {
            "message": data.get("message", ""),
            "sources": [_norm_source(s) for s in data.get("sources", []) if isinstance(s, dict)],
        }
    except Exception as e:
        logger.warning(f"Vane discussions сбой: {e}")
        return {"message": "", "sources": [], "error": str(e)}


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Vane research client")
    ap.add_argument("query")
    ap.add_argument("--mode", default="speed", choices=["speed", "balanced", "quality"])
    args = ap.parse_args()
    result = research(args.query, mode=args.mode)
    print(result["message"][:1200])
    print("\n--- ИСТОЧНИКИ ---")
    for s in result["sources"]:
        print(f"  {s['title'][:70]}\n    {s['url']}")
    if result.get("error"):
        print("ERROR:", result["error"])
