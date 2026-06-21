import os
import json
import time
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("agents.searxng")

# Поддерживается как одиночный URL (SEARXNG_URL), так и пул (SEARXNG_URLS, через запятую).
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")


def _instances() -> List[str]:
    """Список инстансов SearXNG: пул SEARXNG_URLS имеет приоритет, иначе одиночный SEARXNG_URL."""
    pool = os.getenv("SEARXNG_URLS", "").strip()
    if pool:
        urls = [u.strip().rstrip("/") for u in pool.split(",") if u.strip()]
        if urls:
            return urls
    return [SEARXNG_URL.rstrip("/")]


def _query_instance(base_url: str, query: str, num_results: int, timeout: int) -> List[Dict[str, str]]:
    """Один HTTP-запрос к конкретному инстансу SearXNG (с fallback /search → /)."""
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    url = f"{base_url}/search"
    response = requests.get(
        url,
        params={"q": query, "format": "json"},
        headers=headers,
        timeout=timeout,
    )
    if response.status_code == 404:
        logger.info("SearXNG: /search returned 404, trying fallback to root '/'")
        url = f"{base_url}/"
        response = requests.get(
            url,
            params={"q": query, "format": "json"},
            headers=headers,
            timeout=timeout,
        )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", [])[:num_results]:
        # Разные поисковики в SearXNG могут отдавать 'content' или 'snippet'
        snippet = item.get("content", "") or item.get("snippet", "")
        if snippet:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": snippet,
            })
    return results


def search_web(query: str, num_results: int = 10, timeout: int = 10,
               max_retries: int = 3) -> List[Dict[str, str]]:
    """
    Поиск через SearXNG.

    Уровень 1 (отказоустойчивость): экспоненциальный backoff на Timeout/ConnectionError
    (0.5 → 1.5 → 3 c) и ротация по пулу инстансов (SEARXNG_URLS).
    При полном отказе возвращает [] — оркестратор web_search() уйдёт в fallback.
    """
    import requests

    instances = _instances()
    last_err = None
    for base_url in instances:
        backoff = 0.5
        for attempt in range(1, max_retries + 1):
            try:
                results = _query_instance(base_url, query, num_results, timeout)
                logger.info(f"SearXNG[{base_url}]: найдено {len(results)} результатов по '{query}'")
                return results
            except requests.exceptions.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < max_retries:
                    logger.info(f"   ↻ SearXNG[{base_url}] попытка {attempt} не удалась ({last_err}); повтор через {backoff}s")
                    time.sleep(backoff)
                    backoff *= 3
        logger.warning(f"SearXNG[{base_url}]: исчерпаны попытки ({last_err}). Пробуем следующий инстанс/fallback.")

    logger.error(f"Ошибка SearXNG (все инстансы недоступны): {last_err}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Уровень 3 — Альтернативный бэкенд: Gemini Grounding (kie.ai googleSearch).
# Переиспользует те же env, что и freshness.py (KIE_API_KEY / FRESHNESS_API_BASE /
# FRESHNESS_MODEL). Дороже SearXNG, но надёжно и уже подключено в проекте.
# ─────────────────────────────────────────────────────────────────────────────
def _grounding_available() -> bool:
    return bool(os.getenv("KIE_API_KEY"))


def grounded_search(query: str, num_results: int = 10, timeout: float = 60.0) -> List[Dict[str, str]]:
    """Fallback-поиск через Gemini Grounding (Google Search) на kie.ai."""
    if not _grounding_available():
        logger.warning("Grounding fallback недоступен: KIE_API_KEY не задан.")
        return []
    try:
        import httpx
    except Exception as e:
        logger.warning(f"Grounding fallback недоступен (нет httpx): {e}")
        return []

    base = os.getenv("FRESHNESS_API_BASE", "https://api.kie.ai").rstrip("/")
    model = os.getenv("FRESHNESS_MODEL", "gemini-3.1-pro").strip()
    url = f"{base}/{model}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('KIE_API_KEY')}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    payload = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": (
                "Ты — веб-поисковик. Используя поиск Google, верни СТРОГО JSON-объект "
                "вида {\"results\": [{\"title\": \"...\", \"url\": \"...\", \"snippet\": \"...\"}]} "
                "без какого-либо иного текста."
            )}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Найди {num_results} свежих и релевантных веб-результатов по запросу: {query}"
            )}]},
        ],
        "tools": [{"type": "function", "function": {"name": "googleSearch"}}],
        "include_thoughts": False,
        "reasoning_effort": os.getenv("FRESHNESS_REASONING_EFFORT", "low").strip().lower(),
    }
    try:
        with httpx.Client(timeout=timeout) as http:
            resp = http.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            logger.warning(f"Grounding fallback: HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        content = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content")
        if isinstance(content, list):  # некоторые ответы приходят блоками
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not content:
            return []
        import re
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", content)
            data = json.loads(m.group()) if m else {}
        out = []
        for item in (data.get("results") or [])[:num_results]:
            if not isinstance(item, dict):
                continue
            snippet = item.get("snippet", "") or item.get("content", "")
            out.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": snippet,
            })
        logger.info(f"Grounding fallback: получено {len(out)} результатов по '{query}'")
        return out
    except Exception as e:
        logger.warning(f"Grounding fallback ошибка: {type(e).__name__}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Уровень 5 — Прозрачность деградации: оркестратор + health-check.
# ─────────────────────────────────────────────────────────────────────────────
def web_search(query: str, num_results: int = 10, timeout: int = 10) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Оркестратор поиска с прозрачной деградацией.

    Возвращает (results, meta), где meta = {
        "source": "searxng" | "grounding" | "none",
        "degraded": True/False   # True, если SearXNG не отработал и пришлось искать иначе/впустую
    }
    """
    results = search_web(query, num_results=num_results, timeout=timeout)
    if results:
        return results, {"source": "searxng", "degraded": False}

    # SearXNG не отработал — пробуем альтернативный бэкенд (Grounding).
    logger.warning("SearXNG не дал результатов — переключаюсь на Gemini Grounding (fallback).")
    grounded = grounded_search(query, num_results=num_results)
    if grounded:
        return grounded, {"source": "grounding", "degraded": True}

    logger.error("Поиск полностью недоступен (SearXNG и Grounding не дали результата).")
    return [], {"source": "none", "degraded": True}


def health_check(timeout: int = 5) -> bool:
    """Быстрая проверка доступности SearXNG (для preflight)."""
    try:
        res = _query_instance(_instances()[0], "тест", 1, timeout)
        return True if res is not None else False
    except Exception as e:
        logger.warning(f"SearXNG health-check не пройден: {type(e).__name__}: {e}")
        return False
