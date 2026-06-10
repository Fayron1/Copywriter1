import os
import requests
import logging
from typing import List, Dict

logger = logging.getLogger("agents.searxng")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")

def search_web(query: str, num_results: int = 10, timeout: int = 10) -> List[Dict[str, str]]:
    """
    Выполняет поиск через локальный инстанс SearXNG.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        url = f"{SEARXNG_URL}/search"
        response = requests.get(
            url,
            params={
                "q": query,
                "format": "json"
            },
            headers=headers,
            timeout=timeout
        )
        if response.status_code == 404:
            logger.info("SearXNG: /search returned 404, trying fallback to root '/'")
            url = f"{SEARXNG_URL}/"
            response = requests.get(
                url,
                params={
                    "q": query,
                    "format": "json"
                },
                headers=headers,
                timeout=timeout
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
                    "snippet": snippet
                })
            
        logger.info(f"SearXNG: найдено {len(results)} результатов по запросу '{query}'")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка SearXNG: {e}")
        return []
