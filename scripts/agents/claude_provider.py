# -*- coding: utf-8 -*-
"""Claude Sonnet 5.5 + GPT-6 Sol через KIE — провайдер с ретраями.

Клод нестабилен на KIE (~30-50% сбоев), поэтому каждый вызов ретраится
3 раза, при полном фейле — fallback на gpt-6-sol через codex/responses.

Формат Claude (Anthropic):
  POST https://api.kie.ai/claude/v1/messages
  Headers: X-Api-Key, anthropic-version, Content-Type
  Body: {"model": "claude-sonnet-5-5", "messages": [...], "max_tokens": N, "stream": false}
  Response: {"content": [{"type": "text", "text": "..."}], "usage": {...}}

Формат GPT-6 Sol (OpenAI codex/responses):
  POST https://api.kie.ai/codex/v1/responses
  Headers: Authorization: Bearer KIE_API_KEY
  Body: {"model": "gpt-6-sol", "input": [...], "reasoning": {"effort": "high"}, "stream": false}
  Response: {"output": [{"type": "message", "content": [{"type": "output_text", "text": "..."}]}]}
"""
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger("agents.claude_provider")

_CLAUDE_MODEL = "claude-sonnet-5-5"
_GPT6SOL_MODEL = "gpt-6-sol"
_KIE_HOST = os.getenv("KIE_API_HOST", "https://api.kie.ai").rstrip("/")
_KIE_KEY = os.getenv("KIE_API_KEY", "")

_MAX_RETRIES = 5          # попыток Claude на один вызов (KIE нестабилен)
_RETRY_DELAYS = [2, 5, 10, 15, 20]
_FALLBACK_RETRIES = 3     # попыток gpt-6-sol после фейла Claude
_TIMEOUT = 300            # Engineer может думать долго


def _extract_claude_text(data: dict) -> str:
    """Извлечь текст из ответа Claude API."""
    content = data.get("content", [])
    for block in content:
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def _extract_gpt6sol_text(data: dict) -> str:
    """Извлечь текст из ответа codex/responses API."""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text", "")
    return ""


def call_claude_with_fallback(system: str, user: str,
                              max_tokens: int = 16000) -> Optional[str]:
    """Вызвать Claude Sonnet 5.5 с ретраями; при фейле — GPT-6 Sol.

    Возвращает текст или None при полном фейле обоих провайдеров.
    """
    # Phase 1: Claude Sonnet 5.5 (5 попыток с диагностикой)
    result = _try_claude(system, user, max_tokens)
    if result is not None:
        return result

    # Phase 2: GPT-6 Sol fallback (3 попытки)
    logger.warning("  🔄 Claude недоступен после %d попыток — fallback на gpt-6-sol (input: sys=%d, user=%d зн.)",
                   _MAX_RETRIES, len(system), len(user))
    result = _try_gpt6sol(system, user, max_tokens)
    if result is not None:
        return result

    logger.error("  ❌ Оба провайдера недоступны (claude + gpt-6-sol); input: sys=%d, user=%d зн.",
                 len(system), len(user))
    return None


def _try_claude(system: str, user: str, max_tokens: int) -> Optional[str]:
    """Claude Sonnet 5.5 через KIE /claude/v1/messages."""
    url = f"{_KIE_HOST}/claude/v1/messages"
    # KIE требует Authorization: Bearer, НЕ X-Api-Key (кейс статьи 40:
    # X-Api-Key возвращал 401 при 200 статусе)
    headers = {
        "Authorization": f"Bearer {_KIE_KEY}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": _CLAUDE_MODEL,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    logger.info(f"  📤 Claude запрос: sys={len(system):,} user={len(user):,} max_tokens={max_tokens}")

    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.time()
            r = requests.post(url, json=body, headers=headers, timeout=_TIMEOUT)
            elapsed = time.time() - t0
            logger.info(f"  📥 Claude попытка {attempt+1}: статус={r.status_code}, "
                        f"len={len(r.text):,}, время={elapsed:.1f}s")

            if r.status_code in (429, 502, 503):
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                logger.warning(f"  ⏳ Claude {r.status_code}; повтор {attempt + 1}/{_MAX_RETRIES} через {delay}s")
                time.sleep(delay)
                continue
            r.raise_for_status()
            data = r.json()
            # Логируем stop_reason для диагностики обрывов
            stop = data.get("stop_reason", "?")
            text = _extract_claude_text(data)
            logger.info(f"  📥 Claude: stop_reason={stop}, text_len={len(text)}")
            if text:
                return text
            logger.warning(f"  ⚠️ Claude вернул пустой content (попытка {attempt + 1}, "
                           f"stop_reason={stop})")
            time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
        except requests.exceptions.Timeout:
            logger.warning(f"  ⏳ Claude таймаут {_TIMEOUT}s (попытка {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
        except requests.exceptions.RequestException as e:
            logger.warning(f"  ⚠️ Claude ошибка (попытка {attempt + 1}): {e}")
            time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
    return None


def _try_gpt6sol(system: str, user: str, max_tokens: int) -> Optional[str]:
    """GPT-6 Sol через KIE /codex/v1/responses (fallback)."""
    url = f"{_KIE_HOST}/codex/v1/responses"
    headers = {"Authorization": f"Bearer {_KIE_KEY}",
               "Content-Type": "application/json"}
    body = {
        "model": _GPT6SOL_MODEL,
        "input": [
            {"role": "system",
             "content": [{"type": "input_text", "text": system}]},
            {"role": "user",
             "content": [{"type": "input_text", "text": user}]},
        ],
        "reasoning": {"effort": "high"},
        "stream": False,
    }

    for attempt in range(_FALLBACK_RETRIES):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            text = _extract_gpt6sol_text(data)
            if text:
                return text
            logger.warning(f"  ⚠️ GPT-6 Sol пустой content (попытка {attempt + 1})")
            time.sleep(3)
        except requests.exceptions.RequestException as e:
            logger.warning(f"  ⚠️ GPT-6 Sol ошибка (попытка {attempt + 1}): {e}")
            time.sleep(3)
    return None
