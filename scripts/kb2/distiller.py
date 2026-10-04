#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Дистилляция и чистка чанков БЗ.

Чистка (бесплатно, локально): clean_for_index() — склейка переносов-дефисов,
снос колонтитулов/повторяющихся навигационных строк, нормализация пробелов.

Дистилляция (OpenAI, gpt-5.4-mini): LLM получает батч чанков, возвращает
JSON {is_junk, concept, description, application}. Эмбеддинг остаётся
локальным e5 (loader) — OpenAI только текстовую работу делает.

Запуск отдельно: python scripts/kb2/distiller.py --file X.txt
"""
from __future__ import annotations

import json
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("kb2.distiller")

# Провайдеры LLM-дистилляции. Эмбеддинги остаются локальными (e5) —
# LLM только текстовую работу делает: мусор/дистиллят.
PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai": {
        "model": "gpt-5.4-mini",
        "base_url": "",
        "key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
    },
}
DEFAULT_PROVIDER = "openai"
BATCH_SIZE = 4          # чанков в одном запросе
MAX_WORKERS = 8         # параллельных запросов
MAX_RETRIES = 3

PROMPT = """Ты — редактор базы знаний для AI-редакции бизнес-статей.
Тебе даны фрагменты текста (пронумерованы). Для КАЖДОГО верни СТРОГО JSON без markdown:
{"items": [{"n": 1, "is_junk": bool, "concept": str, "description": str, "application": str}, ...]}
- is_junk=true: навигация сайта, меню, колонтитулы, реклама, мусор OCR,
  оглавление без содержания, фрагмент без смысловой нагрузки.
- concept: главная идея одним термином (до 6 слов).
- description: суть своими словами, 1-2 предложения, конкретно, без воды.
- application: где применить в бизнес-статье, 1 предложение.
- Пиши по-русски. Пустые поля — пустой строкой. Число items = числу фрагментов."""

# ── Чистка текста (без LLM) ────────────────────────────────────────────

_PAGE_NUM_RE = re.compile(r"^\s*(?:\d{1,4}|\w)\s*$")
_HYPHEN_JOIN_RE = re.compile(r"([а-яёa-z])-\s*\n\s*([а-яёa-z])", re.I)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_LINE_SPACES_RE = re.compile(r"[ \t\xa0]{2,}")


def _looks_like_nav(line: str) -> bool:
    """Признаки навигационной строки: короткая, без точки, много Капитала."""
    s = line.strip()
    if not s or len(s) > 80:
        return False
    words = s.split()
    if len(words) > 7 or s.endswith((".", "!", "?", ":")):
        return False
    # «Главная  Услуги  Контакты» — почти каждое слово с заглавной
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= max(1, len(words) - 1)


def clean_for_index(text: str) -> str:
    """Механическая чистка: переносы-дефисы, колонтитулы, нав-повторы."""
    if not text:
        return text
    # Склейка слов, разорванных переносом
    text = _HYPHEN_JOIN_RE.sub(r"\1\2", text)
    lines = text.split("\n")
    cleaned: List[str] = []
    seen_counts: Dict[str, int] = {}
    for ln in lines:
        s = ln.strip()
        # Колонтитулы: голые номера страниц/букв
        if _PAGE_NUM_RE.match(s):
            continue
        # Навигационные строки: первым 3 вхождениям верим (реальные
        # заголовки короткие тоже), повторы — мусор
        if _looks_like_nav(s):
            seen_counts[s] = seen_counts.get(s, 0) + 1
            if seen_counts[s] > 1 or len(cleaned) > 0 and s in cleaned[:40]:
                continue
        cleaned.append(ln.rstrip())
    out = "\n".join(cleaned)
    out = _MULTI_BLANK_RE.sub("\n\n", out)
    out = _LINE_SPACES_RE.sub(" ", out)
    return out.strip()


def nav_junk_ratio(text: str) -> float:
    """Доля навигационных строк — грубый локальный фильтр мусора."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 1.0
    nav = sum(1 for l in lines if _looks_like_nav(l))
    return nav / len(lines)


# ── Дистилляция (OpenAI) ───────────────────────────────────────────────

_QUOTA_MARKERS = ("insufficient_quota", "quota", "402", "billing",
                  "insufficient", "balance")
_FAIL_LOCK = threading.Lock()
_EXHAUSTED: set = set()
_CLIENTS: Dict[str, Any] = {}
BATCH_COUNTS: Dict[str, int] = {p: 0 for p in PROVIDERS}


def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(m in s for m in _QUOTA_MARKERS)


def _get_client(provider: str = DEFAULT_PROVIDER):
    if provider in _CLIENTS:
        return _CLIENTS[provider]
    from dotenv import dotenv_values
    from openai import OpenAI
    cfg = PROVIDERS[provider]
    env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    key = env.get(cfg["key_env"], "")
    if not key:
        raise RuntimeError(f"нет ключа {cfg['key_env']} в .env для провайдера {provider}")
    kwargs = {"api_key": key}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    _CLIENTS[provider] = OpenAI(**kwargs)
    return _CLIENTS[provider]


def _provider_order(primary: str) -> List[str]:
    """Primary первым, дальше остальные живые; исчерпавшие баланс — в конец
    (на случай, если баланс вернётся) — но до них дойдёт только фейлровер."""
    rest = [p for p in PROVIDERS if p != primary and p not in _EXHAUSTED]
    if primary in _EXHAUSTED:
        return rest
    return [primary] + rest


def _call_llm(client, model: str, chunks: List[str]) -> tuple[List[Dict[str, Any]], tuple[int, int]]:
    """Один запрос. Quota-ошибки ПРОБУРАСЫВАЕТ наверх (фейлровер),
    прочие — ретраит."""
    numbered = "\n\n".join(
        f"ФРАГМЕНТ {i + 1}:\n{c[:1800]}" for i, c in enumerate(chunks))
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT + "\n\n" + numbered}],
                response_format={"type": "json_object"},
            )
            data = json.loads(r.choices[0].message.content)
            items = data.get("items", [])
            usage = (r.usage.prompt_tokens, r.usage.completion_tokens)
            # выравнивание длины
            while len(items) < len(chunks):
                items.append({"n": len(items) + 1, "is_junk": False,
                              "concept": "", "description": "", "application": ""})
            return items[:len(chunks)], usage
        except Exception as e:  # noqa: BLE001
            if _is_quota_error(e):
                raise
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"батч не удался после {MAX_RETRIES} ретраев: {last_err}")


def _run_batch(batch: List[str], primary: str) -> List[Dict[str, Any]]:
    """Батч с фейлровером: primary → при исчерпании баланса следующий
    провайдер. Пустые метки — только если ВСЕ провайдеры умерли."""
    for prov in _provider_order(primary):
        try:
            items, _usage = _call_llm(_get_client(prov), PROVIDERS[prov]["model"], batch)
            with _FAIL_LOCK:
                BATCH_COUNTS[prov] += 1
            return items
        except Exception as e:  # noqa: BLE001
            if _is_quota_error(e):
                with _FAIL_LOCK:
                    if prov not in _EXHAUSTED:
                        _EXHAUSTED.add(prov)
                        nxt = _provider_order(primary)
                        logger.warning(
                            f"💸 баланс {prov} исчерпан — батчи уходят на "
                            f"{nxt[0] if nxt else 'ничего'}")
                continue
            logger.warning(f"батч {prov}: {str(e)[:90]}")
            time.sleep(2)
    return [{"is_junk": False, "concept": "", "description": "",
             "application": ""} for _ in batch]


def distill_chunks(chunks: List[str], model: str = "",
                   client=None, provider: str = DEFAULT_PROVIDER) -> List[Dict[str, Any]]:
    """Батчевая дистилляция списка текстов. Возвращает по одному словарю
    на чанк: {is_junk, concept, description, application}.
    Фейлровер: OpenAI до исчерпания баланса, затем DeepSeek."""
    model = model or PROVIDERS[provider]["model"]
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for items in ex.map(lambda b: _run_batch(b, provider), batches):
            results.extend(items)
    return results[:len(chunks)]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        prov = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROVIDER
        out = distill_chunks(
            ["Розничная комиссия маркетплейса зависит от категории товара "
             "и модели склада: FBO, FBS или DBS. Ставка округляется до "
             "десятой доли процента и действует с даты публикации тарифа."],
            provider=prov)
        print(f"{prov}: {out[0]}")
        raise SystemExit(0)
    if len(sys.argv) > 1:
        t = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
        print("=== ДО ===")
        print(t[:400])
        print("=== ПОСЛЕ clean_for_index ===")
        print(clean_for_index(t)[:400])
        print(f"nav_junk_ratio: {nav_junk_ratio(t):.2f}")
