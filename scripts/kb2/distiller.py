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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("kb2.distiller")

DISTILL_MODEL = "gpt-5.4-mini"
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

def _get_client():
    from dotenv import dotenv_values
    from openai import OpenAI
    return OpenAI(api_key=dotenv_values(Path(__file__).resolve().parents[2] / ".env")["OPENAI_API_KEY"])


def _call_llm(client, model: str, chunks: List[str]) -> List[Dict[str, Any]]:
    numbered = "\n\n".join(
        f"ФРАГМЕНТ {i + 1}:\n{c[:1800]}" for i, c in enumerate(chunks))
    last_err = None
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
            last_err = e
            time.sleep(2 ** attempt)
    logger.error(f"LLM батч не удался после {MAX_RETRIES}: {last_err}")
    return [{"n": i + 1, "is_junk": False, "concept": "",
             "description": "", "application": ""} for i in range(len(chunks))], (0, 0)


def distill_chunks(chunks: List[str], model: str = DISTILL_MODEL,
                   client=None) -> List[Dict[str, Any]]:
    """Батчевая дистилляция списка текстов. Возвращает по одному словарю
    на чанк: {is_junk, concept, description, application}."""
    if client is None:
        client = _get_client()
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    results: List[Dict[str, Any]] = []

    def run(batch):
        items, usage = _call_llm(client, model, batch)
        return items, usage

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for items, _usage in ex.map(run, batches):
            results.extend(items)
    return results[:len(chunks)]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        t = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
        print("=== ДО ===")
        print(t[:400])
        print("=== ПОСЛЕ clean_for_index ===")
        print(clean_for_index(t)[:400])
        print(f"nav_junk_ratio: {nav_junk_ratio(t):.2f}")
