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


# ── Кэш дистиллятов (для ручной дистилляции агентом без API) ──────────

CACHE_PATH = Path(__file__).resolve().parent / ".distill_cache.json"
QUEUE_DIR = Path(__file__).resolve().parent / ".distill_queue"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def extract_queue(source_rel: str, text: str, batch_size: int = 40) -> list:
    """Нарезать чанки файла и сохранить очереди по батчам.
    Возвращает список путей файлов-очередей (_in.json)."""
    import hashlib
    sys_path = Path(__file__).resolve().parent
    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    from loader import chunk_text, clean_for_index, CHUNK_SIZES  # noqa: E402
    clean = clean_for_index(text)
    chunks = chunk_text(clean, "book", CHUNK_SIZES["book"])
    slug = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", "_", source_rel)[:40].strip("_")
    QUEUE_DIR.mkdir(exist_ok=True)
    outs = []
    for b in range(0, len(chunks), batch_size):
        items = []
        for ch in chunks[b:b + batch_size]:
            h = hashlib.md5(re.sub(r"[\s\W]+", "", ch["text"].lower())
                            .encode("utf-8")).hexdigest()
            items.append({"hash": h, "text": ch["text"]})
        path = QUEUE_DIR / f"{slug}_{b // batch_size:03d}_in.json"
        path.write_text(json.dumps(items, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        outs.append(str(path))
    print(f"очередь: {len(chunks)} чанков -> {len(outs)} батчей ({slug})")
    return outs


def apply_out_files() -> int:
    """Слить *_out.json (ответы агента) в кэш. Возвращает число новых."""
    cache = load_cache()
    added = 0
    for f in sorted(QUEUE_DIR.glob("*_out.json")):
        try:
            items = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"пропуск {f.name}: {e}")
            continue
        for it in items:
            h = it.get("hash")
            if h and h not in cache:
                cache[h] = {k: it.get(k, "") for k in
                            ("is_junk", "concept", "description", "application")}
                added += 1
        f.unlink()
    save_cache(cache)
    print(f"кэш: +{added} записей, всего {len(cache)}")
    return added


def distill_chunks_cached(chunks: list, provider: str = DEFAULT_PROVIDER) -> list:
    """distill_chunks с кэшем: что есть в кэше — берём, остальное — LLM.
    chunks — список СТРОК (текстов чанков), не словарей."""
    import hashlib
    cache = load_cache()
    hashes = [hashlib.md5(re.sub(r"[\s\W]+", "", c.lower())
                          .encode("utf-8")).hexdigest() for c in chunks]
    marks = [cache.get(h) for h in hashes]
    missing_idx = [i for i, m in enumerate(marks) if m is None]
    if missing_idx:
        llm = distill_chunks([chunks[i] for i in missing_idx],
                             provider=provider)
        for i, m in zip(missing_idx, llm):
            marks[i] = m
            cache[hashes[i]] = {k: m.get(k, "") for k in
                                ("is_junk", "concept", "description",
                                 "application")}
        save_cache(cache)
    return [m or {"is_junk": False, "concept": "", "description": "",
                  "application": ""} for m in marks]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        # --extract <файл> [исходный_относительный_путь]
        import fitz
        fp = Path(sys.argv[2])
        rel = sys.argv[3] if len(sys.argv) > 3 else fp.name
        if fp.suffix.lower() == ".pdf":
            d = fitz.open(fp)
            text = "".join(pg.get_text() for pg in d)
            d.close()
        elif fp.suffix.lower() == ".fb2":
            import re as _re
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            text = _re.sub(r"<[^>]+>", " ", raw)
        else:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        extract_queue(rel, text)
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--apply":
        apply_out_files()
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--cache-stats":
        c = load_cache()
        junk = sum(1 for v in c.values() if v.get("is_junk"))
        print(f"кэш: {len(c)} записей, из них мусор {junk}")
        raise SystemExit(0)
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
