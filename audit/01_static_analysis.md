# Аудит Copywriter1 — Этап 1: статанализ и базовая линия

**Дата:** 2026-10-02
**Метод:** `python -m compileall` (синтаксис), `ruff check --select F,E9,B,PLE,PLW`, `bandit -r scripts/ -ll`, прогон `scripts/kb2/regression_tests.py`.
**Влияние на систему:** ноль — только чтение и запуск чистых функций.

---

## 1. Базовая линия (точка отсчёта для всех будущих изменений)

| Проверка | Результат |
|---|---|
| Синтаксис всех `.py` в `scripts/` | ✅ 0 ошибок (`compileall` exit 0) |
| Регресс-тесты линтера | ✅ **160/160 пройдено, 0 провалено** |
| Ruff (баг-фильтр) | 160 находок (см. разбор ниже) |
| Bandit (средний+ приоритет) | 15 находок (см. §5) |

Масштаб: `pipeline.py` — 5 489 строк (349 КБ); `norm_params.py` — 2 000 строк (147 КБ); всего ~16 100 строк в `scripts/agents/` + ~4 000 в `scripts/kb2/`.

**Правило аудита:** после любого изменения кода обязаны выполняться:
1. `python -m compileall -q scripts/` — синтаксис;
2. `python scripts/kb2/regression_tests.py` — 160/160.

---

## 2. 🔴 Находки с рантайм-эффектом

### 2.1. Мёртвая функция `check_expiry` в демоне VPS — `scripts/kb2/kb_update.py:405`

```python
# строка ~393: from qdrant_client import models   ← импортирован только models
flt = Filter(must=[FieldCondition(key="valid_until", range=rng)])  # ← NameError
```

- `Filter` и `FieldCondition` **нигде не импортированы**. Каждый вызов падает с `NameError`.
- Падение глотается соседним `except Exception as e: logger.warning(f"expiry check сбой: {e}")` — функция молча не работает.
- Демон `kb-update.service` на VPS запускает это ежедневно. Счётчик устаревших отчётов (`valid_until < today`) **никогда не считается**, уведомление в Telegram никогда не уходит.
- **Почему система «работает»:** исключение проглочено, основной цикл демона не задет. Классический тихо умерший код.
- **Фикс (направление):** `from qdrant_client.models import Filter, FieldCondition` рядом с импортом `models`. Требует проверки на реальном qdrant-client (у клиента даты — `DatetimeRange`).

### 2.2. Риск незакоммиченного состояния — ~2 600 строк правок

`git diff --stat HEAD`: norm_params.py **+1089**, regression_tests.py **+690**, topic_expander.py **+328**, pipeline.py **+266**, generate.py +81, rag.py +75, loader.py +37, topic_generator.py +16, registry.py +18, prompts*.py +44, tax_calculator.py +9. Плюс 12 новых файлов в `scripts/agents/` (autofix_gate, contradiction_checker, cta_renderer, deliverables, math_validator, procurement_calculator, marketplace_calculator, eaeu_registry, regulatory_events, draft_claim_extractor, html_and_entities) и новые каталоги `scripts/embedding_system/`, `scripts/kb2/` (7 новых утилит) — **не под git вообще**.

- Самое ценное состояние системы живёт только в рабочей копии и `__pycache__`.
- **Рекомендация:** зафиксировать коммитом (или веткой-снапшотом) **до** применения любых фиксов аудита. Самостоятельно ничего не коммичу — только по команде.

---

## 3. 🟡 Латентные дефекты (не рвут рантайм из-за `from __future__ import annotations`, но сломаются при рефакторинге/интроспекции)

| Место | Что | Условие проявления |
|---|---|---|
| `norm_params.py:1243` | `-> Optional[str]`, но `Optional` не импортирован (`from typing import Any, Dict, List`) | `typing.get_type_hints(norm_params)`, инструменты типов, Runtime-валидация |
| `topic_expander.py:450` | `-> List[Dict]`, `typing` не импортирован вообще | то же |
| `pipeline.py:229` | `save_path: 'Path'` — строковая аннотация, `Path` в области видимости нет | `get_type_hints()`, IDE-навигация |

Фикс тривиален (добавить импорты), но делается отдельным атомарным коммитом с прогоном 160 тестов.

---

## 4. Гигиена ruff (нижняя очередь, фиксить пачкой в конце)

| Код | Кол-во | Комментарий |
|---|---|---|
| F401 unused-import | 57 | шум, но маскирует реальные зависимости |
| F541 f-string без подстановок | 30 | косметика |
| PLW2901 loop-variable-overwritten | 20 | в основном паттерн `for line in ...: line = line.strip()` — легитимно, но проверено точечно |
| F841 unused-variable | 16 | возможные потерянные вычисления — выборочно глянуть |
| B007 unused-loop-var | 9 | косметика |
| PLW0603 global-statement | 8 | кэши клиентов (`jev_client`, `vane`, `classifier`, `loader`, `local_embeddings`) — приемлемо, не трогать без нужды |
| B005 `.strip()` с многосимвольной строкой | 3 | `factcheck.py:89,334`, `pipeline.py:2063` — **проверить семантику**: автор мог хотеть `removeprefix/removesuffix` |
| B904/B905/PLW1510 | 8 | мелочь |
| F811 redefinition | 2 | `copywriter_kb/loader.py:40` (Path), `kb_update.py:33` (subprocess) — безобидно, но почистить |

---

## 5. Bandit — безопасность (контекст: личный VPS, внутренние скрипты)

| Серьёзность | Место | Суть | Оценка |
|---|---|---|---|
| Средняя | `kb2/run_articles_17_18.py:42` | B601: paramiko выполняет строку, собранную в рантайме | Проверить источник строки (если константа + имя файла — ОК) |
| Средняя | `kb_update.py:565` | B113: `requests` **без таймаута** в демоне — зависший запрос вешает дневной цикл | Реальная операционная проблема |
| Низкая | 5 скриптов (`kb_watchdog`, `qdrant_tunnel`, `run_articles_17_18`, `stage_rebuild`, `upload_kb_corpus`) | B507: SSH `AutoAddPolicy` — host key не проверяется | Терпимо для своего VPS; MITM-риск теоретический |
| Низкая | `parsers.py:97,236` | B314: `ElementTree.fromstring` на FB2/XML | Парсятся собственные файлы БЗ; риск низкий |
| Низкая | `copywriter_kb/loader.py:189`, `kb2/loader.py:287` | B324: MD5 | Используется для ID чанков, не для безопасности — ОК |
| Низкая | `eaeu_registry.py:76`, `pipeline.py:254` | B310: urllib без ограничения схем | Внутренние URL — ОК |

---

## 6. Состав проверенных скиллов (`.agent/skills`)

Использованы: `systematic-debugging` (фазовое расследование, фиксы только после корневой причины), `lint-and-validate` (инструменты и процедуры), `multi-agent-patterns` (проверка error propagation между агентами), `context-window-management` (разбиение работы по частям). Остальные скиллы папки (`frontend-design`, `ui-designer`, `notebooklm`, `seo-content-writer`, `rag-engineer`, `prompt-engineering`, `kaizen`, `git-pushing`, `readme-master`, `concise-planning`, `prompt-caching`) к задачам аудита не относятся.

---

## 7. Что дальше

Дальше — этап 2 (`pipeline.py`: оркестрация и потоки данных), отчёт в `audit/02_pipeline.md`. Затем этапы 3–6 по плану, сводка — `audit/07_summary.md`.
