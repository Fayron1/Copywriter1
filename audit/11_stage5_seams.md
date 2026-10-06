# Аудит-2, этап 5: стыки модулей (payload, реестр, конфиг, зависимости)

**Дата:** 2026-10-05 · **Влияние на систему: ноль.**

---

## 1. Сверка контрактов payload (registry ↔ loader ↔ rag)

| Поле (registry) | Пишет loader | Индекс | Потребитель | Статус |
|---|---|---|---|---|
| `text`, `source_file`, `source_type` | да | — | все агенты | ✅ |
| `topic` (новое: fact_finder, heart, sheriff, booster) | да, `TOPICS.get(folder, "business")` | KEYWORD (создаётся) | `apply_topic_boost` | ✅ |
| `valid_until` (новое у heart) | да (year_meta/snapshot_meta) | KEYWORD | фильтр свежести rag.py (тройное should) | ✅ |
| `distilled_concept`/`distilled_application` (heart) | да (при distill) | — | heart-контекст | ✅ |
| `kb_layer` | да (fact/craft) | KEYWORD | — (на вырост) | ✅ |
| `chunk_type`, `source_priority`, `effective_date` | **нет** (только legacy `copywriter_kb/classifier.py`) | — | fact_finder читает → None | ⚠️ унаследовано, не новое |
| `chunk_type="style_anchor"` (фильтр pipeline.py:1650) | **никто не пишет** | — | style-запрос всегда пуст → fallback 1652 | ℹ️ см. отчёт 09 (S3) |

**Открытый деплой-вопрос:** payload-индексы создаются идемпотентно при каждом запуске loader'а (`_ensure_payload_indexes`), но для точек, залитых до бэкфила, `topic` появляется только после `_vps_backfill` — тема «тихого пропуска» в `apply_topic_boost` обработана (`ch.get("topic") in topics` — None пропускается).

## 2. Запросы ↔ реестр ↔ промпты

- `rewrite_query` (rag.py) вызывает DeepSeek/KIE напрямую, отдельно от `jev_client` — свой таймаут (`KB_REWRITE_TIMEOUT`, 15с), свой фейлровер, process-level latch на 401/402. Дублирование логики вызова LLM осознанное (комментарий в коде объясняет выбор модели KIE).
- Ревизор (Gemini) и фактчекер получили adversarial-протокол: карточка применимости идёт Fact-Finder → claims → claim pack → Heart (промпт prompts_part1.py:152, 180–183 запрашивает поля). Верификаторские карточки (`state.verified_facts`) в pack не возвращаются — см. 🟡 N3 (отчёт 08).
- Маркеры тем для topic-буста (rag.py: TOPIC_QUERY_MARKERS: psychology/copywriting/marketing/seo) — подмножество значений TOPICS из loader (law, visual, business, reference, reports не бустятся) — консистентно, буст только вверх.

## 3. Конфиг и зависимости

- `scripts/requirements.txt`: `markdown` добавлен (старое замечание закрыто); `qdrant-client>=1.12` покрывает `IsNullCondition`/named-векторы/Prefetch. Локально установлено всё, включая paramiko (в requirements его нет — kb_watchdog одноразовый, допустимо, но пусть будет в документации).
- `env.example` отстал от фактических переменных (см. K3, отчёт 10).
- `.env` — не читался (правило аудита), только факт наличия ключей, нужных цепочкам (TELEGRAM_*, QDRANT_API_KEY) — зафиксировано для трассировки K1.
- `config/styles_config.json` — на месте, потребляется styles-слоем; `dashboard/` — отдельный Vite-фронт, в рантайм-конвейер не входит (вне глубокого аудита, как и в прошлый раз).
- Рабочее дерево: `marketing_kb_inventory.md` имеет незакоммиченные правки (+17/−5) — пользовательский контент, аудитом не трогалось; волна 0 старого аудита (коммит незакоммиченного) актуальна по-прежнему для `_vps_*` и article29–32 артефактов.

## 4. Резюме этапа

Стык «registry ↔ loader» по новым полям закрыт полностью; единственные хвосты — legacy-поля (`chunk_type`/`source_priority`) и мёртвый style_anchor-фильтр, унаследованные до этого аудита. Новый код стыки не ломает: topic-буст тихо деградирует на старых точках, фильтр свежести включается у heart осознанно и безопасно для чанков без valid_until (тройное should).
