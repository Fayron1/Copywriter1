# Состояние дистилляции (обновлено 2026-10-05, план изменён)

## Решение: ручная дистилляция ПРИОСТАНОВЛЕНА

Остаток Ильяхова (и других craft-книг) дистиллируем LLM прямо на VPS,
когда пополнится баланс DeepSeek. Ручной проход по батчам больше не ведётся.

## Что сделано и сохранено

- Кэш дистиллятов: `scripts/kb2/.distill_cache.json` — **1602 записи**
  (Зинсер — 32/32 батчей полностью; Ильяхов — батчи 000-008, 360 чанков).
- Кэш продублирован на VPS: `/root/Copywriter1/scripts/kb2/.distill_cache.json`.
- Инструмент дочинки хэшей: `scripts/kb2/fix_and_apply_out.py`
  (матчит out-записи по 8-символьному префиксу всегда; формат out: is_junk/concept/description/application).
- Всё закоммичено в git.

## Загрузка в Qdrant (VPS, без дистилляции — идёт)

- `craft/writing` — 39 файлов, лоадер на ~16 064/24 334 чанков. **Ильяхов уже в базе: 810 чанков** (raw), Зинсер 780, Сарычева 734, sliwbl 5087, Кларк 553.
- `craft/persuasion` — отдельный лоадер (3599 чанков), лог `/root/craft_persuasion.log`.
- Цепочка nohup: `structure → editorial → seo → visual`, лог `/root/craft_chain.log`.
- Эмбеддинги локальные (multilingual-e5-base, CPU) — ~4-6 мин на батч 64.
- Qdrant на VPS требует API-ключ (grep QDRANT /root/Copywriter1/.env).

## Как запустить дистилляцию на сервере (когда будет баланс DeepSeek)

1. Проверить/вписать DeepSeek-ключ в `/root/Copywriter1/.env` (DEEPSEEK_API_KEY).
2. Запустить по папкам (кэш подхватится автоматически — distill_chunks_cached cache-first):
   ```
   cd /root/Copywriter1
   nohup venv/bin/python scripts/kb2/loader.py --folder craft/writing --distill --provider deepseek > /root/distill_writing.log 2>&1 &
   ```
   Требования: в loader.py FOLDERS у папки distill=true, флаг --distill включает LLM-дистилляцию;
   после дистилляции лоадер сам делает delete-by-source + перезаливку, эмбеддинг строится на дистилляте (concept+description+application).
3. Повторить для остальных craft-папок (persuasion, structure, editorial, seo, visual).
4. Бюджет: остаток Ильяхова по чанкам лоадера ≈ 810 − закэшированные (кэш ручных батчей — другая нарезка, совпадёт частично);
   остальные книги writing ≈ 24 334 − 810 чанков + persuasion 3599 + structure/editorial/seo/visual.

## Примечание про две нарезки fb2

Ручная очередь дистилляции нарезала Ильяхова на ~19 249 мелких чанков с перекрытием,
лоадер — на 810 крупных. Хэши считаются по нормализованному тексту, поэтому часть
ручных дистиллятов не совпадёт с чанками лоадера — LLM-дистилляция на сервере покроет остаток.
Кэш и очередь сохранены: `.distill_cache.json` (в git), `.distill_queue/` (вне git, перегенерируется).

## Если решим вернуться к ручной дистилляции

Смотри git-историю этого файла: там инструкция «как продолжать вручную» (дамп батча →
дистилляты в _out.json → fix_and_apply_out.py → apply_out_files). Очередь батчей 009-481
лежит в `scripts/kb2/.distill_queue/craft_writing_Ильяхов_*_in.json`.
