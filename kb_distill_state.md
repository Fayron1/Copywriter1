# Состояние дистилляции (обновлено 2026-10-06)

## Статус: ЗАПРЕЩЕНА до пополнения DeepSeek

Дистилляция craft-книг выполняется LLM на сервере после пополнения баланса.
Все промпты и инструменты готовы, кэш загружен на VPS.

## Кэш дистиллятов
- `scripts/kb2/.distill_cache.json` — 1602 записи (Зинсер 32/32, Ильяхов 000-008)
- Скопирован на VPS: `/root/Copywriter1/scripts/kb2/.distill_cache.json`
- distill_chunks_cached cache-first: что есть в кэше — бесплатно, остальное — LLM

## Как запустить (когда баланс пополнен)
```bash
cd /root/Copywriter1
nohup venv/bin/python scripts/kb2/loader.py --folder craft/writing --distill --provider deepseek > /root/distill_writing.log 2>&1 &
# повторить для: craft/persuasion, craft/structure, craft/editorial, craft/seo, craft/visual
# и business/marketing/consumer_psychology, business/marketing/ai_for_business
```

## База: 113 536+ точек, 373/378 файлов загружено
- 6 сканов без текстового слоя: Шварц, Клеон, 3 закупочных, гайд юнит-экономики
- 126 файлов дозалиты на VPS (окт 2026), 4 эталона добавлены
