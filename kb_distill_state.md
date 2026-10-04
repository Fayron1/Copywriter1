# Состояние дистилляции БЗ — чекпоинт (2026-10-04, 16:30)

## Стоп-точка
Оба баланса (OpenAI $10 и DeepSeek) исчерпаны в ходе прогона 2026-10-04.
Дистилляция остановлена СОЗНАТЕЛЬНО до пополнения DeepSeek.

## Что готово (дистиллят в payload)
- business/marketing (книги) — 8 648 точек ✅
- craft/persuasion (Барден, Чалдини, Nudge, Scarcity) — 3 373 ✅
- business/methodology (Портер, Голдратт, ТРИЗ…) — 18 763 ✅
- business/marketplaces — 1 968 ✅ (LLM-отбраковка)
- business/marketing/ai_for_business — 19 ✅
- craft/persuasion: barden_playbook выведен в style_client (LLM считала его шаблоном)

## Что осталось (когда пополнится DeepSeek)
1. **craft/writing — дистиллят (~24 309 точек, сейчас сырые очищенные тексты).**
   Прогон: `python scripts/kb2/loader.py --folder craft/writing --distill --provider deepseek`
   Оценка: ~$5–6, время ~2–3 ч.
2. **ДОБАВИТЬ ЧЕКПОИНТ** перед запуском: distiller не имеет резюма — при обрыве
   теряются батчи (уже потеряли ~$5 дважды). План: сохранение готовых меток
   по content_hash в kb2/.distill_state.json + пропуск уже посчитанных.
3. Верификация после прогона: scroll-сэмпл craft/writing → distilled_concept
   заполнен в 25/25.

## Чего НЕ делать при возобновлении
- Не трогать legislation/reference/легальные снимки (law-парсер без дистилляции — так и надо).
- Книги-сканы (Дуарте ×2, Труби, Снайдер, Клеон, Хорн) — без текстового слоя,
  не грузить; вернуть в оборот только текстовые версии (fb2/epub) или OCR.

## Расход 2026-10-04
- OpenAI: $10 — marketing + persuasion + часть purge.
- DeepSeek: ~$9–10 — methodology + брошенный writing-прогон + хвост purge.
