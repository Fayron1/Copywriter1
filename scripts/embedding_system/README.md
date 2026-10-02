# Локальная система эмбеддингов (замена OpenAI)

Эмбеддинги для базы знаний теперь считаются ЛОКАЛЬНО — без OpenAI, без
оплаты за токены, полностью офлайн после первого скачивания модели.

## Что где

| Файл | Назначение |
|---|---|
| `local_embeddings.py` | Локальная модель (sentence-transformers, multilingual-e5-base, 768d, CPU) |
| `../copywriter_kb/config.py` | Переключатель `EMBEDDING_PROVIDER` (local/openai) |
| `../copywriter_kb/loader.py` | Общий вход `get_embeddings_batch(texts, input_type)` |

Модель по умолчанию: `intfloat/multilingual-e5-base` (~1.1 ГБ, отлично для
русского). Альтернативы через env `LOCAL_EMBEDDING_MODEL`:
- `intfloat/multilingual-e5-small` — 384d, быстрее/легче;
- `intfloat/multilingual-e5-large` — 1024d, точнее (для мощного железа).

Техническая деталь: семейство E5 требует префиксы. Заливка документов
идёт с `passage: `, поисковые запросы — с `query: `. Это уже учтено:
`input_type="passage"` в конвейере заливки, `input_type="query"` в RAG.

## Установка (один раз)

```bash
cd scripts
pip install -r embedding_system/requirements.txt
```

Первый запуск скачает модель с HuggingFace в кэш (~1.1 ГБ) — дальше офлайн.

## Пересборка базы знаний с новыми данными

Размерность вектора меняется (3072 → 768), поэтому старую коллекцию
`copywriter_kb` использовать нельзя — база пересоздаётся в новую:

1. Сложить новые книги/материалы в `Books/<Папка агента>/` (маппинг папок —
   в `copywriter_kb/config.py` → AGENT_MAP).
2. В `.env` проекта добавить:
   ```
   EMBEDDING_PROVIDER=local
   KB_COLLECTION_NAME=copywriter_kb_v2
   ```
3. Пересобрать:
   ```bash
   cd scripts
   python -m copywriter_kb.main --agent all --fresh
   ```
4. Проверить: `python -m copywriter_kb.main --mode status`
5. Когда новая база проверена — переключить рантайм на неё же
   (`KB_COLLECTION_NAME` читается и RAG-агентами) и старую `copywriter_kb`
   можно удалить.

Защита от смешения встроена: если попытаться залить в коллекцию с чужой
размерностью, конвейер остановится с понятной ошибкой вместо тихо
сломанного поиска.

## Откат на OpenAI (если понадобится)

`.env`: `EMBEDDING_PROVIDER=openai` — старый платный путь вернётся без
правок кода (но понадобится новая коллекция: dim снова 3072).
