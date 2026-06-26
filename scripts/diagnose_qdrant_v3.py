"""
Диагностика v3: финальная проверка — есть ли ТК РФ под другим именем,
и какой score дают чанки по трудовому запросу относительно порога 0.25.

ЗАПУСК НА СЕРВЕРЕ:
    python diagnose_qdrant_v3.py
"""
import os
import sys
from collections import Counter

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "copywriter_kb"

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30, check_compatibility=False)

print(f"=== ДИАГНОСТИКА v3: поиск ТК РФ и score-профиль ===\n")

# ============================================================
# 1. ВСЕ уникальные source_file в коллекции
# ============================================================
print("=" * 60)
print("1. ВСЕ source_file в коллекции (ищем ТК РФ под любым именем)")
print("=" * 60)
file_counter = Counter()
offset = None
while True:
    try:
        result = client.scroll(
            collection_name=COLLECTION, scroll_filter=None,
            limit=500, offset=offset, with_payload=True, with_vectors=False,
        )
    except Exception as e:
        print(f"scroll err: {e}"); break
    batch = result[0]
    if not batch:
        break
    for p in batch:
        payload = p.payload or {}
        sf = str(payload.get("source_file", "<нет>"))
        file_counter[sf] += 1
    offset = result[1]
    if offset is None:
        break

print(f"Всего уникальных файлов: {len(file_counter)}\n")
# Показываем ВСЕ файлы (их обычно немного)
for sf, cnt in file_counter.most_common(60):
    low = sf.lower()
    marker = ""
    if any(k in low for k in ["труд", "trud", "labour", "tk_rf", "tk-rf", "_tk", "tk.", "кодекс"]):
        marker = "  ← проверь, не ТК ли это"
    print(f"   {cnt:>6}  {sf}{marker}")

# Целенаправленно ищем «труд»
print("\n--- Фильтр по имени: содержит 'труд'/'trud'/'tk' ---")
suspects = [(sf, cnt) for sf, cnt in file_counter.items()
            if any(k in sf.lower() for k in ["труд", "trud", "labour", "tk_rf", "tk-rf", "tk.", "_tk", "тк"])]
if suspects:
    for sf, cnt in suspects:
        print(f"   {cnt:>6}  {sf}")
else:
    print("   ❌ НЕТ файлов с 'труд/trud/tk' в имени. ТК РФ как отдельный файл не загружен.")

# ============================================================
# 2. Score-профиль трудового запроса (важно для порога 0.25)
# ============================================================
print("\n" + "=" * 60)
print("2. Score-профиль запроса 'оплата сверхурочной работы ст.152 ТК РФ'")
print("   (порог fact_finder = 0.25 — всё, что ниже, отбрасывается)")
print("=" * 60)

try:
    from copywriter_kb.loader import get_embeddings_batch
    query = "оплата сверхурочной работы статья 152 Трудового кодекса РФ суммированный учёт"
    print(f"Запрос: {query!r}\n")
    emb = get_embeddings_batch([query])
    if not emb or emb[0] is None:
        print("❌ Нет embedding")
    else:
        vec = emb[0]
        # top 15 БЕЗ фильтра и БЕЗ порога — видим весь скоринг
        res = client.query_points(
            collection_name=COLLECTION, query=vec, query_filter=None,
            limit=15, score_threshold=None, with_payload=True, with_vectors=False,
        )
        pts = res.points if hasattr(res, "points") else res
        print(f"{'score':>7} | {'источник':<30} | {'тип':<8} | текст")
        print("-" * 100)
        above = 0
        below = 0
        for r in pts:
            payload = r.payload or {}
            sf = str(payload.get("source_file", "?"))[:30]
            st = str(payload.get("source_type", "?"))[:8]
            txt = str(payload.get("text", ""))[:50].replace("\n", " ")
            flag = "✅" if r.score >= 0.25 else "❌<порог"
            print(f"{r.score:>7.3f} | {sf:<30} | {st:<8} | {flag} {txt}")
            if r.score >= 0.25:
                above += 1
            else:
                below += 1
        print(f"\nВыше порога 0.25: {above} | Ниже: {below}")
        print(">>> Если релевантного чанка ТК РФ нет даже в top-15 — значит его просто нет в базе.")
except Exception as e:
    print(f"❌ {e}")

# ============================================================
# 3. Содержит ли какой-то чанк точный текст ст. 152 ТК РФ?
# ============================================================
print("\n" + "=" * 60)
print("3. Прямой поиск текста ст. 152 ТК РФ в чанках (scroll + regex)")
print("=" * 60)
import re
patterns = [
    r"сверхурочная работа оплачивается",
    r"первые два часа.*не менее чем в полуторном",
    r"оплата сверхурочной работы",
]
found = {p: 0 for p in patterns}
sample_files = set()
offset = None
checked = 0
while checked < 71448:
    try:
        result = client.scroll(
            collection_name=COLLECTION, scroll_filter=None,
            limit=1000, offset=offset, with_payload=True, with_vectors=False,
        )
    except Exception as e:
        break
    batch = result[0]
    if not batch:
        break
    for p in batch:
        payload = p.payload or {}
        text = str(payload.get("text", "")).lower()
        for pat in patterns:
            if re.search(pat, text):
                found[pat] += 1
                sample_files.add(str(payload.get("source_file", "?")))
        checked += 1
    offset = result[1]
    if offset is None:
        break

print(f"Проверено чанков: {checked}\n")
for pat, cnt in found.items():
    print(f"   '{pat}': {cnt} совпадений")
print(f"\nФайлы, где встречается текст ст.152 ТК РФ: {sample_files if sample_files else 'НЕ НАЙДЕНО'}")
print("\n>>> Если 0 совпадений — ст. 152 ТК РФ как первоисточник ОТСУТСТВУЕТ в базе.")
print(">>> Причина галлюцинаций подтверждена: Heart работает без первоисточника.")
