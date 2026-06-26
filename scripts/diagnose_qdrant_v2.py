"""
Диагностика v2: кросс-таблица source_type × agent_target + реальный семантический поиск.

ЗАПУСК НА СЕРВЕРЕ:
    python diagnose_qdrant_v2.py

Проверяет:
  1) Кросс-таблица: у каких чанков 'law' какой agent_target? (главный вопрос)
  2) Реальный семантический запрос по ТК РФ — что возвращает fact_finder (с фильтром)
     vs без фильтра. Видим разницу = понимаем причину.
  3) Текстовый поиск «трудовой кодекс» по source_file (а не по text) — где лежит ТК РФ.
"""
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Чтобы работали относительные импорты copywriter_kb.loader — запускаем из scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "copywriter_kb"
TODAY = datetime.now().strftime("%Y-%m-%d")

print(f"=== ДИАГНОСТИКА Qdrant v2: {COLLECTION} ===\n")


def _norm(val):
    if val is None:
        return ["<нет>"]
    if isinstance(val, (list, tuple, set)):
        items = [str(v) for v in val]
        return items if items else ["<пустой список>"]
    return [str(val)]


try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, IsNullCondition
    from qdrant_client import models
except ImportError:
    print("❌ qdrant-client не установлен")
    sys.exit(1)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30, check_compatibility=False)


# ============================================================
# 1. КРОСС-ТАБЛИЦА source_type × agent_target (по ВСЕЙ коллекции через scroll)
# ============================================================
print("=" * 60)
print("1. КРОСС-ТАБЛИЦА: source_type × agent_target (по всей коллекции)")
print("=" * 60)

cross = defaultdict(Counter)  # source_type -> Counter(agent_target)
all_source_files_by_type = defaultdict(Counter)  # source_type -> Counter(source_file)
total = 0

offset = None
while True:
    try:
        result = client.scroll(
            collection_name=COLLECTION, scroll_filter=None,
            limit=500, offset=offset, with_payload=True, with_vectors=False,
        )
    except Exception as e:
        print(f"⚠️ Ошибка scroll на offset={offset}: {e}")
        break
    batch = result[0]
    if not batch:
        break
    for p in batch:
        payload = p.payload or {}
        for st in _norm(payload.get("source_type")):
            for at in _norm(payload.get("agent_target")):
                cross[st][at] += 1
            sf = str(payload.get("source_file", "?"))
            all_source_files_by_type[st][sf] += 1
        total += 1
    offset = result[1]
    if offset is None:
        break

print(f"Просмотрено всего точек: {total}\n")

# Собираем все agent_target-ы для заголовка таблицы
all_at = sorted({at for c in cross.values() for at in c})
print(f"{'source_type':<16} | " + " | ".join(f"{a:>12}" for a in all_at) + " |   ИТОГО")
print("-" * (20 + 15 * len(all_at)))
for st, counter in sorted(cross.items()):
    row = " | ".join(f"{counter.get(a, 0):>12}" for a in all_at)
    print(f"{st:<16} | {row} |   {sum(counter.values()):>5}")

print("\n>>> ВАЖНО: смотрим строку 'law' — есть ли там столбец 'fact_finder'?")
print("    Если law есть, но fact_finder=0 → законы невидимы для fact_finder.\n")

# ============================================================
# 2. source_file для source_type='law' — где лежат законы
# ============================================================
print("=" * 60)
print("2. Какие файлы помечены source_type='law'?")
print("=" * 60)
law_files = all_source_files_by_type.get("law", Counter())
if law_files:
    for sf, cnt in law_files.most_common(30):
        marker = "  ← ПОХОЖЕ НА ТК РФ" if any(k in sf.lower() for k in ["труд", "tk", "trud", "labour", "164", "тк"]) else ""
        print(f"   {cnt:>4}  {sf}{marker}")
else:
    print("   Чанков с source_type='law' НЕ НАЙДЕНО!")
print()

# ============================================================
# 3. РЕАЛЬНЫЙ СЕМАНТИЧЕСКИЙ ПОИСК (как делает пайплайн)
# ============================================================
print("=" * 60)
print("3. Реальный семантический поиск: 'статья 152 ТК РФ оплата сверхурочных'")
print("=" * 60)

try:
    from copywriter_kb.loader import get_embeddings_batch
    query = "статья 152 ТК РФ оплата сверхурочной работы при суммированном учёте"
    print(f"Запрос: {query!r}")
    emb = get_embeddings_batch([query])
    if not emb or emb[0] is None:
        print("❌ Не удалось получить embedding (проверьте OPENAI_API_KEY)")
    else:
        vec = emb[0]

        # 3a. БЕЗ фильтра (как Scout/engineer-light)
        print("\n--- 3a. БЕЗ фильтра (top 5) ---")
        res_nofilter = client.query_points(
            collection_name=COLLECTION, query=vec, query_filter=None,
            limit=5, with_payload=True, with_vectors=False,
        )
        for r in (res_nofilter.points if hasattr(res_nofilter, "points") else res_nofilter):
            payload = r.payload or {}
            print(f"   score={r.score:.3f} | source_type={payload.get('source_type')!r} "
                  f"| agent_target={payload.get('agent_target')!r} | "
                  f"{str(payload.get('source_file','?'))[:40]}")
            print(f"        text: {str(payload.get('text',''))[:100]}...")

        # 3b. С фильтром fact_finder (как делает пайплайн)
        print("\n--- 3b. С фильтром agent_target=fact_finder (top 5) ---")
        ff_filter = Filter(must=[FieldCondition(key="agent_target", match=MatchValue(value="fact_finder"))])
        res_ff = client.query_points(
            collection_name=COLLECTION, query=vec, query_filter=ff_filter,
            limit=5, with_payload=True, with_vectors=False,
        )
        pts = res_ff.points if hasattr(res_ff, "points") else res_ff
        if not pts:
            print("   🔴 0 РЕЗУЛЬТАТОВ с фильтром fact_finder — вот почему пайплайн падает в SearXNG!")
        for r in pts:
            payload = r.payload or {}
            print(f"   score={r.score:.3f} | source_type={payload.get('source_type')!r} | "
                  f"{str(payload.get('source_file','?'))[:40]}")
            print(f"        text: {str(payload.get('text',''))[:100]}...")

        # 3c. С фильтром source_type=law (может ли вообще law найтись?)
        print("\n--- 3c. С фильтром source_type=law (top 5) ---")
        law_filter = Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="law"))])
        res_law = client.query_points(
            collection_name=COLLECTION, query=vec, query_filter=law_filter,
            limit=5, with_payload=True, with_vectors=False,
        )
        pts = res_law.points if hasattr(res_law, "points") else res_law
        if not pts:
            print("   🔴 0 результатов по source_type=law по этому запросу")
        for r in pts:
            payload = r.payload or {}
            print(f"   score={r.score:.3f} | agent_target={payload.get('agent_target')!r} | "
                  f"{str(payload.get('source_file','?'))[:40]}")
            print(f"        text: {str(payload.get('text',''))[:100]}...")

except Exception as e:
    print(f"❌ Ошибка при семантическом поиске: {e}")

print("\n" + "=" * 60)
print("Скопируйте ВЕСЬ вывод — по таблице и 3a/3b/3c я дам точное исправление.")
print("=" * 60)
