"""
Диагностика: почему Qdrant возвращает 0 чанков по запросам ТК РФ.

ЗАПУСК НА СЕРВЕРЕ (где есть qdrant-client и .env):
    py scripts/diagnose_qdrant.py

Скрипт НЕ меняет данные — только читает и печатает диагноз.
Проверяет 3 гипотезы:
  1) Фильтр agent_target: есть ли в payload чанков поле agent_target="fact_finder"?
  2) Фильтр valid_until: не истёк ли срок (формат даты, пустые значения)?
  3) Фильтр source_type: какие source_type есть в коллекции?
"""
import os
import sys
from datetime import datetime
from collections import Counter

# Загружаем .env если есть
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "copywriter_kb"
TODAY = datetime.now().strftime("%Y-%m-%d")

print(f"=== ДИАГНОСТИКА Qdrant: {COLLECTION} ===")
print(f"URL: {QDRANT_URL}")
print(f"Сегодня: {TODAY}\n")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest
except ImportError:
    print("❌ qdrant-client не установлен. Установите: pip install qdrant-client")
    sys.exit(1)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

# --- 0. Проверка соединения и коллекции ---
try:
    cols = [c.name for c in client.get_collections().collections]
    print(f"Коллекции: {cols}")
    if COLLECTION not in cols:
        print(f"❌ Коллекция {COLLECTION} не найдена!")
        sys.exit(1)
    info = client.get_collection(COLLECTION)
    print(f"Точек в {COLLECTION}: {info.points_count}\n")
except Exception as e:
    print(f"❌ Не удалось подключиться/получить коллекцию: {e}")
    sys.exit(1)

# --- Скроллим выборку чанков (до 2000) для анализа payload ---
print("=== Анализ payload чанков ===")
all_points = []
offset = None
while len(all_points) < 2000:
    try:
        result = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=None,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        print(f"⚠️ Ошибка scroll: {e}")
        break
    batch = result[0]
    all_points.extend(batch)
    offset = result[1]
    if offset is None or not batch:
        break

print(f"Проскроллено чанков: {len(all_points)}\n")

if not all_points:
    print("❌ Коллекция пуста! Законы не загружены.")
    sys.exit(0)

# --- 1. ГИПОТЕЗА agent_target ---
print("--- ГИПОТЕЗА 1: фильтр agent_target ---")


def _normalize_field(val):
    """Свести строку/список к набору хешируемых строковых ключей."""
    if val is None:
        return ["<ОТСУТСТВУЕТ>"]
    if isinstance(val, (list, tuple, set)):
        items = [str(v) for v in val]
        return items if items else ["<ПУСТОЙ СПИСОК>"]
    return [str(val)]


at_counter = Counter()
for p in all_points:
    payload = p.payload or {}
    for key in _normalize_field(payload.get("agent_target")):
        at_counter[key] += 1
print(f"Распределение agent_target:")
for k, v in at_counter.most_common():
    marker = "  ← fact_finder ищет это" if k == "fact_finder" else ""
    print(f"   {k!r}: {v}{marker}")
print()

# --- 2. ГИПОТЕЗА source_type ---
print("--- ГИПОТЕЗА 2: фильтр source_type ---")
st_counter = Counter()
for p in all_points:
    payload = p.payload or {}
    for key in _normalize_field(payload.get("source_type")):
        st_counter[key] += 1
print(f"Распределение source_type:")
for k, v in st_counter.most_common():
    print(f"   {k!r}: {v}")
print()

# --- 3. ГИПОТЕЗА valid_until ---
print("--- ГИПОТЕЗА 3: фильтр актуальности valid_until ---")
vu_states = {"отсутствует/null": 0, "действует (>= сегодня)": 0, "ИСТЁК (< сегодня)": 0, "нечёткий формат": 0}
expired_examples = []
for p in all_points:
    payload = p.payload or {}
    vu = payload.get("valid_until")
    if vu is None:
        vu_states["отсутствует/null"] += 1
        continue
    # valid_until может быть списком или datetime-объектом — берём первое/строку
    if isinstance(vu, (list, tuple)):
        vu = vu[0] if vu else None
        if vu is None:
            vu_states["отсутствует/null"] += 1
            continue
    # datetime-объект или строка
    try:
        vu_str = vu.strftime("%Y-%m-%d") if hasattr(vu, "strftime") else str(vu)[:10]
    except Exception:
        vu_str = str(vu)[:10]
    try:
        if vu_str >= TODAY:
            vu_states["действует (>= сегодня)"] += 1
        else:
            vu_states["ИСТЁК (< сегодня)"] += 1
            if len(expired_examples) < 5:
                expired_examples.append((vu_str, payload.get("source_file", "?")[:50]))
    except Exception:
        vu_states["нечёткий формат"] += 1

for k, v in vu_states.items():
    marker = "  ← ОПАСНО: эти чанки отсеиваются!" if "ИСТЁК" in k else ""
    print(f"   {k}: {v}{marker}")
if expired_examples:
    print(f"\n   Примеры истёкших (source_file → valid_until):")
    for vu, sf in expired_examples:
        print(f"     {sf}: {vu}")
print()

# --- 4. ИЩЕМ ИМЕННО ТК РФ ---
print("--- ПРОВЕРКА: есть ли вообще ТК РФ в коллекции? ---")
tk_count = 0
tk_examples = []
for p in all_points:
    payload = p.payload or {}
    text = str(payload.get("text", "")).lower()
    sf = str(payload.get("source_file", "")).lower()
    if any(k in text or k in sf for k in ["трудовой код", "тк рф", "trudovoy", "tk_rf", "labour_code"]):
        tk_count += 1
        if len(tk_examples) < 5:
            tk_examples.append({
                "id": p.id,
                "source_file": payload.get("source_file", "?"),
                "agent_target": payload.get("agent_target", "<ОТСУТСТВУЕТ>"),
                "valid_until": payload.get("valid_until", "<ОТСУТСТВУЕТ>"),
                "text_preview": str(payload.get("text", ""))[:80],
            })

print(f"Чанков, похожих на ТК РФ: {tk_count}")
if tk_examples:
    print(f"Примеры первых чанков ТК РФ:")
    for ex in tk_examples:
        print(f"\n   ID: {ex['id']}")
        print(f"   source_file: {ex['source_file']}")
        print(f"   agent_target: {ex['agent_target']!r}   ← должно быть 'fact_finder'")
        print(f"   valid_until: {ex['valid_until']!r}   ← должно быть >= {TODAY} или отсутствовать")
        print(f"   text: {ex['text_preview']}...")
else:
    print("❌ ТК РФ НЕ НАЙДЕН в коллекции вообще! Возможно, загружен в другую коллекцию.")
print()

# --- 5. ВЫВОД ДИАГНОЗА ---
print("=" * 60)
print("=== ИТОГОВЫЙ ДИАГНОЗ ===")
print("=" * 60)
ff_ok = at_counter.get("fact_finder", 0)
vu_expired = vu_states.get("ИСТЁК (< сегодня)", 0)
if ff_ok == 0:
    print(f"🔴 ПРИЧИНА НАЙДЕНА: ни один чанк не имеет agent_target='fact_finder'.")
    print("   Решение: перезагрузить ТК РФ с agent_target='fact_finder' (или другим нужным значением),")
    print("   ИЛИ убрать/ослабить фильтр agent_target в registry.py для fact_finder.")
if vu_expired > 0:
    print(f"🔴 ПРИЧИНА НАЙДЕНА: {vu_expired} чанков с истёкшим valid_until отсеиваются.")
    print("   Решение: обновить valid_until при загрузке (поставить далеко в будущем или null),")
    print("   ИЛИ ИСПРАВИТЬ ДАТУ в payload.")
if tk_count == 0:
    print("🔴 ТК РФ отсутствует в коллекции copywriter_kb — проверьте, в какую коллекцию загружали.")
if ff_ok > 0 and vu_expired == 0 and tk_count > 0:
    print("🟡 Все явные причины не подтвердились. Возможны:")
    print("   - score_threshold=0.25 слишком высок для запроса (семантическое несоответствие)")
    print("   - чанки ТК РФ есть, но без agent_target='fact_finder'")
print("\nСкопируйте вывод этого скрипта — по нему я дам точное исправление.")
