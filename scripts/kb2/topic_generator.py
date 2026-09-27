"""
Генератор тем статей: сид-матрица x три фильтра.

Фильтры (порядок как в стратегии):
1. KB-SUPPORT — тема обязана опираться на базу знаний: запрос через продакшн
   RAG (fact_finder); метрика = средний скор топ-3 чанков. Без опоры тема
   не проходит — это наше УТП против «просто GPT-статей».
2. DEMAND — живой спрос: подсказки Яндекс (suggest) по ядру запроса;
   метрика = число подсказок, пересекающихся с темой (proxy частотности).
3. HOOK — виральный крючок: Jev (noul) оценивает, цепляет ли заголовок
   владельца МСБ (боль/любопытство/страх ошибки). 14 критериев виральности
   в компактной форме.

Запуск (VPS, где есть kb_v2/Яндекс/Jev):
  python scripts/kb2/topic_generator.py --top 30
  python scripts/kb2/topic_generator.py --json topics.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kb.topics")

# ============================================================
# Сид-матрица: сфера x кластер x тип x формула.
# Веса из исследования спроса (2026-09-27): налоги 40% / юр 35% / маркетинг 25%;
# чек-лист 30% / разбор 25% / справочник 15% / кейс 15% / лонгрид 15%.
# ============================================================

SEEDS = [
    # ── НАЛОГИ (40%) ──────────────────────────────────────────
    {"sphere": "налоги", "cluster": "НДС-УСН", "type": "law_review",
     "title": "НДС 5 процентов при УСН: как считать и не переплатить",
     "query": "НДС 5 процентов УСН расчёт лимит"},
    {"sphere": "налоги", "cluster": "НДС-УСН", "type": "law_review",
     "title": "НДС 7 процентов на УСН: кому подходит и когда выгоднее общая ставка",
     "query": "НДС 7 процентов УСН освобождение лимит"},
    {"sphere": "налоги", "cluster": "НДС-УСН", "type": "reference",
     "title": "УСН и НДС в 2026 году: таблица лимитов, ставок и сроков перехода",
     "query": "УСН НДС 2026 лимиты ставки таблица"},
    {"sphere": "налоги", "cluster": "НДС-УСН", "type": "analysis",
     "title": "УСН или ОСНО в 2027 году: как посчитать точку перехода на конкретных цифрах",
     "query": "УСН ОСНО сравнение выбор системы налогообложения"},
    {"sphere": "налоги", "cluster": "НДС-УСН", "type": "checklist",
     "title": "10 пунктов о переходе на уплату НДС при УСН",
     "query": "переход на НДС при УСН порядок действия"},
    {"sphere": "налоги", "cluster": "НДФЛ-шкала", "type": "reference",
     "title": "Прогрессивная шкала НДФЛ 2026: таблица ставок и порогов по зарплатам",
     "query": "прогрессивная шкала НДФЛ ставки пороги"},
    {"sphere": "налоги", "cluster": "НДФЛ-шкала", "type": "checklist",
     "title": "10 пунктов о том, как прогрессивный НДФЛ меняет зарплаты и договоры",
     "query": "НДФЛ прогрессивная шкала удержание работодатель"},
    {"sphere": "налоги", "cluster": "самозанятые", "type": "law_review",
     "title": "Самозанятый свыше 2,4 млн: что делать за месяц до превышения лимита",
     "query": "самозанятый лимит 2,4 млн превышение НПД"},
    {"sphere": "налоги", "cluster": "самозанятые", "type": "checklist",
     "title": "10 пунктов о работе с самозанятыми без риска доначислений",
     "query": "работа с самозанятыми риски статус НПД проверка"},
    {"sphere": "налоги", "cluster": "115-ФЗ", "type": "case_study",
     "title": "Блокировка счёта по 115-ФЗ: разбор типовой ситуации и выход",
     "query": "блокировка счета 115-ФЗ банк причины"},
    {"sphere": "налоги", "cluster": "115-ФЗ", "type": "checklist",
     "title": "10 пунктов о том, как не попасть под блокировку счёта",
     "query": "115-ФЗ блокировка счета профилактика документы"},
    {"sphere": "налоги", "cluster": "ЕНС", "type": "reference",
     "title": "ЕНС и ЕНП в 2026 году: шпаргалка по срокам и типовым ошибкам",
     "query": "ЕНС ЕНП сроки уплата сальдо"},
    {"sphere": "налоги", "cluster": "дедлайны", "type": "reference",
     "title": "Календарь отчетности на 2027 год: сроки для ИП и ООО",
     "query": "производственный календарь 2027 сроки отчетности"},
    {"sphere": "налоги", "cluster": "УСН-льготы", "type": "reference",
     "title": "Пониженные ставки УСН по регионам 2026: таблица, где платить меньше",
     "query": "пониженные ставки УСН регионы таблица"},

    # ── ЮРИДИЧЕСКОЕ (35%) ─────────────────────────────────────
    {"sphere": "юр", "cluster": "РКН/152-ФЗ", "type": "checklist",
     "title": "10 пунктов об уведомлении Роскомнадзора о персональных данных",
     "query": "уведомление Роскомнадзора персональные данные порядок"},
    {"sphere": "юр", "cluster": "РКН/152-ФЗ", "type": "reference",
     "title": "Штрафы РКН в 2026 году: таблица нарушений и сумм по 152-ФЗ",
     "query": "штрафы Роскомнадзор 152-ФЗ персональные данные размер"},
    {"sphere": "юр", "cluster": "РКН/152-ФЗ", "type": "law_review",
     "title": "Авторизация через Google и Apple: почему теперь это штраф до 1,4 млн",
     "query": "авторизация Google Apple штраф персональные данные трансграничная"},
    {"sphere": "юр", "cluster": "РКН/152-ФЗ", "type": "law_review",
     "title": "Плановая проверка Роскомнадзора: шесть блоков, которые проверяют",
     "query": "плановая проверка Роскомнадзора блоки содержание"},
    {"sphere": "юр", "cluster": "РКН/152-ФЗ", "type": "checklist",
     "title": "10 пунктов о сборе персональных данных на сайте",
     "query": "персональные данные на сайте согласие политика"},
    {"sphere": "юр", "cluster": "договоры", "type": "checklist",
     "title": "10 пунктов о договоре с фрилансером: от статуса до акта",
     "query": "договор фрилансер ГПХ статус налоги"},
    {"sphere": "юр", "cluster": "договоры", "type": "law_review",
     "title": "ГПХ или трудовой: как выбрать форму и не проиграть при проверке",
     "query": "ГПХ трудовой договор различения переквалификация"},
    {"sphere": "юр", "cluster": "договоры", "type": "law_review",
     "title": "Электронный трудовой договор: как подписать без ошибки",
     "query": "электронный трудовой договор подписание усиленная подпись"},
    {"sphere": "юр", "cluster": "охрана труда", "type": "checklist",
     "title": "Чек-лист обязательных документов по охране труда",
     "query": "охрана труда обязательные документы чек-лист"},
    {"sphere": "юр", "cluster": "госзаказ", "type": "case_study",
     "title": "Срыв сроков по 44-ФЗ: как одна ошибка в заявке стоила контракта",
     "query": "44-ФЗ срыв сроков госзаказ ответственность"},
    {"sphere": "юр", "cluster": "валютное", "type": "checklist",
     "title": "10 пунктов о валютном контроле для тех, кто платит за рубеж",
     "query": "валютный контроль 173-ФЗ репатриация операции"},

    # ── МАРКЕТИНГ (25%) ───────────────────────────────────────
    {"sphere": "маркетинг", "cluster": "лидогенерация", "type": "analysis",
     "title": "Лидогенерация в 2027 году: какие каналы еще работают, а какие сжигают бюджет",
     "query": "лидогенерация каналы эффективность B2B"},
    {"sphere": "маркетинг", "cluster": "SEO-vs-реклама", "type": "analysis",
     "title": "SEO или таргетированная реклама: где рубль приводит дешевле в 2027",
     "query": "SEO или контекстная реклама сравнение стоимость лида"},
    {"sphere": "маркетинг", "cluster": "маркетплейсы-B2B", "type": "analysis",
     "title": "Маркетплейсы для B2B: ниша, которую конкуренты еще не заняли",
     "query": "маркетплейсы B2B продажи ниша"},
    {"sphere": "маркетинг", "cluster": "маркетплейсы", "type": "reference",
     "title": "Юнит-экономика Wildberries и Ozon: таблица для расчёта за 10 минут",
     "query": "юнит-экономика маркетплейс расчёт комиссия"},
    {"sphere": "маркетинг", "cluster": "позиционирование", "type": "case_study",
     "title": "Неуместный креатив разрушает репутацию: разбор без имён",
     "query": "креатив реклама репутация кризис бренд"},
    {"sphere": "маркетинг", "cluster": "ценообразование", "type": "case_study",
     "title": "Считай порог рентабельности или прогоришь: разбор на цифрах",
     "query": "порог рентабельности точка безубыточности расчет"},
]

# ============================================================
# Фильтры
# ============================================================

def kb_support(query: str, top_k: int = 3) -> float:
    """Средний скор топ-K чанков из kb_v2 (продакшн RAG, fact_finder)."""
    try:
        from agents.rag import query_knowledge
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=f"http://{os.getenv('QDRANT_HOST', '127.0.0.1')}:{os.getenv('QDRANT_PORT', '6333')}",
            api_key=os.getenv("QDRANT_API_KEY") or None, timeout=30)
        chunks = query_knowledge(query, "fact_finder", qdrant_client=client, extra_filters=None)
        if not chunks:
            return 0.0
        scores = [c.get("score", 0) for c in chunks[:top_k]]
        return sum(scores) / len(scores)
    except Exception as e:
        logger.warning(f"kb_support сбой: {e}")
        return 0.0


def demand_hits(query: str) -> int:
    """Число подсказок Яндекса, пересекающихся с темой (proxy спроса).
    Yandex suggest фильтрует TLS-отпечаток requests — зовём настоящий curl."""
    try:
        import subprocess
        import urllib.parse
        core = urllib.parse.quote(" ".join(query.split()[:3]))
        out = subprocess.run(
            ["curl", "-s", "-m", "10",
             f"https://suggest.yandex.ru/suggest-ff?part={core}&utf=1"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        if not out.startswith("["):
            return -1
        data = json.loads(out)
        suggs = data[1] if len(data) > 1 else []
        words = set(re.findall(r"[а-яё]{4,}", query.lower()))
        hits = 0
        for s in suggs:
            sw = set(re.findall(r"[а-яё]{4,}", str(s).lower()))
            if len(words & sw) >= 2:
                hits += 1
        return hits
    except Exception as e:
        logger.warning(f"demand сбой: {e}")
        return -1


def hook_score(title: str) -> float:
    """Jev: цепляет ли заголовок владельца МСБ (боль/любопытство/страх ошибки)."""
    try:
        from agents.jev_client import get_jev_client, noul_question
        client = get_jev_client()
        if client is None:
            return -1.0
        d = client.decide(
            state={"title": title, "audience": "владельцы и директора малого бизнеса в России"},
            questions={
                "hook": noul_question(
                    "Заголовок цепляет: называет боль, риск потери денег, конкретную выгоду "
                    "или любопытство? Обычная описательная формулировка — ответ НЕТ."),
            })
        return d.yes("hook", -1.0)
    except Exception as e:
        logger.warning(f"hook сбой: {e}")
        return -1.0


# ============================================================
# Основной проход
# ============================================================

def evaluate(seeds):
    rows = []
    for s in seeds:
        kb = kb_support(s["query"])
        dm = demand_hits(s["query"])
        hk = hook_score(s["title"])
        # Совокупный скор: опора на базу (x2 — без неё тема не наша),
        # спрос и крючок нормируем грубо: /8 и *1 (0..1)
        score = round(kb * 2 + max(0, dm) / 8.0 + max(0, hk), 3)
        rows.append({**s, "kb": round(kb, 3), "demand": dm, "hook": hk, "score": score})
        logger.info(f"[{s['type']:11s}] kb={kb:.2f} dm={dm:2d} hook={hk:.2f} | {s['title'][:60]}")
    rows.sort(key=lambda r: -r["score"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json", default=None, help="путь для полного JSON-отчёта")
    args = ap.parse_args()

    # .env
    env_file = PROJECT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)
        except ImportError:
            pass

    rows = evaluate(SEEDS)

    print("\n" + "=" * 100)
    print(f"{'СКОР':>6} {'kb':>5} {'спр':>4} {'крюк':>5}  {'тип':11s} {'сфера':9s} ТЕМА")
    print("-" * 100)
    for r in rows[:args.top]:
        kb_mark = "✓" if r["kb"] >= 0.6 else ("~" if r["kb"] >= 0.45 else "✗")
        print(f"{r['score']:6.2f} {r['kb']:5.2f}{kb_mark} {r['demand']:4d} {r['hook']:5.2f}  "
              f"{r['type']:11s} {r['sphere']:9s} {r['title'][:66]}")
    print("=" * 100)
    print("kb: ✓ сильная опора на базу (≥0.60) ~ средняя ✗ нет опоры | спр: подсказок Яндекса | крюк: Jev")

    if args.json:
        Path(args.json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Полный отчёт: {args.json}")


if __name__ == "__main__":
    main()
