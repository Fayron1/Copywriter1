"""Форум-майнер: сбор реальных болей и запросов пользователей.

Идея (2026-10-02): по форумам/отзовикам/сообществам ходит скрипт, собирает
живые боли ЦА по 14 критериям виральности (Маркетинг.txt) и превращает их в
кандидатов тем — с цитатой-источником, URL и скором.

Слои:
1. ПОИСК — SearXNG (VPS :8082) по pain-запросам с site:-фильтрами форумов.
2. ИЗВЛЕЧЕНИЕ — passage'ы с болью: вопросительные треды + pain-маркеры
   («отклонили», «не вернули», «штраф», «что делать», «как избежать»...).
3. СКОР — виральный крючок через Jev (hook_score из topic_generator) +
   критерий-бонусы из Маркетинг.txt (страх потери денег = адреналин,
   «да, у меня тоже так» = дофамин узнавания...).
4. ДЕДУП — сравнение с существующими темами (topic_expander) и прошлыми
   находками (state-файл).
5. ОТЧЁТ — forum_pains_<domain>_<date>.json + .md, топ по скору.

Запуск (VPS): venv/bin/python scripts/kb2/forum_pain_miner.py --domain закупки
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlparse

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kb.forum_miner")

SEARXNG = "http://localhost:8082/search?format=json&q="

# ── Источники болей по доменам (SearXNG site:-фильтры + pain-запросы) ──
DOMAINS = {
    "закупки": {
        "sites": ["forum.garant.ru", "www.audit-it.ru", "www.buhonline.ru",
                  "zakupki.kontur.ru", "www.klerk.ru", "pikabu.ru",
                  "otzovik.com", "vc.ru"],
        "queries": [
            "жалобу в ФАС не приняли что делать",
            "заявку отклонили 44-ФЗ форум",
            "включили в РНП форум опыт",
            "заказчик не вернул обеспечение форум",
            "пени по госконтракту оспорить форум",
            "гарантию не приняли на площадке",
        ],
    },
    "маркетплейсы": {
        "sites": ["otzovik.com", "irecommend.ru", "pikabu.ru", "vc.ru",
                  "silverdata.ru", "forum.sberbank-ast.ru"],
        "queries": [
            "озон штрафы продавцу форум опыт",
            "вайлдберриз невыкуп логистика съела прибыль",
            "выплаты вайлдберриз меньше отчёта форум",
            "карточки заблокировали озон что делать",
            "хранение fbo списали форум неликвид",
            "самозанятый превышение лимита маркетплейс доначисление",
        ],
    },
}

# Pain-маркеры: фразы, рядом с которыми живёт боль (кейс-цитата)
_PAIN_RX = re.compile(
    r"не\s+приняли|отклонили|не\s+вернули|списали|штраф\w*|заблокир|"
    r"доначислили|сгорел\w*|потерял\w*\s+закупк|не\s+успел\w*|пропустил\w*"
    r"|что\s+делать|как\s+избежать|как\s+оспорить|кто\s+виноват|"
    r"съел\w*\s+(?:всю\s+)?(?:прибыль|маржу)|в\s+минус|убыток", re.I)
_QUESTION_RX = re.compile(r"^[^?!\n]{20,300}\?", re.M)
_NOISE_RX = re.compile(
    r"конфиденциальн|cookie|privacy|signup|логин|подписаться|реклама", re.I)

# Виральные критерии (Маркетинг.txt, детерминированные прокси):
# страх денег (адреналин), узнавание «и у меня так» (дофамин),
# самообман/миф (серотонин осознания), совет-сохранёнка (социосигнал)
_CRITERIA_BONUS = {
    "money_fear": re.compile(r"рубл\w+|тыс\.?|млн|₽", re.I),
    "recognition": re.compile(r"все\s+(?:так|как)\s+|у\s+меня\s+тоже|каждый\s+раз", re.I),
    "myth_busting": re.compile(r"думал\w*|казалось|миф|на\s+самом\s+деле|оказалось", re.I),
    "saveable": re.compile(r"чек-?лист|шаг\w+|порядок|алгоритм|инструкция", re.I),
}


def _searxng(query: str, limit: int = 6) -> list:
    try:
        r = requests.get(SEARXNG + quote(query), timeout=20)
        r.raise_for_status()
        return (r.json().get("results") or [])[:limit]
    except Exception as e:
        logger.warning(f"SearXNG недоступен ({e})")
        return []


def _fetch_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
        if r.status_code != 200:
            return ""
        return r.text
    except Exception:
        return ""


def _strip_html(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html)
    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"&nbsp;?", " ", text)
    return re.sub(r"\n{2,}", "\n", text)


def extract_pains(text: str, url: str, domain: str) -> list:
    """Passage'ы с болью: вопрос или pain-маркер + соседний контекст."""
    clean = _strip_html(text)
    out = []
    # 1) прямые вопросы тредов
    for q in _QUESTION_RX.findall(clean)[:8]:
        q = q.strip().replace("\n", " ")
        if _NOISE_RX.search(q) or len(q) < 25:
            continue
        out.append({"domain": domain, "type": "question", "text": q,
                    "url": url, "source": urlparse(url).netloc})
    # 2) pain-маркеры с контекстом
    for m in list(_PAIN_RX.finditer(clean))[:12]:
        s = max(0, m.start() - 150)
        frag = clean[s:m.end() + 250].replace("\n", " ").strip()
        frag = re.sub(r"\s{2,}", " ", frag)
        if len(frag) < 60 or _NOISE_RX.search(frag):
            continue
        out.append({"domain": domain, "type": "pain", "text": frag[:400],
                    "url": url, "source": urlparse(url).netloc})
    return out


def score_pain(pain: dict) -> dict:
    """Виральный скор: базовый hook через Jev + критерии Маркетинг.txt."""
    hooks = {"money_fear": 0.0, "recognition": 0.0,
             "myth_busting": 0.0, "saveable": 0.0}
    for k, rx in _CRITERIA_BONUS.items():
        if rx.search(pain["text"]):
            hooks[k] = 1.0
    bonus = sum(hooks.values())
    question = 1.0 if pain["type"] == "question" else 0.0
    pain["virality"] = round(1.0 + bonus + question, 2)   # 0..5+
    pain["criteria"] = [k for k, v in hooks.items() if v]
    return pain


def _dedup_key(text: str) -> str:
    words = re.findall(r"[а-яё]{5,}", text.lower())
    return " ".join(sorted(set(words[:8])))


def mine(domain: str, max_results: int = 40) -> list:
    cfg = DOMAINS[domain]
    pains, seen_urls = [], set()
    for q in cfg["queries"]:
        for site in cfg["sites"][:3]:
            query = f"{q} site:{site}"
            for res in _searxng(query):
                url = res.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                text = _fetch_text(url)
                if not text:
                    continue
                pains.extend(extract_pains(text, url, domain))
                time.sleep(1.0)                     # щадящий темп
    pains = [score_pain(p) for p in pains]
    # дедуп по событийному ядру
    uniq, seen_keys = [], set()
    for p in sorted(pains, key=lambda x: -x["virality"]):
        k = _dedup_key(p["text"])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        uniq.append(p)
    return uniq[:max_results]


def to_md(pains: list, domain: str) -> str:
    lines = [f"# Форум-майнер: боли {domain} — снимок {date.today().isoformat()}",
             "", "Источники: форумы, отзовики, сообщества. Скор = виральный",
             "крючок по критериям Маркетинг.txt (страх денег, узнавание,",
             "разрушение мифа, сохранёнка) + вопрос-формат.", ""]
    for i, p in enumerate(pains, 1):
        crit = ", ".join(p.get("criteria", [])) or "—"
        lines.append(f"## {i}. [{p['virality']}] {p['source']} ({crit})")
        lines.append(f"> {p['text'][:280]}")
        lines.append(f"Источник: {p['url']}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAINS.keys()))
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pains = mine(args.domain, args.max)
    logger.info(f"Найдено болей: {len(pains)}")
    stamp = date.today().isoformat()
    base = args.out or f"forum_pains_{args.domain}_{stamp}"
    Path(base + ".json").write_text(
        json.dumps(pains, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(base + ".md").write_text(to_md(pains, args.domain), encoding="utf-8")
    logger.info(f"Отчёт: {base}.md / .json")
    for p in pains[:10]:
        logger.info(f"[{p['virality']}] {p['text'][:90]}")


if __name__ == "__main__":
    main()
