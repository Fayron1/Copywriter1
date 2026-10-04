#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Снимок официальных гайдов paid-traffic РФ в маркетинговую БЗ.

Fetch URL -> HTML->text -> knowledge_base/business/marketing/paid_traffic_russia/
  <платформа>/<категория>/<имя>_снимок_<дата>.txt

Запуск:  python scripts/kb2/fetch_paid_traffic_guides.py [--only yd_stoimost,vk_bids]
Повтор:  безопасен — файлы перезаписываются (один снимок на дату).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "knowledge_base" / "business" / "marketing" / "paid_traffic_russia"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

# key -> (url, относительная папка, имя файла без даты)
PAGES = {
    "yd_stoimost": (
        "https://direct.yandex.ru/base/articles/stoimost-kontekstnoy-reklamy-v-yandeks-direkte",
        "yandex_direct/budget_forecast",
        "ЯД_стоимость_контекстной_рекламы",
    ),
    "yd_diy": (
        "https://direct.yandex.ru/base/articles/diy",
        "yandex_direct/official_guides",
        "ЯД_запуск_рекламы_своими_руками",
    ),
    "yd_optoviki": (
        "https://direct.yandex.ru/base/articles/kak-najti-optovyh-pokupatelej",
        "yandex_direct/search",
        "ЯД_кейс_оптовые_покупатели_B2B",
    ),
    "yd_teatr": (
        "https://direct.yandex.ru/base/articles/prodvizhenie-spektakley-i-teatra-marketing-teatralnoy-studii",
        "yandex_direct/cases_b2c",
        "ЯД_кейс_театральная_студия",
    ),
    "yd_yazyk": (
        "https://direct.yandex.ru/base/articles/prodvizhenie-yazykovoy-shkoly-reklama-kursov-inostrannogo-yazyka",
        "yandex_direct/cases_b2c",
        "ЯД_кейс_языковая_школа",
    ),
    "yd_nedvizh": (
        "https://direct.yandex.ru/base/articles/reklama-arendy-kommercheskoj-nedvizhimosti",
        "yandex_direct/cases_b2c",
        "ЯД_кейс_аренда_коммерческой_недвижимости",
    ),
    "yd_epilyaciya": (
        "https://direct.yandex.ru/base/articles/effektivnaya-reklama-lazernoy-epilyacii-i-shugaringa",
        "yandex_direct/cases_b2c",
        "ЯД_кейс_лазерная_эпиляция",
    ),
    "vk_stoimost": (
        "https://ads.vk.ru/insights/stoimost-reklamy",
        "vk_ads/official_guides",
        "VK_стоимость_рекламы_бюджет_ставки",
    ),
    "vk_limits": (
        "https://ads.vk.ru/help/general/start/budget_limits",
        "vk_ads/official_guides",
        "VK_лимиты_бюджета",
    ),
    "vk_bids": (
        "https://ads.vk.ru/help/features/bid_strategy",
        "vk_ads/campaign_goals",
        "VK_стратегия_ставок",
    ),
    "tg_start": (
        "https://ads.telegram.org/getting-started",
        "telegram_ads/official_guides",
        "TG_Ads_getting_started",
    ),
}

MIN_CHARS = 2000  # меньше — почти наверняка JSON-мусор или заглушка


def html_to_text(page: str) -> str:
    page = re.sub(r"(?is)<(script|style|noscript|svg|head|iframe)[^>]*>.*?</\1>", " ", page)
    m = re.search(r"(?is)<main[^>]*>(.*?)</main>", page)
    body = m.group(1) if m else page
    body = re.sub(r"(?is)<br\s*/?>", "\n", body)
    body = re.sub(r"(?is)</(p|div|h[1-6]|li|tr|section|article)>", "\n", body)
    text = re.sub(r"<[^>]+>", " ", body)
    text = _html.unescape(text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch(key: str, url: str, rel_dir: str, name: str, today: str) -> tuple[str, int]:
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    text = html_to_text(r.text)
    title = ""
    tm = re.search(r"<title[^>]*>([^<]+)</title>", r.text, re.I)
    if tm:
        title = _html.unescape(tm.group(1)).strip()
    out_dir = DEST / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_снимок_{today}.txt"
    header = (
        f"# {title or name} — снимок {today}\n\n"
        f"Источник: {url}\n"
        f"Дата снимка: {today}. Официальный материал платформы; тарифы и лимиты\n"
        f"пересматриваются — цифры валидны на дату снимка (valid_until + 90 дней).\n\n"
        f"---\n\n"
    )
    out_path.write_text(header + text, encoding="utf-8")
    return str(out_path.relative_to(ROOT)), len(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="через запятую ключи из PAGES")
    args = ap.parse_args()
    keys = [k.strip() for k in args.only.split(",")] if args.only else list(PAGES)
    today = _dt.date.today().isoformat()
    bad = 0
    for key in keys:
        if key not in PAGES:
            print(f"?? неизвестный ключ {key}")
            bad += 1
            continue
        url, rel_dir, name = PAGES[key]
        try:
            rel_path, n = fetch(key, url, rel_dir, name, today)
            flag = "OK " if n >= MIN_CHARS else "МАЛО"
            if n < MIN_CHARS:
                bad += 1
            print(f"{flag} {rel_path} ({n} символов)")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"ERR {key}: {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
