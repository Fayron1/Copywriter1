#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Снимок материалов CRO / A/B-тестирования в маркетинговую БЗ.

Fetch URL -> HTML->text -> knowledge_base/business/marketing/landing_pages_cro/
  <категория>/<имя>_снимок_<дата>.txt

Запуск:  python scripts/kb2/fetch_cro_guides.py [--only key1,key2]
Повтор:  безопасен — файлы перезаписываются (один снимок на дату).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import requests

from fetch_paid_traffic_guides import HEADERS, MIN_CHARS, html_to_text

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "knowledge_base" / "business" / "marketing" / "landing_pages_cro"

# key -> (url, относительная папка, имя файла без даты)
PAGES = {
    "ya_ab_edu": (
        "https://b2b.yandex.ru/adv/edu/materials/a-b-testirovanie",
        "ab_testing",
        "Яндекс_Реклама_образование_АБ_тестирование",
    ),
    "habr_756434": (
        "https://habr.com/ru/articles/756434/",
        "ab_testing",
        "Хабр_756434_AB_тесты",
    ),
    "optimizely_glossary": (
        "https://www.optimizely.com/optimization-glossary/ab-testing",
        "ab_testing",
        "Optimizely_AB_testing_glossary",
    ),
    "growthbook_intro": (
        "https://www.growthbook.io/blog/what-is-a-b-testing",
        "ab_testing",
        "GrowthBook_what_is_AB_testing",
    ),
    "metrica_index": (
        "https://yandex.ru/support/metrica/",
        "analytics",
        "Метрика_справка_оглавление",
    ),
    "varioqub_index": (
        "https://yandex.ru/support/varioqub/",
        "ab_testing",
        "Вариокуб_справка_оглавление",
    ),
}


def fetch(url: str, rel_dir: str, name: str, today: str) -> tuple[str, int]:
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    r.encoding = "utf-8"
    text = html_to_text(r.text)
    out_dir = DEST / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_снимок_{today}.txt"
    header = (
        f"# {name} — снимок {today}\n\n"
        f"Источник: {url}\n"
        f"Дата снимка: {today}. Внешний материал; интерфейсы и цифры меняются —\n"
        f"валидно на дату снимка.\n\n"
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
            rel_path, n = fetch(url, rel_dir, name, today)
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
