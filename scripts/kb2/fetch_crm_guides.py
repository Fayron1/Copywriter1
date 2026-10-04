#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Снимок официальных справок CRM (Битрикс24, amoCRM) в маркетинговую БЗ.

Fetch URL -> HTML->text -> knowledge_base/business/marketing/crm_and_sales_automation/
  <система>/<категория>/<имя>_снимок_<дата>.txt

Запуск:  python scripts/kb2/fetch_crm_guides.py [--only key1,key2]
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
DEST = ROOT / "knowledge_base" / "business" / "marketing" / "crm_and_sales_automation"

# key -> (url, относительная папка, имя файла без даты)
PAGES = {
    # --- Битрикс24 ---
    "bx_help_24792750": (
        "https://helpdesk.bitrix24.ru/open/24792750/",
        "bitrix24/robots_and_triggers",
        "Битрикс24_helpdesk_24792750",
    ),
    "bx_robots_main": (
        "https://helpdesk.bitrix24.ru/open/24367166/",
        "bitrix24/robots_and_triggers",
        "Битрикс24_роботы_как_автоматизировать",
    ),
    "bx_triggers_main": (
        "https://helpdesk.bitrix24.ru/open/24473054/",
        "bitrix24/robots_and_triggers",
        "Битрикс24_триггеры_как_автоматизировать",
    ),
    "bx_robots_comms": (
        "https://helpdesk.bitrix24.ru/open/19564002/",
        "bitrix24/robots_and_triggers",
        "Битрикс24_роботы_коммуникация_с_клиентом",
    ),
    "bx_triggers_comms": (
        "https://helpdesk.bitrix24.ru/open/21052538/",
        "bitrix24/robots_and_triggers",
        "Битрикс24_триггеры_коммуникация_с_клиентом",
    ),
    "bx_robots_repeat": (
        "https://helpdesk.bitrix24.ru/open/22494682/",
        "bitrix24/sales_automation",
        "Битрикс24_роботы_повторные_продажи",
    ),
    "bx_robots_tasks": (
        "https://helpdesk.bitrix24.ru/open/22892250/",
        "bitrix24/robots_and_triggers",
        "Битрикс24_роботы_и_триггеры_управление_задачами",
    ),
    "bx_voronka_journal": (
        "https://www.bitrix24.ru/journal/kak-postroit-voronku-prodazh-v-nbsp-bitriks24/",
        "bitrix24/pipelines",
        "Битрикс24_как_построить_воронку_продаж",
    ),
    # --- amoCRM ---
    "amo_digital_pipeline": (
        "https://www.amocrm.ru/support/starting_work/digital_pipeline",
        "amocrm/digital_pipeline",
        "amoCRM_digital_pipeline_обзор",
    ),
    "amo_salesbot": (
        "https://www.amocrm.ru/support/digital_pipline/salesbot",
        "amocrm/salesbot",
        "amoCRM_salesbot_обзор",
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
        f"Дата снимка: {today}. Официальный материал вендора; интерфейс и функции\n"
        f"меняются — валидно на дату снимка (valid_until + 180 дней).\n\n"
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
