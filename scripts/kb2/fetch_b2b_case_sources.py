#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Снимок источников кейсов B2B-контента РФ (Mindbox, Unisender).

Fetch URL -> HTML->text -> knowledge_base/business/marketing/b2b_content_cases_russia/
  <категория>/<имя>_снимок_<дата>.txt

Запуск:  python scripts/kb2/fetch_b2b_case_sources.py [--only key1,key2]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import requests

from fetch_paid_traffic_guides import HEADERS, MIN_CHARS, html_to_text

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "knowledge_base" / "business" / "marketing" / "b2b_content_cases_russia"

# key -> (url, относительная папка, имя файла без даты)
PAGES = {
    "case_selectel": (
        "https://mindbox.ru/journal/cases/selectel/",
        "crm_email",
        "Кейс_Selectel_CRM_маркетинг_Mindbox",
    ),
    "case_jivo": (
        "https://mindbox.ru/journal/education/rassylki-b2b/",
        "b2b_saas",
        "Кейс_Jivo_welcome_цепочка_Mindbox_обучение",
    ),
    "case_teachbase": (
        "https://www.unisender.com/ru/blog/kejs-rookee/",
        "lead_magnets",
        "Кейс_Teachbase_контент_лиды_Unisender",
    ),
    "case_rookee": (
        "https://www.unisender.com/ru/blog/kejs-rookee/",
        "crm_email",
        "Кейс_Rookee_сегментация_реактивация_Unisender",
    ),
    "case_completo": (
        "https://www.unisender.com/ru/blog/kejs-completo-i-york/",
        "crm_email",
        "Кейс_Completo_York_карта_писем_Unisender",
    ),
    "case_mpstats": (
        "https://mindbox.ru/journal/cases/mpstats/",
        "marketplaces",
        "Кейс_MPSTATS_email_выручка_Mindbox",
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
        f"Дата снимка: {today}. Кейс поставщика платформы — хранить цифру\n"
        f"ТОЛЬКО вместе с контекстом (компания, период, канал, механика,\n"
        f"ограничения). Чужой результат не переносится как прогноз.\n\n"
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
