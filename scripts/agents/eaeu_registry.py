"""Тонкий клиент OData реестра промышленных товаров ЕАЭС.

Реестр ЕАЭС (евразийская промышленная продукция) — источник подтверждения
происхождения для нацрежима (ПП № 1875). Данные живые: файл-выгрузка устаре-
вает, OData — нет. Модуль используется (1) при генерации статей — проверка
фактов о реестре с датой снимка, (2) в продуктовых чекерах нацрежима,
(3) как справочник полей для fact_finder.

Протокол: GET https://opendata.eaeunion.org/odata/Goodscollection_prod
  $filter=startswith(goodsokpd2,'25.11') &$top=N &$select=...
Авторизации нет; лимитов не задокументировано — щадим: TTL-кэш 24 ч,
ретраи с задержкой.

Проверено вручную 2026-10-01: 94 243 записи; поля registrynumber,
goodsokpd2, tnved, goodsname, name (производитель), address, statemember,
points, publishdate.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agents.eaeu_registry")

BASE = "https://opendata.eaeunion.org/odata"
COLLECTION = "Goodscollection_prod"
DEFAULT_FIELDS = ("registrynumber", "goodsokpd2", "tnved", "goodsname",
                  "name", "statemember", "points", "publishdate")
_TIMEOUT = 25
_RETRIES = 3
_CACHE: Dict[str, tuple] = {}          # key -> (ts, data)
_CACHE_TTL = 24 * 3600

# страны-участники по коду statemember (реестр ЕАЭС)
STATE_MEMBERS = {
    "01": "Армения", "02": "Беларусь", "03": "Казахстан",
    "04": "Кыргызстан", "05": "Россия", "06": "Таджикистан",
}


def build_filter(okpd2: str = "", tnved: str = "",
                 registry_number: str = "", name: str = "") -> str:
    """Собрать $filter из поисковых условий (для юнит-тестов — чистая)."""
    parts: List[str] = []
    if okpd2:
        parts.append(f"startswith(goodsokpd2,'{okpd2}')")
    if tnved:
        parts.append(f"startswith(tnved,'{tnved.replace(' ', '')}')")
    if registry_number:
        parts.append(f"registrynumber eq '{registry_number}'")
    if name:
        parts.append(f"contains(goodsname,'{name}')")
    return " and ".join(parts)


def _http_get(url: str) -> Optional[str]:
    """GET с ретраями; httpx если есть, иначе urllib."""
    last_err = None
    for attempt in range(_RETRIES):
        try:
            try:
                import httpx
                r = httpx.get(url, timeout=_TIMEOUT,
                              headers={"Accept": "application/json",
                                       "User-Agent": "copywriter-kb/1.0"})
                if r.status_code == 200:
                    return r.text
                last_err = f"HTTP {r.status_code}"
            except ImportError:
                import urllib.request
                req = urllib.request.Request(
                    url, headers={"Accept": "application/json",
                                  "User-Agent": "copywriter-kb/1.0"})
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                    return resp.read().decode("utf-8")
        except Exception as e:                     # сеть/таймаут
            last_err = str(e)[:120]
        time.sleep(2 * (attempt + 1))
    logger.warning(f"ЕАЭС реестр недоступен: {last_err}")
    return None


def _query(params: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
    key = json.dumps(params, sort_keys=True)
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    from urllib.parse import quote
    SAFE = "(),'="
    # ключи params уже содержат собственный '$' ($top, $filter, $select)
    url = f"{BASE}/{COLLECTION}?" + "&".join(
        f"{k}={quote(str(v), safe=SAFE)}" for k, v in params.items())
    raw = _http_get(url)
    if raw is None:
        return None
    try:
        data = json.loads(raw).get("value", [])
    except Exception as e:
        logger.warning(f"ЕАЭС реестр: битый JSON ({e})")
        return None
    _CACHE[key] = (now, data)
    return data


def search(okpd2: str = "", tnved: str = "", registry_number: str = "",
           name_contains: str = "", top: int = 20) -> List[Dict[str, Any]]:
    """Поиск товаров в реестре ЕАЭС. Возвращает записи (может быть пусто)."""
    flt = build_filter(okpd2, tnved, registry_number, name_contains)
    params = {"$top": str(top), "$select": ",".join(DEFAULT_FIELDS)}
    if flt:
        params["$filter"] = flt
    res = _query(params)
    return res or []


def count_by_okpd2(okpd2: str) -> Optional[int]:
    """Сколько записей по префиксу ОКПД2 (для фактов в статьях)."""
    flt = build_filter(okpd2=okpd2)
    # @odata.count возвращается в теле — берём напрямую, не через _query
    from urllib.parse import quote as _q
    SAFE = "(),'="
    raw = _http_get(f"{BASE}/{COLLECTION}?$filter={_q(flt, safe=SAFE)}"
                    f"&$top=1&$count=true")
    if raw is None:
        return None
    try:
        return int(json.loads(raw).get("@odata.count", -1))
    except Exception:
        return None


def check_registry_number(registry_number: str) -> Dict[str, Any]:
    """Проверка реестровой записи для чекера нацрежима."""
    recs = search(registry_number=registry_number, top=5)
    rec = recs[0] if recs else None
    return {
        "найдено": bool(rec),
        "запись": rec,
        "страна": STATE_MEMBERS.get(str(rec.get("statemember", "")), "?") if rec else None,
        "производитель": rec.get("name") if rec else None,
        "окпд2": rec.get("goodsokpd2") if rec else None,
        "источник": f"{BASE}/{COLLECTION} (OData, снимок по запросу)",
    }


def fact_block(okpd2: str) -> str:
    """Готовый блок факта для статьи: число записей + примеры, с датой."""
    from datetime import date
    n = count_by_okpd2(okpd2)
    head = (f"Реестр промышленных товаров ЕАЭС (opendata.eaeunion.org, OData, "
            f"данные на {date.today().isoformat()}): ")
    if n is None:
        return head + "источник временно недоступен — цифру не приводить."
    lines = [head + f"по ОКПД2 {okpd2} — {n} записей."]
    for r in search(okpd2=okpd2, top=3):
        country = STATE_MEMBERS.get(str(r.get("statemember", "")), "?")
        lines.append(f"— «{str(r.get('goodsname', ''))[:60]}…», производитель: "
                     f"{str(r.get('name', ''))[:50]}, {country}, реестровая "
                     f"запись № {r.get('registrynumber')}.")
    lines.append("Реестр живой: при цитировании указывать дату обращения; "
                 "подтверждение происхождения для нацрежима — по ПП № 1875.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(fact_block("25.11"))
    print()
    print(json.dumps(check_registry_number("000009226"), ensure_ascii=False,
                     indent=1)[:800])
