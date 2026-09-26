"""
KB Update — демон обновления базы знаний.

Три функции в одном процессе:
1. Telegram-приёмник: владелец присылает боту файл (PDF/FB2/DOCX/ODT/TXT) —
   файл сохраняется в knowledge_base/<папка>, запускается загрузчик kb2,
   в чат приходит отчёт. Команды: /start /help /status /list.
2. Ежедневная проверка официальных публикаций (pravo.gov.ru) по расписанию.
3. Ежедневная hygiene: подсчёт/пометка устаревших отчётов (valid_until).

Запуск (на VPS, systemd: kb-update.service):
  python scripts/kb2/kb_update.py

Env: TELEGRAM_BOT_TOKEN (обязателен), TELEGRAM_CHAT_ID (узнаётся сам по /start),
     KB_ROOT, KB_COLLECTION, PRAVO_DAILY_URL (если задан — парсить список дня).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PROJECT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = PROJECT / "scripts"


def _load_env() -> None:
    env_file = PROJECT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("kb.update")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
KB_ROOT = Path(os.getenv("KB_ROOT", PROJECT / "knowledge_base"))
MANIFEST_PATH = KB_ROOT / "legislation" / "_manifest.json"
STATE_PATH = PROJECT / "scripts" / "kb2" / ".update_state.json"
PRAVO_DAILY_URL = os.getenv("PRAVO_DAILY_URL", "").strip()

API = f"https://api.telegram.org/bot{TOKEN}"
TG_TIMEOUT = 25  # long poll, сек

# Куда сохранять присланный файл: ключ в подписи -> подпапка knowledge_base/
ROUTES = {
    "закон": "legislation", "legislation": "legislation",
    "отчет": "business/reports", "report": "business/reports",
    "справочник": "business/reference", "reference": "business/reference",
    "seo": "craft/seo",
    "маркетинг": "business/marketing", "marketing": "business/marketing",
    "методология": "business/methodology", "письмо": "craft/writing",
}
DEFAULT_ROUTE = "legislation"

MONITOR_KEYWORDS = ["НК РФ", "ТК РФ", "ГК РФ", "КоАП", "115-ФЗ", "ФЗ-115", "152-ФЗ",
                    "402-ФЗ", "129-ФЗ", "44-ФЗ", "223-ФЗ", "209-ФЗ", "Минфин", "ФНС"]


# ============================================================
# Telegram API
# ============================================================

def tg_call(method: str, **params) -> Dict[str, Any]:
    try:
        r = requests.post(f"{API}/{method}", json=params, timeout=35)
        data = r.json()
        if not data.get("ok"):
            logger.warning(f"TG {method}: {data.get('description')}")
        return data
    except Exception as e:
        logger.warning(f"TG {method} сбой: {e}")
        return {"ok": False}


def notify(text: str, chat_id: Optional[str] = None) -> None:
    cid = chat_id or CHAT_ID
    if not cid:
        logger.info(f"[нет chat_id] {text}")
        return
    tg_call("sendMessage", chat_id=cid, text=text[:3900])


def download_tg_file(file_id: str, dest: Path) -> bool:
    data = tg_call("getFile", file_id=file_id)
    if not data.get("ok"):
        return False
    path = data["result"].get("file_path", "")
    if not path:
        return False
    try:
        r = requests.get(f"https://api.telegram.org/file/bot{TOKEN}/{path}", timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        logger.warning(f"Скачивание файла не удалось: {e}")
        return False


# ============================================================
# Manifest
# ============================================================

def load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"documents": {}}


def save_manifest(m: Dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rebuild_manifest_from_folder(m: Dict[str, Any]) -> int:
    """Синхронизировать манифест с файлами в legislation/ (одноразово)."""
    docs = m["documents"]
    added = 0
    folder = KB_ROOT / "legislation"
    if not folder.exists():
        return 0
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.suffix.lower() not in {".pdf", ".fb2", ".docx", ".odt", ".txt"}:
            continue
        key = f.name
        if key not in docs:
            docs[key] = {
                "file": key,
                "sha256": sha256_of(f),
                "edition_hint": next(iter(re.findall(r"(20\d{2})", f.name)), ""),
                "received": datetime.now().strftime("%Y-%m-%d"),
                "monitor": MONITOR_KEYWORDS,
            }
            added += 1
    if added:
        save_manifest(m)
    return added


# ============================================================
# Загрузчик (вызов kb2 loader как подпроцесса)
# ============================================================

def run_loader(folder: str) -> str:
    env = dict(os.environ)
    env.setdefault("QDRANT_HOST", "127.0.0.1")
    env.setdefault("QDRANT_PORT", "6333")
    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "kb2" / "loader.py"), "--folder", folder],
            capture_output=True, text=True, timeout=3600, env=env, encoding="utf-8", errors="replace",
        )
        tail = (proc.stdout or "").strip().splitlines()
        summary = next((l for l in reversed(tail) if "ГОТОВО" in l or "чанков" in l), "")
        return summary or (proc.stderr or "нет вывода").strip()[-300:]
    except Exception as e:
        return f"ошибка запуска загрузчика: {e}"


# ============================================================
# Ежедневные задачи
# ============================================================

def load_state() -> Dict[str, str]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(s: Dict[str, str]) -> None:
    STATE_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def daily_pravo_check(state: Dict[str, str]) -> None:
    """Проверка официального портала опубликования. Пока probe-режим:
    доступность + (если задан PRAVO_DAILY_URL) список документов дня."""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"📰 Проверка pravo.gov.ru за {today}:"]
    base = PRAVO_DAILY_URL or "https://publication.pravo.gov.ru/"
    try:
        r = requests.get(base, timeout=20, headers={"User-Agent": "kb-update/1.0"})
        lines.append(f"• портал доступен (HTTP {r.status_code})")
    except Exception as e:
        lines.append(f"• портал недоступен: {type(e).__name__}")
    notify("\n".join(lines))
    state["pravo_last_check"] = today
    save_state(state)


def daily_expiry_check(state: Dict[str, str]) -> None:
    """Отчёты с истёкшим valid_until уже исключаются фильтром rag.py,
    здесь — подсчёт для гигиены и уведомление."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, Range
        client = QdrantClient(
            url=f"http://{os.getenv('QDRANT_HOST', '127.0.0.1')}:{os.getenv('QDRANT_PORT', '6333')}",
            api_key=os.getenv("QDRANT_API_KEY") or None, timeout=30)
        collection = os.getenv("KB_COLLECTION", "kb_v2")
        flt = Filter(must=[FieldCondition(key="valid_until", range=Range(lt=today))])
        count = client.count(collection_name=collection, count_filter=flt, exact=True).count
        if count:
            notify(f"🕒 {count} точек отчётов устарели (valid_until < {today}) — из поиска они уже исключены фильтром.")
    except Exception as e:
        logger.warning(f"expiry check сбой: {e}")
    state["expiry_last_check"] = today
    save_state(state)


def daily_due(state: Dict[str, str], key: str) -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    return state.get(key) != today


# ============================================================
# Обработка обновлений Telegram
# ============================================================

def route_for(caption: str) -> str:
    cap = (caption or "").lower()
    for key, folder in ROUTES.items():
        if key in cap:
            return folder
    return DEFAULT_ROUTE


def handle_update(upd: Dict[str, Any]) -> None:
    global CHAT_ID
    msg = upd.get("message") or upd.get("document") or {}
    chat = msg.get("chat", {})
    cid = str(chat.get("id", ""))
    if not cid:
        return
    if not CHAT_ID:
        CHAT_ID = cid  # первый написавший становится владельцем
        logger.info(f"Зарегистрирован chat_id: {cid}")

    text = (msg.get("text") or "").strip()
    doc = msg.get("document")

    if text.startswith("/start"):
        m = load_manifest()
        rebuild_manifest_from_folder(m)
        tg_call("sendMessage", chat_id=cid, text=(
            "Бот базы знаний на связи.\n\n"
            "Пришли файл (PDF/FB2/DOCX/ODT/TXT) — залью в базу. По умолчанию в "
            "законодательство; укажи в подписи: «отчет», «справочник», «seo», "
            "«маркетинг», «методология» или «письмо» — положу в нужную папку.\n\n"
            "Команды: /status — состояние базы, /list — документы законодательства."))
        return

    if text.startswith("/status"):
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=f"http://{os.getenv('QDRANT_HOST', '127.0.0.1')}:{os.getenv('QDRANT_PORT', '6333')}",
                api_key=os.getenv("QDRANT_API_KEY") or None, timeout=30)
            info = client.get_collection(os.getenv("KB_COLLECTION", "kb_v2"))
            tg_call("sendMessage", chat_id=cid,
                    text=f"База {os.getenv('KB_COLLECTION', 'kb_v2')}: {info.points_count} точек, статус {info.status}.")
        except Exception as e:
            tg_call("sendMessage", chat_id=cid, text=f"Qdrant недоступен: {e}")
        return

    if text.startswith("/list"):
        m = load_manifest()
        docs = m.get("documents", {})
        if not docs:
            tg_call("sendMessage", chat_id=cid, text="Манифест пуст.")
            return
        lines = [f"• {k} (ред. ~{v.get('edition_hint', '?')}, получен {v.get('received', '?')})"
                 for k, v in sorted(docs.items())][:40]
        tg_call("sendMessage", chat_id=cid, text="Законодательство в базе:\n" + "\n".join(lines))
        return

    if doc:
        filename = doc.get("file_name", "document")
        if not re.search(r"\.(pdf|fb2|docx|odt|txt)$", filename.lower()):
            tg_call("sendMessage", chat_id=cid, text="Формат не поддерживаю. Нужен PDF/FB2/DOCX/ODT/TXT.")
            return
        folder = route_for(msg.get("caption", ""))
        dest_dir = KB_ROOT / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        tg_call("sendMessage", chat_id=cid, text=f"Принял {filename}, качаю…")
        if not download_tg_file(doc["file_id"], dest):
            tg_call("sendMessage", chat_id=cid, text="Не удалось скачать файл, попробуй ещё раз.")
            return
        if folder == DEFAULT_ROUTE:
            m = load_manifest()
            m["documents"][filename] = {
                "file": filename,
                "sha256": sha256_of(dest),
                "edition_hint": next(iter(re.findall(r"(20\d{2})", filename)), ""),
                "received": datetime.now().strftime("%Y-%m-%d"),
                "monitor": MONITOR_KEYWORDS,
            }
            save_manifest(m)
        summary = run_loader(folder)
        notify(f"✅ {filename} → {folder}\n{summary}", chat_id=cid)
        return

    if text:
        tg_call("sendMessage", chat_id=cid, text="Пришли файл или используй /status, /list.")


# ============================================================
# Main loop
# ============================================================

def main() -> int:
    if not TOKEN:
        print("TELEGRAM_BOT_TOKEN не задан")
        return 1
    logger.info("KB Update демон запущен")
    notify("🟢 Демон обновления базы знаний запущен.")

    m = load_manifest()
    if rebuild_manifest_from_folder(m):
        logger.info("Манифест инициализирован из файлов legislation/")

    offset = 0
    state = load_state()
    while True:
        # Ежедневные задачи
        try:
            if daily_due(state, "pravo_last_check") and datetime.now().hour >= 9:
                daily_pravo_check(state)
            if daily_due(state, "expiry_last_check") and datetime.now().hour >= 9:
                daily_expiry_check(state)
        except Exception as e:
            logger.warning(f"ежедневная задача сбой: {e}")

        # Long poll
        try:
            r = requests.get(f"{API}/getUpdates",
                             params={"timeout": TG_TIMEOUT, "offset": offset, "limit": 10},
                             timeout=TG_TIMEOUT + 10)
            data = r.json()
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                try:
                    handle_update(upd)
                except Exception as e:
                    logger.exception(f"обработка обновления: {e}")
        except requests.RequestException:
            time.sleep(5)
        except Exception as e:
            logger.warning(f"getUpdates сбой: {e}")
            time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
