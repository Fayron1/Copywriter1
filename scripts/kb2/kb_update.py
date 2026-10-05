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

import subprocess
import threading
import zipfile

PROJECT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = PROJECT / "scripts"
VENV_PYTHON = PROJECT / "venv" / "bin" / "python"
OUTPUT_ROOT = PROJECT / "output"
TOPICS_MD = PROJECT / "topics.md"


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


def tg_send_document(chat_id: str, path: Path, caption: str = "") -> bool:
    """Отправить файл в чат (multipart)."""
    try:
        with open(path, "rb") as f:
            r = requests.post(
                f"{API}/sendDocument",
                data={"chat_id": chat_id, "caption": caption[:1000]},
                files={"document": (path.name, f)},
                timeout=180,
            )
        return bool(r.json().get("ok"))
    except Exception as e:
        logger.warning(f"sendDocument сбой: {e}")
        return False


# ============================================================
# Ручная генерация статей /gen <тема>
# ============================================================

_GEN_RUNNING = [False]


def _newest_output_dir(before: set) -> Optional[Path]:
    """Самый свежий каталог в output/, которого не было до старта."""
    if not OUTPUT_ROOT.exists():
        return None
    candidates = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir() and d.name not in before]
    if not candidates:
        candidates = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _generation_worker(topic: str, cid: str) -> None:
    _GEN_RUNNING[0] = True
    try:
        before = {d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()} if OUTPUT_ROOT.exists() else set()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Аудит-2 K1: автоотправка воркера отключена (--no-send) — иначе
        # generate шлёт сам (через гейт), а воркер слал ВТОРОЙ ZIP мимо гейта.
        cmd = [str(VENV_PYTHON), str(SCRIPTS / "generate.py"), topic,
               "--chars", "8000", "-p", "deepseek", "--no-send"]
        logger.info(f"[gen] старт: {topic[:80]}")
        proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True,
                              timeout=3600, env=env, encoding="utf-8", errors="replace")
        out_dir = _newest_output_dir(before)
        if proc.returncode != 0 or out_dir is None:
            tail = (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:]
            tg_call("sendMessage", chat_id=cid,
                    text=f"❌ Генерация не удалась (код {proc.returncode}).\n{tail[-700:]}")
            return
        # Аудит-2 K1: собственный ZIP воркера проходит тот же красный гейт,
        # что и автоотправка. Красные замечания = статья не уходит.
        sys.path.insert(0, str(SCRIPTS))
        from agents.autofix_gate import gate, fix_dir
        fix_dir(str(out_dir))
        reds = gate((out_dir / "article.md").read_text(encoding="utf-8"),
                    str(out_dir / "gate_markers.json"))
        if reds:
            red_lines = "\n".join("• " + str(x)[:140] for x in reds[:4])
            tg_call("sendMessage", chat_id=cid,
                    text=(f"🔴 ГЕЙТ: {len(reds)} красных замечаний — статья не "
                          f"отправлена, нужна правка:\n{red_lines}"))
            logger.warning(f"[gen] гейт заблокировал отправку: {len(reds)} красных")
            return
        zip_path = PROJECT / f"article_{out_dir.name[:40]}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out_dir.iterdir()):
                if f.is_file():
                    zf.write(f, arcname=f.name)
        ok = tg_send_document(cid, zip_path, caption=f"✅ Статья готова: «{topic[:80]}»")
        if not ok:
            tg_call("sendMessage", chat_id=cid, text="❌ Архив собран, но не отправился. Проверь вручную: " + str(zip_path))
        else:
            zip_path.unlink(missing_ok=True)
        logger.info(f"[gen] готово: {out_dir.name}")
    except Exception as e:
        logger.exception(f"[gen] сбой: {e}")
        tg_call("sendMessage", chat_id=cid, text=f"❌ Сбой генерации: {type(e).__name__}: {e}"[:900])
    finally:
        _GEN_RUNNING[0] = False


def start_generation(topic: str, cid: str) -> None:
    if _GEN_RUNNING[0]:
        tg_call("sendMessage", chat_id=cid, text="⏳ Уже идёт другая генерация — дождись архива.")
        return
    tg_call("sendMessage", chat_id=cid,
            text=(f"🚀 Запускаю генерацию:\n«{topic[:180]}»\n\n"
                  "Займёт ~10–20 минут. ZIP со статьёй (HTML + MD + паспорт + SEO) придёт сюда."))
    threading.Thread(target=_generation_worker, args=(topic, cid), daemon=True).start()


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
    """Ежедневная проверка источников. pravo.gov.ru фильтрует зарубежные
    датацентр-IP (наш VPS) — фиксируем статус один раз, не тревожим владельца.
    Доступные официальные источники: nalog.gov.ru, consultant.ru, garant.ru,
    minfin.gov.ru (проверены 2026-09-27)."""
    today = datetime.now().strftime("%Y-%m-%d")
    reachable = []
    for url in ["https://www.consultant.ru/", "https://www.garant.ru/",
                "https://minfin.gov.ru/", "https://www.nalog.gov.ru/"]:
        try:
            r = requests.head(url, timeout=10, headers={"User-Agent": "kb-update/1.0"},
                              allow_redirects=True)
            if r.status_code < 500:
                reachable.append(url.split("/")[2])
        except Exception:
            pass
    logger.info(f"[sources] доступны: {reachable or 'НИЧЕГО'}")
    if not reachable and state.get("sources_alert") != today:
        notify("⚠️ Ни один правовой источник не отвечает с VPS — проверь сеть.")
        state["sources_alert"] = today
    state["pravo_last_check"] = today
    save_state(state)


# Мониторимые документы для наблюдателя поправок (через Vane/Яндекс).
# Приоритет источников по группам (рекомендация от 2026-09-27):
#   кодексы/ФЗ — pravo.gov.ru (недоступен с VPS) -> base.garant.ru / consultant.ru;
#   налоговые разъяснения — nalog.gov.ru / minfin.gov.ru;
#   судебная практика — vsrf.ru.
# base.garant.ru доступен с VPS и отдаёт полные тексты (cp1251) — основной
# источник автоскачивания консолидированных редакций.
AMENDMENT_WATCH_DOCS = [
    {"name": "НК РФ", "query": "НК РФ", "sources": "garant.ru, consultant.ru, nalog.gov.ru"},
    {"name": "ТК РФ", "query": "ТК РФ", "sources": "garant.ru, consultant.ru"},
    {"name": "ГК РФ", "query": "ГК РФ", "sources": "garant.ru, consultant.ru"},
    {"name": "КоАП РФ", "query": "КоАП РФ", "sources": "garant.ru, consultant.ru"},
    {"name": "152-ФЗ", "query": "152-ФЗ о персональных данных",
     "sources": "garant.ru, consultant.ru, roskomnadzor.gov.ru"},
    {"name": "44-ФЗ / 223-ФЗ", "query": "44-ФЗ и 223-ФЗ закупки",
     "sources": "garant.ru, consultant.ru, zakupki.gov.ru"},
    {"name": "письма ФНС и Минфина", "query": "новые письма ФНС и Минфина налоги",
     "sources": "nalog.gov.ru, minfin.gov.ru, consultant.ru"},
]
AMENDMENT_WATCH_ENABLED = os.getenv("AMENDMENT_WATCH", "1").strip().lower() not in ("0", "false", "no")


def daily_amendment_watch(state: Dict[str, str]) -> None:
    """Наблюдатель поправок: Vane (Яндекс-метапоиск + синтез с цитатами) по каждому
    мониторимому документу. Уведомляем ТОЛЬКО при находках с датами текущего/будущего
    месяца. Тишина = всё спокойно (еженедельного дайджеста нет — меньше шума)."""
    today = datetime.now().strftime("%Y-%m-%d")
    month_name = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
                  "август", "сентябрь", "октябрь", "ноябрь", "декабрь"][datetime.now().month - 1]
    findings = []
    try:
        sys.path.insert(0, str(SCRIPTS))
        from agents.vane import research as vane_research
        for doc in AMENDMENT_WATCH_DOCS:
            try:
                res = vane_research(
                    f"Приняты ли новые поправки в {doc['query']} в {month_name} {datetime.now().year} года? "
                    f"Только свежие изменения с датами и номерами законов. "
                    f"Приоритетные источники: {doc['sources']}.")
                msg = (res.get("message") or "").strip()
                # Эвристика свежести: месяц/год или фразы о принятии
                fresh = (month_name in msg.lower()
                         or f"{datetime.now().year}" in msg)
                if msg and fresh and "не удалось" not in msg[:60]:
                    findings.append(f"📄 {doc['name']}:\n{msg[:700]}")
            except Exception as e:
                logger.warning(f"[watch] {doc['name']}: {e}")
    except ImportError:
        logger.warning("[watch] agents.vane недоступен — наблюдатель пропущен")

    if findings:
        notify("🔔 Наблюдатель законодательства: возможные свежие поправки.\n\n"
               + "\n\n".join(findings)[:3500]
               + "\n\nПроверь и при необходимости пришли свежую редакцию файла мне. "
                 "Свежие редакции удобно брать на base.garant.ru (полные тексты).")
    state["watch_last_check"] = today
    save_state(state)


def daily_expiry_check(state: Dict[str, str]) -> None:
    """Отчёты с истёкшим valid_until уже исключаются фильтром rag.py,
    здесь — подсчёт для гигиены и уведомление."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models
        client = QdrantClient(
            url=f"http://{os.getenv('QDRANT_HOST', '127.0.0.1')}:{os.getenv('QDRANT_PORT', '6333')}",
            api_key=os.getenv("QDRANT_API_KEY") or None, timeout=30)
        collection = os.getenv("KB_COLLECTION", "kb_v2")
        # Даты: DatetimeRange; на старых версиях клиента — Range с датой в значении
        try:
            rng = models.DatetimeRange(lt=today)
        except AttributeError:
            rng = models.Range(lt=today)
        # Аудит-2 N2: Filter/FieldCondition не импортированы поимённо ->
        # NameError глотался except-ом, expiry check был мёртв.
        flt = models.Filter(must=[models.FieldCondition(key="valid_until", range=rng)])
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
    elif cid != CHAT_ID:
        # ФИКС (аудит 🔴3): чужой чат не может заливать файлы (отравление
        # RAG) и запускать /gen за наш счёт
        logger.warning(f"ЧУЖОЙ chat_id {cid} — игнорирую")
        return

    text = (msg.get("text") or "").strip()
    doc = msg.get("document")

    if text.startswith("/start"):
        m = load_manifest()
        rebuild_manifest_from_folder(m)
        tg_call("sendMessage", chat_id=cid, text=(
            "Бот базы знаний на связи.\n\n"
            "📝 /gen <тема> — запустить генерацию статьи (из списка /topics или свою); "
            "ZIP придёт сюда через ~10–20 минут.\n"
            "📋 /topics — план статей (md-файл).\n"
            "📤 Просто пришли файл (PDF/FB2/DOCX/TXT) — залью в базу. По умолчанию в "
            "законодательство; в подписи укажи: «отчет», «справочник», «seo», "
            "«маркетинг», «методология» или «письмо».\n"
            "📊 /status — состояние базы, /list — документы законодательства."))
        return

    if text.startswith("/topics"):
        if TOPICS_MD.exists():
            tg_send_document(cid, TOPICS_MD, caption="План статей, волна 1 (30 тем). /gen <тема> — запуск генерации.")
        else:
            tg_call("sendMessage", chat_id=cid, text="topics.md не найден на сервере.")
        return

    if text.startswith("/gen"):
        topic = text[4:].strip()
        if not topic:
            tg_call("sendMessage", chat_id=cid,
                    text="Формат: /gen <тема>. Список тем: /topics")
            return
        start_generation(topic, cid)
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
        # ФИКС (аудит 🔴4): path traversal — «../../x.pdf» пишет вне KB_ROOT
        filename = os.path.basename(filename.replace("\\", "/"))
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
                # Метаданные провенанса (рекомендация по контролю актуальности):
                # заполняются из подписи к файлу в формате «source: URL» либо вручную.
                "source_url": next(iter(re.findall(r"source:\s*(https?://\S+)", msg.get("caption", "") or "")), ""),
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
            if AMENDMENT_WATCH_ENABLED and daily_due(state, "watch_last_check") and datetime.now().hour >= 9:
                # Аудит-1: Vane-исследование занимает минуты сети и блокировало
                # long-poll. Запуск в фоновом потоке на копии state.
                _st2 = dict(state)
                threading.Thread(target=daily_amendment_watch, args=(_st2,),
                                 daemon=True).start()
                state["watch_last_check"] = datetime.now().strftime("%Y-%m-%d")
                save_state(state)
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
