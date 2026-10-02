"""Отправка готовой статьи в Telegram-бота (универсальная).

Запуск на VPS:
    cd /root/Copywriter1 && venv/bin/python scripts/kb2/send_article_to_bot.py /root/article16
"""
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from dotenv import load_dotenv
load_dotenv(PROJECT / ".env")

import os
import requests

API = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN', '')}"
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")


def tg_call(method: str, **params) -> dict:
    r = requests.post(f"{API}/{method}", json=params, timeout=35)
    return r.json()


def main(base: str):
    art_dir = sorted(d for d in (Path(base)).iterdir() if d.is_dir())[-1]
    chat = CHAT
    if not chat:
        raise SystemExit("TELEGRAM_CHAT_ID не задан")

    # ГЕЙТ ПУБЛИКАЦИИ (2026-10-02): автофикс известных классов ошибок +
    # блокировка при оставшихся 🔴. Статья с красными замечаниями в бот
    # не уходит (кейс v5: «30 календарных дней» доехало до ревью).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    try:
        from agents.autofix_gate import autofix, gate, fix_dir
        # Кейс 26: article.md вылечен, а article.html в архиве остался
        # старым — ревьюер читал HTML. Фикс на ВСЕ текстовые файлы пакета.
        _, fix_log = fix_dir(str(art_dir))
        for fl in fix_log:
            print("  ✓", fl)
        article = (art_dir / "article.md").read_text(encoding="utf-8")
        html = (art_dir / "article.html")
        reds = gate(article, str(art_dir / "gate_markers.json"))
        if html.exists():
            # сырой HTML ломает линтер шаблонов (<head>/<title> = «переменные»)
            _html_text = html.read_text(encoding="utf-8")
            _html_clean = re.sub(r"<[^>]+>", " ", _html_text)
            reds += gate(_html_clean, str(art_dir / "gate_markers.json"))
        if reds:
            print(f"🔴 ГЕЙТ: {len(reds)} красных замечаний — отправка ОТМЕНЕНА:")
            for r in reds[:6]:
                print("  •", str(r)[:140])
            raise SystemExit(1)
    except ImportError:
        pass

    article = (art_dir / "article.md").read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.+)$", article, re.M)
    title = m.group(1).strip() if m else art_dir.name

    body = re.sub(r"^---.*?---", "", article, flags=re.S)
    chars = len(body)
    has_pkg = (art_dir / "content_package.md").exists()

    cover = (
        f"📄 Статья готова: {title}\n\n"
        f"• {chars:,} символов\n"
        f"• Контент-пакет: {'в комплекте (SEO, meta, FAQ, TG-посты, паспорт)' if has_pkg else 'нет'}\n"
        f"• CTA-футер: {'в конце статьи' if 'Обсудить' in article or 'контент-пакет' in article.lower() else 'выключен'}\n\n"
        f"База: kb_v2 (полная, после пересбора). Ревью по файлам из архива."
    )
    ok1 = tg_call("sendMessage", chat_id=chat, text=cover).get("ok")
    print("sendMessage:", ok1)

    # Уникальное имя архива: mtime статьи в имени — в истории чата
    # несколько зипов с одинаковым именем неразличимы (кейс 26: юзер
    # открыл первый из трёх)
    _mt = max(f.stat().st_mtime for f in art_dir.iterdir() if f.is_file())
    _mt = datetime.fromtimestamp(_mt)
    zip_path = art_dir.parent / f"article_{art_dir.name[:34]}_{_mt:%m%d_%H%M}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in art_dir.iterdir():
            if f.is_file() and f.suffix in (".md", ".html", ".txt", ".json"):
                z.write(f, f.name)
    with open(zip_path, "rb") as fh:
        r = requests.post(
            f"{API}/sendDocument",
            data={"chat_id": chat,
                  "caption": f"Статья + контент-пакет ({art_dir.name[:25]})"},
            files={"document": (zip_path.name, fh, "application/zip")},
            timeout=90)
    print("sendDocument:", r.json().get("ok"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/article15")
