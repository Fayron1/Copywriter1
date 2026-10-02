"""Оркестратор: две статьи последовательно + автоотправка каждой в бота.

Статья 17: Страховые взносы ИП 2026 (налоги)
Статья 18: Оборотные штрафы за утечки ПДн (защита данных)

Запуск локально: python scripts/kb2/run_articles_17_18.py
"""
import json
import time

import paramiko

env = {}
with open(".env", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

ARTICLES = [
    ("17", "Страховые взносы ИП 2026: таблица лимитов, ставок и сроков", "налоги"),
    ("18", "Оборотные штрафы до 3% выручки за повторные утечки данных: кто в зоне риска", "защита данных"),
]


def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(env["VPS_HOST"], username=env["VPS_USER"],
              password=env["VPS_PASSWORD"], timeout=25)
    c.get_transport().set_keepalive(10)
    return c


def run(c, cmd, wait=6, timeout=60):
    """Выполнить команду, вернуть вывод (с ретраем канала)."""
    for attempt in range(3):
        try:
            chan = c.get_transport().open_session(timeout=15)
            chan.settimeout(timeout)
            chan.exec_command(cmd)
            time.sleep(wait)
            out = b""
            while True:
                try:
                    d = chan.recv(65536)
                except Exception:
                    break
                if not d:
                    break
                out += d
            return out.decode(errors="replace")
        except Exception:
            time.sleep(5)
    return ""


def gen_running(c):
    # 'generat[e].py' — regex-трюк: паттерн не матчит саму проверочную команду
    out = run(c, "pgrep -fc 'scripts/generat[e].py'", wait=3)
    try:
        return int(out.strip().splitlines()[-1]) > 0
    except Exception:
        return True  # не выяснили — считаем, что работает


def article_exists(c, num):
    out = run(c, f"ls /root/article{num}/*/article.md 2>/dev/null | head -1", wait=3)
    return "article.md" in out


def send_to_bot(c, num):
    out = run(c, f"cd /root/Copywriter1 && timeout 110 venv/bin/python "
                 f"scripts/kb2/send_article_to_bot.py /root/article{num} 2>&1",
              wait=45, timeout=110)
    return out


def run_article(num, topic, direction):
    print(f"=== СТАТЬЯ {num}: {topic} ===", flush=True)
    c = ssh()
    # запуск (nohup, detached)
    run(c, f"cd /root/Copywriter1 && nohup venv/bin/python scripts/generate.py "
           f"'{topic}' --dir '{direction}' --output /root/article{num} "
           f"> /root/test_article{num}.log 2>&1 < /dev/null & echo STARTED")
    print(time.strftime("%H:%M"), f"article {num} launched", flush=True)
    c.close()

    # ожидание до 75 минут
    deadline = time.time() + 4500
    saw_running = False
    while time.time() < deadline:
        time.sleep(180)
        try:
            c = ssh()
            running = gen_running(c)
            exists = article_exists(c, num)
            c.close()
            if running:
                saw_running = True
            print(time.strftime("%H:%M"), f"article {num}: running={running} md={exists}", flush=True)
            if saw_running and not running and exists:
                time.sleep(15)  # дозапись файлов
                break
        except Exception as e:
            print("check fail:", type(e).__name__, flush=True)

    # отправка
    try:
        c = ssh()
        result = send_to_bot(c, num)
        c.close()
        print(f"SEND {num}:", result.strip().replace(chr(10), " | ")[:200], flush=True)
    except Exception as e:
        print(f"SEND {num} FAIL:", type(e).__name__, flush=True)

    # пауза между статьями
    time.sleep(30)


for num, topic, direction in ARTICLES:
    run_article(num, topic, direction)

print("ALL ARTICLES DONE", flush=True)
