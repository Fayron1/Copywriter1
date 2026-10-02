"""Докачка локального корпуса БЗ на VPS (resumable, с переподключением).

Запуск: python scripts/kb2/upload_kb_corpus.py
Скипает файлы, уже лежащие на VPS с совпадающим размером.
"""
import os
import time

import paramiko

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env = {}
with open(os.path.join(PROJECT, ".env"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

REMOTE_ROOT = "/root/Copywriter1/knowledge_base"
LOCAL_ROOT = os.path.join(PROJECT, "knowledge_base")


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(env["VPS_HOST"], username=env["VPS_USER"],
              password=env["VPS_PASSWORD"], timeout=25)
    return c


def ensure_dir(sftp, path):
    parts = path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def main():
    c = connect()
    sftp = c.open_sftp()
    total = ok = 0

    for folder in ("business", "craft", "style_client", "legislation"):
        local_dir = os.path.join(LOCAL_ROOT, folder)
        if not os.path.isdir(local_dir):
            continue
        for root, _dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, LOCAL_DIR_BASE := LOCAL_ROOT)
            rdir = f"{REMOTE_ROOT}/{rel.replace(os.sep, '/')}".replace("\\", "/")
            ensure_dir(sftp, rdir)
            for fn in sorted(files):
                if fn.endswith((".md",)) and folder == "legislation":
                    continue  # loader не берёт .md — уже есть .txt версия
                lp = os.path.join(root, fn)
                lsize = os.path.getsize(lp)
                rp = f"{rdir}/{fn}"
                total += 1
                try:
                    try:
                        if sftp.stat(rp).st_size == lsize:
                            ok += 1
                            continue
                    except FileNotFoundError:
                        pass
                    sftp.put(lp, rp)
                    ok += 1
                    print("up:", rel.replace(os.sep, "/") + "/" + fn, lsize, flush=True)
                except Exception as e:
                    # переподключение и одна повторная попытка
                    print("reconnect after:", fn, type(e).__name__, flush=True)
                    try:
                        c.close()
                    except Exception:
                        pass
                    time.sleep(8)
                    c = connect()
                    sftp = c.open_sftp()
                    try:
                        sftp.put(lp, rp)
                        ok += 1
                        print("up (retry):", fn, flush=True)
                    except Exception as e2:
                        print("FAILED:", fn, str(e2)[:80], flush=True)

    print(f"DONE: {ok}/{total} files in place", flush=True)
    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
