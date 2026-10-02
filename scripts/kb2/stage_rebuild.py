"""Стадирование финального rebuild-скрипта на VPS (после докачки корпуса).

Запуск: python scripts/kb2/stage_rebuild.py
"""
import os

import paramiko

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env = {}
with open(os.path.join(PROJECT, ".env"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

REBUILD_SH = """#!/bin/bash
# Полный пересбор kb_v2 из корпуса на диске (после инцидента 2026-09-29).
# Без само-kill: pkill бьёт только загрузчик, не себя.
exec 9>/root/kb_rebuild.lock
if ! flock -n 9; then echo "rebuild already running"; exit 0; fi
set -x
# остановить любые одиночные загрузчики, если живы
pkill -9 -f 'loader.py --folder' 2>/dev/null
sleep 2
rm -f /root/kb_load.lock
# чекпойнты обнуляем один раз перед первым прогоном
rm -f /root/Copywriter1/scripts/copywriter_kb/checkpoints/*.json
cd /root/Copywriter1
# попытка 1: полный сброс и залив всего
timeout 7200 venv/bin/python scripts/kb2/loader.py --folder all --fresh
rc=$?
# ретраи без --fresh: докачка с чекпойнтами (uuid5 — идемпотентно)
for i in 2 3 4 5 6; do
  if [ $rc -eq 0 ]; then break; fi
  sleep 30
  timeout 7200 venv/bin/python scripts/kb2/loader.py --folder all
  rc=$?
done
curl -s -H "api-key: $(grep ^QDRANT_API_KEY= /root/Copywriter1/.env | cut -d= -f2)" \\
  http://127.0.0.1:6333/collections/kb_v2 | head -c 300
rm -f /root/kb_load_once.sh
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(env["VPS_HOST"], username=env["VPS_USER"], password=env["VPS_PASSWORD"], timeout=25)
sftp = c.open_sftp()
with sftp.open("/root/kb_load_once.sh", "w") as f:
    f.write(REBUILD_SH)
sftp.chmod("/root/kb_load_once.sh", 0o755)
print("rebuild script staged at /root/kb_load_once.sh")
sftp.close()
c.close()
