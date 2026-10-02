"""Вочдог пересбора kb_v2 (v2): различает ЗАВЕРШЕНИЕ и ПРОПАЖУ скрипта.

Завершение = скрипт удалён И в логе есть CONTINUATION_COMPLETE.
Во всех остальных случаях «script gone» — аномалия: доложить, НЕ трогать монитор.
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


def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(env["VPS_HOST"], username=env["VPS_USER"],
              password=env["VPS_PASSWORD"], timeout=25)
    return c


_c0 = ssh()
_s0 = _c0.open_sftp()
with _s0.open("/root/Copywriter1/.env") as f:
    API_KEY = dict(l.strip().split("=", 1) for l in f.read().decode().splitlines()
                   if "=" in l and not l.startswith("#")).get("QDRANT_API_KEY", "")
_s0.close()
_c0.close()


def count(c):
    import http.client

    ch = c.get_transport().open_channel("direct-tcpip", ("127.0.0.1", 6333), ("127.0.0.1", 0))

    class Sub(http.client.HTTPConnection):
        def connect(self):
            self.sock = ch

    conn = Sub("127.0.0.1", 6333)
    conn.request("GET", "/collections/kb_v2", headers={"api-key": API_KEY})
    return json.loads(conn.getresponse().read()).get("result", {}).get("points_count")


def snapshot():
    c = ssh()
    sftp = c.open_sftp()
    running = False
    for script in ("/root/kb_continue.sh", "/root/kb_round9.sh",
                   "/root/kb_round10.sh"):
        try:
            sftp.stat(script)
            running = True
        except FileNotFoundError:
            pass
    n = count(c)
    complete = False
    try:
        with sftp.open("/root/kb_load_kedo.log") as f:
            complete = "CONTINUATION_COMPLETE" in f.read().decode(errors="replace")[-4000:]
    except FileNotFoundError:
        pass
    sftp.close()
    c.close()
    return running, n, complete


deadline = time.time() + 10 * 3600
last, stale = 0, 0
while time.time() < deadline:
    time.sleep(1500)
    try:
        running, n, complete = snapshot()
        tag = "running" if running else "script gone"
        note = " | stall!" if (n == last and running) else ""
        print(time.strftime("%H:%M"), "points:", n, "|", tag, note, flush=True)
        stale = stale + 1 if (n == last and running) else 0
        last = n
        if not running and complete:
            print("REBUILD COMPLETE, points:", n, flush=True)
            # монитор уже мог быть восстановлен; триггер-строку убираем
            c = ssh()
            sftp = c.open_sftp()
            with open("aura_monitor_original.sh", encoding="utf-8") as f:
                original = f.read()
            with sftp.open("/root/my-server/aura_monitor.sh", "w") as f:
                f.write(original)
            sftp.close()
            c.close()
            print("aura_monitor.sh restored to original", flush=True)
            break
        if not running and not complete:
            print("ANOMALY: script gone without COMPLETE — needs staging", flush=True)
            break
        if stale >= 10:
            print("STALL 5h — attention", flush=True)
            break
    except Exception as e:
        print("check fail:", type(e).__name__, flush=True)
