"""
SSH-туннель до Qdrant на VPS (localhost:6333 внутри сервера).

Поднимает локальный TCP-сервер (по умолчанию 127.0.0.1:16333) и прокидывает
каждое соединение через SSH-канал direct-tcpip. Загрузчик kb2 при этом
работает с QDRANT_HOST=127.0.0.1, QDRANT_PORT=16333.

Параметры читаются из .env проекта:
  VPS_HOST, VPS_USER, VPS_PASSWORD, QDRANT_TUNNEL_PORT

Запуск:
  python scripts/kb2/qdrant_tunnel.py            # держит туннель до Ctrl+C
"""
from __future__ import annotations

import os
import select
import socket
import threading
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent


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


def start_tunnel(host: str, user: str, password: str, local_port: int,
                 remote_host: str = "127.0.0.1", remote_port: int = 6333):
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=20)
    transport = client.get_transport()
    transport.set_keepalive(30)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", local_port))
    server.listen(8)
    print(f"Туннель: 127.0.0.1:{local_port} -> {host}:{remote_host}:{remote_port}")

    def _pipe(src: socket.socket, dst) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except Exception:
            pass
        finally:
            try:
                src.close()
            except Exception:
                pass
            try:
                dst.close()
            except Exception:
                pass

    def _handle(conn: socket.socket) -> None:
        try:
            chan = transport.open_channel(
                "direct-tcpip", (remote_host, remote_port), conn.getpeername())
        except Exception as e:
            print(f"Канал не открыт: {e}")
            conn.close()
            return
        if chan is None:
            conn.close()
            return
        t1 = threading.Thread(target=_pipe, args=(conn, chan), daemon=True)
        t2 = threading.Thread(target=_pipe, args=(chan, conn), daemon=True)
        t1.start()
        t2.start()

    def _accept_loop() -> None:
        while True:
            conn, addr = server.accept()
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()

    threading.Thread(target=_accept_loop, daemon=True).start()
    return client, server


def main() -> None:
    _load_env()
    host = os.getenv("VPS_HOST", "")
    user = os.getenv("VPS_USER", "root")
    password = os.getenv("VPS_PASSWORD", "")
    local_port = int(os.getenv("QDRANT_TUNNEL_PORT", "16333"))
    if not host or not password:
        raise SystemExit("VPS_HOST/VPS_PASSWORD не заданы в .env")

    client, server = start_tunnel(host, user, password, local_port)
    print("Готово. Ctrl+C — закрыть.")
    try:
        while True:
            import time
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nЗакрываю туннель")
        server.close()
        client.close()


if __name__ == "__main__":
    main()
