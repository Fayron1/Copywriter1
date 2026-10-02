#!/bin/bash
# ============================================================
# aura_monitor.sh — проверка здоровья системы Aura раз в 5 минут.
#
# Проверяет: systemd aura-bot / aura-bot-v2, docker-контейнеры,
# HTTP /health у mcp (8000) и core (8001). При деградации — Telegram
# сообщение владельцу (не чаще раза в 30 мин на компонент). При
# восстановлении — сообщение «восстановлено».
#
# Cron: */5 * * * * /root/my-server/aura_monitor.sh >/dev/null 2>&1
# ============================================================
set -u

ENV_FILE="/root/my-server/.env"
STATE_DIR="/root/backups/aura/monitor"
mkdir -p "$STATE_DIR"

# Токен и admin_id берём из .env сервера (значения не светим в процессах)
BOT_TOKEN=$(grep -E '^BOT_TOKEN=' "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
ADMIN_ID=$(grep -E '^ADMIN_ID=' "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"')

if [ -z "$BOT_TOKEN" ] || [ -z "$ADMIN_ID" ]; then
    echo "monitor: BOT_TOKEN/ADMIN_ID не найдены в $ENV_FILE" >&2
    exit 1
fi

ALERT_COOLDOWN=1800  # 30 мин между повторными алертами на компонент

alert() {
    local component="$1" msg="$2"
    local state_file="$STATE_DIR/$component"
    local now=$(date +%s)
    local last=0
    [ -f "$state_file" ] && last=$(cat "$state_file" 2>/dev/null || echo 0)
    if [ $((now - last)) -lt "$ALERT_COOLDOWN" ]; then
        return
    fi
    echo "$now" > "$state_file"
    curl -s -m 10 -o /dev/null "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="$ADMIN_ID" --data-urlencode text="$msg" || true
    logger -t aura_monitor "ALERT: $component — $msg"
}

recover() {
    local component="$1" msg="$2"
    local state_file="$STATE_DIR/$component"
    [ -f "$state_file" ] || return 0   # не было алерта — не шлём recovery
    rm -f "$state_file"
    curl -s -m 10 -o /dev/null "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="$ADMIN_ID" --data-urlencode text="$msg" || true
    logger -t aura_monitor "RECOVERED: $component"
}

# --- 1. systemd-боты ---
for unit in aura-bot aura-bot-v2; do
    if systemctl is-active --quiet "$unit"; then
        recover "$unit" "✅ $unit восстановился и работает."
    else
        alert "$unit" "🚨 Aura: $unit НЕ АКТИВЕН. systemd должен перезапустить сам; если алерт повторяется — зайди на сервер: systemctl status $unit"
    fi
done

# --- 2. Docker-контейнеры ---
for c in aura_mcp_server aura_core aura_v2_mcp aura_v2_core qdrant_service; do
    state=$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null || echo "missing")
    if [ "$state" = "running" ]; then
        recover "docker_$c" "✅ Контейнер $c восстановлен."
    else
        alert "docker_$c" "🚨 Aura: контейнер $c в состоянии '$state'. Логи: docker logs --tail 50 $c"
    fi
done

# --- 3. HTTP health mcp/core (короткий таймаут, чтобы не висеть) ---
if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    recover "http_mcp" "✅ aura-mcp отвечает на /health."
else
    alert "http_mcp" "🚨 Aura: aura-mcp не отвечает на /health (127.0.0.1:8000). Карты/расклады не работают."
fi

if curl -sf -m 5 http://127.0.0.1:8001/health >/dev/null 2>&1; then
    recover "http_core" "✅ aura-core отвечает на /health."
else
    alert "http_core" "🚨 Aura: aura-core не отвечает на /health (127.0.0.1:8001). Чат с Аурой не работает."
fi

# --- 4. Свободное место (алерт при < 5GB) ---
AVAIL_KB=$(df -k / | awk 'NR==2 {print $4}')
if [ "$AVAIL_KB" -lt 5242880 ]; then
    alert "disk" "🚨 Aura: на диске осталось $((AVAIL_KB / 1048576)) GB. Пора чистить."
else
    recover "disk" "✅ Место на диске восстановилось: $((AVAIL_KB / 1048576)) GB."
fi

exit 0
