#!/usr/bin/env bash
# deploy.sh — Update and restart the Groningen University AI Compute Depot
# Usage: sudo bash deploy.sh

set -euo pipefail

APP_DIR="/opt/gridweave-depot"
APP_NAME="gridweave-depot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run with sudo: sudo bash deploy.sh"
    exit 1
fi

echo "==> Deploying ${APP_NAME}…"
cp "${SCRIPT_DIR}/app.py" "${APP_DIR}/app.py"

echo "==> Restarting service…"
systemctl restart "${APP_NAME}"

sleep 2
STATUS=$(systemctl is-active "${APP_NAME}" || true)
echo "==> Status: ${STATUS}"

if [[ "${STATUS}" != "active" ]]; then
    echo "ERROR: service did not start. Check logs:"
    echo "  sudo journalctl -u ${APP_NAME} -n 50 --no-pager"
    exit 1
fi

echo "==> Done. Running at http://$(hostname -I | awk '{print $1}'):$(systemctl show ${APP_NAME} -p ExecStart --value | grep -oP '(?<=--server.port )\d+' || echo 8501)"
