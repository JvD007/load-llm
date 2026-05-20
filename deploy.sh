#!/usr/bin/env bash
# deploy.sh — Update and restart the Groningen University AI Compute Depot
# Usage: sudo bash deploy.sh [--install-hook]
#   --install-hook  Install the pre-push git hook for auto-deploy on push

set -euo pipefail

APP_DIR="/opt/gridweave-depot"
APP_NAME="gridweave-depot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install git hook if requested (does not require root)
if [[ "${1:-}" == "--install-hook" ]]; then
    HOOK_SRC="${SCRIPT_DIR}/hooks/pre-push"
    HOOK_DST="${SCRIPT_DIR}/.git/hooks/pre-push"
    cp "${HOOK_SRC}" "${HOOK_DST}"
    chmod +x "${HOOK_DST}"
    echo "==> pre-push hook installed. Auto-deploy will run on: git push origin main"
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run with sudo: sudo bash deploy.sh"
    echo "       To install the git hook: bash deploy.sh --install-hook"
    exit 1
fi

echo "==> Deploying ${APP_NAME}…"
/usr/bin/cp "${SCRIPT_DIR}/app.py" "${APP_DIR}/app.py"

echo "==> Restarting service…"
/usr/bin/systemctl restart "${APP_NAME}"

sleep 2
STATUS=$(/usr/bin/systemctl is-active "${APP_NAME}" || true)
echo "==> Status: ${STATUS}"

if [[ "${STATUS}" != "active" ]]; then
    echo "ERROR: service did not start. Check logs:"
    echo "  sudo journalctl -u ${APP_NAME} -n 50 --no-pager"
    exit 1
fi

echo "==> Done. Running at http://$(hostname -I | awk '{print $1}'):$(systemctl show ${APP_NAME} -p ExecStart --value | grep -oP '(?<=--server.port )\d+' || echo 8501)"
