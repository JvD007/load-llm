#!/usr/bin/env bash
# install.sh — Groningen University AI Compute Depot
# Installs the app on a clean Ubuntu server and registers it as a systemd service.
#
# Usage: sudo bash install.sh [--port PORT] [--dir DIR] [--user USER]
#   --port  Streamlit listen port  (default: 8501)
#   --dir   Installation directory (default: /opt/gridweave-depot)
#   --user  Service OS user        (default: gridweave)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PORT=8501
APP_DIR="/opt/gridweave-depot"
APP_USER="gridweave"
APP_NAME="gridweave-depot"
WHL_NAME="gridweave_sdk-0.2.0-py3-none-any.whl"
WHL_URL="https://pub-cbb8992ad1bd437b81d58d5b2da09787.r2.dev/tarball/${WHL_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --dir)  APP_DIR="$2"; shift 2 ;;
        --user) APP_USER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SERVICE_FILE="/etc/systemd/system/${APP_NAME}.service"

# ── Root check ─────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run with sudo: sudo bash install.sh"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║   Groningen University — AI Compute Depot installer  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Install dir : ${APP_DIR}"
echo "  Service user: ${APP_USER}"
echo "  Port        : ${PORT}"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "==> [1/6] Installing system dependencies…"
apt-get update -q
apt-get install -y -q python3 python3-venv python3-pip curl

# ── 2. Service user ───────────────────────────────────────────────────────────
echo "==> [2/6] Creating service user '${APP_USER}'…"
if ! id -u "${APP_USER}" &>/dev/null; then
    useradd --system --home-dir "${APP_DIR}" --create-home \
            --shell /usr/sbin/nologin "${APP_USER}"
fi

# ── 3. Application files ──────────────────────────────────────────────────────
echo "==> [3/6] Copying application files to ${APP_DIR}…"
mkdir -p "${APP_DIR}"

if [[ ! -f "${SCRIPT_DIR}/app.py" ]]; then
    echo "ERROR: app.py not found next to install.sh (expected: ${SCRIPT_DIR}/app.py)"
    exit 1
fi
cp "${SCRIPT_DIR}/app.py" "${APP_DIR}/app.py"

# Resolve the wheel — prefer local copy, fall back to download
if [[ -f "${SCRIPT_DIR}/${WHL_NAME}" ]]; then
    echo "         Using local wheel: ${WHL_NAME}"
    cp "${SCRIPT_DIR}/${WHL_NAME}" "${APP_DIR}/${WHL_NAME}"
else
    echo "         Downloading wheel from R2…"
    curl -fsSL -o "${APP_DIR}/${WHL_NAME}" "${WHL_URL}"
fi

# Patch the hardcoded WHL_PATH so _install_deps() finds the wheel at runtime
sed -i "s|WHL_PATH = .*|WHL_PATH = \"${APP_DIR}/${WHL_NAME}\"|" "${APP_DIR}/app.py"

# ── 4. Python virtual environment ─────────────────────────────────────────────
echo "==> [4/6] Creating Python virtual environment…"
python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --quiet --upgrade pip
"${APP_DIR}/.venv/bin/pip" install --quiet \
    streamlit httpx cloudpickle \
    "${APP_DIR}/${WHL_NAME}"

# Streamlit config (suppress browser launch, set port)
mkdir -p "${APP_DIR}/.streamlit"
cat > "${APP_DIR}/.streamlit/config.toml" << TOML
[server]
headless = true
port = ${PORT}
address = "0.0.0.0"

[browser]
gatherUsageStats = false
TOML

# Fix ownership
chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"

# ── 5. systemd service ────────────────────────────────────────────────────────
echo "==> [5/6] Installing systemd service '${APP_NAME}'…"
cat > "${SERVICE_FILE}" << UNIT
[Unit]
Description=Groningen University AI Compute Depot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=HOME=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/streamlit run ${APP_DIR}/app.py
Restart=on-failure
RestartSec=5s

# Harden the service
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=${APP_DIR}

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "${APP_NAME}"

# ── 6. Start ──────────────────────────────────────────────────────────────────
echo "==> [6/6] Starting service…"
systemctl restart "${APP_NAME}"

# Brief wait to check it came up
sleep 3
STATUS=$(systemctl is-active "${APP_NAME}" || true)

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Installation complete!                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Service status : ${STATUS}"
echo "  URL            : http://$(hostname -I | awk '{print $1}'):${PORT}"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status  ${APP_NAME}"
echo "    sudo systemctl restart ${APP_NAME}"
echo "    sudo journalctl -u ${APP_NAME} -f"
echo ""

if [[ "${STATUS}" != "active" ]]; then
    echo "WARNING: service did not start cleanly. Check logs:"
    echo "  sudo journalctl -u ${APP_NAME} -n 50 --no-pager"
    exit 1
fi
