#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="ai-alarm-listener"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

CURRENT_USER="$(whoami)"
WORKDIR="$(pwd)"
PYTHON="${WORKDIR}/.venv/bin/python"

echo "==> Installing ${SERVICE_NAME} systemd service"
echo "    User:       ${CURRENT_USER}"
echo "    WorkingDir: ${WORKDIR}"
echo "    Python:     ${PYTHON}"

# --- Precondition checks ---

if [ ! -f "${PYTHON}" ]; then
  echo ""
  echo "ERROR: Python executable not found at ${PYTHON}"
  echo "       Please create a virtual environment first:"
  echo "         python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [ ! -f "${WORKDIR}/models/alarm_model.pt" ]; then
  echo ""
  echo "WARNING: Trained model not found at ${WORKDIR}/models/alarm_model.pt"
  echo "         The service will start but will fail until a model is trained."
  echo "         Run: python train.py"
  echo ""
fi

# --- Generate and install the service unit file ---

echo "==> Writing ${SERVICE_FILE}"

sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=AI Alarm Sound Recognition Listener
After=network.target sound.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${WORKDIR}
ExecStart=${PYTHON} ${WORKDIR}/listen.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# --- Enable and start the service ---

echo "==> Reloading systemd daemon"
sudo systemctl daemon-reload

echo "==> Enabling ${SERVICE_NAME} (auto-start on boot)"
sudo systemctl enable "${SERVICE_NAME}"

echo "==> Starting ${SERVICE_NAME}"
sudo systemctl start "${SERVICE_NAME}"

echo ""
echo "==> Service status:"
sudo systemctl status "${SERVICE_NAME}" --no-pager

echo ""
echo "Install complete. Useful commands:"
echo "  Status:  sudo systemctl status ${SERVICE_NAME}"
echo "  Logs:    journalctl -u ${SERVICE_NAME} -f"
echo "  Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo "  Restart: sudo systemctl restart ${SERVICE_NAME}"
