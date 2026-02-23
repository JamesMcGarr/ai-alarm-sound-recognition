#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="ai-alarm-listener"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "==> Uninstalling ${SERVICE_NAME} systemd service"

# --- Check the service is actually installed ---

if [ ! -f "${SERVICE_FILE}" ]; then
  echo "Service not installed (${SERVICE_FILE} not found). Nothing to do."
  exit 0
fi

# --- Stop the service (ignore errors if it is not currently running) ---

echo "==> Stopping ${SERVICE_NAME}"
sudo systemctl stop "${SERVICE_NAME}" || true

# --- Disable auto-start on boot (ignore errors if it was never enabled) ---

echo "==> Disabling ${SERVICE_NAME}"
sudo systemctl disable "${SERVICE_NAME}" || true

# --- Remove the unit file ---

echo "==> Removing ${SERVICE_FILE}"
sudo rm -f "${SERVICE_FILE}"

# --- Reload systemd so the unit disappears from the registry ---

echo "==> Reloading systemd daemon"
sudo systemctl daemon-reload
sudo systemctl reset-failed 2>/dev/null || true

echo ""
echo "Uninstall complete. The system is back to its state before install.sh was run."
echo "  - Service unit removed: ${SERVICE_FILE}"
echo "  - Service will no longer start on boot."
echo ""
echo "Project files (models, data, logs, .env) have NOT been removed."
