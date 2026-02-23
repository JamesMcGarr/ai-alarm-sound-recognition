"""
siren/controller.py
-------------------
Controls the Tapo smart plug that the loud siren is plugged into.

Uses TapoDevice from tapo_client.py.  A boolean flag tracks the last known
state of the plug so commands are only sent when there is an actual state
change — the plug is never spammed with repeated on/off calls.

Required environment variables (set in .env or shell):
    API_USERNAME        Tapo account email
    API_PASSWORD        Tapo account password
    DEVICE_IP_ADDRESS   LAN IP address of the Tapo plug the siren is on
"""

from __future__ import annotations

import asyncio
import logging
import os

from src.siren.tapo_client import TapoDevice

logger = logging.getLogger(__name__)


class SirenController:
    """
    Manages the on/off state of the Tapo siren plug.

    Only sends a command to the device when the desired state differs from the
    last known state, preventing the network from being flooded with repeated
    requests during continuous alarm detection windows.

    Parameters
    ----------
    api_username:
        Tapo account email.  Falls back to ``API_USERNAME`` env var.
    api_password:
        Tapo account password.  Falls back to ``API_PASSWORD`` env var.
    device_ip:
        LAN IP of the Tapo plug.  Falls back to ``DEVICE_IP_ADDRESS`` env var.
    """

    def __init__(
        self,
        api_username: str | None = None,
        api_password: str | None = None,
        device_ip: str | None = None,
    ) -> None:
        from src.siren.tapo_client import TapoClient

        username = api_username or os.environ.get("API_USERNAME")
        password = api_password or os.environ.get("API_PASSWORD")
        ip = device_ip or os.environ.get("DEVICE_IP_ADDRESS")

        if not username or not password or not ip:
            raise ValueError(
                "Tapo credentials and device IP are required. "
                "Set API_USERNAME, API_PASSWORD, and DEVICE_IP_ADDRESS "
                "as environment variables (or pass them directly)."
            )

        client = TapoClient.__new__(TapoClient)
        from tapo import ApiClient
        client.api_client = ApiClient(username, password)

        self._device_ip = ip
        self._api_client = client.api_client
        self._device: TapoDevice = asyncio.run(self._init_device())

        # Tracks what we last told the plug to do.
        # Starts as None (unknown) so the first call always sends a command.
        self._siren_on: bool | None = None

        logger.info("SirenController ready (device: %s)", ip)

    async def _init_device(self):
        from src.siren.tapo_client import TapoDevice
        device = TapoDevice(self._api_client, self._device_ip, "Siren")
        await device.initialize()
        return device

    # ── Public API ────────────────────────────────────────────────────────────

    def turn_on(self) -> None:
        """
        Turn the siren plug ON.

        No-op if the plug is already known to be on.
        """
        if self._siren_on is True:
            logger.debug("Siren already ON — skipping command.")
            return
        logger.info("Turning siren ON.")
        asyncio.run(self._device.on())
        self._siren_on = True

    def turn_off(self) -> None:
        """
        Turn the siren plug OFF.

        No-op if the plug is already known to be off.
        """
        if self._siren_on is False:
            logger.debug("Siren already OFF — skipping command.")
            return
        logger.info("Turning siren OFF.")
        asyncio.run(self._device.off())
        self._siren_on = False

    @property
    def is_on(self) -> bool | None:
        """Last known siren state.  ``None`` means unknown (not yet set)."""
        return self._siren_on
