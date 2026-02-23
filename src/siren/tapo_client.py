import asyncio
import datetime
import logging
import os
from tapo import ApiClient

logger = logging.getLogger(__name__)


class TapoDevice:

    device = None
    session_duration: datetime.timedelta = datetime.timedelta(minutes=0)
    last_operation_time: datetime.datetime = None

    def __init__(self, client: ApiClient, device_ip: str, description: str = ""):
        self.client = client
        self.device_ip = device_ip
        self.description = description
        self.device = None

    async def initialize(self):
        self.device = await self.client.p110(self.device_ip)

    async def on(self):
        await self._execute_with_retry('turn on', self.device.on)

    async def off(self):
        await self._execute_with_retry('turn off', self.device.off)

    async def _execute_with_retry(self, operation_name: str, operation):
        async def with_refresh_session():
            await self.device.refresh_session()
            return await operation()

        async def with_reinitialize():
            self.device = await self.client.p110(self.device_ip)
            return await operation()

        retry_strategies = [
            operation,
            with_refresh_session,
            with_refresh_session,
            with_reinitialize,
            with_reinitialize
        ]

        for strategy in retry_strategies:
            try:
                return await strategy()
            except Exception as e:
                logger.warning("Error during %s: %s", operation_name, e)
                await asyncio.sleep(5)
        logger.error("All attempts to %s failed.", operation_name)


class TapoClient:

    def __init__(self):
        username = os.environ.get("API_USERNAME")
        password = os.environ.get("API_PASSWORD")
        if not username or not password:
            raise ValueError("API_USERNAME and API_PASSWORD must be set in the environment variables.")
        self.api_client = ApiClient(username, password)

    async def create_device(self):
        device_ip = os.environ.get("DEVICE_IP_ADDRESS")
        if not device_ip:
            raise ValueError("DEVICE_IP_ADDRESS must be set in the environment variables.")
        device = TapoDevice(self.api_client, device_ip, 'Laptop Charger')
        await device.initialize()
        return device
