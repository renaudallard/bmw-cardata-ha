"""Thin async HTTP client for Traccar REST API."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import aiohttp

_LOGGER = logging.getLogger(__name__)

_KNOTS_TO_KMH = 1.852
_REQUEST_TIMEOUT = 10


@dataclass
class TraccarPosition:
    """A single Traccar position report."""

    latitude: float
    longitude: float
    speed_kmh: float
    device_time: str
    valid: bool


class TraccarClient:
    """Async client for a self-hosted Traccar server."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        token: str,
    ) -> None:
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    async def get_device_id_by_name(self, name: str) -> int | None:
        """Look up a Traccar device ID by its name."""
        try:
            async with self._session.get(
                f"{self._base_url}/api/devices",
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                devices = await resp.json()
        except Exception as err:
            _LOGGER.warning("Traccar: failed to list devices: %s", err)
            return None

        for dev in devices:
            if dev.get("name") == name:
                return int(dev["id"])
        return None

    async def get_latest_position(self, device_id: int) -> TraccarPosition | None:
        """Get the most recent position for a device."""
        try:
            async with self._session.get(
                f"{self._base_url}/api/positions",
                headers=self._headers,
                params={"deviceId": str(device_id)},
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                positions = await resp.json()
        except Exception as err:
            _LOGGER.debug("Traccar: failed to get position for device %d: %s", device_id, err)
            return None

        if not positions:
            return None

        pos = positions[-1]
        return TraccarPosition(
            latitude=float(pos["latitude"]),
            longitude=float(pos["longitude"]),
            speed_kmh=float(pos.get("speed", 0)) * _KNOTS_TO_KMH,
            device_time=str(pos.get("deviceTime", "")),
            valid=bool(pos.get("valid", False)),
        )
