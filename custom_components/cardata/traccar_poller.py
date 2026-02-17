"""Per-VIN Traccar polling manager for speed-bucketed consumption learning."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .utils import redact_vin

if TYPE_CHECKING:
    from .magic_soc import MagicSOCPredictor
    from .traccar_client import TraccarClient

_LOGGER = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 60


class TraccarPoller:
    """Manage per-VIN polling tasks that feed speed data to MagicSOCPredictor."""

    def __init__(
        self,
        client: TraccarClient,
        magic_soc: MagicSOCPredictor,
    ) -> None:
        self._client = client
        self._magic_soc = magic_soc
        self._device_ids: dict[str, int] = {}  # VIN -> Traccar device ID
        self._tasks: dict[str, asyncio.Task] = {}  # VIN -> polling task

    def set_device_id(self, vin: str, device_id: int) -> None:
        """Map a VIN to a Traccar device ID."""
        self._device_ids[vin] = device_id

    def start_polling(self, vin: str) -> None:
        """Start polling Traccar for a VIN (creates an asyncio.Task)."""
        if vin not in self._device_ids:
            return
        if vin in self._tasks and not self._tasks[vin].done():
            return
        self._tasks[vin] = asyncio.get_running_loop().create_task(
            self._poll_loop(vin),
            name=f"traccar_poll_{redact_vin(vin)}",
        )
        _LOGGER.debug("Traccar: started polling for %s", redact_vin(vin))

    def stop_polling(self, vin: str) -> None:
        """Stop polling Traccar for a VIN."""
        task = self._tasks.pop(vin, None)
        if task and not task.done():
            task.cancel()
            _LOGGER.debug("Traccar: stopped polling for %s", redact_vin(vin))

    def stop_all(self) -> None:
        """Stop all polling tasks (shutdown)."""
        for vin in list(self._tasks):
            self.stop_polling(vin)

    async def _poll_loop(self, vin: str) -> None:
        """Poll Traccar for latest position at regular intervals."""
        device_id = self._device_ids[vin]
        try:
            while True:
                try:
                    pos = await self._client.get_latest_position(device_id)
                    if pos is not None and pos.valid:
                        self._magic_soc.update_traccar_speed(vin, pos.speed_kmh)
                    else:
                        self._magic_soc.update_traccar_speed(vin, None)
                except Exception:
                    self._magic_soc.update_traccar_speed(vin, None)

                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass
