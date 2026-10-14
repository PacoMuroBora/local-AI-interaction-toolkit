"""Thread-safe OSC client for TouchDesigner.

Address root is `/ai`, not the installation name.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from loguru import logger
from pythonosc.udp_client import SimpleUDPClient

StateName = Literal["loading", "listening", "stt", "thinking", "synthesizing", "speaking"]

ADDR_STATE = "/ai/state"
ADDR_USER_TEXT = "/ai/user_text"
ADDR_AI_TEXT = "/ai/ai_text"
ADDR_PLAY = "/ai/play"
# Int channels for OSC In CHOP. OSC bool (T/F) has no payload and TD often drops it.
ADDR_LISTENING = "/ai/listening"
ADDR_USER_SPEAKING = "/ai/user_speaking"
ADDR_STT = "/ai/stt"
ADDR_THINKING = "/ai/thinking"
ADDR_SYNTHESIZING = "/ai/synthesizing"
ADDR_AI_SPEAKING = "/ai/ai_speaking"
ADDR_IN_LEVEL = "/ai/in_level"
ADDR_OUT_LEVEL = "/ai/out_level"
ADDR_VISION_TEXT = "/ai/vision_text"
ADDR_VISION = "/ai/vision"
ADDR_VISION_ACTIVE = "/ai/vision_active"

OscArg = str | int | float

FlagSnapshot = tuple[tuple[str, int], ...]


class TouchDesignerOscClient:
    """Async wrapper around python-osc UDP client."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        self._host = host
        self._port = port
        self._client = SimpleUDPClient(host, port)
        self._lock = asyncio.Lock()
        self._flags: FlagSnapshot | None = None
        self._play = 0

    def reconfigure(self, host: str, port: int) -> None:
        if host == self._host and port == self._port:
            return
        self._host = host
        self._port = port
        self._client = SimpleUDPClient(host, port)
        logger.info("OSC client retargeted to {}:{}", host, port)

    async def _send(self, address: str, value: OscArg) -> None:
        await self._send_many(((address, value),))

    async def _send_many(
        self,
        messages: tuple[tuple[str, OscArg], ...],
        *,
        log: bool = True,
    ) -> None:
        async with self._lock:
            def _flush() -> None:
                for address, value in messages:
                    self._client.send_message(address, value)

            await asyncio.to_thread(_flush)
        if log:
            for address, value in messages:
                logger.debug("OSC {} {}", address, value)

    async def send_state(self, state: StateName) -> None:
        await self._send(ADDR_STATE, state)

    async def send_flags(
        self,
        *,
        listening: bool,
        user_speaking: bool,
        stt: bool,
        thinking: bool,
        synthesizing: bool,
        ai_speaking: bool,
    ) -> None:
        """Push the full 0/1 snapshot. OSC In CHOP holds the last value until a new one arrives."""
        snapshot: FlagSnapshot = (
            (ADDR_LISTENING, 1 if listening else 0),
            (ADDR_USER_SPEAKING, 1 if user_speaking else 0),
            (ADDR_STT, 1 if stt else 0),
            (ADDR_THINKING, 1 if thinking else 0),
            (ADDR_SYNTHESIZING, 1 if synthesizing else 0),
            (ADDR_AI_SPEAKING, 1 if ai_speaking else 0),
        )
        if snapshot == self._flags:
            return
        self._flags = snapshot
        await self._send_many(snapshot)

    async def send_user_text(self, text: str) -> None:
        await self._send(ADDR_USER_TEXT, text)

    async def send_ai_text(self, text: str) -> None:
        await self._send(ADDR_AI_TEXT, text)

    async def send_play(self) -> None:
        self._play = 1
        await self._send(ADDR_PLAY, 1)

    async def clear_play(self, *, force: bool = False) -> None:
        if not force and self._play == 0:
            return
        self._play = 0
        await self._send(ADDR_PLAY, 0)

    async def send_levels(self, input_level: float, output_level: float) -> None:
        """30 Hz meter. Not logged — it would flood the console."""
        await self._send_many(
            (
                (ADDR_IN_LEVEL, float(input_level)),
                (ADDR_OUT_LEVEL, float(output_level)),
            ),
            log=False,
        )

    async def send_vision_text(self, text: str) -> None:
        await self._send(ADDR_VISION_TEXT, text)

    async def send_vision_pulse(self) -> None:
        """Rising edge then clear so OSC In CHOP can trigger on Off→On."""
        await self._send(ADDR_VISION, 1)
        await self._send(ADDR_VISION, 0)

    async def send_vision_active(self, active: bool) -> None:
        await self._send(ADDR_VISION_ACTIVE, 1 if active else 0)
