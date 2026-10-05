"""Thread-safe OSC client for TouchDesigner."""

from __future__ import annotations

import asyncio
from typing import Literal

from loguru import logger
from pythonosc.udp_client import SimpleUDPClient

StateName = Literal["listening", "thinking", "speaking"]

ADDR_STATE = "/luzifer/state"
ADDR_USER_TEXT = "/luzifer/user_text"
ADDR_AI_TEXT = "/luzifer/ai_text"
ADDR_PLAY = "/luzifer/play"


class TouchDesignerOscClient:
    """Async wrapper around python-osc UDP client."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        self._host = host
        self._port = port
        self._client = SimpleUDPClient(host, port)
        self._lock = asyncio.Lock()

    def reconfigure(self, host: str, port: int) -> None:
        if host == self._host and port == self._port:
            return
        self._host = host
        self._port = port
        self._client = SimpleUDPClient(host, port)
        logger.info("OSC client retargeted to {}:{}", host, port)

    async def _send(self, address: str, value: str | int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._client.send_message, address, value)
            logger.debug("OSC {} {}", address, value)

    async def send_state(self, state: StateName) -> None:
        await self._send(ADDR_STATE, state)

    async def send_user_text(self, text: str) -> None:
        await self._send(ADDR_USER_TEXT, text)

    async def send_ai_text(self, text: str) -> None:
        await self._send(ADDR_AI_TEXT, text)

    async def send_play(self) -> None:
        await self._send(ADDR_PLAY, 1)
