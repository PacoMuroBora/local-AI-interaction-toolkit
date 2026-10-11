"""Inbound OSC server for TouchDesigner → Luzifer commands."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

ADDR_SAY = "/luzifer/in/say"
ADDR_SPEAK = "/luzifer/in/speak"
ADDR_RESET = "/luzifer/in/reset"
ADDR_INTERRUPT = "/luzifer/in/interrupt"
ADDR_LISTEN = "/luzifer/in/listen"
ADDR_VISION = "/luzifer/in/vision"

CommandHandler = Callable[[str, Any], Awaitable[None] | None]


class TouchDesignerOscServer:
    """Listen for TD OSC Out messages and dispatch to the pipeline."""

    def __init__(self, on_command: CommandHandler) -> None:
        self._on_command = on_command
        self._transport: Any | None = None
        self._server: AsyncIOOSCUDPServer | None = None
        self._host = "127.0.0.1"
        self._port = 9001
        self._loop: asyncio.AbstractEventLoop | None = None

    def _dispatch(self, address: str, *args: Any) -> None:
        value: Any = args[0] if args else None
        # TD often streams /luzifer/in/vision 0 every frame — drop before scheduling.
        if address == ADDR_VISION:
            fire = True
            if isinstance(value, (int, float)):
                fire = bool(int(value))
            elif isinstance(value, str):
                fire = value.strip() not in {"0", "false", "False", ""}
            elif value is None:
                fire = False
            if not fire:
                return
        loop = self._loop
        if loop is None:
            return

        async def _run() -> None:
            try:
                result = self._on_command(address, value)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.exception("OSC command failed {}: {}", address, exc)

        def _schedule() -> None:
            asyncio.create_task(_run())

        loop.call_soon_threadsafe(_schedule)

    async def start(self, host: str, port: int) -> None:
        await self.stop()
        self._host = host
        self._port = port
        self._loop = asyncio.get_running_loop()
        dispatcher = Dispatcher()
        dispatcher.map(ADDR_SAY, self._dispatch)
        dispatcher.map(ADDR_SPEAK, self._dispatch)
        dispatcher.map(ADDR_RESET, self._dispatch)
        dispatcher.map(ADDR_INTERRUPT, self._dispatch)
        dispatcher.map(ADDR_LISTEN, self._dispatch)
        dispatcher.map(ADDR_VISION, self._dispatch)
        self._server = AsyncIOOSCUDPServer((host, port), dispatcher, self._loop)
        self._transport, _protocol = await self._server.create_serve_endpoint()
        logger.info("OSC inbound listening on {}:{}", host, port)

    async def stop(self) -> None:
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("OSC server stop error: {}", exc)
        self._transport = None
        self._server = None

    async def reconfigure(self, host: str, port: int) -> None:
        if host == self._host and port == self._port and self._transport is not None:
            return
        await self.start(host, port)
