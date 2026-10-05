"""App entry: asyncio pipeline + FastAPI dashboard in one process."""

from __future__ import annotations

import asyncio
import signal
import sys

import uvicorn
from loguru import logger

from luzifer.config import ConfigStore, load_settings
from luzifer.dashboard import create_dashboard
from luzifer.pipeline import Pipeline


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )


async def _run() -> None:
    _configure_logging()
    settings = load_settings()
    store = ConfigStore(settings)
    pipeline = Pipeline(store)
    app = create_dashboard(store, pipeline)

    config = uvicorn.Config(
        app,
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    pipeline_task = asyncio.create_task(pipeline.run(), name="luzifer-pipeline")
    server_task = asyncio.create_task(server.serve(), name="luzifer-dashboard")

    logger.info(
        "Dashboard http://{}:{}  OSC {}:{}",
        settings.dashboard.host,
        settings.dashboard.port,
        settings.osc.host,
        settings.osc.port,
    )

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows: signal handlers limited; rely on KeyboardInterrupt path
            pass

    done, pending = await asyncio.wait(
        {pipeline_task, server_task, asyncio.create_task(stop_event.wait())},
        return_when=asyncio.FIRST_COMPLETED,
    )

    server.should_exit = True
    await pipeline.stop()
    for task in pending:
        task.cancel()
    await asyncio.gather(pipeline_task, server_task, return_exceptions=True)
    logger.info("Luzifer stopped")


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
