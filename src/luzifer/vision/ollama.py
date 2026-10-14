"""Ollama VLM caption client (separate from the chat model)."""

from __future__ import annotations

import asyncio
import base64
import io
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from loguru import logger

from luzifer.config import VisionSettings

# Moondream often emits punctuation-only / emoji junk on vague prompts.
_JUNK = re.compile(r"^[\s.!?;:,\-_/\\|@#$%^&*()+=\[\]{}'\"`~<>]+$")
_WORD = re.compile(r"[A-Za-zÀ-ÿ]{2,}")


def is_usable_caption(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return False
    if _JUNK.match(t):
        return False
    if len(_WORD.findall(t)) < 2:
        return False
    return True


# Same budget and pin as the chat client. A later request that omits keep_alive
# resets Ollama's 5 minute default and drops the other model.
_TIMEOUT = httpx.Timeout(300.0, connect=5.0)
_KEEP_ALIVE = -1


def runner_options(
    settings: VisionSettings,
    *,
    temperature: float,
    num_predict: int,
) -> dict[str, Any]:
    """Same load-time contract as the chat client. A later call that drops these reloads."""
    return {
        "temperature": temperature,
        "num_predict": int(num_predict),
        "num_ctx": int(settings.num_ctx),
        "num_gpu": int(settings.num_gpu),
        "num_batch": int(settings.num_batch),
    }


def _tiny_jpeg() -> bytes:
    """Small valid JPEG so a VLM warmup exercises the image path, not a text-only load."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (24, 24, 24)).save(buf, format="JPEG", quality=40)
    return buf.getvalue()


class VisionOllama:
    """POST /api/chat (fallback /api/generate) with images for a small local VLM."""

    def __init__(self, lock: asyncio.Lock | None = None) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url: str | None = None
        self._lock = lock if lock is not None else asyncio.Lock()

    async def _get_client(self, base_url: str) -> httpx.AsyncClient:
        if self._client is None or self._base_url != base_url:
            if self._client is not None:
                await self._client.aclose()
            self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=_TIMEOUT)
            self._base_url = base_url
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def warmup(self, settings: VisionSettings, *, base_url: str) -> bool:
        """Load the VLM and pin it. Junk captions still count — we only need the weights resident."""
        model = (settings.model or "").strip()
        if not model:
            logger.warning("Vision warmup skipped (vision.model empty)")
            return False
        async with self._lock:
            t0 = time.perf_counter()
            try:
                jpeg = _tiny_jpeg()
            except Exception as exc:  # noqa: BLE001
                logger.error("Vision warmup jpeg failed: {}", exc)
                return False
            client = await self._get_client(base_url)
            b64 = base64.b64encode(jpeg).decode("ascii")
            try:
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "keep_alive": _KEEP_ALIVE,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Reply with ok.",
                                "images": [b64],
                            }
                        ],
                        "options": runner_options(settings, temperature=0, num_predict=1),
                    },
                )
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Vision warmup failed for {} ({:.1f}s): {}",
                    model,
                    time.perf_counter() - t0,
                    exc,
                )
                return False
        logger.info(
            "Vision model ready in {:.1f}s ({}, num_gpu={}, num_ctx={}, num_batch={})",
            time.perf_counter() - t0,
            model,
            settings.num_gpu,
            settings.num_ctx,
            settings.num_batch,
        )
        return True

    async def describe(
        self,
        jpeg: bytes,
        settings: VisionSettings,
        *,
        base_url: str,
        proceed: Callable[[], bool] | None = None,
    ) -> str | None:
        """Caption a still. None means skipped so the talker can have the runner.

        The lock is taken before `proceed` runs. A caption that already started
        cannot be cancelled; overlapping /api/chat calls wedge the Ollama runner.
        """
        async with self._lock:
            if proceed is not None and not proceed():
                logger.info("Vision caption skipped (talker busy)")
                return None
            return await self._describe(jpeg, settings, base_url=base_url)

    async def _describe(
        self,
        jpeg: bytes,
        settings: VisionSettings,
        *,
        base_url: str,
    ) -> str:
        client = await self._get_client(base_url)
        b64 = base64.b64encode(jpeg).decode("ascii")
        prompt = (settings.prompt or "").strip() or (
            "Describe the scene in one or two short factual English sentences. "
            "Name people, pose, clothing, objects, and background. No poetry."
        )
        options = runner_options(
            settings,
            temperature=0.1,
            num_predict=max(32, int(settings.num_predict)),
        )

        # 1) chat API
        try:
            response = await client.post(
                "/api/chat",
                json={
                    "model": settings.model,
                    "stream": False,
                    "keep_alive": _KEEP_ALIVE,
                    "messages": [
                        {"role": "user", "content": prompt, "images": [b64]},
                    ],
                    "options": options,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = str(data.get("message", {}).get("content", "") or "").strip()
            if is_usable_caption(content):
                return content
            logger.warning("Vision chat junk/empty: {!r}", content[:80])
        except Exception as exc:  # noqa: BLE001
            logger.error("Vision /api/chat failed: {}", exc)

        # 2) generate API (moondream is often happier here)
        try:
            response = await client.post(
                "/api/generate",
                json={
                    "model": settings.model,
                    "stream": False,
                    "keep_alive": _KEEP_ALIVE,
                    "prompt": prompt,
                    "images": [b64],
                    "options": options,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = str(data.get("response", "") or "").strip()
            if is_usable_caption(content):
                return content
            logger.warning("Vision generate junk/empty: {!r}", content[:80])
        except Exception as exc:  # noqa: BLE001
            logger.error("Vision /api/generate failed: {}", exc)

        return ""
