"""Async Ollama chat client."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from luzifer.config import LlmSettings

ChatMessage = dict[str, str]

# Cold-load of a 14B can sit well past a normal generation. The request blocks
# inside Ollama until weights are resident, so this is a load budget, not a token budget.
_TIMEOUT = httpx.Timeout(300.0, connect=5.0)
# -1 pins the model until the server restarts. Omitting it resets Ollama's 5 min default
# and the next call evicts whatever we just warmed.
_KEEP_ALIVE = -1


def runner_options(
    settings: LlmSettings,
    *,
    temperature: float,
    num_predict: int,
    top_p: float | None = None,
) -> dict[str, Any]:
    """Load-time knobs. Omit them on a later call and Ollama builds a new runner."""
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": int(num_predict),
        "num_ctx": int(settings.num_ctx),
        "num_gpu": int(settings.num_gpu),
    }
    if top_p is not None:
        options["top_p"] = top_p
    return options


def models_match(requested: str, loaded: str) -> bool:
    """True when an /api/ps name is the tag we asked Ollama to load."""

    def norm(name: str) -> str:
        name = name.strip()
        if name.endswith(":latest"):
            name = name[: -len(":latest")]
        return name

    a, b = norm(requested), norm(loaded)
    if not a or not b:
        return False
    return a == b or b.endswith("/" + a) or a.endswith("/" + b)


class OllamaEngine:
    """Talk to a local Ollama server over HTTP.

    `lock` is shared with the vision client. Ollama loads one runner at a time on
    a single GPU; overlapping /api/chat calls (14B talker + VLM) wedge that runner.
    """

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

    async def loaded_models(self, base_url: str) -> list[str]:
        """Names currently resident in the Ollama runner. Raises on HTTP failure."""
        client = await self._get_client(base_url)
        response = await client.get("/api/ps")
        response.raise_for_status()
        names: list[str] = []
        for item in response.json().get("models", []):
            for key in ("name", "model"):
                name = str(item.get(key) or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    async def unload(self, model: str, base_url: str) -> None:
        """Drop a resident runner. No-op when it is not loaded (does not pull weights)."""
        name = (model or "").strip()
        if not name:
            return
        if not any(models_match(name, loaded) for loaded in await self.loaded_models(base_url)):
            return
        async with self._lock:
            client = await self._get_client(base_url)
            response = await client.post(
                "/api/generate",
                json={"model": name, "keep_alive": 0},
            )
            response.raise_for_status()
        logger.info("Unloaded {}", name)

    async def warmup(self, settings: LlmSettings) -> bool:
        """Load the chat model and pin it. One token, no history. Serialized on `lock`."""
        model = (settings.model or "").strip()
        if not model:
            logger.warning("Chat warmup skipped (llm.model empty)")
            return False
        async with self._lock:
            t0 = time.perf_counter()
            try:
                client = await self._get_client(settings.base_url)
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "keep_alive": _KEEP_ALIVE,
                        "messages": [{"role": "user", "content": "Reply with ok."}],
                        "options": runner_options(settings, temperature=0, num_predict=1),
                    },
                )
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Chat warmup failed for {} ({:.1f}s): {}",
                    model,
                    time.perf_counter() - t0,
                    exc,
                )
                return False
        logger.info(
            "Chat model ready in {:.1f}s ({}, num_gpu={}, num_ctx={})",
            time.perf_counter() - t0,
            model,
            settings.num_gpu,
            settings.num_ctx,
        )
        return True

    async def chat(
        self,
        user_text: str,
        system_prompt: str,
        settings: LlmSettings,
        history: list[ChatMessage] | None = None,
    ) -> str:
        async with self._lock:
            return await self._chat(user_text, system_prompt, settings, history)

    async def _chat(
        self,
        user_text: str,
        system_prompt: str,
        settings: LlmSettings,
        history: list[ChatMessage] | None,
    ) -> str:
        client = await self._get_client(settings.base_url)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": settings.model,
            "stream": False,
            "keep_alive": _KEEP_ALIVE,
            "messages": messages,
            "options": runner_options(
                settings,
                temperature=settings.temperature,
                num_predict=settings.max_tokens,
                top_p=settings.top_p,
            ),
        }
        t0 = time.perf_counter()
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            _log_chat_speed(settings.model, data, time.perf_counter() - t0)
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama chat failed after {:.1f}s: {}", time.perf_counter() - t0, exc)
            return "I am momentarily silent. Try again in a moment."


def _log_chat_speed(model: str, data: dict[str, Any], elapsed_s: float) -> None:
    eval_ns = float(data.get("eval_duration") or 0)
    eval_count = int(data.get("eval_count") or 0)
    tok_s = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
    logger.info("Chat {} {:.1f}s, {} tok, {:.1f} tok/s", model, elapsed_s, eval_count, tok_s)
    if 0 < tok_s < 8:
        logger.warning(
            "Chat decode is {:.1f} tok/s. The talker is starved for VRAM — "
            "lower vision.num_gpu (0 puts the VLM on CPU) and restart.",
            tok_s,
        )
