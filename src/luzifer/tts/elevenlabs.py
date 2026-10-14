"""ElevenLabs cloud TTS → raw PCM, then same speech.wav path as local engines."""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
from loguru import logger

from luzifer.config import ElevenLabsSettings

BASE_URL = "https://api.elevenlabs.io"
PCM_RATE = 22050
# ElevenLabs clamps speed roughly to this range on Flash/Turbo/Multilingual.
_SPEED_MIN = 0.7
_SPEED_MAX = 1.2

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def clamp_speed(speed: float) -> float:
    return max(_SPEED_MIN, min(_SPEED_MAX, float(speed)))


def pcm16_to_float(raw: bytes) -> np.ndarray:
    if not raw:
        return np.zeros(0, dtype=np.float32)
    # Odd trailing byte would truncate; ignore the last incomplete sample.
    n = len(raw) - (len(raw) % 2)
    arr = np.frombuffer(raw[:n], dtype=np.int16).astype(np.float32) / 32768.0
    return arr


class ElevenLabsClient:
    """Async ElevenLabs text-to-speech (full clip, not streamed)."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=_TIMEOUT)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def synthesize_pcm(
        self,
        text: str,
        *,
        settings: ElevenLabsSettings,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """Return (float32 mono samples, sample_rate). Raises on missing key / HTTP error."""
        api_key = (settings.api_key or "").strip()
        voice_id = (settings.voice_id or "").strip()
        if not api_key:
            raise RuntimeError("ElevenLabs api_key is empty")
        if not voice_id:
            raise RuntimeError("ElevenLabs voice_id is empty")

        client = await self._get_client()
        params = {"output_format": f"pcm_{PCM_RATE}"}
        body = {
            "text": text,
            "model_id": (settings.model_id or "eleven_flash_v2_5").strip(),
            "voice_settings": {
                "stability": float(settings.stability),
                "similarity_boost": float(settings.similarity_boost),
                "speed": clamp_speed(speed),
            },
        }
        headers = {
            "xi-api-key": api_key,
            "Accept": "application/octet-stream",
            "Content-Type": "application/json",
        }
        url = f"/v1/text-to-speech/{voice_id}"
        resp = await client.post(url, params=params, json=body, headers=headers)
        if resp.status_code >= 400:
            # Never log the API key; status + short body is enough.
            detail = (resp.text or "")[:200]
            raise RuntimeError(f"ElevenLabs HTTP {resp.status_code}: {detail}")
        audio = pcm16_to_float(resp.content)
        return audio, PCM_RATE

    async def list_voices(self, api_key: str) -> list[dict[str, Any]]:
        key = (api_key or "").strip()
        if not key:
            raise RuntimeError("ElevenLabs api_key is empty")
        client = await self._get_client()
        resp = await client.get("/v1/voices", headers={"xi-api-key": key})
        if resp.status_code >= 400:
            detail = (resp.text or "")[:200]
            raise RuntimeError(f"ElevenLabs HTTP {resp.status_code}: {detail}")
        data = resp.json()
        voices = data.get("voices") or []
        out: list[dict[str, Any]] = []
        for v in voices:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("voice_id") or "").strip()
            name = str(v.get("name") or vid).strip()
            if vid:
                out.append({"voice_id": vid, "name": name})
        out.sort(key=lambda row: row["name"].lower())
        return out
