"""faster-whisper STT with mock fallback."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from loguru import logger

from luzifer.config import SttSettings


class WhisperEngine:
    """Local speech-to-text. Falls back to a mock transcript if unavailable."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded_key: tuple[str, str, str] | None = None
        self._mock = False

    def _ensure_model(self, settings: SttSettings) -> None:
        key = (settings.model_size, settings.device, settings.compute_type)
        if self._model is not None and self._loaded_key == key:
            return
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                settings.model_size,
                device=settings.device,
                compute_type=settings.compute_type,
            )
            self._loaded_key = key
            self._mock = False
            logger.info("Whisper model loaded: {}", settings.model_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Whisper unavailable ({}), using mock STT", exc)
            self._model = None
            self._loaded_key = key
            self._mock = True

    def _transcribe_sync(self, audio: np.ndarray, settings: SttSettings) -> str:
        self._ensure_model(settings)
        if self._mock or self._model is None:
            duration = float(audio.size) / 16000.0
            return f"[mock transcript {duration:.1f}s]"

        segments, _info = self._model.transcribe(
            audio.astype(np.float32),
            language=settings.language,
            vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

    async def transcribe(self, audio: np.ndarray, settings: SttSettings) -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio, settings)
