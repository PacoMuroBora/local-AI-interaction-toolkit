"""faster-whisper STT with mock fallback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from luzifer.config import SttSettings


@dataclass(frozen=True)
class Transcript:
    """One utterance. Language is from this clip only — never history."""

    text: str
    language: str | None = None  # ISO 639-1 from Whisper, e.g. "en" / "de"
    language_probability: float | None = None


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
            logger.info(
                "Whisper model loaded: {} device={} compute={}",
                settings.model_size,
                settings.device,
                settings.compute_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Whisper unavailable ({}), using mock STT", exc)
            self._model = None
            self._loaded_key = key
            self._mock = True

    def _transcribe_sync(self, audio: np.ndarray, settings: SttSettings) -> Transcript:
        self._ensure_model(settings)
        if self._mock or self._model is None:
            duration = float(audio.size) / 16000.0
            locked = (settings.language or "").split("-")[0].lower() or None
            return Transcript(text=f"[mock transcript {duration:.1f}s]", language=locked)

        segments, info = self._model.transcribe(
            audio.astype(np.float32),
            language=settings.language,
            vad_filter=False,
            beam_size=1,
            best_of=1,
            patience=1.0,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected = getattr(info, "language", None)
        if detected:
            detected = str(detected).split("-")[0].lower()
        # If language is locked in config, Whisper still fills info.language with that lock.
        prob = getattr(info, "language_probability", None)
        try:
            prob_f = float(prob) if prob is not None else None
        except (TypeError, ValueError):
            prob_f = None
        logger.info(
            "STT lang={} p={:.2f} text={!r}",
            detected,
            prob_f or 0.0,
            text[:80],
        )
        return Transcript(text=text, language=detected, language_probability=prob_f)

    async def transcribe(self, audio: np.ndarray, settings: SttSettings) -> Transcript:
        return await asyncio.to_thread(self._transcribe_sync, audio, settings)
