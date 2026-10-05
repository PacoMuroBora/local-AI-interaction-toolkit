"""Local TTS (Kokoro / Piper) with beep-wav fallback."""

from __future__ import annotations

import asyncio
import math
import struct
import wave
from pathlib import Path

import numpy as np
from loguru import logger

from luzifer.config import TtsSettings


class TtsEngine:
    """Synthesize speech to a WAV file for TouchDesigner Audio File In."""

    def __init__(self) -> None:
        self._kokoro = None
        self._piper = None

    def _write_beep(self, path: Path, duration_s: float = 0.6, freq: float = 440.0) -> float:
        sample_rate = 22050
        n = int(sample_rate * duration_s)
        samples = [
            int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
            for i in range(n)
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n}h", *samples))
        return duration_s

    def _write_float_wav(self, path: Path, audio: np.ndarray, sample_rate: int) -> float:
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return float(len(pcm) / sample_rate)

    def _synthesize_kokoro(self, text: str, path: Path, settings: TtsSettings) -> float:
        try:
            from kokoro_onnx import Kokoro

            if self._kokoro is None:
                self._kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
            samples, sample_rate = self._kokoro.create(
                text,
                voice=settings.voice,
                speed=settings.speed,
            )
            return self._write_float_wav(path, np.asarray(samples, dtype=np.float32), int(sample_rate))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kokoro failed ({}), writing beep", exc)
            return self._write_beep(path)

    def _synthesize_piper(self, text: str, path: Path, settings: TtsSettings) -> float:
        try:
            from piper import PiperVoice

            model_path = Path("models") / f"{settings.voice}.onnx"
            if self._piper is None or getattr(self._piper, "_luzifer_path", None) != str(model_path):
                self._piper = PiperVoice.load(str(model_path))
                self._piper._luzifer_path = str(model_path)  # type: ignore[attr-defined]

            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as wf:
                self._piper.synthesize_wav(text, wf)
                frames = wf.getnframes()
                rate = wf.getframerate()
            return float(frames / rate) if rate else 0.5
        except Exception as exc:  # noqa: BLE001
            logger.warning("Piper failed ({}), writing beep", exc)
            return self._write_beep(path)

    def _synthesize_sync(self, text: str, path: Path, settings: TtsSettings) -> float:
        engine = settings.engine.lower().strip()
        if engine == "piper":
            return self._synthesize_piper(text, path, settings)
        return self._synthesize_kokoro(text, path, settings)

    async def synthesize(self, text: str, path: Path, settings: TtsSettings) -> float:
        """Write speech.wav and return duration in seconds."""
        if not text.strip():
            return self._write_beep(path, duration_s=0.2)
        return await asyncio.to_thread(self._synthesize_sync, text, path, settings)
