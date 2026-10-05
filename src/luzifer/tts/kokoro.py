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

    def _write_beep(self, path: Path, duration_s: float = 0.6, freq: float = 440.0) -> tuple[float, float]:
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
        return duration_s, 16000 / 32768.0

    def _write_float_wav(self, path: Path, audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(audio, -1.0, 1.0)
        peak = float(np.max(np.abs(clipped))) if clipped.size else 0.0
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return float(len(pcm) / sample_rate), peak

    def _synthesize_kokoro(self, text: str, path: Path, settings: TtsSettings) -> tuple[float, float]:
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

    def _synthesize_piper(self, text: str, path: Path, settings: TtsSettings) -> tuple[float, float]:
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
            duration = float(frames / rate) if rate else 0.5
            # cheap peak approx from file — read int16 peak only
            peak = 0.4
            try:
                with wave.open(str(path), "rb") as rf:
                    raw = rf.readframes(min(rf.getnframes(), rate))  # ~1s sample
                if raw:
                    arr = np.frombuffer(raw, dtype=np.int16)
                    peak = float(np.max(np.abs(arr))) / 32768.0 if arr.size else 0.4
            except Exception:  # noqa: BLE001
                pass
            return duration, peak
        except Exception as exc:  # noqa: BLE001
            logger.warning("Piper failed ({}), writing beep", exc)
            return self._write_beep(path)

    def _synthesize_sync(self, text: str, path: Path, settings: TtsSettings) -> tuple[float, float]:
        engine = settings.engine.lower().strip()
        if engine == "piper":
            return self._synthesize_piper(text, path, settings)
        return self._synthesize_kokoro(text, path, settings)

    async def synthesize(self, text: str, path: Path, settings: TtsSettings) -> tuple[float, float]:
        """Write speech.wav and return (duration_s, peak_level)."""
        if not text.strip():
            return self._write_beep(path, duration_s=0.2)
        return await asyncio.to_thread(self._synthesize_sync, text, path, settings)

    def _play_local_sync(self, path: Path, device_id: int | None) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
        except Exception as exc:  # noqa: BLE001
            # soundfile optional — fall back to wave + numpy
            logger.debug("soundfile unavailable ({}), using wave playback", exc)
            try:
                import sounddevice as sd

                with wave.open(str(path), "rb") as wf:
                    rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    channels = wf.getnchannels()
                    width = wf.getsampwidth()
                if width == 2:
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                sd.play(audio, rate, device=device_id, blocking=True)
            except Exception as play_exc:  # noqa: BLE001
                logger.warning("Local monitor playback failed: {}", play_exc)
            return

        try:
            data, rate = sf.read(str(path), dtype="float32")
            sd.play(data, rate, device=device_id, blocking=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local monitor playback failed: {}", exc)

    async def play_local(self, path: Path, device_id: int | None = None) -> None:
        """Play a wav on a local output device (dashboard monitor)."""
        await asyncio.to_thread(self._play_local_sync, path, device_id)
