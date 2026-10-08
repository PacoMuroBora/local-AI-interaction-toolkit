"""Local TTS (Kokoro / Piper) with beep-wav fallback."""

from __future__ import annotations

import asyncio
import math
import struct
import time
import wave
from pathlib import Path

import numpy as np
from loguru import logger

from luzifer.audio.envelope import peak_envelope_samples
from luzifer.config import TtsSettings
from luzifer.metrics import LEVEL_HZ

SynthResult = tuple[float, float, np.ndarray]


class TtsEngine:
    """Synthesize speech to a WAV file for TouchDesigner Audio File In."""

    def __init__(self) -> None:
        self._kokoro = None
        self._piper = None
        self._kokoro_key: tuple[str, str, str, str] | None = None

    @staticmethod
    def _onnx_providers(settings: TtsSettings) -> list[str]:
        device = (settings.device or "cpu").lower().strip()
        if device in {"cuda", "gpu"}:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device in {"dml", "directml"}:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _write_beep(self, path: Path, duration_s: float = 0.6, freq: float = 440.0) -> SynthResult:
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
        audio = np.asarray(samples, dtype=np.float32) / 32768.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return duration_s, peak, peak_envelope_samples(audio, sample_rate, LEVEL_HZ)

    def _write_float_wav(self, path: Path, audio: np.ndarray, sample_rate: int) -> SynthResult:
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
        peak = float(np.max(np.abs(clipped))) if clipped.size else 0.0
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        duration = float(len(pcm) / sample_rate)
        return duration, peak, peak_envelope_samples(clipped, sample_rate, LEVEL_HZ)

    def _ensure_kokoro(self, settings: TtsSettings) -> None:
        model = "models/kokoro-v1.0.onnx"
        voices = "models/voices-v1.0.bin"
        device = (settings.device or "cpu").lower().strip()
        compute = (settings.compute_type or "float32").lower().strip()
        key = (model, voices, device, compute)
        if self._kokoro is not None and self._kokoro_key == key:
            return

        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        # onnxruntime-gpu 1.30 expects CUDA 13 + cuDNN 9 DLLs (pip extras), not the
        # system CUDA 12.x toolkit alone. Preload the nvidia-* wheel DLLs first.
        try:
            ort.preload_dlls()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ORT preload_dlls: {}", exc)

        providers = self._onnx_providers(settings)
        available = ort.get_available_providers()
        usable = [p for p in providers if p in available]
        if not usable:
            logger.warning(
                "TTS providers {} unavailable (have {}); falling back to CPU",
                providers,
                available,
            )
            usable = ["CPUExecutionProvider"]

        t0 = time.perf_counter()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_options: list[dict[str, str]] = [{} for _ in usable]
        if usable[0] == "CUDAExecutionProvider" and compute == "float16":
            # Hint only — graph weights stay as exported; EP may still prefer FP16 mats.
            provider_options[0] = {"cudnn_conv_use_max_workspace": "1"}

        session = ort.InferenceSession(
            model,
            sess_options=sess_options,
            providers=usable,
            provider_options=provider_options,
        )
        # kokoro-onnx.from_session reads this private attr for KoKoroConfig paths
        session._model_path = model  # type: ignore[attr-defined]
        self._kokoro = Kokoro.from_session(session, voices)
        self._kokoro_key = key
        active = session.get_providers()
        if device in {"cuda", "gpu"} and "CUDAExecutionProvider" not in active:
            logger.warning(
                "TTS requested cuda but session is {}; install onnxruntime-gpu[cuda,cudnn]",
                active,
            )
        logger.info(
            "Kokoro loaded in {:.2f}s device={} compute={} providers={}",
            time.perf_counter() - t0,
            device,
            compute,
            active,
        )

    def _synthesize_kokoro(self, text: str, path: Path, settings: TtsSettings) -> SynthResult:
        try:
            self._ensure_kokoro(settings)
            t0 = time.perf_counter()
            lang = (settings.lang or "en-us").strip() or "en-us"
            samples, sample_rate = self._kokoro.create(
                text,
                voice=settings.voice,
                speed=settings.speed,
                lang=lang,
            )
            audio = np.asarray(samples, dtype=np.float32)
            result = self._write_float_wav(path, audio, int(sample_rate))
            logger.info(
                "Kokoro synth {:.2f}s → {:.2f}s audio (voice={} lang={} speed={})",
                time.perf_counter() - t0,
                result[0],
                settings.voice,
                lang,
                settings.speed,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kokoro failed ({}), writing beep", exc)
            return self._write_beep(path)

    def _synthesize_piper(self, text: str, path: Path, settings: TtsSettings) -> SynthResult:
        try:
            from piper import PiperVoice

            model_path = Path("models") / f"{settings.voice}.onnx"
            if self._piper is None or getattr(self._piper, "_luzifer_path", None) != str(model_path):
                self._piper = PiperVoice.load(str(model_path))
                self._piper._luzifer_path = str(model_path)  # type: ignore[attr-defined]

            path.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            with wave.open(str(path), "wb") as wf:
                self._piper.synthesize_wav(text, wf)
                frames = wf.getnframes()
                rate = wf.getframerate()
            duration = float(frames / rate) if rate else 0.5
            peak = 0.4
            envelope = np.zeros(1, dtype=np.float32)
            try:
                with wave.open(str(path), "rb") as rf:
                    raw = rf.readframes(rf.getnframes())
                    rate = rf.getframerate() or rate
                if raw:
                    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    peak = float(np.max(np.abs(arr))) if arr.size else 0.4
                    envelope = peak_envelope_samples(arr, int(rate), LEVEL_HZ)
            except Exception:  # noqa: BLE001
                pass
            logger.info(
                "Piper synth {:.2f}s → {:.2f}s audio",
                time.perf_counter() - t0,
                duration,
            )
            return duration, peak, envelope
        except Exception as exc:  # noqa: BLE001
            logger.warning("Piper failed ({}), writing beep", exc)
            return self._write_beep(path)

    def _synthesize_sync(self, text: str, path: Path, settings: TtsSettings) -> SynthResult:
        engine = settings.engine.lower().strip()
        if engine == "piper":
            return self._synthesize_piper(text, path, settings)
        return self._synthesize_kokoro(text, path, settings)

    def preload_sync(self, settings: TtsSettings) -> None:
        """Load the active TTS backend so the first spoken line isn't cold."""
        engine = settings.engine.lower().strip()
        if engine == "piper":
            model_path = Path("models") / f"{settings.voice}.onnx"
            if not model_path.exists():
                logger.warning("Piper voice missing at {}, skip preload", model_path)
                return
            from piper import PiperVoice

            self._piper = PiperVoice.load(str(model_path))
            self._piper._luzifer_path = str(model_path)  # type: ignore[attr-defined]
            logger.info("Piper preloaded {}", settings.voice)
            return
        try:
            self._ensure_kokoro(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kokoro preload failed: {}", exc)

    async def preload(self, settings: TtsSettings) -> None:
        await asyncio.to_thread(self.preload_sync, settings)

    async def synthesize(self, text: str, path: Path, settings: TtsSettings) -> SynthResult:
        """Write speech.wav and return (duration_s, peak_level, envelope)."""
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
