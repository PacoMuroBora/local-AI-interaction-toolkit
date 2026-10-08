"""Silero VAD + sounddevice capture with software gating."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
from loguru import logger

from luzifer.config import AudioSettings

GateFn = Callable[[], bool]
SpeechFn = Callable[[bool], Awaitable[None]]


class VoiceActivityDetector:
    """Capture a single utterance while the pipeline gate is open.

    Uses Silero via torch.hub when available; otherwise RMS energy mock.
    """

    FRAME_SAMPLES = 512

    def __init__(self) -> None:
        self._model: Any | None = None
        self._use_silero = False
        self._stream: Any | None = None
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._device_id: int | None = None
        self._sample_rate = 16000
        self._channels = 1
        self._started = False
        self.metrics: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch

            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            model.eval()
            self._model = model
            self._use_silero = True
            logger.info("Silero VAD loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Silero VAD unavailable ({}), using energy mock", exc)
            self._model = None
            self._use_silero = False

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            logger.debug("sounddevice status: {}", status)
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        if self.metrics is not None:
            # peak abs — O(n) on 512 samples, no alloc beyond the frame copy we already do
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            self.metrics.note_input_peak(peak)
        try:
            self._q.put_nowait(mono.astype(np.float32))
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(mono.astype(np.float32))

    def ensure_stream(self, audio: AudioSettings) -> None:
        """Open or reopen the input stream when device/rate change."""
        needs_reopen = (
            not self._started
            or self._device_id != audio.input_device_id
            or self._sample_rate != audio.sample_rate
            or self._channels != audio.channels
        )
        if not needs_reopen:
            return
        self.stop_stream()
        self._device_id = audio.input_device_id
        self._sample_rate = audio.sample_rate
        self._channels = audio.channels
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self.FRAME_SAMPLES,
                device=self._device_id,
                callback=self._callback,
            )
            self._stream.start()
            self._started = True
            logger.info(
                "Mic stream open device={} rate={}",
                self._device_id,
                self._sample_rate,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to open mic stream: {}", exc)
            self._stream = None
            self._started = False

    def stop_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stream stop error: {}", exc)
        self._stream = None
        self._started = False
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _speech_prob(self, frame: np.ndarray) -> float:
        if self._use_silero and self._model is not None:
            import torch

            tensor = torch.from_numpy(frame)
            with torch.no_grad():
                return float(self._model(tensor, self._sample_rate).item())
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        return min(1.0, rms * 12.0)

    async def capture_utterance(
        self,
        audio: AudioSettings,
        *,
        is_listening: GateFn,
        on_speech: SpeechFn | None = None,
    ) -> np.ndarray | None:
        """Block until an utterance ends, or return None if the gate closes."""
        if not is_listening():
            return None

        self.ensure_stream(audio)
        if not self._started:
            await asyncio.sleep(0.25)
            return None

        silence_limit = max(1, int(audio.silence_duration_ms / 1000 * audio.sample_rate / self.FRAME_SAMPLES))
        threshold = audio.vad_threshold
        chunks: list[np.ndarray] = []
        triggered = False
        silent_frames = 0

        try:
            while is_listening():
                try:
                    frame = await asyncio.to_thread(self._q.get, True, 0.2)
                except queue.Empty:
                    continue

                # pad / trim to FRAME_SAMPLES
                if frame.size < self.FRAME_SAMPLES:
                    frame = np.pad(frame, (0, self.FRAME_SAMPLES - frame.size))
                elif frame.size > self.FRAME_SAMPLES:
                    frame = frame[: self.FRAME_SAMPLES]

                prob = self._speech_prob(frame)
                if not triggered:
                    if prob >= threshold:
                        triggered = True
                        chunks.append(frame)
                        silent_frames = 0
                        if on_speech is not None:
                            await on_speech(True)
                    continue

                chunks.append(frame)
                if prob < threshold:
                    silent_frames += 1
                    if silent_frames >= silence_limit:
                        break
                else:
                    silent_frames = 0

            if not is_listening() or not chunks:
                return None
            return np.concatenate(chunks)
        finally:
            if triggered and on_speech is not None:
                await on_speech(False)
