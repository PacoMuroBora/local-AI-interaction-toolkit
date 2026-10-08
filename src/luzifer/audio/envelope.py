"""Peak envelope of a 16-bit wav, one float per hop. For OSC meters, not playback."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def peak_envelope(path: Path, hz: float = 30.0) -> np.ndarray:
    """Max |sample| per hop, linear 0..1. Empty or unreadable files return [0]."""
    try:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate() or 1
            channels = wf.getnchannels() or 1
            width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
    except Exception:  # noqa: BLE001
        return np.zeros(1, dtype=np.float32)

    if width != 2 or not raw:
        return np.zeros(1, dtype=np.float32)

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        usable = (pcm.size // channels) * channels
        pcm = np.abs(pcm[:usable].reshape(-1, channels)).max(axis=1)
    else:
        pcm = np.abs(pcm)

    hop = max(1, int(round(sample_rate / hz)))
    hops = pcm.size // hop
    if hops <= 0:
        peak = float(pcm.max()) if pcm.size else 0.0
        return np.array([peak], dtype=np.float32)
    return pcm[: hops * hop].reshape(hops, hop).max(axis=1).astype(np.float32)
