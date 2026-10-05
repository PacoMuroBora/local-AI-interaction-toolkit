"""Lightweight live metrics (GPU / VRAM / audio levels). Approximations only."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from typing import Any


class MetricsStore:
    """Thread-safe-ish floats updated from audio callbacks + nvidia-smi."""

    def __init__(self) -> None:
        self.input_level = 0.0
        self.output_level = 0.0
        self._out_hold = 0.0
        self.gpu_name = ""
        self.gpu_util = 0.0
        self.vram_used_mb = 0.0
        self.vram_total_mb = 0.0
        self.gpu_available = False

    def note_input_peak(self, peak: float) -> None:
        # Cheap peak-hold; decay happens in tick()
        p = abs(float(peak))
        if p > self.input_level:
            self.input_level = min(1.0, p)

    def note_output_peak(self, peak: float) -> None:
        p = min(1.0, abs(float(peak)))
        self._out_hold = max(self._out_hold, p)
        self.output_level = self._out_hold

    def tick(self, *, speaking: bool = False) -> None:
        self.input_level *= 0.82
        if speaking and self._out_hold > 0.05:
            # Approximate envelope while TTS plays (no sample streaming)
            self.output_level = max(0.08, self._out_hold * 0.75 + self.output_level * 0.25)
            self._out_hold *= 0.985
        else:
            self.output_level *= 0.78
            self._out_hold *= 0.9

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "metrics",
            "input_level": round(self.input_level, 3),
            "output_level": round(self.output_level, 3),
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_util": round(self.gpu_util, 1),
            "vram_used_mb": round(self.vram_used_mb, 0),
            "vram_total_mb": round(self.vram_total_mb, 0),
        }


def _query_nvidia_smi() -> dict[str, Any] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None
        return {
            "name": parts[0],
            "util": float(parts[1]),
            "vram_used": float(parts[2]),
            "vram_total": float(parts[3]),
        }
    except Exception:  # noqa: BLE001
        return None


async def poll_gpu(store: MetricsStore) -> None:
    data = await asyncio.to_thread(_query_nvidia_smi)
    if data is None:
        store.gpu_available = False
        return
    store.gpu_available = True
    store.gpu_name = str(data["name"])
    store.gpu_util = float(data["util"])
    store.vram_used_mb = float(data["vram_used"])
    store.vram_total_mb = float(data["vram_total"])
