"""Latest vision caption shared by pipeline, OSC, and dashboard."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Observation:
    text: str
    captured_at: float
    mae: float = 0.0

    @property
    def age(self) -> float:
        return time.monotonic() - self.captured_at


class ObservationStore:
    def __init__(self) -> None:
        self._latest: Observation | None = None

    @property
    def latest(self) -> Observation | None:
        return self._latest

    @property
    def text(self) -> str:
        return self._latest.text if self._latest else ""

    def update(self, text: str, *, mae: float = 0.0) -> Observation:
        obs = Observation(text=text.strip(), captured_at=time.monotonic(), mae=mae)
        self._latest = obs
        return obs

    def context_block(self, max_age_s: float) -> str:
        obs = self._latest
        if obs is None or not obs.text:
            return ""
        if max_age_s > 0 and obs.age > max_age_s:
            return ""
        return (
            "\n\nWhat you currently see (fact, not the visitor speaking; "
            "do not invent beyond it):\n"
            f"{obs.text}"
        )

    def clear(self) -> None:
        self._latest = None
