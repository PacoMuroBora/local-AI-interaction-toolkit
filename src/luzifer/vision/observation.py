"""Latest vision caption shared by pipeline, OSC, and dashboard."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

VISION_TOKEN = "{{vision}}"
_BLANK_GAP = re.compile(r"\n{3,}")


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

    def touch(self) -> bool:
        """Scene unchanged: keep the caption, restart its age. No VLM call."""
        obs = self._latest
        if obs is None or not obs.text:
            return False
        obs.captured_at = time.monotonic()
        return True

    def context_block(self, max_age_s: float) -> str:
        """Private fact for a visitor reply. Not a cue to narrate the room."""
        obs = self._latest
        if obs is None or not obs.text:
            return ""
        if max_age_s > 0 and obs.age > max_age_s:
            return ""
        return (
            "What you currently see (private fact, not someone speaking). "
            "Yes-and one detail as a prop in the reply. Do not recite the note, "
            "do not mention the room, and do not add people or objects that are not in it:\n"
            f"{obs.text}"
        )

    def clear(self) -> None:
        self._latest = None


def apply_vision_placeholder(prompt: str, block: str) -> tuple[str, bool]:
    """Replace {{vision}}. Second value is whether the token was present."""
    if VISION_TOKEN not in (prompt or ""):
        return prompt, False
    text = prompt.replace(VISION_TOKEN, (block or "").strip())
    if not (block or "").strip():
        text = _BLANK_GAP.sub("\n\n", text).strip()
    return text, True
