"""One active aim at a time. The prompt opts in with {{current_goal}}."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from luzifer.config import GoalsSettings

GOAL_TOKEN = "{{current_goal}}"
_BLANK_GAP = re.compile(r"\n{3,}")


def clean_goals(items: list[str] | None) -> list[str]:
    out: list[str] = []
    for item in items or []:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def render_goal_block(goal: str) -> str:
    """Model-facing aim. Empty when there is nothing to pursue."""
    text = (goal or "").strip()
    if not text:
        return ""
    return (
        "Current aim, private. Do not name it and do not say you have one. "
        f"{text} "
        "This replaces any earlier aim. One move toward it in this reply. "
        "If they are not there yet, take the single step that gets you closer."
    )


def apply_goal_placeholder(prompt: str, goal: str) -> tuple[str, bool]:
    """Replace {{current_goal}}. Second value is whether the token was present."""
    if GOAL_TOKEN not in (prompt or ""):
        return prompt, False
    block = render_goal_block(goal)
    text = prompt.replace(GOAL_TOKEN, block)
    if not block:
        text = _BLANK_GAP.sub("\n\n", text).strip()
    return text, True


@dataclass
class GoalClock:
    """Sequential aims. Timer and reset both advance; a one-item list stays put."""

    index: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _timer_on: bool | None = None

    def text(self, goals: GoalsSettings) -> str:
        items = clean_goals(goals.items)
        if not items:
            return ""
        return items[self.index % len(items)]

    def tick(self, goals: GoalsSettings, now: float | None = None) -> str | None:
        """Advance on the interval. Returns the new aim, or None if it did not move."""
        now = time.monotonic() if now is None else now
        items = clean_goals(goals.items)
        running = bool(goals.enabled and goals.interval_s > 0 and len(items) > 1)
        if self._timer_on is None:
            self._timer_on = running
        if items:
            self.index %= len(items)
        if not running:
            self._timer_on = False
            return None
        if not self._timer_on:
            # Timer just started. Do not skip the aim that is already showing.
            self._timer_on = True
            self.started_at = now
            return None
        elapsed = now - self.started_at
        if elapsed < goals.interval_s:
            return None
        steps = int(elapsed // goals.interval_s)
        self.index = (self.index + steps) % len(items)
        self.started_at += steps * float(goals.interval_s)
        return items[self.index]

    def advance(self, goals: GoalsSettings, now: float | None = None) -> str:
        """Next aim, wrapping. Resets the timer. A single aim stays and the clock restarts."""
        now = time.monotonic() if now is None else now
        items = clean_goals(goals.items)
        if not items:
            self.index = 0
            self.started_at = now
            return ""
        if len(items) == 1:
            self.index = 0
        else:
            self.index = (self.index + 1) % len(items)
        self.started_at = now
        return items[self.index]

    def status(self, goals: GoalsSettings, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        items = clean_goals(goals.items)
        count = len(items)
        index = self.index % count if count else 0
        timer_on = bool(goals.enabled and goals.interval_s > 0 and count > 1)
        remaining: float | None = None
        if timer_on:
            elapsed = max(0.0, now - self.started_at)
            remaining = max(0.0, float(goals.interval_s) - elapsed)
        return {
            "enabled": bool(goals.enabled),
            "text": items[index] if count else "",
            "index": index,
            "count": count,
            "interval_s": float(goals.interval_s),
            "remaining_s": None if remaining is None else round(remaining, 1),
            "timer_on": timer_on,
        }
