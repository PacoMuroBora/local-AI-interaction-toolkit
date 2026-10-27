"""One active campaign at a time. A beat stays for ``hold`` spoken replies.

A campaign line is ``title | beat | beat``. The prompt opts in with {{current_goal}}.
"""

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


def split_campaign(item: str) -> tuple[str, list[str]]:
    """``title | beat | beat``. A line with no bar is a one-beat campaign."""
    parts = [part.strip() for part in str(item or "").split("|") if part.strip()]
    if not parts:
        return "", []
    if len(parts) == 1:
        return parts[0], parts
    return parts[0], parts[1:]


def render_goal_block(goal: str) -> str:
    """Model-facing beat. Empty when there is nothing to pursue."""
    text = (goal or "").strip()
    if not text:
        return ""
    return f"Private. Do not say this. {text}"


def insert_before_goal(prompt: str, extra: str) -> str:
    """Place extra immediately before {{current_goal}} so the beat stays last."""
    extra = extra or ""
    prompt = prompt or ""
    if not extra:
        return prompt
    if GOAL_TOKEN in prompt:
        return prompt.replace(GOAL_TOKEN, f"{extra}\n\n{GOAL_TOKEN}", 1)
    return prompt + extra


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
    """Sequential campaigns. A beat covers ``hold`` replies. Timer and reset change the campaign."""

    index: int = 0
    beat: int = 0
    # Spoken replies already spent on the current beat.
    uses: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _timer_on: bool | None = None

    def _parts(self, goals: GoalsSettings) -> tuple[str, list[str]]:
        items = clean_goals(goals.items)
        if not items:
            self.index = 0
            self.beat = 0
            self.uses = 0
            return "", []
        self.index %= len(items)
        title, beats = split_campaign(items[self.index])
        if not beats:
            self.beat = 0
            self.uses = 0
            return title, []
        self.beat %= len(beats)
        return title, beats

    def text(self, goals: GoalsSettings) -> str:
        """The beat injected into the prompt."""
        _, beats = self._parts(goals)
        if not beats:
            return ""
        return beats[self.beat]

    def tick(self, goals: GoalsSettings, now: float | None = None) -> str | None:
        """Advance the campaign on the interval. Returns the new beat, or None if it did not move."""
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
            # Timer just started. Do not skip the campaign that is already showing.
            self._timer_on = True
            self.started_at = now
            return None
        elapsed = now - self.started_at
        if elapsed < goals.interval_s:
            return None
        steps = int(elapsed // goals.interval_s)
        self.index = (self.index + steps) % len(items)
        self.started_at += steps * float(goals.interval_s)
        self.beat = 0
        self.uses = 0
        return self.text(goals)

    def advance(self, goals: GoalsSettings, now: float | None = None) -> str:
        """Next campaign, wrapping. Resets the beat and the timer."""
        now = time.monotonic() if now is None else now
        items = clean_goals(goals.items)
        self.beat = 0
        self.uses = 0
        if not items:
            self.index = 0
            self.started_at = now
            return ""
        if len(items) == 1:
            self.index = 0
        else:
            self.index = (self.index + 1) % len(items)
        self.started_at = now
        return self.text(goals)

    def advance_beat(self, goals: GoalsSettings) -> str | None:
        """Count a spoken reply. Returns the new beat only when the hold is spent."""
        if not goals.enabled:
            return None
        _, beats = self._parts(goals)
        if not beats:
            return None
        hold = max(1, int(goals.hold))
        self.uses += 1
        if self.uses < hold:
            return None
        self.uses = 0
        if len(beats) <= 1:
            self.beat = 0
            return None
        self.beat = (self.beat + 1) % len(beats)
        return beats[self.beat]

    def status(self, goals: GoalsSettings, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        items = clean_goals(goals.items)
        count = len(items)
        title, beats = self._parts(goals)
        beat_count = len(beats)
        timer_on = bool(goals.enabled and goals.interval_s > 0 and count > 1)
        remaining: float | None = None
        if timer_on:
            elapsed = max(0.0, now - self.started_at)
            remaining = max(0.0, float(goals.interval_s) - elapsed)
        return {
            "enabled": bool(goals.enabled),
            "text": beats[self.beat] if beat_count else "",
            "campaign": title,
            "beat": self.beat if beat_count else 0,
            "beat_count": beat_count,
            "uses": self.uses,
            "hold": max(1, int(goals.hold)),
            "index": self.index if count else 0,
            "count": count,
            "interval_s": float(goals.interval_s),
            "remaining_s": None if remaining is None else round(remaining, 1),
            "timer_on": timer_on,
        }
