"""Operator-triggered engagement decks: quiz, preach, spectacle.

Cursors live in ``runtime/engagement.json`` so a restart does not repeat card 0.
The overlay prompt exists only for the one LLM beat the card needs.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from luzifer.config import (
    ROOT_DIR,
    EngagementSettings,
    PreachCard,
    QuizCard,
    SpectacleCard,
)

EngagementMode = Literal["idle", "quiz", "preach", "spectacle"]
EngagementPhase = Literal[
    "idle",
    "ask",
    "await",
    "comment",
    "story",
    "reply",
    "hold",
    "line",
    "vision",
]

DEFAULT_CURSOR_PATH = ROOT_DIR / "runtime" / "engagement.json"
_DECK_KEYS = ("quiz", "preach", "spectacle")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def preach_hook(text: str) -> str:
    """Last sentence of a preach script — the barbed leave-behind."""
    body = (text or "").strip()
    if not body:
        return ""
    parts = [p.strip() for p in _SENTENCE_END.split(body) if p.strip()]
    return parts[-1] if parts else body


def render_engagement_prompt(template: str, values: dict[str, str]) -> str:
    """Replace ``{{token}}`` placeholders. Missing keys become empty strings."""
    out = template or ""
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value or "")
    # Strip leftover known tokens so a blank vision does not leave braces.
    for key in ("question", "truth", "angle", "vision", "title", "hook"):
        out = out.replace("{{" + key + "}}", "")
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _deck_cards(settings: EngagementSettings, mode: str) -> list[Any]:
    if mode == "quiz":
        return list(settings.quiz or [])
    if mode == "preach":
        return list(settings.preach or [])
    if mode == "spectacle":
        return list(settings.spectacle or [])
    return []


def _card_id(card: Any) -> str:
    return str(getattr(card, "id", "") or "").strip()


def _load_cursors(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("engagement cursor load failed: {}", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_cursors(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class EngagementRuntime:
    """In-memory piece state + persisted deck cursors."""

    path: Path = field(default_factory=lambda: DEFAULT_CURSOR_PATH)
    mode: EngagementMode = "idle"
    phase: EngagementPhase = "idle"
    card_id: str = ""
    last_mode: EngagementMode | None = None
    last_card_id: str = ""
    last_rendered_prompt: str = ""
    show: bool = False
    show_s: float = 0.0
    show_ends_at: float | None = None
    await_deadline: float | None = None
    quiz_card: QuizCard | None = None
    preach_card: PreachCard | None = None
    spectacle_card: SpectacleCard | None = None
    cursors: dict[str, int] = field(default_factory=lambda: {k: 0 for k in _DECK_KEYS})

    def __post_init__(self) -> None:
        self.load()

    def load(self) -> None:
        raw = _load_cursors(self.path)
        for key in _DECK_KEYS:
            try:
                self.cursors[key] = max(0, int(raw.get(key, 0)))
            except (TypeError, ValueError):
                self.cursors[key] = 0
        last_mode = str(raw.get("last_mode") or "").strip()
        if last_mode in _DECK_KEYS:
            self.last_mode = last_mode  # type: ignore[assignment]
        self.last_card_id = str(raw.get("last_card_id") or "").strip()

    def save(self) -> None:
        payload = {
            "quiz": int(self.cursors.get("quiz", 0)),
            "preach": int(self.cursors.get("preach", 0)),
            "spectacle": int(self.cursors.get("spectacle", 0)),
            "last_mode": self.last_mode or "",
            "last_card_id": self.last_card_id,
        }
        try:
            _save_cursors(self.path, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("engagement cursor save failed: {}", exc)

    @property
    def active(self) -> bool:
        return self.mode != "idle" or self.phase != "idle"

    def is_awaiting(self) -> bool:
        return self.phase == "await" and self.mode in {"quiz", "preach"}

    def peek(self, settings: EngagementSettings, mode: str) -> tuple[Any | None, int, int]:
        """Next card without advancing. Returns (card, index, count)."""
        cards = _deck_cards(settings, mode)
        count = len(cards)
        if not count:
            return None, 0, 0
        index = self.cursors.get(mode, 0) % count
        return cards[index], index, count

    def find_card(
        self,
        settings: EngagementSettings,
        mode: str,
        card_id: str,
    ) -> tuple[Any | None, int, int]:
        cards = _deck_cards(settings, mode)
        count = len(cards)
        if not count:
            return None, 0, 0
        wanted = (card_id or "").strip()
        for i, card in enumerate(cards):
            if _card_id(card) == wanted:
                return card, i, count
        return None, 0, count

    def take_next(self, settings: EngagementSettings, mode: str) -> tuple[Any | None, int, int]:
        """Return the next card and advance the cursor past it."""
        card, index, count = self.peek(settings, mode)
        if card is None or count <= 0:
            return None, 0, 0
        self.cursors[mode] = (index + 1) % count
        self.last_mode = mode  # type: ignore[assignment]
        self.last_card_id = _card_id(card)
        self.save()
        return card, index, count

    def take_replay(self, settings: EngagementSettings, mode: str | None = None) -> tuple[Any | None, int, int]:
        """Replay the last fired card without moving the cursor."""
        use_mode = mode or self.last_mode
        if use_mode not in _DECK_KEYS:
            return None, 0, 0
        card, index, count = self.find_card(settings, use_mode, self.last_card_id)
        if card is None:
            # Fall back to peek without advancing if last id vanished.
            return self.peek(settings, use_mode)
        return card, index, count

    def skip(self, settings: EngagementSettings, mode: str) -> tuple[Any | None, int, int]:
        """Advance past the next card without playing it. Returns the skipped card."""
        return self.take_next(settings, mode)

    def begin(
        self,
        mode: EngagementMode,
        card: Any,
        *,
        phase: EngagementPhase,
    ) -> None:
        self.mode = mode
        self.phase = phase
        self.card_id = _card_id(card)
        self.quiz_card = card if mode == "quiz" else None
        self.preach_card = card if mode == "preach" else None
        self.spectacle_card = card if mode == "spectacle" else None
        self.show = False
        self.show_s = 0.0
        self.show_ends_at = None
        self.await_deadline = None
        self.last_rendered_prompt = ""
        self.last_mode = mode if mode != "idle" else self.last_mode
        if self.card_id:
            self.last_card_id = self.card_id
            self.save()

    def set_phase(self, phase: EngagementPhase) -> None:
        self.phase = phase

    def arm_await(self, await_s: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.phase = "await"
        self.await_deadline = now + max(5.0, float(await_s))

    def bump_await(self, await_s: float, now: float | None = None) -> None:
        if not self.is_awaiting():
            return
        now = time.monotonic() if now is None else now
        self.await_deadline = now + max(5.0, float(await_s))

    def await_expired(self, now: float | None = None) -> bool:
        if not self.is_awaiting() or self.await_deadline is None:
            return False
        now = time.monotonic() if now is None else now
        return now >= self.await_deadline

    def await_remaining_s(self, now: float | None = None) -> float | None:
        if not self.is_awaiting() or self.await_deadline is None:
            return None
        now = time.monotonic() if now is None else now
        return max(0.0, self.await_deadline - now)

    def begin_hold(self, seconds: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        secs = max(0.0, float(seconds))
        self.phase = "hold"
        self.show = True
        self.show_s = secs
        self.show_ends_at = now + secs if secs > 0 else now

    def end_hold(self) -> None:
        self.show = False
        self.show_s = 0.0
        self.show_ends_at = None

    def show_remaining_s(self, now: float | None = None) -> float | None:
        if not self.show or self.show_ends_at is None:
            return None
        now = time.monotonic() if now is None else now
        return max(0.0, self.show_ends_at - now)

    def clear(self) -> None:
        """End the piece. Cursor position stays."""
        self.mode = "idle"
        self.phase = "idle"
        self.card_id = ""
        self.quiz_card = None
        self.preach_card = None
        self.spectacle_card = None
        self.show = False
        self.show_s = 0.0
        self.show_ends_at = None
        self.await_deadline = None
        # Keep last_rendered_prompt for the Floor preview until the next fire.

    def status(self, settings: EngagementSettings, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        decks: dict[str, Any] = {}
        for key in _DECK_KEYS:
            card, index, count = self.peek(settings, key)
            preview = ""
            title = ""
            if card is not None:
                title = _card_id(card) or getattr(card, "title", "") or ""
                if key == "quiz":
                    preview = str(getattr(card, "question", "") or "")
                elif key == "preach":
                    preview = str(getattr(card, "title", "") or "")
                else:
                    preview = str(getattr(card, "title", "") or "")
            decks[key] = {
                "next_index": index,
                "count": count,
                "next_id": _card_id(card) if card else "",
                "next_title": title,
                "next_preview": preview[:160],
            }
        return {
            "mode": self.mode,
            "phase": self.phase,
            "card_id": self.card_id,
            "last_mode": self.last_mode or "",
            "last_card_id": self.last_card_id,
            "last_rendered_prompt": self.last_rendered_prompt,
            "show": bool(self.show),
            "show_s": float(self.show_s) if self.show else 0.0,
            "show_remaining_s": None
            if self.show_remaining_s(now) is None
            else round(self.show_remaining_s(now) or 0.0, 1),
            "await_remaining_s": None
            if self.await_remaining_s(now) is None
            else round(self.await_remaining_s(now) or 0.0, 1),
            "await_s": float(settings.await_s),
            "decks": decks,
        }
