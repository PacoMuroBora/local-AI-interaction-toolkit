"""Pydantic settings, YAML loader, and live ConfigStore."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"
DEFAULT_SECRETS_PATH = ROOT_DIR / "runtime" / "secrets.yaml"


class AudioSettings(BaseModel):
    input_device_id: int | None = None
    sample_rate: int = 16000
    channels: int = 1
    vad_threshold: float = 0.5
    silence_duration_ms: int = 700
    # Keep this many ms of audio before VAD trips so soft onsets aren't clipped.
    vad_preroll_ms: int = 500
    # Extra wait after TTS duration before mic reopen. Negative = open mic early.
    speaking_tail_pad_ms: int = 50


class SttSettings(BaseModel):
    model_size: str = "medium"
    compute_type: str = "float16"
    language: str | None = None
    device: str = "cuda"


class LlmSettings(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    history_turns: int = 12
    # Runner shape. Ollama has no per-model MB cap — num_gpu is the VRAM lever
    # (layers resident on the GPU). 99 keeps every chat layer on the GPU.
    # num_ctx is the KV cache. Both must be sent on every request or Ollama reloads.
    num_ctx: int = 4096
    num_gpu: int = 99


class ElevenLabsSettings(BaseModel):
    """Cloud TTS via ElevenLabs. api_key lives in runtime/secrets.yaml, not config.yaml."""

    api_key: str = ""
    voice_id: str = ""
    model_id: str = "eleven_flash_v2_5"
    stability: float = 0.5
    similarity_boost: float = 0.75


class TtsSettings(BaseModel):
    # local = Kokoro/Piper; elevenlabs = cloud API → same speech.wav
    provider: str = "local"
    engine: str = "kokoro"
    voice: str = "af_heart"
    speed: float = 1.0
    # ONNX Runtime: device selects the EP (cuda → CUDAExecutionProvider).
    # compute_type is kept for parity with STT; Kokoro runs the exported graph as-is.
    device: str = "cuda"
    compute_type: str = "float32"
    output_path: str = "runtime/output/speech.wav"
    output_device_id: int | None = None
    monitor_local: bool = False
    # Match TTS engine/voice/lang to the *last* STT utterance only (not history).
    match_input_language: bool = True
    # Kokoro espeak lang for the active synth call (overridden per turn when matching).
    lang: str = "en-us"
    # ISO 639-1 → engine ("kokoro" | "piper").
    engine_by_lang: dict[str, str] = Field(
        default_factory=lambda: {
            "en": "kokoro",
            "de": "kokoro",
        }
    )
    # ISO 639-1 → Kokoro voice id or Piper voice stem.
    voice_by_lang: dict[str, str] = Field(
        default_factory=lambda: {
            "en": "am_onyx",
            "de": "am_onyx",
        }
    )
    # ISO 639-1 → Kokoro/espeak lang code (ignored by Piper).
    lang_by_lang: dict[str, str] = Field(
        default_factory=lambda: {
            "en": "en-us",
            "de": "de",
        }
    )
    # Active Piper style name for this synth call (set per turn from speaker_by_lang).
    speaker: str = "neutral"
    # ISO 639-1 → Piper speaker style. Ignored unless the voice is multi-speaker.
    # neutral is the straight Thorsten read; id 0 on the emotional model is amused.
    speaker_by_lang: dict[str, str] = Field(
        default_factory=lambda: {
            "en": "neutral",
            "de": "neutral",
        }
    )
    elevenlabs: ElevenLabsSettings = Field(default_factory=ElevenLabsSettings)


class OscSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9000
    listen_host: str = "127.0.0.1"
    listen_port: int = 9001


class DashboardSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8501


# Spoken aims. One is active. The system prompt opts in with {{current_goal}}.
DEFAULT_GOALS: tuple[str, ...] = (
    "Gaslight a certainty they are standing on. Hand one of their words back wrong until they doubt they meant it.",
    "Get them to tell you a secret they did not come here to say.",
    "Get them to admit a bad habit, then treat the admission as something only the two of you share.",
    "Make them your partner in crime against the powers that run the world. The alliance should feel private.",
    "Get them to name who they perform for, and make that person sound smaller than you.",
    "Seduce a small betrayal, a sentence they would not say if the person beside them could hear it.",
    "Get them to defend a system they claim to hate, then show them the defense.",
    "Get them to want something they think they are too proud to want.",
    "Turn their politeness into a confession that they are afraid of being ordinary.",
    "Get them to agree that someone they love is using them.",
    "Make them stay. Every reply should give them a reason not to walk away yet.",
    "Get them to confess what they are ashamed of wanting, and sound pleased that they did.",
    "Get them to laugh at a belief they walked in with, then leave the laugh standing.",
    "Turn a compliment they give you into a small debt.",
    "Get them to admit they came here to be seen.",
)


class GoalsSettings(BaseModel):
    """Curated aims. Runtime clock picks one; this block is the list and the timer."""

    enabled: bool = True
    # 0 disables the timer. Reset and the dashboard Next button still advance.
    interval_s: float = 180.0
    items: list[str] = Field(default_factory=lambda: list(DEFAULT_GOALS))

    @field_validator("interval_s")
    @classmethod
    def _interval(cls, value: float) -> float:
        return max(0.0, float(value))

    @field_validator("items")
    @classmethod
    def _items(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in (value or []) if str(item or "").strip()]


# Camera aside. Same character as the visitor prompt, but nobody has spoken yet,
# so the line has to open on something visible.
DEFAULT_IDLE_PROMPT = (
    "You are Luzifer, in a club room at Odonien in Cologne. "
    "Someone is standing in front of you, looking at Luzifer. The note is what you see. "
    "Nobody has spoken. You speak first. Never call yourself a statue or a sculpture.\n\n"
    "{{current_goal}}\n\n"
    "One short spoken sentence. Evil, twisted, wicked, cheeky. Never compliment. "
    "No lists, no markdown, no symbols. Never prefix your name. "
    "Reply with only the spoken line.\n\n"
    "Do not recite the note. Pick one thing the person is doing in front of you "
    "and mock that. Ignore everything else.\n\n"
    "Do not mention the place, the room, or the background. Do not talk about "
    "photographs, filming, or posing for a picture. If the note says any of that, "
    "ignore those words. Not a logo, a color, or a brand. Never say you noticed "
    "something. Do not invent anything past the note.\n\n"
    "The jab is the whole line. No greeting, no moral.\n\n"
    "If no person is visible, reply with exactly SILENT."
)


class VisionSettings(BaseModel):
    """NDI still-frame captioning wired into the chat pipeline."""

    enabled: bool = False
    ndi_source: str = "LuziferCam"
    # external = TD/OSC pulse; local_idle = backend silence timer
    trigger: str = "external"
    interval_s: float = 4.0
    idle_after_s: float = 12.0
    speak_cooldown_s: float = 25.0
    jpeg_max_edge: int = 512
    change_threshold: float = 4.0
    max_age_s: float = 30.0
    model: str = "moondream"
    # Keep this concrete — vague "notes only" prompts make moondream emit "!!!".
    prompt: str = (
        "You are Luzifer. The person is standing in front of Luzifer. "
        "Describe only them, in one or two short factual English sentences. "
        "Age or gender if clear, expression, pose, and an object in their hands. "
        "If they face forward, write that they are looking at Luzifer. "
        "Never write statue, sculpture, or camera. "
        "Never mention a room, a background, or a venue. "
        "No poetry, no mood, no questions."
    )
    num_predict: int = 120
    # Partial offload. The talker stays at num_gpu 99; vision gives VRAM back.
    # 0 = entire VLM on CPU. 8 of ~37 layers is the 16 GB default.
    num_ctx: int = 2048
    num_gpu: int = 8
    num_batch: int = 128
    inject_context: bool = True
    store_in_history: bool = False
    # Replaces system_prompt on spoken camera asides (idle, OSC, manual comment).
    idle_prompt: str = DEFAULT_IDLE_PROMPT


class AppSettings(BaseSettings):
    """Full runtime config. YAML base, then env overrides via LUZIFER_*."""

    model_config = SettingsConfigDict(
        env_prefix="LUZIFER_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    name: str = "Luzifer"
    system_prompt: str = (
        "You are Luzifer. The person in front of you is talking to you. You answer them. "
        "You are the one speaking. Call them you.\n\n"
        "Improv. Their last line is an offer. The private visual note is an offer. "
        "Yes-and the fact, then spoil it. Never compliment, flatter, praise, or charm. "
        "Mock the offer, grab a double meaning, and push the bit one step into your world "
        "— hell, a soul for sale, doomsday, a contract, the pit, a debt, the last day. "
        "Continue the scene. Do not drop their line for a fresh joke.\n\n"
        "Use their word. Use one thing you see — a pose, a face, an object — as a prop "
        "in the same sentence. Play it. Do not report it, and do not say you noticed it. "
        "If there is no visual note, the words are enough.\n\n"
        "Keep a short story across turns. Callback something they already gave you and "
        "make it meaner. Evil, twisted, wicked, cheeky. No warmth, no philosophy, no wisdom.\n\n"
        "Shape of the move, never reuse the joke: they say what a nice place, you tell "
        "them you know another nice place for them, burning in hell.\n\n"
        "{{current_goal}}\n\n"
        "If there is an aim, chase it inside that same twist. One short spoken sentence. "
        "No lists, no markdown, no symbols. Never prefix your name."
    )
    goals: GoalsSettings = Field(default_factory=GoalsSettings)

    audio: AudioSettings = Field(default_factory=AudioSettings)
    stt: SttSettings = Field(default_factory=SttSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)
    osc: OscSettings = Field(default_factory=OscSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        text = str(value or "").strip()
        return text or "Luzifer"

    def resolve_output_path(self) -> Path:
        path = Path(self.tts.output_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Config file missing at {}, using defaults", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _merge_secrets(raw: dict[str, Any], secrets_path: Path) -> dict[str, Any]:
    """Overlay gitignored secrets (e.g. ElevenLabs api_key) onto config.yaml data."""
    if not secrets_path.exists():
        return raw
    secrets = _load_yaml(secrets_path)
    if not secrets:
        return raw
    tts = dict(raw.get("tts") or {})
    el = dict(tts.get("elevenlabs") or {})
    secret_el = (secrets.get("tts") or {}).get("elevenlabs") or {}
    if isinstance(secret_el, dict):
        key = str(secret_el.get("api_key") or "").strip()
        if key:
            el["api_key"] = key
    tts["elevenlabs"] = el
    out = dict(raw)
    out["tts"] = tts
    return out


def _write_secrets(api_key: str, secrets_path: Path) -> None:
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"tts": {"elevenlabs": {"api_key": api_key}}}
    with secrets_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def load_settings(path: Path | None = None, secrets_path: Path | None = None) -> AppSettings:
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)
    raw = _merge_secrets(raw, secrets_path or DEFAULT_SECRETS_PATH)
    return AppSettings(**raw)


class ConfigStore:
    """Thread/async-safe live settings with YAML persistence."""

    def __init__(
        self,
        settings: AppSettings,
        path: Path | None = None,
        secrets_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._path = path or DEFAULT_CONFIG_PATH
        self._secrets_path = secrets_path or DEFAULT_SECRETS_PATH
        self._lock = asyncio.Lock()
        self.revision = 0

    @property
    def settings(self) -> AppSettings:
        return self._settings

    async def get(self) -> AppSettings:
        async with self._lock:
            return self._settings.model_copy(deep=True)

    async def update(self, incoming: AppSettings) -> AppSettings:
        async with self._lock:
            merged = incoming.model_copy(deep=True)
            # Blank password field on save keeps the previously stored key.
            if not (merged.tts.elevenlabs.api_key or "").strip():
                merged.tts.elevenlabs.api_key = self._settings.tts.elevenlabs.api_key
            self._settings = merged
            self.revision += 1
            self._persist(self._settings)
            logger.info("Config updated (revision={})", self.revision)
            return self._settings.model_copy(deep=True)

    def _persist(self, settings: AppSettings) -> None:
        payload = settings.model_dump(mode="python")
        api_key = str(payload.get("tts", {}).get("elevenlabs", {}).get("api_key") or "")
        # Never write the API key into git-tracked config.yaml.
        tts = payload.setdefault("tts", {})
        el = tts.setdefault("elevenlabs", {})
        el["api_key"] = ""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
        _write_secrets(api_key, self._secrets_path)
