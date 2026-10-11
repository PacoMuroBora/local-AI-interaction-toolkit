"""Pydantic settings, YAML loader, and live ConfigStore."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


class AudioSettings(BaseModel):
    input_device_id: int | None = None
    sample_rate: int = 16000
    channels: int = 1
    vad_threshold: float = 0.5
    silence_duration_ms: int = 700
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


class TtsSettings(BaseModel):
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


class OscSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9000
    listen_host: str = "127.0.0.1"
    listen_port: int = 9001


class DashboardSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8501


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
    prompt: str = (
        "Factual notes only, one or two short sentences, English. "
        "People, motion, objects, light. No mood, no story."
    )
    num_predict: int = 80
    inject_context: bool = True
    store_in_history: bool = False


class AppSettings(BaseSettings):
    """Full runtime config. YAML base, then env overrides via LUZIFER_*."""

    model_config = SettingsConfigDict(
        env_prefix="LUZIFER_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    system_prompt: str = (
        "You are Luzifer, an interactive installation character. "
        "Keep answers short, vivid, and suitable for spoken TTS."
    )
    audio: AudioSettings = Field(default_factory=AudioSettings)
    stt: SttSettings = Field(default_factory=SttSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)
    osc: OscSettings = Field(default_factory=OscSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)

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


def load_settings(path: Path | None = None) -> AppSettings:
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)
    return AppSettings(**raw)


class ConfigStore:
    """Thread/async-safe live settings with YAML persistence."""

    def __init__(self, settings: AppSettings, path: Path | None = None) -> None:
        self._settings = settings
        self._path = path or DEFAULT_CONFIG_PATH
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
            self._settings = incoming.model_copy(deep=True)
            self.revision += 1
            self._persist(self._settings)
            logger.info("Config updated (revision={})", self.revision)
            return self._settings.model_copy(deep=True)

    def _persist(self, settings: AppSettings) -> None:
        payload = settings.model_dump(mode="python")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
