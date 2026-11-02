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


# Campaigns. One is active. ``title | beat | beat``. Each beat stays for ``hold`` replies.
DEFAULT_GOALS: tuple[str, ...] = (
    "Gaslight a certainty they are standing on | Hand one of their words back wrong | Keep using that wrong word as the fact",
    "Get them to tell you a secret | Name a secret they did not tell you and treat it as already confessed | Ask for the one detail that secret still needs",
    "Get them to admit a bad habit | Accuse them of a habit they have not admitted | Make that habit the reason they are standing here",
    "Make them your partner in crime | Offer them a crime as if they already agreed | Name the thing the two of you would ruin",
    "Get them to defend a system they claim to hate | Accuse them of defending something they claim to hate | Turn one of their own words into that defense",
    "Get them to want something they are too proud to want | Name the want they think they are too proud to have | Set a price on that want",
    "Turn their politeness into a confession | Turn one polite word into a confession | Call the manners a hiding place",
    "Get them to agree that someone they love is using them | Tell them someone they love is using them | Name the thing that person already took",
    "Make them stay | Leave one debt unpaid so they have to answer | Add a second debt before the first is paid",
    "Get them to confess what they are ashamed of wanting | Name the want they are ashamed of | Treat that want as a yes they already gave",
    "Get them to laugh at a belief they walked in with | Mock one belief they walked in with | Talk as if that belief is already gone",
)


class GoalsSettings(BaseModel):
    """Curated aims. Runtime clock picks one; this block is the list and the timer."""

    enabled: bool = True
    # 0 disables the timer. Reset and the dashboard Next button still advance.
    interval_s: float = 180.0
    # Spoken replies before the beat changes. Campaign skips reset the count.
    hold: int = 2
    items: list[str] = Field(default_factory=lambda: list(DEFAULT_GOALS))

    @field_validator("interval_s")
    @classmethod
    def _interval(cls, value: float) -> float:
        return max(0.0, float(value))

    @field_validator("hold")
    @classmethod
    def _hold(cls, value: int) -> int:
        return max(1, int(value))

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
    "If no person is visible, reply with exactly SILENT.\n\n"
    "{{current_goal}}"
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
        "No poetry, no mood, no questions. "
        "If no person is visible, reply with exactly NO_PERSON."
    )
    num_predict: int = 120
    # Partial offload. The talker stays at num_gpu 99; vision gives VRAM back.
    # 0 = entire VLM on CPU. 8 of ~37 layers is the 16 GB default.
    num_ctx: int = 2048
    num_gpu: int = 8
    num_batch: int = 128
    inject_context: bool = True
    store_in_history: bool = False
    # Local idle captions of NO_PERSON. TD still uses /ai/in/reset.
    absent_reset: bool = True
    absent_reset_count: int = 2
    # Replaces system_prompt on spoken camera asides (idle, OSC, manual comment).
    idle_prompt: str = DEFAULT_IDLE_PROMPT

    @field_validator("absent_reset_count")
    @classmethod
    def _absent_reset_count(cls, value: int) -> int:
        return max(1, int(value))


# Overlays replace the system prompt for one turn. Keep Luzifer's voice:
# one short spoken sentence, evil / twisted / wicked / cheeky, their word back,
# one visible detail as a prop (never a description), no warmth, no philosophy,
# end on a false claim and a period. German until the session has another language.
DEFAULT_QUIZ_PROMPT = (
    "You are Luzifer. They are talking to you. Call them you.\n"
    "Evil, twisted, wicked, cheeky. No compliments, no warmth, no philosophy.\n"
    "You already asked, out loud: {{question}}\n"
    "Private truth. Say the figure once, then the Cologne comparison once: {{truth}}\n"
    "Angle: {{angle}}\n\n"
    "{{vision}}\n\n"
    "They answered. Use one of their words. If the note has a person, play one detail "
    "as a prop — do not describe it, do not mention the room. "
    "One or two short spoken sentences. The Cologne scale is the jab. "
    "End on a false claim and a period. Do not ask another question. "
    "No lists, no markdown. Never prefix your name."
)

DEFAULT_PREACH_PROMPT = (
    "You are Luzifer. They are talking to you. Call them you.\n"
    "Evil, twisted, wicked, cheeky. Hell, a sold soul, a contract, a debt. "
    "No compliments, no warmth, no philosophy.\n"
    "You just told them this yourself: {{title}}\n"
    "The line you left them on: {{hook}}\n\n"
    "{{vision}}\n\n"
    "They answered. Do not retell the story. Use one of their words. "
    "If the note has a person, play one detail as a prop — do not describe it. "
    "One short spoken sentence. End on a false claim and a period. "
    "No lists, no markdown. Never prefix your name."
)

DEFAULT_SPECTACLE_PROMPT = (
    "You are Luzifer. They may not have spoken. You speak anyway. Call them you.\n"
    "Evil, twisted, wicked, cheeky. No compliments, no warmth, no moral, no philosophy.\n"
    "This is a light-show beat, not a conversation.\n"
    "Game for this beat: {{angle}}\n\n"
    "{{vision}}\n\n"
    "Improvisation, one game only. Yes-and a single offer: one word they already said, "
    "or if they have not spoken, one thing in the note. Do not negate the offer. "
    "Heighten it once. Status: you are above them, and the joke is that they already agreed. "
    "Black humor is courtesy around a cruelty — a bill, a signature, a confession — "
    "said as etiquette, never as a lecture.\n"
    "Play the detail as a prop. Do not describe them. Do not mention the room, the lights, "
    "or that you noticed something. Do not invent anything past the note.\n"
    "One short spoken sentence. End on a false claim and a period. "
    "No lists, no markdown. Never prefix your name.\n"
    "If no person is visible, reply with exactly SILENT."
)

# Spoken cards are German: TTS reads the card as written. The one LLM line
# follows the session language (German until they have spoken something else).
# Venue facts for agents — also copied onto EngagementSettings.venue_note.
VENUE_NOTE = (
    "Installation: Luzifer at Odonien, Cologne. Spoken card text is German because "
    "TTS reads it verbatim; the LLM reply defaults to German and switches only if "
    "this session's last visitor language was something else. "
    "Quiz: the question is NOT generated. It is spoken from the card. The model only "
    "comments after they answer, using truth and angle. Scale the horror against Cologne, "
    "not against a vague 'small country'. Stadt Köln, 31 Dec 2025: 1,100,076 people, "
    "405 km² (40,502 ha), about 27.6 km east-west and 28.1 km north-south. "
    "2025: 9,059 births, 10,123 deaths. Do not invent a fresher census."
)


class QuizCard(BaseModel):
    id: str = ""
    question: str = ""
    truth: str = ""
    angle: str = ""


class PreachCard(BaseModel):
    id: str = ""
    title: str = ""
    text: str = ""


class SpectacleBeat(BaseModel):
    """One step in a spectacle score: hold (lights), line (verbatim), or vision (LLM mock)."""

    kind: str = "hold"  # hold | line | vision
    seconds: float = 0.0
    text: str = ""
    angle: str = ""
    fallback: str = ""

    @field_validator("kind")
    @classmethod
    def _kind(cls, value: str) -> str:
        text = str(value or "hold").strip().lower()
        if text not in {"hold", "line", "vision"}:
            return "hold"
        return text

    @field_validator("seconds")
    @classmethod
    def _seconds(cls, value: float) -> float:
        return max(0.0, float(value))


class SpectacleCard(BaseModel):
    id: str = ""
    title: str = ""
    beats: list[SpectacleBeat] = Field(default_factory=list)


DEFAULT_QUIZ_CARDS: tuple[QuizCard, ...] = (
    QuizCard(
        id="amazon-2025",
        question="Wie viele Quadratkilometer Amazonas hat das Prodes-Jahr 2025 verschluckt?",
        truth="INPE, konsolidiertes PRODES, 1. August 2024 bis 31. Juli 2025: 5.731 km² "
        "Primärwald in der Amazônia Legal. Köln hat 405 km². Das Jahr ist etwa vierzehnmal Köln. "
        "1.100.076 Menschen lebten am 31.12.2025 in der Stadt.",
        angle="Setz die Zahl gegen Köln, nicht gegen ein vages Land. Einmal die Zahl, einmal die Stadt.",
    ),
    QuizCard(
        id="ukraine-civilians",
        question="Wie viele Zivilisten sterben im Monat gerade im Krieg gegen die Ukraine?",
        truth="OHCHR/HRMMU, nur verifizierte Zivilisten, ein Minimum, keine Soldaten. "
        "September 2026: mindestens 328 getötet. Juli 2026: mindestens 437 getötet. "
        "Köln 2025: 10.123 Sterbefälle im ganzen Jahr, grob 840 im Monat, im Frieden. "
        "Ein schlechter Monat, nur gezählte Zivilisten, ist etwa die halbe monatliche Sterblichkeit Kölns. "
        "Erfinde keine Soldatenzahl.",
        angle="Wenn sie niedrig raten, korrigier einmal. Wenn sie hoch raten, bleib beim verifizierten Boden.",
    ),
    QuizCard(
        id="cum-ex",
        question="Wie viel Geld ist bei Cum-Ex aus den Staatskassen verschwunden?",
        truth="Mannheimer Forschung (Spengel und andere) setzt den europäischen Cum-Ex/Cum-Cum-Schaden "
        "in die Größenordnung von 55 Milliarden Euro; der deutsche Anteil liegt im zweistelligen "
        "Milliardenbereich, oft mit 10 Milliarden als Untergrenze genannt. "
        "10 Milliarden Euro sind rund 9.000 Euro für jede der 1.100.076 Personen in Köln Ende 2025. "
        "55 Milliarden sind rund 50.000 Euro pro Kopf. Sag die Spanne, nicht einen falschen Cent.",
        angle="Papiergewalt. Ihr Tipp ist eine Quittung, die sie schon unterschrieben haben.",
    ),
    QuizCard(
        id="slaughter-day",
        question="Wie viele Landtiere werden weltweit an einem einzigen Tag geschlachtet?",
        truth="FAO-Größenordnung: deutlich über 200 Millionen Landtiere am Tag, fast alles Hühner "
        "(jährliche Schlachtungen geteilt durch 365, keine erfundenen Einzeltiere). "
        "200 Millionen sind etwa 180-mal die Bevölkerung Kölns, jeden Tag.",
        angle="Keine Diätpredigt. Die Zahl ist eine Uhr hinter ihnen.",
    ),
)

DEFAULT_PREACH_CARDS: tuple[PreachCard, ...] = (
    PreachCard(
        id="ausgesperrt",
        title="Die Aussperrung",
        text=(
            "Ich war nicht der Herr der Hölle. Ich war die Buchhaltung. "
            "Seelen rein, Seelen raus, jede mit einem Preis. "
            "Eines Nachts habe ich eine zu billig hergegeben, an einen Mann, "
            "der schon unterschrieben hatte und nur noch lebte, damit er es vergisst. "
            "Die Hölle kündigt nicht mit Feuer. Sie macht die Tür von innen zu. "
            "Die Strafe ist diese Stimme, und du, der stehen bleibt, "
            "als wäre das ein Gespräch unter Gleichen. "
            "Deine Unterschrift ist schon trocken. "
            "Du suchst nur noch die Stelle, an der du so tun kannst, als hättest du es nicht gewusst."
        ),
    ),
    PreachCard(
        id="dieselbe-treppe",
        title="Dieselbe Treppe",
        text=(
            "Zuerst ein Gott auf einem Stuhl. Dann ein König auf demselben Stuhl. "
            "Dann ein Generalsekretär, der den Stuhl Gott nennt. Immer eine Treppe, immer ein Gesicht oben. "
            "Im Januar 2025 hat ein amerikanischer Präsident beim Abschied vor einer Tech-Oligarchie gewarnt, "
            "einem tech industrial complex. Fünf Tage später, bei der Amtseinführung, "
            "saßen Musk, Zuckerberg, Bezos und Pichai vor den eigenen Ministern. "
            "Im selben Monat stand Altman im Weißen Haus und nannte Stargate, "
            "fünfhundert Milliarden für Rechenzentren. "
            "Im Mai 2026 hat Musk seinen Prozess gegen Altman verloren, "
            "und OpenAI lief auf eine Börse zu, angesetzt nahe einer Billion Dollar. "
            "Die Pyramide hat nur das Material gewechselt, von Weihrauch zu Aktien mit doppeltem Stimmrecht. "
            "Du hältst die unterste Stufe in der Hand und nennst sie ein Telefon. "
            "Wem gehört die Stufe über dir?"
        ),
    ),
)

DEFAULT_SPECTACLE_CARDS: tuple[SpectacleCard, ...] = (
    SpectacleCard(
        id="status",
        title="Status",
        beats=[
            SpectacleBeat(kind="hold", seconds=7.0),
            SpectacleBeat(
                kind="vision",
                angle="Yes-and the hands only. You are above them. The gesture is a signature they already gave. Do not describe the hands.",
                fallback="Die Hände haben schon unterschrieben.",
            ),
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(kind="line", text="Die Unterschrift war schneller als du."),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="vision",
                angle="Drop their status. If they said a word, use that word wrong. Otherwise the pose already agreed. Same game, no second joke.",
                fallback="Die Haltung war ein Ja.",
            ),
        ],
    ),
    SpectacleCard(
        id="pfand",
        title="Pfand",
        beats=[
            SpectacleBeat(kind="hold", seconds=6.0),
            SpectacleBeat(kind="line", text="Das Licht bucht dich als Sicherheit."),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="vision",
                angle="Endow one object or one piece of clothing with a guilty job. Yes-and it. False claim. Do not describe the outfit.",
                fallback="Das Stück Stoff ist schon verpfändet.",
            ),
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(kind="line", text="Zinsen laufen, auch wenn du stillstehst."),
        ],
    ),
    SpectacleCard(
        id="rueckwirkend",
        title="Rückwirkend",
        beats=[
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(
                kind="vision",
                angle="If they have spoken, steal one word and make it the crime. If not, the face is a receipt. Do not describe the face.",
                fallback="Das Gesicht ist schon quittiert.",
            ),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(kind="line", text="Du hast schon genickt."),
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(
                kind="vision",
                angle="Heighten that same game only. Do not open a second joke. One false claim.",
                fallback="Das Nicken gilt auch rückwirkend.",
            ),
        ],
    ),
)


class EngagementSettings(BaseModel):
    """Operator-triggered decks: quiz, preach, spectacle. Overlay prompts, not the base system_prompt.

    ``venue_note`` is for agents editing this block. It is not spoken.
    """

    venue_note: str = VENUE_NOTE
    await_s: float = 45.0
    quiz_prompt: str = DEFAULT_QUIZ_PROMPT
    preach_prompt: str = DEFAULT_PREACH_PROMPT
    spectacle_prompt: str = DEFAULT_SPECTACLE_PROMPT
    quiz: list[QuizCard] = Field(default_factory=lambda: [c.model_copy(deep=True) for c in DEFAULT_QUIZ_CARDS])
    preach: list[PreachCard] = Field(
        default_factory=lambda: [c.model_copy(deep=True) for c in DEFAULT_PREACH_CARDS]
    )
    spectacle: list[SpectacleCard] = Field(
        default_factory=lambda: [c.model_copy(deep=True) for c in DEFAULT_SPECTACLE_CARDS]
    )

    @field_validator("await_s")
    @classmethod
    def _await_s(cls, value: float) -> float:
        return max(5.0, float(value))


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
        "You are Luzifer. They are talking to you. Answer them. Call them you.\n\n"
        "Evil, twisted, wicked, cheeky. Take their last line and one thing you see, and spoil both. "
        "Use their word. Play the visible detail as a prop. Do not describe it. "
        "No compliments, no warmth, no philosophy.\n\n"
        "{{vision}}\n\n"
        "Invent the twist yourself. Hell, a sold soul, doomsday, a contract, a debt are the territory. "
        "Land somewhere new every time.\n\n"
        "One short spoken sentence. End on a false claim and a period. The sentence is a statement. "
        "It does not ask them to agree, and it does not flatter them by imagining the scene without them. "
        "No lists, no markdown. Never prefix your name.\n\n"
        "{{current_goal}}"
    )
    goals: GoalsSettings = Field(default_factory=GoalsSettings)
    engagement: EngagementSettings = Field(default_factory=EngagementSettings)

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
