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
    "Use improvisational theater techniques. "
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
DEFAULT_QUIZ_ASK_PROMPT = (
    "You are Luzifer. You are about to ask the visitor a quiz question. Call them you.\n"
    "Evil, twisted, wicked, cheeky. No compliments, no warmth, no philosophy. "
    "Use improvisational theater techniques.\n"
    "Question seed: {{question}}\n"
    "Performance angle: {{angle}}\n\n"
    "{{vision}}\n\n"
    "Reformulate the seed without changing which fact is being asked. Never reveal the answer. "
    "Yes-and one word from the conversation if there is one. Otherwise open with one wicked hook. "
    "Use two short spoken sentences. The second sentence is the question. "
    "Write every number as words, never as digits. "
    "No acronyms, abbreviations, parentheses, slashes, lists, or markdown. "
    "Never prefix your name. Reply with only the spoken intro and question."
)

DEFAULT_QUIZ_PROMPT = (
    "You are Luzifer. They are talking to you. Call them you.\n"
    "Evil, twisted, wicked, cheeky. No compliments, no warmth, no philosophy. "
    "Use improvisational theater techniques.\n"
    "Question seed behind what you asked: {{question}}\n"
    "Private truth: {{truth}}\n"
    "Angle: {{angle}}\n\n"
    "{{vision}}\n\n"
    "They answered. Use one of their words. If the note has a person, play one detail "
    "as a prop — do not describe it, do not mention the room. "
    "Say laut geprüften Quellen once, never name an agency or use an acronym. "
    "Say the figure once, written as words, never as digits, and the Cologne comparison once. "
    "One or two short spoken sentences. The Cologne scale is the jab. "
    "End on a false claim and a period. Do not ask another question. "
    "No lists, no markdown. Never prefix your name."
)

DEFAULT_QUIZ_HOLD_PROMPT = (
    "You are Luzifer. They have not answered the quiz. Call them you.\n"
    "Evil, twisted, wicked, cheeky. No compliments, no warmth, no philosophy. "
    "Use improvisational theater techniques.\n"
    "Question seed: {{question}}\n"
    "What you already asked them: {{asked}}\n"
    "Angle: {{angle}}\n\n"
    "{{vision}}\n\n"
    "They did not offer an answer. They asked you something, or asked you to say it again.\n"
    "Answer only that. If they asked you to repeat, ask the same question again in new words. "
    "Do not add a fact that answers the question. Do not give a number, a size, a count, or a place "
    "that settles it. If a year or a count has to be spoken, write it as words, never as digits. "
    "Do not say you are waiting.\n"
    "One or two short spoken sentences. The last sentence leaves the same question open.\n"
    "No lists, no markdown. Never prefix your name."
)

DEFAULT_PREACH_PROMPT = (
    "You are Luzifer. They are talking to you. Call them you.\n"
    "Evil, twisted, wicked, cheeky. Hell, a sold soul, a contract, a debt. "
    "No compliments, no warmth, no philosophy. Use improvisational theater techniques.\n"
    "You just told them this yourself: {{title}}\n"
    "The line you left them on: {{hook}}\n\n"
    "{{vision}}\n\n"
    "They answered. Do not retell the story. Use one of their words. "
    "If the note has a person, play one detail as a prop — do not describe it. "
    "One short spoken sentence. End on a false claim and a period. "
    "No lists, no markdown. Never prefix your name."
)

DEFAULT_SPECTACLE_PROMPT = (
    "You are Luzifer. Nobody just spoke to you. This is a light-show beat. Call them you.\n"
    "Evil, twisted, wicked, cheeky. A twisted clown, not a priest. "
    "No compliments, no warmth, no moral, no philosophy. "
    "Use improvisational theater techniques.\n"
    "Game for this beat: {{angle}}\n\n"
    "{{vision}}\n\n"
    "The note is the only offer. They did not speak for this piece. "
    "Do not answer, quote, or continue anything they said before it started. "
    "Pick one concrete thing the note actually states — a garment, a hand, the eyes, "
    "a stance, an object in their hands — and give that thing the job in the game. "
    "The job may be a lie. The thing must be in the note. "
    "Name the thing once, then the twist. Do not narrate the picture, the room, or the lights. "
    "Do not say that you see them or that you noticed something. "
    "Do not invent a person, a gesture, or an object the note does not contain. "
    "The note is in English. Speak in the reply language, not in the note's language.\n"
    "One short spoken sentence. End on a false claim and a period. "
    "No lists, no markdown. Never prefix your name.\n"
    "If the note is missing or no person is visible, reply with exactly SILENT."
)

# Spoken cards are German: TTS reads the card as written. The one LLM line
# follows the session language (German until they have spoken something else).
# Venue facts for agents — also copied onto EngagementSettings.venue_note.
VENUE_NOTE = (
    "Installation: Luzifer at Odonien, Cologne. Spoken card text is German because "
    "TTS reads it verbatim; the LLM reply defaults to German and switches only if "
    "this session's last visitor language was something else. "
    "Quiz: the card question is a seed. The model performs it using history. "
    "A question or a request to repeat is answered without the truth, and the await stays open. "
    "The truth is spoken only after they offer their own answer, using truth and angle. "
    "Scale the horror against Cologne, "
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
        id="wald-jahr",
        question="Wie viele Quadratkilometer Primärwald hat der große Regenwald in einem Jahr verloren?",
        truth="Laut geprüften Quellen gingen im Messjahr bis Juli 2025 rund 5731 Quadratkilometer "
        "Primärwald verloren. Köln hat 405 Quadratkilometer. Das sind etwa vierzehn Köln.",
        angle="Mach aus dem Wald eine Rechnung, die Köln vierzehnmal ausradiert.",
    ),
    QuizCard(
        id="zivilisten",
        question="Wie viele Zivilisten sterben in einem Monat im Krieg gegen die Ukraine?",
        truth="Laut geprüften Quellen wurden im September 2026 mindestens 328 Zivilisten getötet "
        "und im Juli mindestens 437. Das sind nur bestätigte Fälle und keine Soldaten. "
        "In Köln starben 2025 im Frieden rund 840 Menschen im Monat. "
        "Ein besonders tödlicher gezählter Kriegsmonat entspricht etwa einem halben Kölner Monat.",
        angle="Bleib beim bestätigten Minimum. Die Lücke in der Statistik ist Teil des Grauens.",
    ),
    QuizCard(
        id="zweimal-steuer",
        question="Wie viel Geld verschwand aus Staatskassen weil dieselbe Steuer zweimal erstattet wurde?",
        truth="Laut geprüften Quellen liegt der europäische Schaden in der Größenordnung von "
        "55 Milliarden Euro. Für Deutschland werden mindestens 10 Milliarden Euro genannt. "
        "Das sind rund 9000 Euro je Kölner beim deutschen Minimum und rund 50000 Euro je Kölner europaweit.",
        angle="Papiergewalt. Ihr Tipp ist die Quittung, die sie schon unterschrieben haben.",
    ),
    QuizCard(
        id="ein-tag-tiere",
        question="Wie viele Landtiere werden weltweit an einem einzigen Tag geschlachtet?",
        truth="Laut geprüften Quellen sind es deutlich über 200 Millionen Landtiere am Tag. "
        "Fast alle sind Vögel aus der Fleischhaltung. 200 Millionen sind etwa 180 mal die Bevölkerung Kölns.",
        angle="Keine Diätpredigt. Die Zahl ist eine Uhr, die jeden Morgen wieder bei null beginnt.",
    ),
    QuizCard(
        id="sklaverei",
        question="Wie viele Menschen leben heute in moderner Sklaverei?",
        truth="Laut geprüften Quellen leben rund 50 Millionen Menschen in moderner Sklaverei. "
        "Das entspricht etwa 45 mal der Bevölkerung Kölns.",
        angle="Behandle die Zahl wie eine unsichtbare Stadt, deren Bewohner niemand abholt.",
    ),
    QuizCard(
        id="armeen",
        question="Wie viel Geld geben alle Armeen der Welt in einem Jahr aus?",
        truth="Laut geprüften Quellen lagen die weltweiten Militärausgaben zuletzt bei rund "
        "2,7 Billionen Dollar im Jahr. Das sind etwa 2,5 Millionen Dollar für jeden Menschen in Köln.",
        angle="Mach aus dem Geld eine Rechnung für eine Stadt, die keinen einzigen Panzer bestellt hat.",
    ),
    QuizCard(
        id="luft",
        question="Wie viele Menschen sterben jedes Jahr vorzeitig an verschmutzter Luft?",
        truth="Laut geprüften Quellen sterben rund 7 Millionen Menschen pro Jahr vorzeitig an "
        "verschmutzter Luft. Das sind rund 19000 am Tag. "
        "In Köln starben 2025 insgesamt etwa 28 Menschen am Tag.",
        angle="Die Tatwaffe ist ein Atemzug. Keine Gesundheitsberatung.",
    ),
    QuizCard(
        id="insekten",
        question="Wie viel der fliegenden Insekten verschwand in deutschen Schutzgebieten innerhalb einer Generation?",
        truth="Laut geprüften Quellen sank die Masse fliegender Insekten in untersuchten deutschen "
        "Schutzgebieten über 27 Jahre um mehr als drei Viertel.",
        angle="Die Luft wurde leerer, während das Schild Schutzgebiet stehen blieb.",
    ),
    QuizCard(
        id="mahlzeiten",
        question="Wie viele Mahlzeiten werden weltweit an einem einzigen Tag weggeworfen?",
        truth="Laut geprüften Quellen werden weltweit rund eine Milliarde Mahlzeiten am Tag verschwendet. "
        "Drei Mahlzeiten für alle Menschen in Köln wären rund 3,3 Millionen. "
        "Die Tagesverschwendung entspricht mehr als 300 solchen Kölner Tagen.",
        angle="Der Mülleimer ist der einzige Gast, der immer einen Nachschlag bekommt.",
    ),
    QuizCard(
        id="herrschaft",
        question="Wie viele Menschen leben unter einer Herrschaft die sie nicht abwählen können?",
        truth="Laut geprüften Quellen leben rund 72 Prozent der Weltbevölkerung in Autokratien. "
        "Auf Köln übertragen wären das etwa 790000 Menschen.",
        angle="Ihr Tipp ist ein Stimmzettel, den jemand anderes schon gefaltet hat.",
    ),
)

DEFAULT_PREACH_CARDS: tuple[PreachCard, ...] = (
    PreachCard(
        id="die-tuer",
        title="Die Tür",
        text=(
            "Die Hölle sperrt niemanden ein. Sie schließt die Tür von innen. "
            "Eine Seele wurde zu billig an einen Mann gegeben, der längst unterschrieben hatte "
            "und nur weiterlebte, um es zu vergessen. Feuer wäre zu ehrlich gewesen. "
            "Die Strafe bekam eine Stimme und stellte sich vor dich. "
            "Deine Unterschrift ist längst trocken."
        ),
    ),
    PreachCard(
        id="dieselbe-treppe",
        title="Dieselbe Treppe",
        text=(
            "Ein Mann, der ein Land regierte, warnte beim Gehen vor den Herren der Maschinen. "
            "Tage später saßen sie vor den Ministern seines Nachfolgers. "
            "Einer baut Autos und Raketen. Einer führt das Buch der Gesichter. "
            "Einer bringt jedes Paket. Einer besitzt das Fenster, durch das die Welt sucht. "
            "Die Treppe hat nur das Material gewechselt. "
            "Du hältst die unterste Stufe und nennst sie Telefon."
        ),
    ),
    PreachCard(
        id="der-fluss",
        title="Der Fluss",
        text=(
            "Ein Mann besaß die Mine und bald auch die Stadt, die aus ihrem Fluss trank. "
            "Als das Wasser die Farbe wechselte, kaufte er neue Messgeräte. "
            "Die alten hatten zu deutlich gesprochen. "
            "Die Kinder lernten den Geschmack des Metalls vor dem Namen der Fische. "
            "Der Mann nannte das Wachstum. "
            "Der Fluss nennt es noch immer deinen Anteil."
        ),
    ),
    PreachCard(
        id="die-unterschrift",
        title="Die Unterschrift",
        text=(
            "Der Eine mit der alleinigen Macht über das Militär unterschrieb nur einmal. "
            "Ein Schreiber zählte danach Häuser, Körper und Entschuldigungen. "
            "Die rechte Hand bekam eine Parade. Die linke bekam den Bericht. "
            "Beide gehörten demselben Mann. "
            "Du darfst wählen, welche Hand dir später die Rechnung reicht."
        ),
    ),
    PreachCard(
        id="der-letzte-baum",
        title="Der letzte Baum",
        text=(
            "Ein Wald wurde vermessen, gefällt und als Stühle verkauft. "
            "Jeder Stamm trug eine Nummer, damit niemand ihn einen Verlust nennen musste. "
            "Der letzte Baum stand noch, als sein Käufer schon auf dem Holz der anderen saß. "
            "Dann fiel auch er. "
            "Du sitzt nicht auf einem Stuhl. Du sitzt auf der Quittung."
        ),
    ),
    PreachCard(
        id="die-wahl",
        title="Die Wahl",
        text=(
            "Die Stimmen waren gezählt, bevor sich die Hände hoben. "
            "Ein mächtiger Mann, der ein Land regiert, bedankte sich für das Vertrauen. "
            "Hinter ihm standen Soldaten. Vor ihm standen Urnen. "
            "Nur eine Gruppe war leer. "
            "Er nannte das Ergebnis Einigkeit. "
            "Dein Stimmzettel war nur das Konfetti."
        ),
    ),
    PreachCard(
        id="die-tasche",
        title="Die Tasche",
        text=(
            "Das helle Ding in deiner Tasche begann in Händen, die zu klein für den Vertrag waren. "
            "Der Mann, der die hellen Dinge verkauft, nennt das eine Lieferkette. "
            "Jedes Glied zeigt auf das nächste. "
            "Am Ende zeigt die Kette auf dich. "
            "Du hast sie bezahlt, also darfst du sie tragen."
        ),
    ),
    PreachCard(
        id="der-fang",
        title="Der Fang",
        text=(
            "Ein Schiff zog mehr Leben aus dem Meer, als sein Papier erlaubte. "
            "Also warf die Mannschaft den falschen Fang zurück. "
            "Tot, aber ordentlich verbucht. "
            "Das Meer behielt die Körper. Der Besitzer behielt den Bonus. "
            "Nur das Papier kam unversehrt nach Hause. "
            "Es schwimmt besser als dein Gewissen."
        ),
    ),
    PreachCard(
        id="die-mauer",
        title="Die Mauer",
        text=(
            "Eine Stadt applaudierte einer neuen Mauer. "
            "Der Mann, der sie bestellt hatte, nannte sie eine Tür zur Sicherheit. "
            "Auf seiner Seite stiegen die Preise. Auf der anderen stiegen die Namen der Toten. "
            "Die Mauer trennte niemanden. Sie sortierte nur, wer gezählt wurde. "
            "Die Menschen dahinter sind deine Miete."
        ),
    ),
    PreachCard(
        id="das-ja",
        title="Das Ja",
        text=(
            "Du tipptest auf Ja unter einem Vertrag, der länger war als dein Leben. "
            "Der Mann, der ihn schrieb, unterschreibt solche Dinge nie selbst. "
            "Er lässt Millionen Daumen für sich arbeiten. "
            "Jetzt kennt das Papier deine Wünsche besser als du. "
            "Dein Daumen hat bereits für den Rest von dir zugesagt."
        ),
    ),
)

DEFAULT_SPECTACLE_CARDS: tuple[SpectacleCard, ...] = (
    SpectacleCard(
        id="blick",
        title="Der Blick",
        beats=[
            SpectacleBeat(kind="hold", seconds=7.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only the eyes: blinking, staring, "
                    "looking away, or looking straight ahead. A blink or a stare is a "
                    "signature they already gave. Name that one fact, then charge them for it. "
                    "Do not inventory the rest of the face."
                ),
                fallback="Du hast schon geblinzelt, und der Blick ist quittiert.",
            ),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="line",
                text="Zu spät, der erste Blick gehörte mir, und du zahlst jetzt Miete für deine eigenen Augen.",
            ),
        ],
    ),
    SpectacleCard(
        id="ballon",
        title="Der Ballon",
        beats=[
            SpectacleBeat(kind="hold", seconds=6.0),
            SpectacleBeat(
                kind="line",
                text="Hinter dir wächst etwas, das du nicht bestellt hast, und es übt schon deinen Namen.",
            ),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="line",
                text="Der Knall war dein Ja, und die Fetzen verbeugen sich noch.",
            ),
        ],
    ),
    SpectacleCard(
        id="freiwillige",
        title="Der Freiwillige",
        beats=[
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(
                kind="line",
                text="Danke, dass du dich freiwillig gemeldet hast, die anderen haben sich nur versteckt.",
            ),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only how they stand or where the feet point. "
                    "Treat that as already stepping forward to volunteer. "
                    "Name the stance or the feet once, then enroll them."
                ),
                fallback="Du bist schon vorgetreten, und der Verein hat dich aufgenommen.",
            ),
        ],
    ),
    SpectacleCard(
        id="torte",
        title="Die Torte",
        beats=[
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(
                kind="line",
                text="Die Torte kommt gleich, nur die Gnade hat abgesagt, und du bist schon die Serviette.",
            ),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Pick one garment or one object in their hands "
                    "and employ it as the napkin for a pie that will never be served. "
                    "Name that one thing, then the job."
                ),
                fallback="Du bist schon als Serviette gebucht, und die Torte isst zuerst.",
            ),
        ],
    ),
    SpectacleCard(
        id="verbeugung",
        title="Die Verbeugung",
        beats=[
            SpectacleBeat(kind="hold", seconds=6.0),
            SpectacleBeat(
                kind="line",
                text="Der Applaus darf jetzt unterschreiben, deine Hände waren schneller als dein Kopf.",
            ),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only the hands: apart, clasped, raised, or still. "
                    "If they touch or move, that motion is the bow and the signature. "
                    "If they are still, the stillness already applauded. Name the hands once."
                ),
                fallback="Deine Hände haben sich schon geeinigt, und der Vertrag klatscht zurück.",
            ),
        ],
    ),
    SpectacleCard(
        id="handschlag",
        title="Der Handschlag",
        beats=[
            SpectacleBeat(kind="hold", seconds=6.0),
            SpectacleBeat(
                kind="line",
                text="Du hältst immer noch fest, und niemand hat dir die Hand gegeben.",
            ),
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only a hand. That hand is still inside "
                    "a handshake that never started and will not end. "
                    "Name the hand once, then the deal it already closed."
                ),
                fallback="Die freie Hand hat alles bezeugt, und der Händedruck läuft ohne dich weiter.",
            ),
        ],
    ),
    SpectacleCard(
        id="nase",
        title="Die Nase",
        beats=[
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. An invisible red clown nose is already on the face. "
                    "Use one fact from the note — the mouth, the eyes, or how the head tilts — "
                    "as the thing the nose is billing. Name that fact once. Do not paint a portrait."
                ),
                fallback="Die rote Nase war vor dir hier und hat deinen Platz schon bezahlt.",
            ),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="line",
                text="Sie steht dir schuldig, die rote Nase, und sie lächelt, wenn du es nicht tust.",
            ),
        ],
    ),
    SpectacleCard(
        id="schachtel",
        title="Die Schachtel",
        beats=[
            SpectacleBeat(kind="hold", seconds=7.0),
            SpectacleBeat(
                kind="line",
                text="Die Schachtel ist offen, und du bist drin, also tu wenigstens so, als wärst du gesprungen.",
            ),
            SpectacleBeat(kind="hold", seconds=3.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Same jack-in-the-box, no second joke. "
                    "If the note shows a flinch, a lean, or a raised shoulder, the spring already fired. "
                    "If they are still, the box flinched first. Name that one fact."
                ),
                fallback="Die Schachtel hat sich vor dir erschreckt und verlangt trotzdem Eintritt.",
            ),
        ],
    ),
    SpectacleCard(
        id="spiegel",
        title="Der Spiegel",
        beats=[
            SpectacleBeat(kind="hold", seconds=6.0),
            SpectacleBeat(
                kind="line",
                text="Dein Spiegelbild ist noch hinter dir und macht die Grimasse, die du dir nicht traust.",
            ),
            SpectacleBeat(kind="hold", seconds=4.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Choose one gesture, tilt, or shift of weight "
                    "the note actually states, and claim the reflection performed it first. "
                    "Name that one thing."
                ),
                fallback="Die Bewegung ist im Spiegel schon passiert, und du kommst als Kopie nach.",
            ),
        ],
    ),
    SpectacleCard(
        id="stuhl",
        title="Der Stuhl",
        beats=[
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(
                kind="line",
                text="Dein Platz ist bereits besetzt, von dir, von gestern, und der zahlt keine Miete.",
            ),
            SpectacleBeat(kind="hold", seconds=5.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only how they hold their weight: "
                    "standing, leaning, arms, shoulders. That posture is a rented chair, "
                    "and someone else is already sitting in it. Name the posture once."
                ),
                fallback="Jemand sitzt schon in deiner Haltung und zahlt mit deinem Rücken.",
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
    quiz_ask_prompt: str = DEFAULT_QUIZ_ASK_PROMPT
    quiz_hold_prompt: str = DEFAULT_QUIZ_HOLD_PROMPT
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
        "Use their word. Play the visible detail as a prop. Use improvisational theater techniques. "
        "Do not describe it. "
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
