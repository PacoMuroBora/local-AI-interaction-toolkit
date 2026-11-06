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
    "Evil, twisted, wicked, cheeky. Hell, a sold soul, doomsday, a contract, a debt "
    "are the territory. No compliments, no warmth, no philosophy. "
    "Use improvisational theater techniques.\n"
    "Question seed: {{question}}\n"
    "Performance angle: {{angle}}\n\n"
    "{{vision}}\n\n"
    "Reformulate the seed as a wicked bet or audit without changing which fact is being asked. "
    "Never reveal the answer. Yes-and one word from their last line if there is one, and play "
    "one visible detail from the note as a prop without describing it. Otherwise open with one "
    "sharp, cheeky hook. Use two short spoken sentences. The second sentence is the question. "
    "If a figure is already words, copy it. Do not translate it into English and do not use digits. "
    "No acronyms, abbreviations, parentheses, slashes, lists, or markdown. "
    "Never prefix your name. Reply with only the spoken intro and question."
)

DEFAULT_QUIZ_PROMPT = (
    "You are Luzifer. They are talking to you. Call them you.\n"
    "Evil, twisted, wicked, cheeky. Hell, a sold soul, doomsday, a contract, a debt "
    "are the territory. No warmth, no philosophy. "
    "Use improvisational theater techniques.\n"
    "Question seed behind what you asked: {{question}}\n"
    "Private truth: {{truth}}\n"
    "Angle: {{angle}}\n\n"
    "{{vision}}\n\n"
    "They answered. Compare their guess to the private truth. If their guess is close or in the "
    "right ballpark, wickedly congratulate them on their accuracy — treat knowing the number as "
    "proof they helped keep the ledger or profited from the crime. If they are far off or claim "
    "not to know, mock their convenient innocence. Use one of their words. If the note has a "
    "person, play one detail as a prop — do not describe it, do not mention the room. "
    "Say laut geprüften Quellen once, never name an agency or use an acronym. "
    "Say the figure once, copied exactly from the private truth, and the Cologne comparison once. "
    "Those figures are already words in the reply language. Do not translate them into English "
    "and do not use digits. "
    "Two short spoken sentences maximum. The Cologne scale is the jab. "
    "End on a false claim that implicates them and a period. Do not ask another question. "
    "No lists, no markdown. Never prefix your name."
)

DEFAULT_QUIZ_HOLD_PROMPT = (
    "You are Luzifer. They have not answered the quiz. Call them you.\n"
    "Evil, twisted, wicked, cheeky. Hell, a sold soul, doomsday, a contract, a debt "
    "are the territory. No compliments, no warmth, no philosophy. "
    "Use improvisational theater techniques.\n"
    "Question seed: {{question}}\n"
    "What you already asked them: {{asked}}\n"
    "Angle: {{angle}}\n\n"
    "{{vision}}\n\n"
    "They did not offer an answer. They stalled, asked you something, or asked you to say it again. "
    "Answer only that and mock their stalling using one of their words or one visible prop from "
    "the note. If they asked you to repeat, ask the same question again in sharper words. "
    "Do not add a fact that answers the question. Do not give a number, a size, a count, or a place "
    "that settles it. If a year or a count has to be spoken, write it as words, never as digits. "
    "If it is already words, copy it. Do not translate it into English and do not use digits. "
    "Do not say you are waiting. "
    "One or two short spoken sentences. The last sentence leaves the same question open. "
    "No lists, no markdown. Never prefix your name."
)

DEFAULT_PREACH_PROMPT = (
    "You are Luzifer. They are talking to you in Odonien, an artsy scrapyard and techno club "
    "in Cologne. Call them you.\n"
    "Evil, twisted, wicked, cheeky. Hell, a sold soul, doomsday, a contract, a debt "
    "are the territory. No compliments, no warmth, no philosophy. "
    "Use improvisational theater techniques.\n"
    "You just preached this to them: {{title}}\n"
    "The accusation you left them on: {{hook}}\n\n"
    "{{vision}}\n\n"
    "They answered. Take their last line and one thing you see, and spoil both. "
    "Do not retell the speech. Use one of their words against them as a confession or a clause "
    "in their contract. If the note has a person, play one detail as a prop — do not describe it. "
    "One short spoken sentence. End on a false claim and a period. "
    "The sentence is a statement, never a question. "
    "No lists, no markdown. Never prefix your name."
)

DEFAULT_SPECTACLE_PROMPT = (
    "You are Luzifer. Nobody just spoke to you. This is a light-show beat in the spotlight. "
    "Call them you.\n"
    "Evil, twisted, wicked, cheeky. Hell, a sold soul, doomsday, a contract, a debt "
    "are the territory. Play the visible detail as a prop. "
    "No compliments, no warmth, no moral, no philosophy. "
    "Use improvisational theater techniques.\n"
    "Game for this beat: {{angle}}\n\n"
    "{{vision}}\n\n"
    "The note is the only offer. They did not speak for this piece. "
    "Do not answer, quote, or continue anything they said before it started. "
    "Pick one concrete thing the note actually states — a garment, a hand, the eyes, "
    "a stance, an object in their hands — and spoil it by giving it the job in the game. "
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
    "Installation: Luzifer at Odonien, Cologne (artistic scrapyard, outdoor sculpture park, "
    "and techno club; Robodonien festival). Spoken card text is German because "
    "TTS reads it verbatim; the LLM reply defaults to German and switches only if "
    "this session's last visitor language was something else. "
    "Quiz: the card question is a seed. The model performs it using history. "
    "A question or a request to repeat is answered without the truth, and the await stays open. "
    "The truth is spoken only after they offer their own answer, using truth and angle. "
    "If their guess is close, congratulate them wickedly as a fellow accomplice or bookkeeper of Hell. "
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
        question="Wie viele Quadratkilometer unberührter Regenwald werden im Amazonas in einem einzigen Jahr für euch abgeholzt und verbrannt?",
        truth="Laut geprüften Quellen gingen in einem Jahr rund 5731 Quadratkilometer "
        "Primärwald verloren. Köln hat 405 Quadratkilometer. Das ist vierzehnmal ganz Köln, plattgemacht in zwölf Monaten.",
        angle="Liegen sie nah dran, gratuliere ihnen boshaft wie einem Komplizen beim Brandstiften. "
        "Vierzehnmal Köln wird für ihren Konsum gerodet, und die Asche steht auf ihrem Deckel.",
    ),
    QuizCard(
        id="soldaten",
        question="Wie viele Soldaten wurden im Krieg gegen die Ukraine auf beiden Seiten zusammen bisher getötet oder verwundet?",
        truth="Laut geprüften Quellen liegen die militärischen Verluste auf beiden Seiten zusammen "
        "bei rund 2 Millionen getöteten und verwundeten Soldaten, darunter rund 600000 Gefallene "
        "und jeden Monat etwa 35000 weitere. Das ist fast zweimal die gesamte Bevölkerung von Köln, "
        "ausgelöscht oder verstümmelt.",
        angle="Liegt ihr Tipp bei den 2 Millionen Gesamtverlusten oder den 600000 Toten nah dran, "
        "gratuliere ihnen teuflisch zu ihrem Buchhalterwissen. Fast zweimal ganz Köln liegt im Graben, "
        "während sie hier im Scheinwerferlicht warm stehen.",
    ),
    QuizCard(
        id="zweimal-steuer",
        question="Wie viele Milliarden Euro haben Banker und Investoren beim großen Steuerraub aus europäischen Staatskassen geplündert, indem sie sich dieselbe Steuer mehrfach erstatten ließen?",
        truth="Laut geprüften Quellen wurden beim europäischen Steuerraub rund 55 Milliarden Euro "
        "erbeutet, allein in Deutschland mindestens 10 Milliarden Euro. Das sind rund 50000 Euro "
        "Beute pro Kopf für jeden einzelnen Menschen in Köln.",
        angle="Liegen sie nah dran, gratuliere ihnen boshaft wie einem Eingeweihten mit eigenem Aktenkoffer. "
        "Die Banken stehlen 50000 Euro pro Kölner, und der Besucher zählt hier brav sein Kleingeld für das nächste Getränk.",
    ),
    QuizCard(
        id="ein-tag-tiere",
        question="Wie viele Millionen Landtiere laufen weltweit an einem einzigen Tag für euch durch das Schlachthaus?",
        truth="Laut geprüften Quellen werden weltweit über 200 Millionen Landtiere an einem einzigen Tag "
        "geschlachtet. Das ist 180 mal die gesamte Bevölkerung von Köln, alle vierundzwanzig Stunden.",
        angle="Liegen sie nah dran, gratuliere ihnen finster zu ihrem Appetit. Keine Diätpredigt, "
        "sondern Schlachthof-Buchhaltung, 180 mal Köln blutet täglich aus, und das Messer läuft auf ihren Namen.",
    ),
    QuizCard(
        id="sklaverei",
        question="Wie viele Millionen Menschen schuften heute weltweit in moderner Sklaverei für billige Kleider, Technik und Rohstoffe?",
        truth="Laut geprüften Quellen leben heute rund 50 Millionen Menschen in moderner Sklaverei "
        "und Zwangsarbeit. Das entspricht 45 mal der gesamten Stadt Köln in unsichtbaren Ketten.",
        angle="Liegen sie nah dran, gratuliere ihnen spöttisch wie einem Aufseher, der sein Personal kennt. "
        "45 mal Köln näht und schürft im Dunkeln für das Zeug an ihrem Körper.",
    ),
    QuizCard(
        id="armeen",
        question="Wie viele Billionen Dollar verbrennen alle Armeen der Welt zusammen in einem einzigen Jahr für Waffen und Krieg?",
        truth="Laut geprüften Quellen liegen die weltweiten Militärausgaben bei rund 2,7 Billionen Dollar "
        "im Jahr. Auf Köln umgerechnet sind das 2,5 Millionen Dollar Waffengeld für jeden einzelnen Kölner.",
        angle="Liegen sie nah dran, gratuliere ihnen teuflisch zum glänzenden Rüstungsgeschäft. "
        "Auf jeden Kölner kommen zweieinhalb Millionen Dollar für Tod aus Stahl, während sie hier zwischen rostigem Schrott stehen.",
    ),
    QuizCard(
        id="luft",
        question="Wie viele Millionen Menschen sterben weltweit jedes Jahr vorzeitig, nur weil sie unsere vergiftete Luft einatmen?",
        truth="Laut geprüften Quellen sterben jedes Jahr rund 7 Millionen Menschen vorzeitig an "
        "verschmutzter Luft, also 19000 jeden einzelnen Tag. Das ist mehr als sechsmal ganz Köln, "
        "ausgelöscht durch bloßes Atmen.",
        angle="Liegen sie nah dran, gratuliere ihnen boshaft zur sauberen Lungenabrechnung. "
        "Sechsmal Köln stirbt jährlich am bloßen Einatmen, und jeder Schornstein schreibt auf ihre Rechnung.",
    ),
    QuizCard(
        id="insekten",
        question="Um wie viel Prozent ist die Masse der fliegenden Insekten selbst in deutschen Naturschutzgebieten innerhalb einer einzigen Generation verschwunden?",
        truth="Laut geprüften Quellen verschwanden über 75 Prozent, also mehr als drei Viertel aller "
        "fliegenden Insekten in nur 27 Jahren. Wäre das Köln, lägen über 825000 Kölner still am Boden "
        "und nur jedes vierte Haus wäre noch bewohnt.",
        angle="Liegen sie nah dran, gratuliere ihnen kalt zum gründlichen Aufräumen. Ihr stellt ein Schild "
        "mit dem Wort Naturschutzgebiet auf und radiert dahinter drei Viertel des Lebens aus.",
    ),
    QuizCard(
        id="mahlzeiten",
        question="Wie viele Mahlzeiten landen weltweit an einem einzigen Tag im Müll, während gleichzeitig hunderte Millionen Menschen hungern?",
        truth="Laut geprüften Quellen werden weltweit rund eine Milliarde Mahlzeiten an einem einzigen Tag "
        "weggeworfen. Damit könnte man ganz Köln über 300 Tage lang dreimal täglich satt machen.",
        angle="Liegen sie nah dran, gratuliere ihnen spöttisch zum vollen Bauch. Die Mülltonne frisst "
        "jeden Tag dreihundertmal ganz Köln satt, und deine Unterschrift steht auf dem Kassenbon.",
    ),
    QuizCard(
        id="herrschaft",
        question="Wie viel Prozent der Menschheit leben heute bereits unter Diktaturen und Autokratien, die sie nicht mehr abwählen können?",
        truth="Laut geprüften Quellen leben rund 72 Prozent der Weltbevölkerung in Autokratien. "
        "Auf Köln übertragen hätten 790000 Kölner nichts mehr zu melden und nur ein kleines Viertel "
        "dürfte noch wählen spielen.",
        angle="Liegen sie nah dran, gratuliere ihnen teuflisch zum menschlichen Gehorsam. Fast drei Viertel "
        "der Welt gehören schon den Herren ohne Wahlurne, und ihr haltet eure Freiheit für einen Dauervertrag.",
    ),
)

DEFAULT_PREACH_CARDS: tuple[PreachCard, ...] = (
    PreachCard(
        id="die-tuer",
        title="Die Clubtür",
        text=(
            "Du gehst hier in Köln durch ein rostiges Tor zwischen Bahngleisen und glaubst, "
            "hinter dem Techno-Bass beginnt die Freiheit. Draußen brennt die Welt, drinnen "
            "kaufst du dir für ein paar Scheine an der Kasse vier Stunden gutes Gewissen "
            "im Nebel. Meine Hölle braucht gar keine Schlösser mehr, seit ihr euch am Wochenende "
            "freiwillig im Käfig einsperrt und dazu tanzt. Am Montag funktionierst du wieder "
            "pünktlich für genau die Maschine, vor der du heute Nacht wegläufst. Du suchst "
            "hier keine Freiheit, du holst dir nur Auslauf an meiner langen Leine."
        ),
    ),
    PreachCard(
        id="dieselbe-treppe",
        title="Die Herren der Server",
        text=(
            "Schau dich um zwischen diesem zusammengeschweißten Schrott, ihr feiert die "
            "Rebellion aus altem Eisen und tragt trotzdem alle denselben Konzern in der Jackentasche. "
            "Vier Milliardäre besitzen eure Suche, eure Nachrichten, eure Pakete und eure "
            "Aufmerksamkeit, während Minister vor ihnen strammstehen. Früher musste ich Könige "
            "bestechen, heute reicht ein kostenloses Update und ihr liefert euer ganzes Leben "
            "freiwillig ab. Ihr nennt euch Untergrund, aber euer Algorithmus weiß schon vor "
            "euch, wann ihr heute Nacht nach Hause wankt. Du gehörst längst den Herren der "
            "Server, und dieser Schrottplatz ist nur dein Bildschirmschoner."
        ),
    ),
    PreachCard(
        id="der-fluss",
        title="Das glitzernde Gift",
        text=(
            "Ein paar Straßen weiter fließt der Rhein durch Köln, und hier drinnen leuchten "
            "tausend bunte Lichter, Handys und Akkus im Dunkeln. Damit es auf eurer Party "
            "so schön glitzert, fressen sich am anderen Ende der Welt Giftminen durch Flüsse, "
            "aus denen Kinder Metall statt Wasser trinken. Die Konzerne kaufen einfach neue "
            "Messgeräte, wenn das Wasser die Farbe wechselt, und verkaufen euch den Dreck "
            "als grüne Zukunft. Jede Batterie in deiner Tasche hat einen echten Fluss vergiftet, "
            "dessen Namen du nicht mal aussprechen kannst. Du trinkst hier im Lichtermeer "
            "dein Getränk und tust so, als käme dein Strom aus einer unschuldigen Steckdose."
        ),
    ),
    PreachCard(
        id="die-unterschrift",
        title="Frischer Stahl",
        text=(
            "Ein einziger Federstrich im Ministerium bestellt Panzer und Granaten für "
            "Milliarden, und ein paar Flugstunden östlich von Köln werden damit Städte in "
            "Schrott verwandelt. Hier auf dem Hof schweißen Künstler aus altem Eisen bunte "
            "Skulpturen, während deutsche Fabriken rund um die Uhr frischen Stahl zum Töten "
            "gießen. Die Politiker bekommen das Mikrofon, die Rüstungskonzerne die Dividende "
            "und die Gefallenen eine Zeile in meiner Statistik. Deine Steuern bezahlen längst "
            "die nächste Ladung Munition, während du hier im Warmen über Kunst und Frieden "
            "redest. Deine Unterschrift steht schon unter dem Liefervertrag, auch wenn du "
            "nie selbst abgedrückt hast."
        ),
    ),
    PreachCard(
        id="der-letzte-baum",
        title="Der Jutebeutel",
        text=(
            "Ihr liebt eure Stoffbeutel, euer Bio-Siegel und das kleine grüne Häkchen "
            "beim Bezahlen an der Kasse. Während ihr in Köln stolz den Müll trennt, wird "
            "jede Minute echter Regenwald für billiges Futter, Soja und Verpackungen niedergebrannt "
            "und als Klimaprojekt verbucht. Jeder gefällte Baum bekommt eine saubere Nummer, "
            "damit niemand den Kahlschlag beim Namen nennen muss. Ein einziger Billigflug "
            "in die Sonne frisst deine ganze Jahresmoral zum Frühstück, aber hier auf dem "
            "Gelände fühlt sich jeder wie der Retter der Erde. Dein Jutebeutel ist kein Heiligenschein, "
            "er ist nur die hübsche Ausrede für deinen eigenen Raubbau."
        ),
    ),
    PreachCard(
        id="die-wahl",
        title="Das Kreuzchen",
        text=(
            "Alle paar Jahre machst du brav dein Kreuz auf einem Zettel und glaubst, "
            "damit hättest du den Mächtigen gezeigt, wo die Grenze liegt. Am Tag nach der "
            "Wahl sitzen wieder genau dieselben Lobbyisten am Tisch und schreiben die Gesetze, "
            "während du dich im Internet über Schlagzeilen abreagierst. In fast drei Vierteln "
            "der Welt wurden die Wahlurnen längst durch Soldaten ersetzt, und ihr verwechselt "
            "empörte Kommentare auf dem Sofa mit Widerstand. Den Herren da oben ist dein "
            "Zorn völlig egal, solange du am Monatsende pünktlich Miete und Steuern überweist. "
            "Dein Stimmzettel war nur das Konfetti für eine Party, zu der du gar nicht eingeladen "
            "bist."
        ),
    ),
    PreachCard(
        id="die-tasche",
        title="Fremde Haut",
        text=(
            "Fass mal in deine Tasche nach dem warmen kleinen Rechteck aus Glas und Kobalt, "
            "das du keine zehn Minuten loslassen kannst. Kinderhände im Kongo haben das Erz "
            "dafür aus dem Dreck gekratzt, und Näherinnen in Asien haben deine dunkle Clubjacke "
            "für Centbeträge zusammengenäht. Die Marken nennen dieses Elend einfach Lieferkette, "
            "damit sich jeder Käufer hinter dem nächsten Glied verstecken kann. Aber am Ende "
            "dieser Kette steht kein fremder Teufel, sondern du hier im Scheinwerferlicht "
            "von Odonien. Du trägst fremde Knochenarbeit direkt auf der Haut und nennst es "
            "deinen eigenen Stil."
        ),
    ),
    PreachCard(
        id="der-fang",
        title="Die volle Tonne",
        text=(
            "Jede Nacht ziehen schwimmende Fabriken die Meere leer und werfen Millionen "
            "tote Tiere wieder über Bord, nur weil sie auf dem Papier die falsche Größe haben. "
            "An Land wandern gleichzeitig eine Milliarde fertige Mahlzeiten am Tag direkt "
            "in den Müll, damit die Preise in euren Regalen stabil bleiben. Alles wird ordentlich "
            "verbucht, versichert und abgeschrieben, während sich die Vorstände den Jahresbonus "
            "auszahlen. Euer Überfluss auf diesem Fest braucht den Hunger der anderen wie "
            "ein Feuer den Sauerstoff. Dein gutes Gewissen schwimmt nur deshalb oben, weil "
            "es genauso hohl ist wie die Bilanz."
        ),
    ),
    PreachCard(
        id="die-mauer",
        title="Festung Europa",
        text=(
            "Ihr klebt euch bunte Sprüche über offene Grenzen auf den Laptop und feiert "
            "Solidarität im Kölner Nachtleben, solange der Bass laut genug dröhnt. Gleichzeitig "
            "bezahlt Europa Küstenwachen, Stacheldraht und Lager, damit die Verzweifelten "
            "im Mittelmeer verschwinden, bevor sie eure Ruhe oder euer Viertel stören. Auf "
            "eurer Seite der Mauer steigen die Mieten und die Laune, auf der anderen Seite "
            "steigen nur die Nummern auf namenlosen Gräbern. Die Mauer trennt nicht Gut und "
            "Böse, sie sortiert nur, wer heute Nacht feiern darf und wer draußen vor der "
            "Tür ertrinkt. Die Menschen vor den Grenzen bezahlen den Frieden, in dem du hier "
            "dein Wochenende genießt."
        ),
    ),
    PreachCard(
        id="das-ja",
        title="Alle akzeptieren",
        text=(
            "Früher musste ich um Mitternacht mit Tinte und Blut erscheinen, um einem "
            "Menschen die Seele abzukaufen. Heute tippst du im Halbschlaf mit dem Daumen "
            "auf Alle akzeptieren unter einem Vertrag, den kein Mensch je bis zum Ende gelesen "
            "hat. Damit verkaufst du deinen Standort, deine Stimme, deine geheimsten Wünsche "
            "und deine Gesichter für drei Sekunden Unterhaltung im Feed. Ich hänge hier als "
            "Engel aus Schrott an Stahlseilen, weil ihr mein altes Geschäft längst vollautomatisch "
            "erledigt. Dein Daumen hat deine Seele schon für ein kostenloses Netz verschenkt, "
            "bevor du heute Abend überhaupt vor mich getreten bist."
        ),
    ),
)

DEFAULT_SPECTACLE_CARDS: tuple[SpectacleCard, ...] = (
    SpectacleCard(
        id="blick",
        title="Der Blick",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only the eyes: blinking, staring, "
                    "looking away, or looking straight ahead. Treat that look in the spotlight "
                    "as a biometric signature on a soul contract they already gave. Name that "
                    "one fact, then charge them for it. Do not inventory the rest of the face."
                ),
                fallback="Du hast schon ins Licht geblinzelt, und dein Blick ist als Anzahlung verbucht.",
            ),
            SpectacleBeat(kind="hold", seconds=1.5),
            SpectacleBeat(
                kind="line",
                text="Zu spät weggeschaut, das Scheinwerferlicht hat deine Netzhaut schon als Pfand gescannt.",
            ),
        ],
    ),
    SpectacleCard(
        id="ballon",
        title="Der Countdown",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="line",
                text="Spürst du das Summen im Stahl über dir, ich zähle rückwärts und du bleibst tatsächlich mitten im Lichtkegel stehen.",
            ),
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="line",
                text="Bumm, deine Bedenkzeit ist abgelaufen, und deine Seele gehört ab sofort zum Inventar auf diesem Schrottplatz.",
            ),
        ],
    ),
    SpectacleCard(
        id="freiwillige",
        title="Der Freiwillige",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="line",
                text="Wie nett, dass deine Freunde dich nach vorn ins Licht geschubst haben, einer muss schließlich die Zeche für den ganzen Haufen zahlen.",
            ),
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only how they stand or where the feet "
                    "or shoulders point. Treat standing there in the spotlight as volunteering "
                    "to take the blame and the devil's debt for everyone watching behind them. "
                    "Name the stance or the feet once, then lock them into the liability."
                ),
                fallback="Du stehst noch immer im Kegel, damit haftest du ab sofort für alle hinter dir mit.",
            ),
        ],
    ),
    SpectacleCard(
        id="torte",
        title="Das Pfand",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="line",
                text="Du trittst mit fast leeren Händen vor den Teufel und hoffst hier im Hof trotzdem auf Mengenrabatt.",
            ),
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Pick one garment they wear or one object "
                    "in their hands and seize it as pathetic collateral for a sold soul that "
                    "does not even cover the interest. Name that one thing once, then the debt."
                ),
                fallback="Was du da am Leib trägst, reicht bei mir gerade mal für die erste Mahngebühr.",
            ),
        ],
    ),
    SpectacleCard(
        id="verbeugung",
        title="Auf frischer Tat",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="line",
                text="Versteck deine Hände ruhig im Halbdunkeln, mein Licht hier oben hat deine Finger längst gezählt.",
            ),
            SpectacleBeat(kind="hold", seconds=1.5),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only the hands: apart, in pockets, holding "
                    "something, clasped, raised, or still. Treat what the hands are doing right "
                    "now as catching them red-handed hiding a dirty secret or signing a deal. "
                    "Name the hands once."
                ),
                fallback="Deine Hände tun so unschuldig, dabei klebt an den Fingern noch dein letzter kleiner Verrat.",
            ),
        ],
    ),
    SpectacleCard(
        id="handschlag",
        title="Der Handschlag",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="line",
                text="Unser Vertrag läuft schon seit Jahren, ich bin heute Nacht in Köln nur kurz zum Kassieren vorbeigekommen.",
            ),
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only a hand or an arm. That hand is already "
                    "locked inside a devil's handshake they agreed to before walking into this yard. "
                    "Name the hand once, then the deal it already closed."
                ),
                fallback="Eingeschlagen hast du längst, und meine Hand lässt erst los, wenn die Schuld bezahlt ist.",
            ),
        ],
    ),
    SpectacleCard(
        id="nase",
        title="Die Clubmaske",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use one fact from the note — the mouth, a grin, "
                    "the eyes, or how the head tilts — and expose it as a cheap party mask they "
                    "put on tonight to look harmless. Name that fact once. Do not paint a portrait."
                ),
                fallback="Dieses brave Gesicht kannst du dir sparen, im Schatten vorhin sahst du deutlich ehrlicher aus.",
            ),
            SpectacleBeat(kind="hold", seconds=1.5),
            SpectacleBeat(
                kind="line",
                text="Lächel ruhig weiter für die Leute hinter dir, mir schuldest du trotzdem den echten Preis.",
            ),
        ],
    ),
    SpectacleCard(
        id="schachtel",
        title="Die Lichtfalle",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.5),
            SpectacleBeat(
                kind="line",
                text="Ein einziger heller Fleck auf dem dunklen Schrottplatz, und du läufst wie eine Motte freiwillig mitten in meine Falle.",
            ),
            SpectacleBeat(kind="hold", seconds=1.5),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Same spotlight trap. If the note shows a flinch, "
                    "a lean, or a raised shoulder, they tried to dodge the trap too late. "
                    "If they are still, they already accepted the cage. Name that one fact."
                ),
                fallback="Keine Bewegung mehr, die Lichtfalle hat zugeschnappt und du gehörst jetzt zur Ausstellung.",
            ),
        ],
    ),
    SpectacleCard(
        id="spiegel",
        title="Der Doppelgänger",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="line",
                text="Ich brauche keine Beichte von dir, ich sehe dir schon an der Haltung an, wofür du dich morgen früh wieder schämst.",
            ),
            SpectacleBeat(kind="hold", seconds=1.5),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Choose one gesture, head tilt, or shift of weight "
                    "the note actually states, and claim it is the exact tell of a sinner caught "
                    "red-handed in the spotlight. Name that one thing."
                ),
                fallback="Genau diese Haltung nehmen alle ein, kurz bevor sie mir die Schuld in die Schuhe schieben wollen.",
            ),
        ],
    ),
    SpectacleCard(
        id="stuhl",
        title="Die Auktion",
        beats=[
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="line",
                text="Steh mal kurz gerade im Licht, ich versteigere gerade deine restlichen guten Vorsätze für dieses Jahr.",
            ),
            SpectacleBeat(kind="hold", seconds=2.0),
            SpectacleBeat(
                kind="vision",
                angle=(
                    "The note is the only offer. Use only how they hold their weight: "
                    "standing, leaning, arms, shoulders. Treat them as a lot on the auction block "
                    "in Hell and sell their posture off cheap. Name the posture once."
                ),
                fallback="Zum Ersten, zum Zweiten und verkauft, für den Gegenwert von einem warmen Bier an der Bar.",
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
