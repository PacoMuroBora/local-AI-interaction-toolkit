"""Central async turn-taking orchestrator with mic gating."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from loguru import logger

from luzifer.audio.vad import VoiceActivityDetector
from luzifer.config import ConfigStore, TtsSettings, VisionSettings
from luzifer.llm.ollama import ChatMessage, OllamaEngine, models_match
from luzifer.metrics import LEVEL_HZ, MetricsStore
from luzifer.osc.td_client import TouchDesignerOscClient
from luzifer.stt.whisper import WhisperEngine
from luzifer.tts.kokoro import TtsEngine
from luzifer.vision.frames import frame_changed, jpeg_bytes, mean_abs_diff
from luzifer.vision.ndi import NdiGrabber
from luzifer.vision.observation import ObservationStore
from luzifer.vision.ollama import VisionOllama, is_usable_caption

StateName = Literal["loading", "listening", "stt", "thinking", "synthesizing", "speaking"]
StateListener = Callable[[StateName, str, str], Awaitable[None] | None]
CommandKind = Literal["say", "speak", "reset", "interrupt", "listen", "vision"]

_DE_HINT = re.compile(
    r"[äöüÄÖÜß]|\b(ich|nicht|und|das|ist|ein|eine|der|die|mit|auch|wie|was|warum|hallo|bitte)\b",
    re.IGNORECASE,
)
_EN_HINT = re.compile(
    r"\b(the|and|you|what|is|are|this|that|with|have|why|how|hello|please)\b",
    re.IGNORECASE,
)


def guess_lang_from_text(text: str) -> str | None:
    """Cheap fallback for typed/OSC lines (no Whisper). Prefer explicit DE markers."""
    if not text or not text.strip():
        return None
    if any(c in text for c in "äöüÄÖÜß"):
        return "de"
    de_hits = len(_DE_HINT.findall(text))
    en_hits = len(_EN_HINT.findall(text))
    if de_hits > en_hits:
        return "de"
    if en_hits > de_hits:
        return "en"
    return None


def normalize_lang(code: str | None) -> str | None:
    if not code:
        return None
    return str(code).split("-")[0].lower().strip() or None


def resolve_tts_for_lang(tts: TtsSettings, lang: str | None) -> TtsSettings:
    """Copy TTS settings with engine/voice/lang/style overridden for this turn's language."""
    out = tts.model_copy(deep=True)
    code = normalize_lang(lang)
    # Legacy engine/voice fields track the English column when matching is off.
    style_code = code if (out.match_input_language and code) else "en"
    if out.match_input_language and code:
        engine = out.engine_by_lang.get(code)
        if engine:
            out.engine = engine
        voice = out.voice_by_lang.get(code)
        if voice:
            out.voice = voice
        kokoro_lang = out.lang_by_lang.get(code)
        if kokoro_lang:
            out.lang = kokoro_lang
    speaker = (out.speaker_by_lang.get(style_code) or "").strip()
    if speaker:
        out.speaker = speaker
    return out


def language_system_suffix(lang: str | None) -> str:
    code = normalize_lang(lang)
    if code == "de":
        return "\n\nReply in German only for this turn."
    if code == "en":
        return "\n\nReply in English only for this turn."
    return ""


@dataclass
class PipelineCommand:
    kind: CommandKind
    text: str = ""
    enabled: bool | None = None
    # vision: caption only vs caption + spoken reaction; force skips idle/cooldown/change gate
    comment: bool = False
    force: bool = False


class Pipeline:
    """Listen → think → speak loop. Mic is only open while listening."""

    def __init__(
        self,
        store: ConfigStore,
        *,
        vad: VoiceActivityDetector | None = None,
        stt: WhisperEngine | None = None,
        llm: OllamaEngine | None = None,
        tts: TtsEngine | None = None,
        osc: TouchDesignerOscClient | None = None,
    ) -> None:
        self.store = store
        self.vad = vad or VoiceActivityDetector()
        self.stt = stt or WhisperEngine()
        self.tts = tts or TtsEngine()
        # One Ollama runner. Chat and vision must never load or generate at the same time.
        self._ollama_lock = asyncio.Lock()
        self.llm = llm or OllamaEngine(self._ollama_lock)
        self.llm._lock = self._ollama_lock
        settings = store.settings
        self.osc = osc or TouchDesignerOscClient(settings.osc.host, settings.osc.port)
        self.metrics = MetricsStore()
        self.vad.metrics = self.metrics
        self.state: StateName = "loading"
        self._models_ready = asyncio.Event()
        self.last_user_text = ""
        self.last_ai_text = ""
        # Language of the most recent user input only (Whisper or text heuristic).
        self.last_language: str | None = None
        self.observations = ObservationStore()
        self._ndi = NdiGrabber()
        self._vision_llm = VisionOllama(self._ollama_lock)
        self._running = False
        self._mic_enabled = True
        self._listeners: list[StateListener] = []
        self._history: list[ChatMessage] = []
        self._commands: asyncio.Queue[PipelineCommand] = asyncio.Queue()
        self._active_task: asyncio.Task[Any] | None = None
        self._level_task: asyncio.Task[Any] | None = None
        self._vision_task: asyncio.Task[Any] | None = None
        self._interrupt = asyncio.Event()
        self._user_speaking = False
        self._last_activity_at = time.monotonic()
        self._last_vision_speak_at = 0.0
        self._last_vision_attempt_at = 0.0
        self._prev_small: np.ndarray | None = None
        self._vision_busy = False
        self._caption_lock = asyncio.Lock()
        self._vision_listeners: list[Callable[[], Awaitable[None] | None]] = []

    def on_state(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    def on_vision(self, listener: Callable[[], Awaitable[None] | None]) -> None:
        self._vision_listeners.append(listener)

    async def _notify(self) -> None:
        for listener in list(self._listeners):
            result = listener(self.state, self.last_user_text, self.last_ai_text)
            if asyncio.iscoroutine(result):
                await result

    async def _notify_vision(self) -> None:
        for listener in list(self._vision_listeners):
            result = listener()
            if asyncio.iscoroutine(result):
                await result
        # Also push full state so dashboard flags stay in sync
        await self._notify()

    @property
    def mic_on(self) -> bool:
        return self.is_listening()

    @property
    def stt_active(self) -> bool:
        return self.state == "stt"

    @property
    def thinking(self) -> bool:
        return self.state == "thinking"

    @property
    def synthesizing(self) -> bool:
        return self.state == "synthesizing"

    @property
    def ai_speaking(self) -> bool:
        return self.state == "speaking"

    @property
    def vision_analyzing(self) -> bool:
        return self._vision_busy

    @property
    def vision_ready(self) -> bool:
        return not self._vision_busy

    def is_listening(self) -> bool:
        return self._running and self.state == "listening" and self._mic_enabled

    async def _publish_flags(self, *, user_speaking: bool = False) -> None:
        listening = self.state == "listening"
        await self.osc.send_flags(
            listening=listening,
            user_speaking=user_speaking and listening,
            stt=self.state == "stt",
            thinking=self.state == "thinking",
            synthesizing=self.state == "synthesizing",
            ai_speaking=self.state == "speaking",
        )

    async def _on_user_speech(self, active: bool) -> None:
        self._user_speaking = active
        if active:
            self._last_activity_at = time.monotonic()
        await self._publish_flags(user_speaking=active)

    async def set_state(self, state: StateName) -> None:
        self.state = state
        if state != "listening":
            self.vad.stop_stream()
        await self.osc.send_state(state)
        await self._publish_flags(user_speaking=False)
        if state != "speaking":
            self.metrics.disarm_output()
            await self.osc.clear_play()
        await self._notify()
        logger.info("state={}", state)

    async def enqueue(self, command: PipelineCommand) -> None:
        if command.kind == "interrupt":
            self._interrupt.set()
            if self._active_task is not None and not self._active_task.done():
                self._active_task.cancel()
        await self._commands.put(command)

    async def say(self, text: str) -> None:
        await self.enqueue(PipelineCommand(kind="say", text=text.strip()))

    async def speak(self, text: str) -> None:
        await self.enqueue(PipelineCommand(kind="speak", text=text.strip()))

    async def reset_session(self) -> None:
        await self.enqueue(PipelineCommand(kind="reset"))

    async def interrupt(self) -> None:
        await self.enqueue(PipelineCommand(kind="interrupt"))

    async def set_listen(self, enabled: bool) -> None:
        await self.enqueue(PipelineCommand(kind="listen", enabled=enabled))

    async def analyze_vision(self) -> str:
        """Dashboard CTA: caption now. Does not touch the mic."""
        await self._models_ready.wait()
        settings = await self.store.get()
        logger.info("Vision analyze CTA (source={!r})", settings.vision.ndi_source)
        return await self._caption_frame(settings.vision, force=True) or ""

    async def comment_vision(self) -> str:
        """Dashboard CTA: caption + spoken reaction. Returns caption."""
        await self._models_ready.wait()
        settings = await self.store.get()
        logger.info("Vision comment CTA (source={!r})", settings.vision.ndi_source)
        text = await self._caption_frame(settings.vision, force=True) or ""
        if not text:
            return ""
        # Wake mic loop so the speak turn can start promptly
        self.vad.stop_stream()
        await self.enqueue(
            PipelineCommand(kind="vision", comment=True, force=True, text=text)
        )
        return text

    async def request_vision_from_osc(self) -> None:
        """TD pulse: analyze + comment when trigger=external."""
        await self._models_ready.wait()
        settings = await self.store.get()
        if not settings.vision.enabled:
            logger.info("OSC vision ignored (vision.enabled=false)")
            return
        if settings.vision.trigger != "external":
            logger.info("OSC vision ignored (trigger={})", settings.vision.trigger)
            return
        self.vad.stop_stream()
        await self.enqueue(PipelineCommand(kind="vision", comment=True, force=True))

    def _trim_history(self, max_turns: int) -> None:
        if max_turns <= 0:
            self._history.clear()
            return
        max_messages = max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    async def _clear_session(self) -> None:
        self._history.clear()
        self.last_user_text = ""
        self.last_ai_text = ""
        self.last_language = None
        self.observations.clear()
        await self.osc.send_user_text("")
        await self.osc.send_ai_text("")
        await self.osc.send_vision_text("")
        await self.set_state("listening")
        logger.info("Session reset")

    def _set_turn_language(self, lang: str | None, *, source: str) -> None:
        """Overwrite turn language from this input only — never blend with history."""
        code = normalize_lang(lang)
        self.last_language = code
        logger.info("turn language={} source={}", code, source)

    async def _wait_or_interrupt(self, seconds: float) -> bool:
        """Sleep up to seconds; return True if interrupt won."""
        if seconds <= 0:
            return self._interrupt.is_set()
        try:
            await asyncio.wait_for(self._interrupt.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return self._interrupt.is_set()

    async def _run_cancellable(self, coro: Awaitable[None]) -> None:
        task = asyncio.create_task(coro)
        self._active_task = task
        try:
            await task
        except asyncio.CancelledError:
            await self.set_state("listening")
        finally:
            if self._active_task is task:
                self._active_task = None

    async def _boot(self, message: str) -> None:
        self.last_ai_text = message
        logger.info(message)
        if self.state != "loading":
            await self.set_state("loading")
            return
        await self.osc.send_state("loading")
        await self._publish_flags(user_speaking=False)
        await self._notify()

    async def _ensure_chat_resident(self, settings: Any) -> None:
        """Vision warmup can evict the talker when Ollama only has one model slot."""
        llm = settings.llm
        try:
            loaded = await self.llm.loaded_models(llm.base_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list resident Ollama models: {}", exc)
            return
        logger.info("Ollama resident: {}", loaded)
        if any(models_match(llm.model, name) for name in loaded):
            return
        logger.warning(
            "Chat model {} left VRAM when {} loaded. Ollama is keeping one model. "
            "Set OLLAMA_MAX_LOADED_MODELS=2 and restart Ollama so both stay hot. "
            "Re-pinning the chat model; vision will load on demand, one call at a time.",
            llm.model,
            settings.vision.model,
        )
        await self._boot(f"loading {llm.model}")
        await self.llm.warmup(llm)

    async def _warmup(self, settings: Any) -> None:
        """TTS → Whisper → chat → vision, strictly one GPU load at a time.

        The mic and the vision loop stay down until this returns. Dashboard vision
        CTAs wait on `_models_ready` so they can't sneak a second Ollama load in
        between steps.
        """
        try:
            await self._boot("loading TTS")
            await self.tts.preload(settings.tts)

            stt = settings.stt
            await self._boot(f"loading whisper {stt.model_size}")
            await self.stt.preload(stt)

            await self._boot(f"loading {settings.llm.model}")
            chat_ok = await self.llm.warmup(settings.llm)

            vision = settings.vision
            if vision.enabled and vision.model.strip():
                await self._boot(f"loading {vision.model}")
                await self._vision_llm.warmup(vision, base_url=settings.llm.base_url)
                if chat_ok:
                    await self._ensure_chat_resident(settings)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Warmup failed: {}", exc)
        finally:
            self.last_ai_text = ""
            # Idle timer starts when the system is actually up, not at process start.
            self._last_activity_at = time.monotonic()
            self._models_ready.set()

    async def run(self) -> None:
        self._running = True
        try:
            settings = await self.store.get()
            self.osc.reconfigure(settings.osc.host, settings.osc.port)
            await self.osc.clear_play(force=True)
            await self._warmup(settings)
            await self.set_state("listening")
            self._level_task = asyncio.create_task(self._stream_levels(), name="levels")
            self._vision_task = asyncio.create_task(self._vision_loop(), name="vision")
            logger.info("Pipeline ready")

            while self._running:
                try:
                    await self._loop_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Turn failed: {}", exc)
                    await self.set_state("listening")
                    await asyncio.sleep(0.5)
        finally:
            self._models_ready.set()
            self._running = False
            self.vad.stop_stream()
            self.metrics.disarm_output()
            for task_attr in ("_level_task", "_vision_task"):
                task = getattr(self, task_attr)
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    setattr(self, task_attr, None)
            self._ndi.close()
            try:
                await self.osc.send_flags(
                    listening=False,
                    user_speaking=False,
                    stt=False,
                    thinking=False,
                    synthesizing=False,
                    ai_speaking=False,
                )
                await self.osc.send_levels(0.0, 0.0)
                await self.osc.send_vision_active(False)
                await self.osc.clear_play(force=True)
            except Exception as exc:  # noqa: BLE001
                logger.debug("OSC shutdown clear failed: {}", exc)

    async def stop(self) -> None:
        self._running = False
        self._interrupt.set()
        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()
        self.vad.stop_stream()
        self._ndi.close()
        await self.llm.close()
        await self._vision_llm.close()

    async def _loop_once(self) -> None:
        settings = await self.store.get()
        self.osc.reconfigure(settings.osc.host, settings.osc.port)

        # Drain non-blocking control commands first
        while not self._commands.empty():
            cmd = self._commands.get_nowait()
            handled = await self._handle_control(cmd)
            if handled == "consume":
                continue
            if handled == "turn":
                return

        await self.set_state("listening")
        self._interrupt.clear()

        capture_task = asyncio.create_task(
            self.vad.capture_utterance(
                settings.audio,
                is_listening=self.is_listening,
                on_speech=self._on_user_speech,
            ),
            name="capture",
        )
        command_task = asyncio.create_task(self._commands.get(), name="command")
        self._active_task = None

        done, pending = await asyncio.wait(
            {capture_task, command_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if command_task in done:
            cmd = command_task.result()
            self.vad.stop_stream()
            handled = await self._handle_control(cmd)
            if handled in {"turn", "consume"}:
                return
            return

        # Capture won — still drain any command that landed while we cancelled the waiter
        while not self._commands.empty():
            cmd = self._commands.get_nowait()
            self.vad.stop_stream()
            handled = await self._handle_control(cmd)
            if handled == "turn":
                return
            if handled == "consume":
                continue

        audio = capture_task.result()
        if audio is None or not self._running:
            await asyncio.sleep(0.05)
            return

        await self.set_state("stt")
        transcript = await self.stt.transcribe(audio, settings.stt)
        user_text = transcript.text.strip()
        self._set_turn_language(transcript.language, source="stt")
        await self._run_cancellable(self._run_say_turn(user_text))

    async def _handle_control(self, cmd: PipelineCommand) -> Literal["consume", "turn", "none"]:
        if cmd.kind == "reset":
            await self._clear_session()
            return "consume"
        if cmd.kind == "interrupt":
            self._interrupt.set()
            await self.set_state("listening")
            return "consume"
        if cmd.kind == "listen":
            self._mic_enabled = bool(cmd.enabled)
            if not self._mic_enabled:
                self.vad.stop_stream()
            logger.info("mic_enabled={}", self._mic_enabled)
            await self._notify()
            return "consume"
        if cmd.kind == "say":
            if not cmd.text:
                return "consume"
            self._set_turn_language(guess_lang_from_text(cmd.text), source="text")
            await self._run_cancellable(self._run_say_turn(cmd.text))
            return "turn"
        if cmd.kind == "speak":
            if not cmd.text:
                return "consume"
            # Speak-only: keep last mic language unless the text itself is clearly DE/EN.
            guessed = guess_lang_from_text(cmd.text)
            if guessed:
                self._set_turn_language(guessed, source="text")
            await self._run_cancellable(self._run_speak_only(cmd.text))
            return "turn"
        if cmd.kind == "vision":
            await self._run_cancellable(
                self._handle_vision_command(
                    comment=cmd.comment,
                    force=cmd.force,
                    caption=cmd.text or None,
                )
            )
            return "turn"
        return "none"

    async def _run_say_turn(self, user_text: str) -> None:
        settings = await self.store.get()
        self.last_user_text = user_text
        await self.osc.send_user_text(user_text)
        await self._notify()

        if not user_text:
            logger.info("Empty transcript, skipping LLM")
            await self.set_state("listening")
            return

        await self.set_state("thinking")
        history = list(self._history) if settings.llm.history_turns > 0 else []
        # Language instruction is per-call only — not stored in history / not averaged.
        system = settings.system_prompt + language_system_suffix(self.last_language)
        if settings.vision.enabled and settings.vision.inject_context:
            system += self.observations.context_block(settings.vision.max_age_s)
        try:
            ai_text = await self.llm.chat(
                user_text,
                system,
                settings.llm,
                history=history,
            )
        except asyncio.CancelledError:
            await self.set_state("listening")
            raise

        if self._interrupt.is_set():
            await self.set_state("listening")
            return

        self.last_ai_text = ai_text
        await self.osc.send_ai_text(ai_text)
        await self._notify()

        if settings.llm.history_turns > 0:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": ai_text})
            self._trim_history(settings.llm.history_turns)

        await self._speak_text(ai_text)

    async def _run_speak_only(self, text: str) -> None:
        self.last_ai_text = text
        await self.osc.send_ai_text(text)
        await self._notify()
        await self._speak_text(text)

    async def _speak_text(self, text: str) -> None:
        if self._interrupt.is_set():
            await self.set_state("listening")
            return

        settings = await self.store.get()
        tts = resolve_tts_for_lang(settings.tts, self.last_language)
        await self.set_state("synthesizing")
        out_path = settings.resolve_output_path()
        try:
            duration, peak, envelope = await self.tts.synthesize(text, out_path, tts)
        except asyncio.CancelledError:
            await self.set_state("listening")
            raise

        if self._interrupt.is_set():
            await self.set_state("listening")
            return

        self.metrics.note_output_peak(peak)
        await self.set_state("speaking")
        self.metrics.arm_output(envelope, 1.0 / LEVEL_HZ)
        await self.osc.send_play()
        pad = settings.audio.speaking_tail_pad_ms / 1000.0

        if settings.tts.monitor_local:
            # Local playback is the clock — resume listening as soon as it ends.
            try:
                await self.tts.play_local(out_path, settings.tts.output_device_id)
            except asyncio.CancelledError:
                await self.set_state("listening")
                raise
            if pad > 0 and await self._wait_or_interrupt(pad):
                await self.set_state("listening")
                return
        else:
            # TD plays the wav; wait estimated duration (+ optional pad) then open mic.
            if await self._wait_or_interrupt(max(0.05, duration + pad)):
                await self.set_state("listening")
                return

        self._last_activity_at = time.monotonic()
        await self.set_state("listening")

    def _idle_eligible(self, settings: VisionSettings) -> bool:
        if not settings.enabled or settings.trigger != "local_idle":
            return False
        if self.state != "listening" or not self._mic_enabled or self._user_speaking:
            return False
        idle_for = time.monotonic() - self._last_activity_at
        return idle_for >= max(0.0, settings.idle_after_s)

    def _vision_cooldown_ok(self, settings: VisionSettings) -> bool:
        if settings.speak_cooldown_s <= 0:
            return True
        return (time.monotonic() - self._last_vision_speak_at) >= settings.speak_cooldown_s

    async def _handle_vision_command(
        self,
        *,
        comment: bool,
        force: bool,
        caption: str | None = None,
    ) -> None:
        settings = await self.store.get()
        if not settings.vision.enabled and not force:
            await self.set_state("listening")
            return
        if not force:
            if not self._idle_eligible(settings.vision) or not self._vision_cooldown_ok(settings.vision):
                await self.set_state("listening")
                return
        # Pre-captioned CTAs pass text=; OSC / idle path captions here
        if not caption:
            caption = await self._caption_frame(settings.vision, force=force)
        if not caption:
            logger.info("Vision command: no caption")
            # Don't bounce state/mic when already listening
            if self.state != "listening":
                await self.set_state("listening")
            return
        if comment:
            await self._run_vision_turn(caption, force=force)
        else:
            if self.state != "listening":
                await self.set_state("listening")

    async def _caption_frame(self, vision: VisionSettings, *, force: bool) -> str | None:
        """Grab NDI still → VLM caption. Returns caption text or None.

        Waits for the caption lock (CTAs never get a busy-reject). Idle path
        skips the VLM when the frame is unchanged.
        """
        async with self._caption_lock:
            self._vision_busy = True
            self._last_vision_attempt_at = time.monotonic()
            await self.osc.send_vision_active(True)
            await self._notify_vision()
            try:
                settings = await self.store.get()
                rgb = await asyncio.to_thread(self._ndi.latest_rgb, vision.ndi_source)
                if rgb is None:
                    logger.info("No NDI frame from {!r}", vision.ndi_source)
                    return None

                changed, small = frame_changed(rgb, self._prev_small, vision.change_threshold)
                mae = (
                    999.0
                    if self._prev_small is None
                    else mean_abs_diff(small, self._prev_small)
                )
                self._prev_small = small

                if not force and not changed:
                    logger.info("Vision frame unchanged (mae={:.2f}), skip VLM", mae)
                    return None

                jpeg = await asyncio.to_thread(
                    jpeg_bytes, rgb, vision.jpeg_max_edge, 70
                )
                text = await self._vision_llm.describe(
                    jpeg,
                    vision,
                    base_url=settings.llm.base_url,
                )
                if not text or not is_usable_caption(text):
                    logger.info(
                        "Vision VLM unusable (model={!r}): {!r}",
                        vision.model,
                        (text or "")[:80],
                    )
                    return None

                self.observations.update(text, mae=mae)
                await self.osc.send_vision_text(text)
                await self.osc.send_vision_pulse()
                await self._notify()
                logger.info("Vision caption: {}", text[:120])
                return text
            except Exception as exc:  # noqa: BLE001
                logger.exception("Vision caption failed: {}", exc)
                return None
            finally:
                self._vision_busy = False
                try:
                    await self.osc.send_vision_active(False)
                except Exception:  # noqa: BLE001
                    pass
                await self._notify_vision()

    async def _run_vision_turn(self, observation: str, *, force: bool) -> None:
        settings = await self.store.get()
        vision = settings.vision
        if not force:
            if not self._idle_eligible(vision) or not self._vision_cooldown_ok(vision):
                await self.set_state("listening")
                return

        user = (
            "[visual observation — camera facts, not a visitor speaking]\n"
            f"{observation}\n\n"
            "In character, speak 1–2 short sentences that clearly react to those facts "
            "(what you see / who is there). Use real words. "
            "Do not answer with punctuation, emojis, or filler alone. "
            "If the scene is empty or not worth a line, reply with exactly SILENT."
        )
        system = settings.system_prompt + language_system_suffix(self.last_language)
        system += (
            "\n\nA [visual observation] is what the camera sees right now. "
            "Treat it as ground truth; do not invent people or objects beyond it."
        )

        await self.set_state("thinking")
        history = list(self._history) if settings.llm.history_turns > 0 else []
        try:
            ai_text = await self.llm.chat(user, system, settings.llm, history=history)
        except asyncio.CancelledError:
            await self.set_state("listening")
            raise

        if self._interrupt.is_set():
            await self.set_state("listening")
            return

        stripped = ai_text.strip()
        if stripped.upper() == "SILENT" or not stripped or not is_usable_caption(stripped):
            logger.info("Vision turn silent/junk: {!r}", stripped[:60])
            await self.set_state("listening")
            return

        self.last_ai_text = stripped
        await self.osc.send_ai_text(stripped)
        await self._notify()

        if vision.store_in_history and settings.llm.history_turns > 0:
            self._history.append({"role": "user", "content": user})
            self._history.append({"role": "assistant", "content": stripped})
            self._trim_history(settings.llm.history_turns)

        self._last_vision_speak_at = time.monotonic()
        await self._speak_text(stripped)

    async def _vision_loop(self) -> None:
        """Local-idle sampler: caption in-background (no mic tear-down).

        Only enqueues a speak turn when a new caption is worth saying.
        """
        while True:
            try:
                settings = await self.store.get()
                vision = settings.vision
                if not vision.enabled or vision.trigger != "local_idle":
                    await asyncio.sleep(0.5)
                    continue

                if self.state != "listening" or self._user_speaking:
                    await asyncio.sleep(0.25)
                    continue

                if not self._idle_eligible(vision):
                    await asyncio.sleep(0.5)
                    continue

                # Pace attempts by interval (even when frame unchanged / VLM fails)
                since_attempt = time.monotonic() - self._last_vision_attempt_at
                if since_attempt < max(0.5, vision.interval_s):
                    await asyncio.sleep(0.25)
                    continue

                if not self._vision_cooldown_ok(vision):
                    await asyncio.sleep(0.5)
                    continue

                if self._active_task is not None or not self._commands.empty():
                    await asyncio.sleep(0.5)
                    continue

                # Caption without going through the command queue (keeps mic open)
                text = await self._caption_frame(vision, force=False)
                if text and self.state == "listening" and self._vision_cooldown_ok(vision):
                    self.vad.stop_stream()
                    await self.enqueue(
                        PipelineCommand(kind="vision", comment=True, force=True, text=text)
                    )
                await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("vision loop: {}", exc)
                await asyncio.sleep(1.0)

    async def _stream_levels(self) -> None:
        interval = 1.0 / LEVEL_HZ
        while True:
            try:
                await self.osc.send_levels(
                    self.metrics.take_input_level(),
                    self.metrics.output_level_now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("level stream: {}", exc)
            await asyncio.sleep(interval)
