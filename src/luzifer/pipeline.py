"""Central async turn-taking orchestrator with mic gating."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from luzifer.audio.vad import VoiceActivityDetector
from luzifer.config import ConfigStore
from luzifer.llm.ollama import ChatMessage, OllamaEngine
from luzifer.metrics import MetricsStore
from luzifer.osc.td_client import TouchDesignerOscClient
from luzifer.stt.whisper import WhisperEngine
from luzifer.tts.kokoro import TtsEngine

StateName = Literal["listening", "thinking", "speaking"]
StateListener = Callable[[StateName, str, str], Awaitable[None] | None]
CommandKind = Literal["say", "speak", "reset", "interrupt", "listen"]


@dataclass
class PipelineCommand:
    kind: CommandKind
    text: str = ""
    enabled: bool | None = None


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
        self.llm = llm or OllamaEngine()
        self.tts = tts or TtsEngine()
        settings = store.settings
        self.osc = osc or TouchDesignerOscClient(settings.osc.host, settings.osc.port)
        self.metrics = MetricsStore()
        self.vad.metrics = self.metrics
        self.state: StateName = "listening"
        self.last_user_text = ""
        self.last_ai_text = ""
        self._running = False
        self._mic_enabled = True
        self._listeners: list[StateListener] = []
        self._history: list[ChatMessage] = []
        self._commands: asyncio.Queue[PipelineCommand] = asyncio.Queue()
        self._active_task: asyncio.Task[Any] | None = None
        self._interrupt = asyncio.Event()

    def on_state(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    async def _notify(self) -> None:
        for listener in list(self._listeners):
            result = listener(self.state, self.last_user_text, self.last_ai_text)
            if asyncio.iscoroutine(result):
                await result

    @property
    def mic_on(self) -> bool:
        return self.is_listening()

    @property
    def thinking(self) -> bool:
        return self.state == "thinking"

    @property
    def ai_speaking(self) -> bool:
        return self.state == "speaking"
    def is_listening(self) -> bool:
        return self._running and self.state == "listening" and self._mic_enabled

    async def set_state(self, state: StateName) -> None:
        self.state = state
        if state != "listening":
            self.vad.stop_stream()
        await self.osc.send_state(state)
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
        await self.osc.send_user_text("")
        await self.osc.send_ai_text("")
        await self.set_state("listening")
        logger.info("Session reset")

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

    async def run(self) -> None:
        self._running = True
        settings = await self.store.get()
        self.osc.reconfigure(settings.osc.host, settings.osc.port)
        await self.set_state("listening")
        logger.info("Pipeline started")

        while self._running:
            try:
                await self._loop_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Turn failed: {}", exc)
                await self.set_state("listening")
                await asyncio.sleep(0.5)

    async def stop(self) -> None:
        self._running = False
        self._interrupt.set()
        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()
        self.vad.stop_stream()
        await self.llm.close()

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
            self.vad.capture_utterance(settings.audio, is_listening=self.is_listening),
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

        audio = capture_task.result()
        if audio is None or not self._running:
            await asyncio.sleep(0.05)
            return

        await self.set_state("thinking")
        user_text = (await self.stt.transcribe(audio, settings.stt)).strip()
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
            await self._run_cancellable(self._run_say_turn(cmd.text))
            return "turn"
        if cmd.kind == "speak":
            if not cmd.text:
                return "consume"
            await self._run_cancellable(self._run_speak_only(cmd.text))
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
        try:
            ai_text = await self.llm.chat(
                user_text,
                settings.system_prompt,
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
        await self.set_state("speaking")
        out_path = settings.resolve_output_path()
        try:
            duration, peak = await self.tts.synthesize(text, out_path, settings.tts)
        except asyncio.CancelledError:
            await self.set_state("listening")
            raise

        if self._interrupt.is_set():
            await self.set_state("listening")
            return

        self.metrics.note_output_peak(peak)
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

        await self.set_state("listening")
