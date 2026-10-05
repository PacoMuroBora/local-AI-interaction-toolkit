"""Central async turn-taking orchestrator with mic gating."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

from loguru import logger

from luzifer.audio.vad import VoiceActivityDetector
from luzifer.config import ConfigStore
from luzifer.llm.ollama import OllamaEngine
from luzifer.osc.td_client import TouchDesignerOscClient
from luzifer.stt.whisper import WhisperEngine
from luzifer.tts.kokoro import TtsEngine

StateName = Literal["listening", "thinking", "speaking"]
StateListener = Callable[[StateName, str, str], Awaitable[None] | None]


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
        self.state: StateName = "listening"
        self.last_user_text = ""
        self.last_ai_text = ""
        self._running = False
        self._listeners: list[StateListener] = []

    def on_state(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    async def _notify(self) -> None:
        for listener in list(self._listeners):
            result = listener(self.state, self.last_user_text, self.last_ai_text)
            if asyncio.iscoroutine(result):
                await result

    def is_listening(self) -> bool:
        return self._running and self.state == "listening"

    async def set_state(self, state: StateName) -> None:
        self.state = state
        if state != "listening":
            self.vad.stop_stream()
        await self.osc.send_state(state)
        await self._notify()
        logger.info("state={}", state)

    async def run(self) -> None:
        self._running = True
        settings = await self.store.get()
        self.osc.reconfigure(settings.osc.host, settings.osc.port)
        await self.set_state("listening")
        logger.info("Pipeline started")

        while self._running:
            try:
                await self._turn()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Turn failed: {}", exc)
                await self.set_state("listening")
                await asyncio.sleep(0.5)

    async def stop(self) -> None:
        self._running = False
        self.vad.stop_stream()
        await self.llm.close()

    async def _turn(self) -> None:
        settings = await self.store.get()
        self.osc.reconfigure(settings.osc.host, settings.osc.port)

        await self.set_state("listening")
        audio = await self.vad.capture_utterance(
            settings.audio,
            is_listening=self.is_listening,
        )
        if audio is None or not self._running:
            await asyncio.sleep(0.05)
            return

        await self.set_state("thinking")
        user_text = (await self.stt.transcribe(audio, settings.stt)).strip()
        self.last_user_text = user_text
        await self.osc.send_user_text(user_text)
        await self._notify()

        if not user_text:
            logger.info("Empty transcript, skipping LLM")
            await self.set_state("listening")
            return

        ai_text = await self.llm.chat(user_text, settings.system_prompt, settings.llm)
        self.last_ai_text = ai_text
        await self.osc.send_ai_text(ai_text)
        await self._notify()

        await self.set_state("speaking")
        out_path = settings.resolve_output_path()
        duration = await self.tts.synthesize(ai_text, out_path, settings.tts)
        await self.osc.send_play()
        pad = settings.audio.speaking_tail_pad_ms / 1000.0
        await asyncio.sleep(max(0.1, duration + pad))
        await self.set_state("listening")
