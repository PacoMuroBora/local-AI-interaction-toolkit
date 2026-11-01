# Architecture

One asyncio process. `luzifer.main:main` loads `config.yaml` into a `ConfigStore`, starts `Pipeline.run`, and serves the FastAPI dashboard on port **8501**. Inbound OSC (`TouchDesignerOscServer`) starts with the dashboard and listens on **9001**. Outbound OSC (`TouchDesignerOscClient`) targets **9000**.

Read this when you need the turn, not the OSC address table (`OSC_SPEC.md`) or the dashboard look (`DASHBOARD.md`).

## Find a change

| You want to… | Open |
|---|---|
| Reorder a turn, gate the mic, queue say / speak / vision | `src/luzifer/pipeline.py` — `Pipeline` |
| Change how an utterance is cut | `src/luzifer/audio/vad.py` — `VoiceActivityDetector.capture_utterance` |
| Change transcription | `src/luzifer/stt/whisper.py` — `WhisperEngine` |
| Change chat HTTP, keep-alive, runner options | `src/luzifer/llm/ollama.py` — `OllamaEngine`, `runner_options` |
| Change how `{{current_goal}}` is rendered or advanced | `src/luzifer/goals.py` — `GoalClock`, `apply_goal_placeholder` |
| Change how `{{vision}}` is rendered | `src/luzifer/vision/observation.py` — `ObservationStore` |
| Change NDI connect / grab | `src/luzifer/vision/ndi.py` — `NdiGrabber` |
| Change “did the frame move?” or JPEG size | `src/luzifer/vision/frames.py` |
| Change the VLM call | `src/luzifer/vision/ollama.py` — `VisionOllama` |
| Change local TTS or the wav write | `src/luzifer/tts/kokoro.py` — `TtsEngine` |
| Change ElevenLabs | `src/luzifer/tts/elevenlabs.py` — `ElevenLabsClient` |
| Change outbound OSC payload or types | `src/luzifer/osc/td_client.py` |
| Change which inbound addresses exist | `src/luzifer/osc/td_server.py` |
| Change what an inbound address does | `src/luzifer/dashboard.py` — `_make_osc_handler` |
| Change meters or the 30 Hz envelope | `src/luzifer/metrics.py`, `src/luzifer/audio/envelope.py` |
| Change model lists or downloads | `src/luzifer/models.py` |
| Add a setting | `src/luzifer/config.py` (`AppSettings`), then the control in `dashboard.py` |
| Change the operator UI | `src/luzifer/dashboard.py` — `DASHBOARD_HTML` is the whole page |

`config.yaml` is the live file the dashboard writes. Pydantic defaults in `config.py` apply when a key is missing. `runtime/secrets.yaml` holds `tts.elevenlabs.api_key` and is never written into `config.yaml`.

## Tasks inside the process

| Task | Where | What it does |
|---|---|---|
| `luzifer-pipeline` | `Pipeline.run` | Warmup, then the listen loop |
| `luzifer-dashboard` | uvicorn | HTTP + `/ws` |
| `levels` | `Pipeline._stream_levels` | `/ai/in_level` and `/ai/out_level` at 30 Hz |
| `vision` | `Pipeline._vision_loop` | Idle caption + optional spoken aside |
| `metrics` | `create_dashboard` startup | Dashboard broadcast every 500 ms; `nvidia-smi` about every 2 s |
| `capture` | one-shot per listen | VAD, raced against the command queue |

Blocking work (Whisper, TTS, NDI, JPEG, nvidia-smi) goes through `asyncio.to_thread`. Ollama and OSC are async.

## Listen loop

`Pipeline._loop_once`:

1. Drain `_commands`. `reset`, `interrupt`, and `listen` are consumed in place. `say`, `speak`, and `vision` run a turn and return.
2. `set_state("listening")` and clear the interrupt. The VAD stream is allowed only in this state.
3. Race `vad.capture_utterance` against the next command. The loser is cancelled. A command that arrives as capture wins is still drained before STT.
4. On audio: `stt` → `WhisperEngine.transcribe` → `last_language` from Whisper → `_run_say_turn`.

`set_state` always stops the VAD stream when the new state is not `listening`, publishes OSC (`/ai/state` plus the int flag snapshot), and notifies dashboard listeners. Leaving `speaking` disarms the output envelope and sends `/ai/play` = 0.

### Say turn

`_run_say_turn` (mic transcript, dashboard test line, `/ai/in/say`):

1. Publish `/ai/user_text`. Empty text skips the LLM and returns to listening.
2. `thinking`. Build the system string (below) and call `OllamaEngine.chat` with up to `llm.history_turns` pairs. History is empty when `history_turns` is 0.
3. Publish `/ai/ai_text`. Append the pair and trim to `history_turns * 2` messages.
4. `_speak_text`, then `GoalClock.advance_beat`.

`_run_speak_only` (`/ai/in/speak`, “Speak only”) writes the wav and does not call the LLM, append history, or advance the beat.

### Speak

`_speak_text`:

1. `resolve_tts_for_lang` copies TTS settings for `last_language` (this utterance only, not the history).
2. `synthesizing` → `TtsEngine.synthesize` → `runtime/output/speech.wav` (or `tts.output_path`).
3. `speaking`, arm the dry-wav envelope, `/ai/play` = 1.
4. Wait `duration + speaking_tail_pad_ms`. Negative pad reopens the mic before the wav ends. `monitor_local` plays the wav on `output_device_id` in parallel.
5. Stamp `_last_activity_at` and return to `listening`.

Interrupt cancels `_active_task` and forces `listening`.

## Prompt assembly

Visitor turns start from `system_prompt`. Camera asides start from `vision.idle_prompt` (`idle_system_prompt`) and do not see chat history.

Order, all in `pipeline.py`:

1. If `vision.inject_context`, replace `{{vision}}` with `ObservationStore.context_block` (`vision/observation.py`). Missing token: caption is not injected; one warning.
2. Insert the language line (`Reply in German/English only for this turn.`) immediately before `{{current_goal}}` via `insert_before_goal`. Typed and OSC lines use `guess_lang_from_text` when Whisper did not run.
3. Replace `{{current_goal}}` with `Private. Do not say this. {beat}` (`goals.render_goal_block`). Missing token: beat is not injected; one warning per slot (`system prompt` or `idle prompt`).

A vision aside that is not forced may answer `SILENT` (or junk `is_usable_caption` rejects). That skips TTS and still counts as “already said” for this caption. A forced comment (dashboard or OSC) retries once instead of staying silent.

## Vision

`trigger: local_idle` runs `_vision_loop`. `trigger: external` only reacts to `/ai/in/vision` (`request_vision_from_osc`), and only when `vision.enabled`.

A sample is allowed (`_vision_sample_allowed`) when:

- state is `listening` or `speaking` (not `stt`, `thinking`, `synthesizing`, `loading`)
- the visitor is not currently tripping VAD
- the command queue is empty
- no non-speaking turn owns `_active_task`
- 2 s have passed since VAD dropped (`_VISION_HANDOFF_S`), so a caption cannot take the Ollama lock before the talker

`_caption_frame` (under `_caption_lock`):

1. `NdiGrabber.latest_rgb`. No frame → stop.
2. `frame_changed` (mean absolute error on a small gray image). Unchanged and not `force`: `ObservationStore.touch` (refresh age, no VLM) and return the old caption.
3. Set `_vision_busy`, send `/ai/vision_active` 1.
4. JPEG via `jpeg_bytes`, then `VisionOllama.describe`. Both engines share `Pipeline._ollama_lock`.
5. Usable text → `observations.update`, `/ai/vision_text`, `/ai/vision` pulse. Junk caption keeps the previous thumbnail so a static scene retries next interval.

Idle speech (`_idle_comment_ready`) also needs `idle_after_s` since `_last_activity_at`, `speak_cooldown_s` since the last aside, a fresh caption (`max_age_s`), and a caption that is not `_last_idle_comment`. It enqueues `PipelineCommand(kind="vision", comment=True)`.

Dashboard **Analyze** is caption only (`force=True`). **Analyze + Comment** captions, then enqueues a forced speak. Forced grabs skip the idle, cooldown, and change gates. They still wait for `_models_ready`.

## Goals

`GoalClock` (`goals.py`). A config line is `title | beat | beat`. The title is the campaign name; each beat is what `{{current_goal}}` becomes. A line with no `|` is a one-beat campaign.

- A beat stays for `goals.hold` successful visitor replies (`advance_beat` at the end of `_run_say_turn` only).
- `goals.interval_s` (0 = off), Reset, `/ai/in/goal` 1, and the dashboard **Next** button skip to the next campaign and restart at its first beat.
- Reset (`_clear_session`) also clears history, transcript fields, language, and the observation. Next and `/ai/in/goal` do not.
- `skip_goal` / the timer do not go through `_commands`. Reset, interrupt, and listen do.

## Warmup and VRAM

`_warmup` holds the mic and the vision loop until `_models_ready`. Order: TTS preload, Whisper preload, unload the VLM, chat warmup, vision warmup, then `_ensure_chat_resident` if the VLM evicted the talker. The footer warmup bar reads `warmup_status()` until `_finish_warmup`.

`llm.num_ctx`, `llm.num_gpu`, and the vision runner fields must be sent on every Ollama call (`runner_options`). Omitting them makes Ollama build a new runner. `keep_alive` is `-1` so a warmed model is not evicted on Ollama’s 5 minute default.

Saving config compares `_runner_budget` (chat model, URL, `num_ctx`, `num_gpu`, vision enabled/model/`num_ctx`/`num_gpu`/`num_batch`). A change calls `Pipeline.rebalance_models`: wait for the current turn, unload vision, load chat, load vision, re-pin chat.

## Language → TTS

`resolve_tts_for_lang` in `pipeline.py`. With `tts.match_input_language`, `engine_by_lang`, `voice_by_lang`, and `lang_by_lang` pick Kokoro vs Piper and the voice for `en` or `de`. `speaker_by_lang` is the Piper style (`neutral` is the straight read). `tts.provider: elevenlabs` uses `ElevenLabsClient` and the same wav path. `tts.engine` / `tts.voice` track the English column when matching is off.

## Dashboard data path

`create_dashboard` registers `pipeline.on_state` and `on_vision`. Those broadcast JSON on `/ws` (state, transcripts, flags, goal, warmup). A 500 ms loop adds meters. The page also `fetch`es `/api/config` on load and `PUT`s it on Save. Route list is the `@app.*` block at the bottom of `dashboard.py`.

GPU layer or context edits that call `rebalance_models` are the only config save that reloads models. Other fields are read from `ConfigStore` on the next turn.
