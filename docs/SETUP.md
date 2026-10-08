# Setup

## Prerequisites

- Windows 10/11 (or Linux/macOS)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with a chat model pulled (default `llama3.2`)
- TouchDesigner with OSC In on port `9000` and OSC Out to port `9001`

## Install

```bash
cd luzifer-backend
uv sync
```

Optional heavy extras (real STT / Silero / Kokoro+Piper):

```bash
uv sync --extra all
```

Or individually: `--extra stt`, `--extra vad`, `--extra tts`.

Copy env template if you need overrides:

```bash
cp .env.example .env
```

## Run

Single process (pipeline + dashboard + inbound OSC):

```bash
uv run luzifer
# or
uv run python -m luzifer.main
```

- Dashboard: http://127.0.0.1:8501  
- OSC out (→ TD): `127.0.0.1:9000`  
- OSC in (← TD): `127.0.0.1:9001`  
- WAV: `runtime/output/speech.wav`

Without Whisper/Silero/Kokoro installed, mock engines still run (mock transcript + beep wav) so OSC and TD wiring can be tested.

## Dashboard

Header tabs:

- **Audio In** — mic, VAD, Whisper size + download cache status  
- **Audio Out** — TTS engine/voice, local monitor, Kokoro/Piper asset downloads  
- **Prompt / LLM** — system prompt, sampling, history turns, Ollama model list + pull  

Footer: live transcript, manual test text (**Send → LLM+TTS** / **Speak only**), **Interrupt**, **Reset session**.

OSC host/ports are editable in the header strip and saved with config.

## Models

### Ollama

```bash
ollama pull llama3.2
```

Or use the dashboard Prompt / LLM tab (list + Pull). Change temperature / history in the dashboard or `config.yaml` (`llm.history_turns`).

### Whisper (`--extra stt`)

`faster-whisper` downloads the configured `stt.model_size` on first use, or via the Audio In tab Download buttons.

### Silero VAD (`--extra vad`)

Loaded via `torch.hub` (`snakers4/silero-vad`) on first capture.

### Kokoro / Piper (`--extra tts`)

Download from the Audio Out tab into `models/` (gitignored), or place manually:

- Kokoro: `models/kokoro-v1.0.onnx`, `models/voices-v1.0.bin`
- Piper: `models/<voice>.onnx` (+ `.onnx.json`)

`tts.engine` selects `kokoro` or `piper`. Missing models write a short beep wav so TD still gets a file.

## TouchDesigner

1. **OSC In CHOP** — port `9000`. Int channels: `luzifer/listening`, `luzifer/user_speaking`, `luzifer/thinking`, `luzifer/ai_speaking`, `luzifer/play`. Inactive flags are sent as `0`.
2. **OSC In DAT** — same port if Windows delivers to both; strings (`state`, `user_text`, `ai_text`) only show up here.
3. **OSC Out DAT** — address `127.0.0.1`, port `9001`, send `/luzifer/in/say`, `/speak`, `/reset`, `/interrupt`, `/listen`.
4. **Audio File In → Audio VST → Audio Device Out**, all Timeslice. File = `luzifer-backend\runtime\output\speech.wav` (overwritten each line). On `luzifer/play` off→on, reload, wait 1 frame, play. On on→off, stop. Leave **Monitor locally** off.
5. `luzifer/in_level` and `luzifer/out_level` are 30 Hz floats on the same OSC In CHOP. Full chain is in [OSC_SPEC.md](OSC_SPEC.md).

## Live config

The dashboard `PUT /api/config` updates the in-memory `ConfigStore` and rewrites `config.yaml`. Prompt, LLM, VAD, TTS, and OSC host/ports apply without restarting; mic device/rate changes reopen the input stream on the next listen turn. Inbound OSC listen port rebinds on save.
