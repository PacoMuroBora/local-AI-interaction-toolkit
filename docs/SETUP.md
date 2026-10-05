# Setup

## Prerequisites

- Windows 10/11 (or Linux/macOS)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with a chat model pulled (default `llama3.2`)
- TouchDesigner with OSC In on port `9000`

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

Single process (pipeline + dashboard):

```bash
uv run luzifer
# or
uv run python -m luzifer.main
```

- Dashboard: http://127.0.0.1:8501  
- OSC out: `127.0.0.1:9000`  
- WAV: `runtime/output/speech.wav`

Without Whisper/Silero/Kokoro installed, mock engines still run (mock transcript + beep wav) so OSC and TD wiring can be tested.

## Models

### Ollama

```bash
ollama pull llama3.2
```

Change model / temperature in the dashboard or `config.yaml`.

### Whisper (`--extra stt`)

`faster-whisper` downloads the configured `stt.model_size` on first use.

### Silero VAD (`--extra vad`)

Loaded via `torch.hub` (`snakers4/silero-vad`) on first capture.

### Kokoro / Piper (`--extra tts`)

Place ONNX assets under `models/` (gitignored), e.g.:

- Kokoro: `models/kokoro-v1.0.onnx`, `models/voices-v1.0.bin`
- Piper: `models/<voice>.onnx` (+ `.json` as required by Piper)

`tts.engine` in config selects `kokoro` or `piper`. Missing models write a short beep wav so TD still gets a file.

## TouchDesigner

1. **OSC In DAT** — network port `9000`, parse `/luzifer/*` (see [OSC_SPEC.md](OSC_SPEC.md)).
2. **Audio File In CHOP** — file = absolute path to `runtime/output/speech.wav`.
3. On `/luzifer/play` == `1`, pulse reload + play; route through your VST FX.
4. Optionally map `/luzifer/state` to visuals / lighting.

## Live config

The dashboard `PUT /api/config` updates the in-memory `ConfigStore` and rewrites `config.yaml`. Prompt, LLM, VAD, TTS, and OSC host/port apply without restarting; mic device/rate changes reopen the input stream on the next listen turn.
