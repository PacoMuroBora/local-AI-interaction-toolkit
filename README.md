# Local AI Interaction Toolkit

Local async speech stack for a TouchDesigner installation: mic, speech-to-text, a local LLM, speech output, optional NDI vision, OSC, and a config dashboard.

Luzifer is one character on this toolkit (`config.yaml` → `name` and `system_prompt`). The OSC namespace is `/ai/…` and does not follow the installation name.

**Pipeline:** Silero VAD → Whisper STT → Ollama LLM → Kokoro/Piper TTS → OSC + `speech.wav`

One process runs the turn-taking loop, a FastAPI config dashboard on port **8501**, and an inbound OSC server on port **9001**.

## Quick start

```bash
uv sync
uv run luzifer
```

Open http://127.0.0.1:8501 — tabs for Audio In, Audio Out, Prompt/LLM, Vision, OSC; pull models; send test text; reset session. The header name is the installation label and is saved with the rest of the config.

For real models: `uv sync --extra all` (see [docs/SETUP.md](docs/SETUP.md)). Vision/NDI: `uv sync --extra vision` + `ollama pull moondream`.

## Layout

```
src/luzifer/          # config, pipeline, engines, dashboard, models
runtime/output/       # speech.wav for TD Audio File In
docs/                 # setup + OSC spec
.cursor/rules/        # architecture, OSC gating, commit cadence
```

## OSC

| Direction | Default | Addresses |
|-----------|---------|-----------|
| → TD | `127.0.0.1:9000` | `/ai/state`, `/user_text`, `/ai_text`, `/listening`, `/user_speaking`, `/stt`, `/thinking`, `/synthesizing`, `/ai_speaking`, `/play`, `/in_level`, `/out_level`, `/vision_text`, `/vision`, `/vision_active` |
| ← TD | `127.0.0.1:9001` | `/ai/in/say`, `/speak`, `/reset`, `/interrupt`, `/listen`, `/vision` |

Full table + TD wiring: [docs/OSC_SPEC.md](docs/OSC_SPEC.md)
