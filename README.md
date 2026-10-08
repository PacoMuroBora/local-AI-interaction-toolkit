# Luzifer Backend

Local async speech microservice for the **Luzifer** TouchDesigner installation.

**Pipeline:** Silero VAD → Whisper STT → Ollama LLM → Kokoro/Piper TTS → OSC + `speech.wav`

One process runs the turn-taking loop, a FastAPI config dashboard on port **8501**, and an inbound OSC server on port **9001**.

## Quick start

```bash
uv sync
uv run luzifer
```

Open http://127.0.0.1:8501 — tabs for Audio In, Audio Out, Prompt/LLM; pull models; send test text; reset session.

For real models: `uv sync --extra all` (see [docs/SETUP.md](docs/SETUP.md)).

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
| → TD | `127.0.0.1:9000` | `/luzifer/state`, `/user_text`, `/ai_text`, `/listening`, `/user_speaking`, `/thinking`, `/ai_speaking`, `/play`, `/in_level`, `/out_level` |
| ← TD | `127.0.0.1:9001` | `/luzifer/in/say`, `/speak`, `/reset`, `/interrupt`, `/listen` |

Full table + TD wiring: [docs/OSC_SPEC.md](docs/OSC_SPEC.md)
