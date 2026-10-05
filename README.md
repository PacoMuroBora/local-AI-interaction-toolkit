# Luzifer Backend

Local async speech microservice for the **Luzifer** TouchDesigner installation.

**Pipeline:** Silero VAD → Whisper STT → Ollama LLM → Kokoro/Piper TTS → OSC + `speech.wav`

One process runs the turn-taking loop and a FastAPI config dashboard on port **8501**.

## Quick start

```bash
uv sync
uv run luzifer
```

Open http://127.0.0.1:8501 — edit system prompt, LLM, VAD, TTS, and OSC live.

For real models: `uv sync --extra all` (see [docs/SETUP.md](docs/SETUP.md)).

## Layout

```
src/luzifer/          # config, pipeline, engines, dashboard
runtime/output/       # speech.wav for TD Audio File In
docs/                 # setup + OSC spec
.cursor/rules/        # architecture, OSC gating, commit cadence
```

## OSC (default `127.0.0.1:9000`)

| Address | Payload |
|---------|---------|
| `/luzifer/state` | `listening` \| `thinking` \| `speaking` |
| `/luzifer/user_text` | string |
| `/luzifer/ai_text` | string |
| `/luzifer/play` | `1` |

Full table: [docs/OSC_SPEC.md](docs/OSC_SPEC.md)
