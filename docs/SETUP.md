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

On an NVIDIA GPU set `stt.device: cuda` and `stt.compute_type: float16` (dashboard Audio In). This uses **CTranslate2 CUDA**, not PyTorch — the CPU torch build is fine for Silero VAD. RTX 4080: `medium` + float16 is typically well under 2 GB VRAM and ~0.3s per short utterance. Leave `stt.language` empty for auto-detect (needed for bilingual DE/EN voice switching). Lock it (e.g. `en` / `de`) only if you want a fixed language.

### Bilingual TTS (DE/EN)

With `tts.match_input_language: true` (default), each turn uses Whisper’s language from **that utterance only** — never an average over chat history. That picks:

- Kokoro `lang` via `tts.lang_by_lang` (`en` → `en-us`, `de` → `de`)
- Voice via `tts.voice_by_lang` (same speaker for both is fine with Kokoro; for Piper use `en_US-lessac-medium` / `de_DE-thorsten-medium`)

The LLM also gets a one-shot “reply in German/English this turn” suffix that is **not** written into history.

### Silero VAD (`--extra vad`)

Loaded via `torch.hub` (`snakers4/silero-vad`) on first capture.

### Kokoro / Piper (`--extra tts`)

Download from the Audio Out tab into `models/` (gitignored), or place manually:

- Kokoro: `models/kokoro-v1.0.onnx`, `models/voices-v1.0.bin`
- Piper: `models/<voice>.onnx` (+ `.onnx.json`)

`tts.engine` selects `kokoro` or `piper`. Missing models write a short beep wav so TD still gets a file.

Audio Out **Device** / **Compute Type** mirror STT knobs. Kokoro uses ONNX Runtime: `cuda` → `CUDAExecutionProvider`.

`onnxruntime-gpu` 1.30 is built for **CUDA 13 + cuDNN 9**. A system CUDA 12.x toolkit alone is not enough (`cublasLt64_13.dll` missing → silent CPU fallback). Install the pip CUDA/cuDNN wheels (stop `luzifer` first so DLLs unlock):

```bash
uv pip uninstall onnxruntime
uv sync --extra tts-gpu
# or: uv pip install "onnxruntime-gpu[cuda,cudnn]"
```

Validate in the log on preload/first synth:

```
Kokoro loaded … providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If you still see `Failed to create CUDAExecutionProvider` / only `CPUExecutionProvider`, CUDA is **not** in use. `cpu` always works. Piper ignores device/compute.

## TouchDesigner

1. **OSC In CHOP** — port `9000`. Int channels: `luzifer/listening`, `luzifer/user_speaking`, `luzifer/stt`, `luzifer/thinking`, `luzifer/synthesizing`, `luzifer/ai_speaking`, `luzifer/play`. Inactive flags are sent as `0`.
2. **OSC In DAT** — same port if Windows delivers to both; strings (`state`, `user_text`, `ai_text`) only show up here.
3. **OSC Out DAT** — address `127.0.0.1`, port `9001`, send `/luzifer/in/say`, `/speak`, `/reset`, `/interrupt`, `/listen`.
4. **Audio File In → Audio VST → Audio Device Out**, all Timeslice. File = `luzifer-backend\runtime\output\speech.wav` (overwritten each line). On `luzifer/play` off→on, reload, wait 1 frame, play. On on→off, stop. Leave **Monitor locally** off.
5. `luzifer/in_level` and `luzifer/out_level` are 30 Hz floats on the same OSC In CHOP. Full chain is in [OSC_SPEC.md](OSC_SPEC.md).

## Live config

The dashboard `PUT /api/config` updates the in-memory `ConfigStore` and rewrites `config.yaml`. Prompt, LLM, VAD, TTS, and OSC host/ports apply without restarting; mic device/rate changes reopen the input stream on the next listen turn. Inbound OSC listen port rebinds on save.
