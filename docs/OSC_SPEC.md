# OSC specification

Luzifer sends UDP OSC to TouchDesigner (default `127.0.0.1:9000`). Wire an **OSC In DAT** (or OSC In CHOP) to this port.

## Channels

| Address | Type | Example | TouchDesigner behavior |
|---------|------|---------|------------------------|
| `/luzifer/state` | string | `listening` | Drive UI / animation modes. Values: `listening`, `thinking`, `speaking`. |
| `/luzifer/user_text` | string | `Who are you?` | Show visitor transcript (text DAT / text TOP). |
| `/luzifer/ai_text` | string | `I am Luzifer…` | Show AI reply before/during playback. |
| `/luzifer/play` | int | `1` | Pulse: reload `runtime/output/speech.wav` in **Audio File In CHOP** and start playback (VST FX chain as needed). |

## Ordering

Typical turn:

1. `/luzifer/state` → `thinking`
2. `/luzifer/user_text` → transcript
3. `/luzifer/ai_text` → reply
4. `/luzifer/state` → `speaking`
5. File write completes → `/luzifer/play` → `1`
6. After wav duration + tail pad → `/luzifer/state` → `listening`

## Anti-feedback

While `state` is `thinking` or `speaking`, Luzifer stops the microphone stream so room speakers cannot re-trigger VAD.

## WAV path

Absolute or project-relative path configured as `tts.output_path` (default `runtime/output/speech.wav`). Point Audio File In at that file and re-read on each `/luzifer/play`.
