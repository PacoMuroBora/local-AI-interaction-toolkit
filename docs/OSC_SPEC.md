# OSC specification

Bidirectional UDP OSC between Luzifer and TouchDesigner.

| Direction | Default | Operator |
|-----------|---------|----------|
| Luzifer → TD | `127.0.0.1:9000` | TD **OSC In DAT** |
| TD → Luzifer | `127.0.0.1:9001` | TD **OSC Out DAT** |

## Outbound (Luzifer → TouchDesigner)

Wire an **OSC In DAT** (or OSC In CHOP) to port `9000`.

| Address | Type | Example | TouchDesigner behavior |
|---------|------|---------|------------------------|
| `/luzifer/state` | string | `listening` | Drive UI / animation modes. Values: `listening`, `thinking`, `speaking`. |
| `/luzifer/user_text` | string | `Who are you?` | Show visitor transcript (text DAT / text TOP). |
| `/luzifer/ai_text` | string | `I am Luzifer…` | Show AI reply before/during playback. |
| `/luzifer/play` | int | `1` | Pulse: reload `runtime/output/speech.wav` in **Audio File In CHOP** and start playback (VST FX chain as needed). |

### Ordering

Typical turn:

1. `/luzifer/state` → `thinking`
2. `/luzifer/user_text` → transcript
3. `/luzifer/ai_text` → reply
4. `/luzifer/state` → `speaking`
5. File write completes → `/luzifer/play` → `1`
6. After wav duration + tail pad → `/luzifer/state` → `listening`

## Inbound (TouchDesigner → Luzifer)

Add an **OSC Out DAT**. Network Address `127.0.0.1`, Port `9001`. One message per row (address + argument).

| Address | Arg | What it does |
|---------|-----|--------------|
| `/luzifer/in/say` | string | Visitor line. Skips mic/STT. Runs LLM + TTS + `/luzifer/play`. Appends chat history. |
| `/luzifer/in/speak` | string | Say this line with TTS only. No LLM, no history. |
| `/luzifer/in/reset` | int `1` | Clear chat history and empty transcript fields. |
| `/luzifer/in/interrupt` | int `1` | Abort think/speak, return to `listening`. |
| `/luzifer/in/listen` | int `1` or `0` | Unmute (`1`) or mute (`0`) the microphone. |

### Example OSC Out DAT rows

```
/luzifer/in/say	who are you
/luzifer/in/speak	The gate is open.
/luzifer/in/reset	1
/luzifer/in/interrupt	1
/luzifer/in/listen	0
```

Send with a CHOP Execute / DAT Execute that calls `op('oscout1').sendOSC(address, [arg])`, or pulse the OSC Out DAT send.

## Anti-feedback

While `state` is `thinking` or `speaking`, Luzifer stops the microphone stream so room speakers cannot re-trigger VAD. `/luzifer/in/listen 0` also keeps the mic closed while `state` may still be `listening`.

## WAV path

Absolute or project-relative path configured as `tts.output_path` (default `runtime/output/speech.wav`). Point Audio File In at that file and re-read on each `/luzifer/play`. Optional dashboard **Monitor locally** plays the same file on a PC output device (off by default).
