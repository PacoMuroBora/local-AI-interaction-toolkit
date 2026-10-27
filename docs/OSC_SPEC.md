# OSC specification

Bidirectional UDP OSC between the toolkit and TouchDesigner. The address root is `/ai`. The installation name (Luzifer, or whatever `name` is set to) is not part of the protocol.

| Direction | Default | Operator |
|-----------|---------|----------|
| Toolkit → TD | `127.0.0.1:9000` | TD **OSC In DAT** |
| TD → Toolkit | `127.0.0.1:9001` | TD **OSC Out DAT** |

## Outbound (toolkit → TouchDesigner)

Wire an **OSC In DAT** (or OSC In CHOP) to port `9000`.

| Address | Type | Example | TouchDesigner behavior |
|---------|------|---------|------------------------|
| `/ai/state` | string | `listening` | DAT only. Values: `loading`, `listening`, `stt`, `thinking`, `synthesizing`, `speaking`. `loading` is startup warmup; `/ai/listening` stays `0`. |
| `/ai/user_text` | string | `Who are you?` | DAT only. Visitor transcript. |
| `/ai/ai_text` | string | `Here I am.` | DAT only. AI reply. |
| `/ai/listening` | int | `1` or `0` | CHOP. `1` while the pipeline is in `listening`. |
| `/ai/user_speaking` | int | `1` or `0` | CHOP. `1` only while VAD hears the visitor. `0` the moment they stop, and whenever state is not `listening`. |
| `/ai/stt` | int | `1` or `0` | CHOP. `1` while Whisper is transcribing. |
| `/ai/thinking` | int | `1` or `0` | CHOP. `1` while the LLM is generating. |
| `/ai/synthesizing` | int | `1` or `0` | CHOP. `1` while TTS is writing `speech.wav`. |
| `/ai/ai_speaking` | int | `1` or `0` | CHOP. `1` during playback + tail (after the file exists). |
| `/ai/play` | int | `1` then `0` | CHOP. `1` once `speech.wav` is written. Back to `0` when speaking ends, so the next line is a rising edge. |
| `/ai/in_level` | float | `0.0`–`1.0` | CHOP. Mic peak, 30 Hz. `0` within ~120 ms of the mic closing. |
| `/ai/out_level` | float | `0.0`–`1.0` | CHOP. Dry-wav peak envelope, 30 Hz, clocked from the `play` edge. `0` while synthesizing and after the file ends. Not the post-VST level. |
| `/ai/vision_text` | string | `A person stands…` | DAT. Latest NDI still caption from the vision VLM. |
| `/ai/vision` | int | `1` then `0` | CHOP. Pulse on each new caption (rising edge). |
| `/ai/vision_active` | int | `1` or `0` | CHOP. `1` while the vision VLM HTTP call is in flight. |

Ints are `0`/`1`, not OSC bool. A bool has no payload and OSC In CHOP often never makes a channel. Every flag snapshot sends all six addresses, so an inactive flag is explicitly `0`. OSC In CHOP sample-and-holds: if you only ever send `1`, the channel stays `1`.

`listening` and `user_speaking` can both be `1`. `stt` / `thinking` / `synthesizing` / `ai_speaking` are exclusive. Idle with the mic armed is `listening=1` and the other flags `0`. Clean shutdown sends every flag `0` and `play=0`. A hard kill cannot send that; the CHOP keeps the last numbers.

### Ordering

Typical turn:

1. `/ai/listening` → `1`, other flags → `0`
2. Visitor starts talking → `/ai/user_speaking` → `1`
3. Visitor stops → `/ai/user_speaking` → `0`, `/ai/stt` → `1`, `/ai/listening` → `0`
4. `/ai/user_text`, then `/ai/thinking` → `1`, `/ai/stt` → `0`
5. `/ai/ai_text`, then `/ai/synthesizing` → `1`, `/ai/thinking` → `0`
6. File ready → `/ai/ai_speaking` → `1`, `/ai/synthesizing` → `0`, `/ai/play` → `1`, `/ai/out_level` starts
7. After wav duration + tail → `/ai/play` → `0`, `/ai/out_level` → `0`, `/ai/ai_speaking` → `0`, `/ai/listening` → `1`

### OSC In CHOP

Port `9000`. Strings never become channels; that's why the CHOP in the screenshot only showed `ai/play`. After a restart the int channels show up as:

- `ai/listening`
- `ai/user_speaking`
- `ai/stt`
- `ai/thinking`
- `ai/synthesizing`
- `ai/ai_speaking`
- `ai/play`
- `ai/in_level`
- `ai/out_level`

```python
op('oscin1')['ai/thinking']
```

Rename CHOP, from → to:

```
ai/listening ai/user_speaking ai/stt ai/thinking ai/synthesizing ai/ai_speaking ai/play
listening user stt thinking synth speaking play
```

Export the bool channels straight onto blends.

`in_level` / `out_level` update at 30 Hz even when the value is `0`, so a late CHOP still fills in. Speech peaks are often under `0.3`; scale them in a Math CHOP. `out_level` is the dry file, starting at the `play` edge. After a VST, meter with an Analyze CHOP on the VST output.

### Playback (VST chain)

Play the wav in TouchDesigner. The backend only writes the file.

`D:\Admin\Documents\TouchDesigner\Luzifer_Robodonien2026\AI\luzifer-backend\runtime\output\speech.wav`

Same path every line (overwritten). Kokoro writes 16-bit mono at 24 kHz. Leave dashboard **Monitor locally** off or the line plays twice.

```
audiofilein1  (Timeslice on, Play Mode Sequential, Repeat off, Speed 1)
  -> audiovst1
  -> audiodeviceout1
```

Every operator in that chain needs Timeslice. The VST gets one giant buffer otherwise and you hear nothing live.

Trigger on `play` rising, not on `ai_speaking`. `ai_speaking` goes high while the file is still being written. Reload, wait one frame so the new bytes are what gets decoded, then play. Stop on the falling edge so an interrupt doesn't keep the VST sounding into an open mic.

CHOP Execute on the OSC In (or the rename). Enable **Off to On** and **On to Off**:

```python
def onOffToOn(channel, sampleIndex, val, prev):
    if channel.name not in ('ai/play', 'play'):
        return
    a = op('audiofilein1')
    a.par.play = False
    a.par.reload.pulse()
    run(
        "a = op('audiofilein1')\n"
        "a.par.cue = 0\n"
        "a.par.cuepulse.pulse()\n"
        "a.par.play = True",
        delayFrames=1,
    )

def onOnToOff(channel, sampleIndex, val, prev):
    if channel.name not in ('ai/play', 'play'):
        return
    op('audiofilein1').par.play = False
```

Keep the OSC In DAT for `user_text`, `ai_text`, and the `state` string. On Windows both operators can share port `9000`. If one of them stays empty, only one bind won — leave the CHOP on `9000`.

## Inbound (TouchDesigner → toolkit)

Add an **OSC Out DAT**. Network Address `127.0.0.1`, Port `9001`. One message per row (address + argument).

| Address | Arg | What it does |
|---------|-----|--------------|
| `/ai/in/say` | string | Visitor line. Skips mic/STT. Runs LLM + TTS + `/ai/play`. Appends chat history. |
| `/ai/in/speak` | string | Say this line with TTS only. No LLM, no history. |
| `/ai/in/reset` | int `1` | Clear chat history, empty transcript fields, and advance to the next campaign. |
| `/ai/in/goal` | int `1` | Skip to the next campaign. The beat resets. Chat history stays. `0` is ignored. |
| `/ai/in/interrupt` | int `1` | Abort think/speak, return to `listening`. |
| `/ai/in/listen` | int `1` or `0` | Unmute (`1`) or mute (`0`) the microphone. |
| `/ai/in/vision` | int `1` | Grab NDI still, caption, speak a reaction. Only when `vision.enabled` and `vision.trigger=external`. TD should gate on idle + presence. |

### Example OSC Out DAT rows

```
/ai/in/say	who are you
/ai/in/speak	The gate is open.
/ai/in/reset	1
/ai/in/goal	1
/ai/in/interrupt	1
/ai/in/listen	0
/ai/in/vision	1
```

Send with a CHOP Execute / DAT Execute that calls `op('oscout1').sendOSC(address, [arg])`, or pulse the OSC Out DAT send.

## Anti-feedback

While `state` is `thinking` or `speaking`, the toolkit stops the microphone stream so room speakers cannot re-trigger VAD. `/ai/in/listen 0` also keeps the mic closed while `state` may still be `listening`.

## WAV path

Absolute or project-relative path configured as `tts.output_path` (default `runtime/output/speech.wav`). Point Audio File In at that file and re-read on each `/ai/play`. Optional dashboard **Monitor locally** plays the same file on a PC output device (off by default).
