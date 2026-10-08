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
| `/luzifer/state` | string | `listening` | DAT only. Values: `listening`, `thinking`, `speaking`. |
| `/luzifer/user_text` | string | `Who are you?` | DAT only. Visitor transcript. |
| `/luzifer/ai_text` | string | `I am Luzifer…` | DAT only. AI reply. |
| `/luzifer/listening` | int | `1` or `0` | CHOP. `1` while the pipeline is in `listening`. |
| `/luzifer/user_speaking` | int | `1` or `0` | CHOP. `1` only while VAD hears the visitor. `0` the moment they stop, and whenever state is not `listening`. |
| `/luzifer/thinking` | int | `1` or `0` | CHOP. `1` during STT + LLM. |
| `/luzifer/ai_speaking` | int | `1` or `0` | CHOP. `1` for the whole speak phase (synth, playback, tail). |
| `/luzifer/play` | int | `1` then `0` | CHOP. `1` once `speech.wav` is written. Back to `0` when speaking ends, so the next line is a rising edge. |
| `/luzifer/in_level` | float | `0.0`–`1.0` | CHOP. Mic peak, 30 Hz. `0` within ~120 ms of the mic closing. |
| `/luzifer/out_level` | float | `0.0`–`1.0` | CHOP. Dry-wav peak envelope, 30 Hz, clocked from the `play` edge. `0` while synthesizing and after the file ends. Not the post-VST level. |

Ints are `0`/`1`, not OSC bool. A bool has no payload and OSC In CHOP often never makes a channel. Every flag snapshot sends all four addresses, so an inactive flag is explicitly `0`. OSC In CHOP sample-and-holds: if you only ever send `1`, the channel stays `1`.

`listening` and `user_speaking` can both be `1`. The other combinations are exclusive. Idle with the mic armed is `listening=1` and the other three `0`. Clean shutdown sends all four `0` and `play=0`. A hard kill cannot send that; the CHOP keeps the last numbers.

### Ordering

Typical turn:

1. `/luzifer/listening` → `1`, other flags → `0`
2. Visitor starts talking → `/luzifer/user_speaking` → `1`
3. Visitor stops → `/luzifer/user_speaking` → `0`, then `/luzifer/thinking` → `1`, `/luzifer/listening` → `0`
4. `/luzifer/user_text`, `/luzifer/ai_text`
5. `/luzifer/ai_speaking` → `1`, `/luzifer/thinking` → `0` (wav not ready yet, `play` still `0`)
6. File write completes → `/luzifer/play` → `1` and `/luzifer/out_level` starts the envelope
7. After wav duration + tail → `/luzifer/play` → `0`, `/luzifer/out_level` → `0`, `/luzifer/ai_speaking` → `0`, `/luzifer/listening` → `1`

### OSC In CHOP

Port `9000`. Strings never become channels; that's why the CHOP in the screenshot only showed `luzifer/play`. After a restart the int channels show up as:

- `luzifer/listening`
- `luzifer/user_speaking`
- `luzifer/thinking`
- `luzifer/ai_speaking`
- `luzifer/play`
- `luzifer/in_level`
- `luzifer/out_level`

```python
op('oscin1')['luzifer/thinking']
```

Rename CHOP, from → to:

```
luzifer/listening luzifer/user_speaking luzifer/thinking luzifer/ai_speaking luzifer/play
listening user thinking ai play
```

Switch TOP index (one channel, Math or Expression CHOP):

```python
op('rename1')['user'] + op('rename1')['thinking']*2 + op('rename1')['ai']*3
```

`0` idle, `1` person, `2` thinking, `3` AI. Export the bool channels straight onto blends.

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
    if channel.name not in ('luzifer/play', 'play'):
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
    if channel.name not in ('luzifer/play', 'play'):
        return
    op('audiofilein1').par.play = False
```

Keep the OSC In DAT for `user_text`, `ai_text`, and the `state` string. On Windows both operators can share port `9000`. If one of them stays empty, only one bind won — leave the CHOP on `9000`.

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
