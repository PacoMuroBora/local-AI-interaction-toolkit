# Dashboard design

Operator page for the installation. One dark screen, no build step. The document is the `DASHBOARD_HTML` string in `src/luzifer/dashboard.py` (markup, `<style>`, and `<script>`). Routes and the websocket hub are `create_dashboard` in the same file.

Do not add a stylesheet, a JS bundle, or a component framework. New UI goes in that string and uses the tokens below.

## Chrome

Top to bottom, full viewport (`body` is a column, `main` is `flex: 1`):

1. **Header** — installation name (`#name`, class `brand`, saved with config), tab row, `#state` badge, `#status` text, **Save** (`#save`, `.secondary`, `margin-left: auto`).
2. **Metrics strip** (`.metrics`) — In, Out, GPU, VRAM. Not part of a tab. In is green, Out is blue, GPU is amber, VRAM is purple.
3. **One panel** — the active tab. Panels are a vertical stack of folds, not a card grid.
4. **Footer** — warmup bar (hidden once warmup finishes), live transcript, activity flags, direct-playback switch, debug actions. Stays visible on every tab.

Tabs: `floor`, `audio-in`, `audio-out`, `prompt-llm`, `vision`, `osc`. Markup is `button.tab[data-tab="{id}"]` and `section#panel-{id}.panel`. Floor is first and carries `active` — fire quiz / preach / spectacle, edit decks, see the live phase and last overlay prompt. A quiz card stores a question seed; the LLM performs the spoken intro with session history before waiting for the answer. The footer engagement strip mirrors mode / phase / card / show on every tab.

## Tokens

| Role | Value | Use |
|---|---|---|
| Page | `#0f1115` | `body`, inputs, inset cards, meter tracks |
| Panel | `#161a22` | `.panel`, idle tab, model row |
| Chrome | `#12151c` | footer, metrics strip |
| Border | `#2a2f3a` | panels, inputs, folds, flags |
| Text | `#e8eaed` | values, input text |
| Label | `#9aa0a6` | `label`, fold titles, section `h2` |
| Hint | `#6e7681` | `.hint` |
| Accent | `#1f6feb` | primary button, active tab, focus-level blue |
| Accent text | `#79b8ff` | badge, code in tables, links-in-hints, speak flag |
| Hover | `#388bfd` | primary button hover |
| Ok | `#3fb950` / `#238636` | mic + vision-ready flags, In meter, `.pill.ok` |
| Warn | `#d29922` / `#9e6a03` | thinking + vision-analyzing, warmup bar, GPU meter, `.goal-warn` |
| Danger | `#da3633` / `#f85149` | Reset, `.pill.missing` |
| STT | `#a371f7` | STT flag, VRAM meter |
| TTS | `#f778ba` | TTS flag |

Radius: `8px` inputs, buttons, meters-adjacent cards; `10px` folds and `.lang-card`; `12px` panels; `999px` pills, flags, the direct-playback switch. Font: `ui-sans-serif, system-ui, sans-serif`. Code, model names, and meter values: `ui-monospace, SFMono-Regular, Menlo, monospace`.

At `max-width: 960px`, `.grid` collapses to one column. Folds are already one column. `.lang-cols` stays two columns (English | German).

## Components

**Fold** — `details.fold` > `summary` > `.fold-title` plus optional `.fold-note`, then `.fold-body`. The chevron is the `summary::before` glyph, not a real marker. `data-fold` is required and unique. `initFolds` stores open/closed in `localStorage` under `luzifer.folds` and overrides the HTML `open` attribute on the next visit. Put `open` on the fold an operator needs first (mic, TTS, system prompt, NDI, outbound OSC); leave model lists and address tables closed.

**Field** — `label` then `input` / `select` / `textarea`. The `id` is the dotted `AppSettings` path (`audio.vad_threshold`, `tts.engine_by_lang.en`). `PATHS` in the script is what Save reads and load writes. Checkboxes and device `<select>`s are applied outside that list; if you add one, wire it in `loadConfig` / `collectConfig` the way `goals.enabled` and `audio.input_device_id` already are. `goals_items` is a textarea, not `goals.items` — one campaign per line.

**Hint** — `<p class="hint">` under a control whose effect is timing, gating, or storage. One or two sentences. Skip hints for “this is the temperature”.

**Buttons** — default is the primary action in that row (Save is the exception: it is `.secondary` so it doesn’t shout). `.secondary` for Refresh, Interrupt, Sources, Next. `.danger` for Reset session only. Disable a button while its job is in flight (vision Analyze / Comment while `vision_analyzing`).

**Flags** — `<span class="flag {kind}"><span class="dot"></span>Label</span>`. Add `on` to light it. Kinds: `mic`, `stt`, `think`, `tts`, `speak` in the footer; `vision-ready`, `vision-analyzing` on the Vision tab. Drive them from `setActivityFlags` / `setVisionFlags`. Do not invent a new color if an existing kind matches the phase.

**Model row** — `.model-list` > `.model-row` > `.name` + `.pill.ok` or `.pill.missing`. Use `renderModelRows`.

**Lang card** — `.lang-cols` > `.lang-card` with an `h3` and a `.hint` that says when the column is used (`Whisper reports en`).

**Live block** — `.live` in the footer: Goal, User (with `#user_lang`), AI, Vision. Read-only. The same vision string also fills `#vision_result` on the Vision tab (`setVisionResult`).

**Table** — `.io-table` for OSC maps. Address in `code`, type in `.typ`. If you change a row, change `docs/OSC_SPEC.md` and `.cursor/rules/02-touchdesigner-osc.mdc` in the same edit.

**Switch** — the direct-playback control is the only toggle that is not a native checkbox. It writes `tts.monitor_local` immediately (`saveMonitorLocal`), not on Save.

## Copy

Labels are short title case (`Silence Duration (ms)`, `GPU layers`). Fold titles are the group name (`Microphone / VAD`). The header badge is the pipeline state word (`listening`, `stt`, `thinking`, `synthesizing`, `speaking`, `loading`). Status (`#status`) is a short result of the last action (`session reset`, `next goal`), not a log.

## Adding a setting

1. Field on the right model in `config.py`, with a default. Validators live there, not in the page.
2. Control in the matching fold. `id` = dotted path. Number inputs get `step`.
3. Append the path to `PATHS` unless load/save is special-cased.
4. If changing it should reload Ollama, add it to `_runner_budget`. Anything else is picked up on the next turn from `ConfigStore`.
5. Hint only when the operator can mis-time it (tail pad, idle after, GPU layers, secret file).

## Adding a tab or a fold

Tab: button in `nav.tabs`, `section.panel` with `id="panel-{data-tab}"`. The existing click handler toggles `.active`; don’t add a router.

Fold: `details.fold` and a new `data-fold` string. Don’t reuse an id — `luzifer.folds` would reopen the wrong section.

## Live data

`/ws` is the only push channel. `connectWs` reconnects 1.5 s after close.

| Message | What the page does |
|---|---|
| `type: "metrics"` | meters, and goal / engagement / empty-room streak / flags / vision text when those keys are present |
| `state`, `user_text`, `ai_text`, `language` | badge and transcript |
| `warmup` | `#warmup-bar` (amber). Hide when the payload is null |
| `goal` | `#goal_now`, fold note, footer |
| `type` + `status` + `progress` | `#status` during model pull/download; refresh lists when `status == "done"` |

Config is not pushed. Load with `GET /api/config`, save with `PUT /api/config`. NDI source and direct playback save themselves. Test line, interrupt, reset, next goal, and vision CTAs are their own `POST`s (`/api/debug/say`, `/api/debug/speak`, `/api/debug/interrupt`, `/api/session/reset`, `/api/goals/next`, `/api/vision/analyze`, `/api/vision/comment`).

## Secrets and destructive actions

ElevenLabs API key: `type="password"`, `autocomplete="off"`, placeholder says a blank value keeps the stored key. The key is written to `runtime/secrets.yaml`. Don’t render it into a `.live` block or a hint.

Reset is the only red button. Its `title` says it clears history and advances the campaign. Next goal is `.secondary` and does not clear history.
