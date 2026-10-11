"""FastAPI live configuration dashboard (embedded in the same process)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field

from luzifer.config import AppSettings, ConfigStore
from luzifer import models as model_mgr
from luzifer.metrics import poll_gpu
from luzifer.osc.td_server import (
    ADDR_INTERRUPT,
    ADDR_LISTEN,
    ADDR_RESET,
    ADDR_SAY,
    ADDR_SPEAK,
    ADDR_VISION,
    TouchDesignerOscServer,
)
from luzifer.vision.ndi import list_ndi_sources
from luzifer.pipeline import Pipeline

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Luzifer Dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #0f1115; color: #e8eaed; min-height: 100vh; display:flex; flex-direction:column; }
    header { padding: .85rem 1.25rem; border-bottom: 1px solid #2a2f3a; display:flex; gap:.75rem; align-items:center; flex-wrap:wrap; }
    .tabs { display:flex; gap:.35rem; }
    .tab { background:#161a22; color:#9aa0a6; border:1px solid #2a2f3a; border-radius:8px; padding:.45rem .85rem; cursor:pointer; font-weight:600; }
    .tab.active { background:#1f6feb; color:white; border-color:#1f6feb; }
    .badge { padding: .25rem .6rem; border-radius: 999px; background:#1f6feb33; color:#79b8ff; font-size:.85rem; }
    .osc-strip { display:flex; gap:.5rem; align-items:center; margin-left:auto; font-size:.8rem; color:#9aa0a6; flex-wrap:wrap; }
    .osc-strip input { width:7.5rem; background:#0f1115; color:#e8eaed; border:1px solid #2a2f3a; border-radius:6px; padding:.3rem .45rem; }
    .osc-strip input.wide { width:9rem; }
    main { flex:1; padding:1rem 1.25rem; }
    .panel { display:none; background:#161a22; border:1px solid #2a2f3a; border-radius:12px; padding:1rem; }
    .panel.active { display:block; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:1rem; }
    .lang-cols { display:grid; grid-template-columns: 1fr 1fr; gap:.75rem; margin-top:.75rem; }
    .lang-card { background:#0f1115; border:1px solid #2a2f3a; border-radius:10px; padding:.75rem .85rem; }
    .lang-card h3 { margin:0 0 .35rem; font-size:.95rem; color:#e8eaed; }
    .lang-card .hint { margin:0 0 .55rem; font-size:.72rem; color:#6e7681; }
    .hint { margin:.35rem 0 .55rem; font-size:.72rem; color:#6e7681; }
    .hint code { font-size:.7rem; }
    h2 { margin:0 0 .75rem; font-size:1rem; color:#9aa0a6; text-transform:uppercase; letter-spacing:.04em; }
    label { display:block; font-size:.8rem; color:#9aa0a6; margin:.55rem 0 .2rem; }
    input, textarea, select { width:100%; box-sizing:border-box; background:#0f1115; color:#e8eaed;
      border:1px solid #2a2f3a; border-radius:8px; padding:.55rem .65rem; }
    textarea { min-height:160px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    button, .btn { background:#1f6feb; color:white; border:0; border-radius:8px;
      padding:.55rem .9rem; font-weight:600; cursor:pointer; }
    button:hover, .btn:hover { background:#388bfd; }
    button.secondary { background:#2a2f3a; }
    button.danger { background:#da3633; }
    .row { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; margin-top:.75rem; }
    .model-list { display:flex; flex-direction:column; gap:.4rem; max-height:280px; overflow:auto; margin-top:.5rem; }
    .model-row { display:flex; gap:.5rem; align-items:center; background:#0f1115; border:1px solid #2a2f3a; border-radius:8px; padding:.45rem .6rem; }
    .model-row .name { flex:1; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.85rem; }
    .pill { font-size:.7rem; padding:.15rem .45rem; border-radius:999px; text-transform:uppercase; letter-spacing:.03em; }
    .pill.ok { background:#23863644; color:#3fb950; }
    .pill.missing { background:#da363344; color:#f85149; }
    .live { white-space:pre-wrap; background:#0f1115; border-radius:8px; padding:.75rem; min-height:2.5rem; border:1px solid #2a2f3a; }
    footer { border-top:1px solid #2a2f3a; padding:1rem 1.25rem 1.25rem; background:#12151c; }
    footer .transcript-head { display:flex; align-items:center; gap:.75rem; flex-wrap:wrap; margin-bottom:.75rem; }
    footer .transcript-head h2 { margin:0; }
    .flags { display:flex; gap:.5rem; align-items:center; }
    .flag { display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .7rem; border-radius:999px;
      border:1px solid #2a2f3a; background:#0f1115; color:#6e7681; font-size:.8rem; font-weight:600; }
    .flag .dot { width:.55rem; height:.55rem; border-radius:50%; background:#3d444d; }
    .flag.on.mic { color:#3fb950; border-color:#23863666; background:#23863622; }
    .flag.on.mic .dot { background:#3fb950; box-shadow:0 0 8px #3fb950aa; }
    .flag.on.stt { color:#a371f7; border-color:#8957e566; background:#8957e522; }
    .flag.on.stt .dot { background:#a371f7; box-shadow:0 0 8px #a371f7aa; }
    .flag.on.think { color:#d29922; border-color:#9e6a0366; background:#9e6a0322; }
    .flag.on.think .dot { background:#d29922; box-shadow:0 0 8px #d29922aa; }
    .flag.on.tts { color:#f778ba; border-color:#bf398966; background:#bf398922; }
    .flag.on.tts .dot { background:#f778ba; box-shadow:0 0 8px #f778baaa; }
    .flag.on.speak { color:#79b8ff; border-color:#1f6feb66; background:#1f6feb22; }
    .flag.on.speak .dot { background:#79b8ff; box-shadow:0 0 8px #79b8ffaa; }
    .direct-play { display:inline-flex; align-items:center; gap:.55rem; margin-left:auto;
      color:#9aa0a6; font-size:.8rem; font-weight:600; cursor:pointer; user-select:none; }
    .direct-play input { position:absolute; opacity:0; width:0; height:0; }
    .switch { width:2.2rem; height:1.2rem; border-radius:999px; background:#2a2f3a; border:1px solid #3d444d;
      position:relative; transition:background .15s ease; flex-shrink:0; }
    .switch::after { content:""; position:absolute; top:1px; left:1px; width:.95rem; height:.95rem;
      border-radius:50%; background:#9aa0a6; transition:transform .15s ease, background .15s ease; }
    .direct-play input:checked + .switch { background:#1f6feb66; border-color:#1f6feb; }
    .direct-play input:checked + .switch::after { transform:translateX(1rem); background:#79b8ff; }
    .direct-play input:checked ~ .direct-play-label { color:#79b8ff; }
    footer .debug { display:flex; gap:.5rem; align-items:flex-end; flex-wrap:wrap; margin-top:.75rem; }
    footer .debug input { flex:1; min-width:220px; }
    #status { color:#9aa0a6; font-size:.85rem; }
    .metrics { display:flex; gap:1rem; align-items:center; flex-wrap:wrap; padding:.55rem 1.25rem;
      border-bottom:1px solid #2a2f3a; background:#12151c; font-size:.8rem; color:#9aa0a6; }
    .meter { display:flex; align-items:center; gap:.4rem; min-width:140px; }
    .meter label { margin:0; color:#9aa0a6; font-size:.75rem; text-transform:uppercase; letter-spacing:.03em; min-width:2.2rem; }
    .bar { flex:1; height:8px; background:#0f1115; border:1px solid #2a2f3a; border-radius:4px; overflow:hidden; min-width:80px; }
    .bar > i { display:block; height:100%; width:0%; background:linear-gradient(90deg,#238636,#3fb950); }
    .bar.out > i { background:linear-gradient(90deg,#1f6feb,#79b8ff); }
    .bar.gpu > i { background:linear-gradient(90deg,#9e6a03,#d29922); }
    .bar.vram > i { background:linear-gradient(90deg,#8957e5,#a371f7); }
    .metrics .val { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.75rem; color:#e8eaed; min-width:4.5rem; }
    @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <strong>Luzifer</strong>
    <nav class="tabs">
      <button class="tab active" data-tab="audio-in">Audio In</button>
      <button class="tab" data-tab="audio-out">Audio Out</button>
      <button class="tab" data-tab="prompt-llm">Prompt / LLM</button>
      <button class="tab" data-tab="vision">Vision</button>
    </nav>
    <span id="state" class="badge">connecting…</span>
    <span id="status"></span>
    <div class="osc-strip">
      <span>OSC → TD</span>
      <input id="osc.host" class="wide" placeholder="host" />
      <input id="osc.port" type="number" placeholder="9000" />
      <span>TD → OSC</span>
      <input id="osc.listen_host" class="wide" placeholder="listen host" />
      <input id="osc.listen_port" type="number" placeholder="9001" />
      <button id="save" class="secondary">Save</button>
    </div>
  </header>

  <div class="metrics">
    <div class="meter"><label>In</label><div class="bar" id="bar-in"><i></i></div><span class="val" id="val-in">0%</span></div>
    <div class="meter"><label>Out</label><div class="bar out" id="bar-out"><i></i></div><span class="val" id="val-out">0%</span></div>
    <div class="meter"><label>GPU</label><div class="bar gpu" id="bar-gpu"><i></i></div><span class="val" id="val-gpu">—</span></div>
    <div class="meter"><label>VRAM</label><div class="bar vram" id="bar-vram"><i></i></div><span class="val" id="val-vram">—</span></div>
    <span id="gpu-name" style="margin-left:auto"></span>
  </div>

  <main>
    <section id="panel-audio-in" class="panel active">
      <div class="grid">
        <div>
          <h2>Microphone / VAD</h2>
          <label>Input Device</label><select id="audio.input_device_id"></select>
          <label>Sample Rate</label><input id="audio.sample_rate" type="number" step="1" />
          <label>VAD Threshold</label><input id="audio.vad_threshold" type="number" step="0.05" />
          <label>Silence Duration (ms)</label><input id="audio.silence_duration_ms" type="number" step="50" />
          <label>Speaking Tail Pad (ms)</label><input id="audio.speaking_tail_pad_ms" type="number" step="50" />
        </div>
        <div>
          <h2>Whisper STT</h2>
          <label>Model Size</label>
          <select id="stt.model_size">
            <option>tiny</option><option>tiny.en</option>
            <option>base</option><option>base.en</option>
            <option>small</option><option>small.en</option>
            <option>medium</option><option>medium.en</option>
            <option>large-v2</option><option>large-v3</option>
            <option>distil-large-v3</option>
          </select>
          <label>Device</label>
          <select id="stt.device">
            <option value="cuda">cuda</option>
            <option value="cpu">cpu</option>
          </select>
          <label>Compute Type</label>
          <select id="stt.compute_type">
            <option value="float16">float16</option>
            <option value="int8">int8</option>
            <option value="float32">float32</option>
            <option value="int8_float16">int8_float16</option>
          </select>
          <label>Language (empty = auto)</label><input id="stt.language" />
          <h2 style="margin-top:1rem">Whisper Cache</h2>
          <div id="whisper-models" class="model-list"></div>
        </div>
      </div>
    </section>

    <section id="panel-audio-out" class="panel">
      <div class="grid">
        <div>
          <h2>TTS</h2>
          <label style="display:flex;gap:.5rem;align-items:center;margin-top:.35rem">
            <input id="tts.match_input_language" type="checkbox" style="width:auto" />
            Match engine/voice to last input language (not history)
          </label>
          <div class="lang-cols">
            <div class="lang-card">
              <h3>English</h3>
              <p class="hint">Used when Whisper reports <code>en</code></p>
              <label>Engine</label>
              <select id="tts.engine_by_lang.en">
                <option value="kokoro">kokoro</option>
                <option value="piper">piper</option>
              </select>
              <label>Voice</label>
              <select id="tts.voice_by_lang.en"></select>
            </div>
            <div class="lang-card">
              <h3>German</h3>
              <p class="hint">Used when Whisper reports <code>de</code></p>
              <label>Engine</label>
              <select id="tts.engine_by_lang.de">
                <option value="kokoro">kokoro</option>
                <option value="piper">piper</option>
              </select>
              <label>Voice</label>
              <select id="tts.voice_by_lang.de"></select>
            </div>
          </div>
          <label>Speed</label><input id="tts.speed" type="number" step="0.05" />
          <label>Device</label>
          <select id="tts.device">
            <option value="cuda">cuda</option>
            <option value="cpu">cpu</option>
            <option value="dml">dml (DirectML)</option>
          </select>
          <label>Compute Type</label>
          <select id="tts.compute_type">
            <option value="float32">float32</option>
            <option value="float16">float16</option>
          </select>
          <label>WAV Output Path</label><input id="tts.output_path" />
          <label>Local Output Device</label><select id="tts.output_device_id"></select>
          <label style="display:flex;gap:.5rem;align-items:center;margin-top:.85rem">
            <input id="tts.monitor_local" type="checkbox" style="width:auto" />
            Monitor locally (play speech.wav on this machine)
          </label>
        </div>
        <div>
          <h2>TTS Assets</h2>
          <div id="tts-models" class="model-list"></div>
        </div>
      </div>
    </section>

    <section id="panel-prompt-llm" class="panel">
      <div class="grid">
        <div>
          <h2>System Prompt</h2>
          <textarea id="system_prompt"></textarea>
          <h2 style="margin-top:1rem">LLM</h2>
          <label>Base URL</label><input id="llm.base_url" />
          <label>Model</label><input id="llm.model" />
          <label>Temperature</label><input id="llm.temperature" type="number" step="0.05" />
          <label>Top-P</label><input id="llm.top_p" type="number" step="0.05" />
          <label>Max Tokens</label><input id="llm.max_tokens" type="number" step="1" />
          <label>History Turns (0 = none)</label><input id="llm.history_turns" type="number" step="1" />
        </div>
        <div>
          <h2>Ollama Models</h2>
          <div class="row">
            <input id="pull_name" placeholder="model name to pull…" style="flex:1" />
            <button id="pull_btn" type="button">Pull</button>
            <button id="refresh_llm" class="secondary" type="button">Refresh</button>
          </div>
          <div id="llm-models" class="model-list"></div>
        </div>
      </div>
    </section>

    <section id="panel-vision" class="panel">
      <div class="grid">
        <div>
          <h2>NDI Vision</h2>
          <label style="display:flex;gap:.5rem;align-items:center;margin-top:.35rem">
            <input id="vision.enabled" type="checkbox" style="width:auto" />
            Enabled
          </label>
          <label>NDI Source</label>
          <div class="row">
            <input id="vision.ndi_source" placeholder="LuziferCam" style="flex:1" />
            <button id="refresh_ndi" class="secondary" type="button">Sources</button>
          </div>
          <select id="ndi_sources" style="margin-top:.4rem"></select>
          <label>Trigger</label>
          <select id="vision.trigger">
            <option value="external">external (OSC from TD)</option>
            <option value="local_idle">local_idle (backend silence timer)</option>
          </select>
          <p class="hint">External: TD decides idle + presence, then pulses <code>/luzifer/in/vision</code>. Local idle: backend captions when quiet.</p>
          <label>Idle After (s)</label><input id="vision.idle_after_s" type="number" step="0.5" />
          <label>Speak Cooldown (s)</label><input id="vision.speak_cooldown_s" type="number" step="0.5" />
          <label>Sample Interval (s)</label><input id="vision.interval_s" type="number" step="0.5" />
          <label>Change Threshold (0 = off)</label><input id="vision.change_threshold" type="number" step="0.5" />
          <label>Max Caption Age (s)</label><input id="vision.max_age_s" type="number" step="1" />
        </div>
        <div>
          <h2>Vision Model</h2>
          <label>Ollama VLM</label><input id="vision.model" placeholder="moondream" />
          <label>JPEG Max Edge</label><input id="vision.jpeg_max_edge" type="number" step="32" />
          <label>Num Predict</label><input id="vision.num_predict" type="number" step="1" />
          <label>Caption Prompt</label><textarea id="vision.prompt" rows="4"></textarea>
          <label style="display:flex;gap:.5rem;align-items:center;margin-top:.85rem">
            <input id="vision.inject_context" type="checkbox" style="width:auto" />
            Inject latest caption into next visitor turn
          </label>
          <label style="display:flex;gap:.5rem;align-items:center;margin-top:.35rem">
            <input id="vision.store_in_history" type="checkbox" style="width:auto" />
            Store vision comments in chat history
          </label>
          <h2 style="margin-top:1rem">Manual</h2>
          <div class="row">
            <button id="vision_analyze" type="button">Analyze</button>
            <button id="vision_comment" type="button">Analyze + Comment</button>
          </div>
          <p class="hint">Buttons always work (force). They queue behind an in-progress turn.</p>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <div class="transcript-head">
      <h2>Live Transcript</h2>
      <div class="flags">
        <span id="flag-mic" class="flag mic"><span class="dot"></span>Mic</span>
        <span id="flag-stt" class="flag stt"><span class="dot"></span>STT</span>
        <span id="flag-think" class="flag think"><span class="dot"></span>Thinking</span>
        <span id="flag-tts" class="flag tts"><span class="dot"></span>TTS</span>
        <span id="flag-speak" class="flag speak"><span class="dot"></span>AI speaking</span>
      </div>
      <label class="direct-play" title="Play speech.wav on this machine (skip waiting on TD)">
        <input id="direct_playback" type="checkbox" />
        <span class="switch"></span>
        <span class="direct-play-label">Direct playback</span>
      </label>
    </div>
    <div class="live"><strong>User:</strong> <span id="user_lang" style="opacity:.55;margin-right:.35rem"></span><span id="user_text"></span></div>
    <div class="live" style="margin-top:.5rem"><strong>AI:</strong> <span id="ai_text"></span></div>
    <div class="live" style="margin-top:.5rem"><strong>Vision:</strong> <span id="vision_text" style="opacity:.85"></span></div>
    <div class="debug">
      <div style="flex:1;min-width:260px">
        <label>Test text (skip mic)</label>
        <input id="test_text" placeholder="Type a visitor line…" />
      </div>
      <button id="send_say" type="button">Send → LLM+TTS</button>
      <button id="send_speak" class="secondary" type="button">Speak only</button>
      <button id="vision_analyze_footer" class="secondary" type="button">Analyze</button>
      <button id="vision_comment_footer" class="secondary" type="button">Analyze + Comment</button>
      <button id="interrupt" class="secondary" type="button">Interrupt</button>
      <button id="reset" class="danger" type="button">Reset session</button>
    </div>
  </footer>

<script>
const PATHS = [
  "system_prompt",
  "llm.model","llm.base_url","llm.temperature","llm.top_p","llm.max_tokens","llm.history_turns",
  "audio.sample_rate","audio.vad_threshold","audio.silence_duration_ms","audio.speaking_tail_pad_ms",
  "stt.model_size","stt.device","stt.compute_type","stt.language",
  "tts.speed","tts.device","tts.compute_type","tts.output_path",
  "tts.engine_by_lang.en","tts.engine_by_lang.de",
  "osc.host","osc.port","osc.listen_host","osc.listen_port",
  "vision.ndi_source","vision.trigger","vision.idle_after_s","vision.speak_cooldown_s",
  "vision.interval_s","vision.change_threshold","vision.max_age_s","vision.model",
  "vision.jpeg_max_edge","vision.num_predict","vision.prompt"
];
let KOKORO_VOICES = [];
let PIPER_VOICES = [];

function get(obj, path){ return path.split(".").reduce((o,k)=>o?.[k], obj); }
function set(obj, path, value){
  const parts = path.split(".");
  let cur = obj;
  for (let i=0;i<parts.length-1;i++) cur = cur[parts[i]];
  const key = parts[parts.length-1];
  const el = document.getElementById(path);
  if (!el) return;
  if (el.type === "number") value = Number(value);
  if (el.type === "checkbox") value = !!el.checked;
  if (path === "audio.input_device_id" || path === "tts.output_device_id") {
    value = value === "" || value === "null" ? null : Number(value);
  }
  if (path === "stt.language") value = value === "" ? null : value;
  cur[key] = value;
}
function setStatus(msg){ document.getElementById("status").textContent = msg || ""; }

document.querySelectorAll(".tab").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("panel-" + btn.dataset.tab).classList.add("active");
  };
});

async function loadDevices(kind, selectId, selected){
  const res = await fetch(`/api/audio/devices?kind=${kind}`);
  const devices = await res.json();
  const sel = document.getElementById(selectId);
  sel.innerHTML = '<option value="null">Default</option>';
  for (const d of devices){
    const opt = document.createElement("option");
    opt.value = d.index;
    opt.textContent = `${d.index}: ${d.name}`;
    sel.appendChild(opt);
  }
  sel.value = selected == null ? "null" : String(selected);
}

function renderModelRows(el, rows, onSelect, onDownload){
  el.innerHTML = "";
  for (const row of rows){
    const div = document.createElement("div");
    div.className = "model-row";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = row.label;
    const pill = document.createElement("span");
    pill.className = "pill " + (row.downloaded ? "ok" : "missing");
    pill.textContent = row.downloaded ? "downloaded" : "missing";
    div.appendChild(name);
    div.appendChild(pill);
    if (onSelect){
      const use = document.createElement("button");
      use.className = "secondary";
      use.type = "button";
      use.textContent = "Use";
      use.onclick = () => onSelect(row);
      div.appendChild(use);
    }
    if (onDownload && !row.downloaded){
      const dl = document.createElement("button");
      dl.type = "button";
      dl.textContent = "Download";
      dl.onclick = () => onDownload(row);
      div.appendChild(dl);
    }
    el.appendChild(div);
  }
}

async function refreshLlmModels(){
  const models = await (await fetch("/api/models/llm")).json();
  renderModelRows(
    document.getElementById("llm-models"),
    models.map(m => ({...m, label: m.name})),
    (row) => { document.getElementById("llm.model").value = row.name; setStatus("selected " + row.name); },
    async (row) => {
      setStatus("pulling " + row.name);
      await fetch("/api/models/llm/pull", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({name: row.name})});
    }
  );
}

async function refreshWhisperModels(){
  const models = await (await fetch("/api/models/whisper")).json();
  renderModelRows(
    document.getElementById("whisper-models"),
    models.map(m => ({...m, label: m.size})),
    (row) => { document.getElementById("stt.model_size").value = row.size; },
    async (row) => {
      setStatus("downloading whisper " + row.size);
      await fetch("/api/models/whisper/download", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({size: row.size})});
    }
  );
}

function fillVoiceSelect(selectId, engine, current){
  const sel = document.getElementById(selectId);
  if (!sel) return;
  const voices = engine === "piper" ? PIPER_VOICES : KOKORO_VOICES;
  sel.innerHTML = "";
  for (const v of voices){
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }
  if (current && [...sel.options].some(o => o.value === current)) sel.value = current;
  else if (current && !voices.includes(current)){
    const opt = document.createElement("option");
    opt.value = current;
    opt.textContent = current;
    sel.appendChild(opt);
    sel.value = current;
  } else if (voices.length) {
    sel.value = voices[0];
  }
}

function fillLangVoiceSelects(cfg){
  const enEngine = get(cfg, "tts.engine_by_lang.en")
    || document.getElementById("tts.engine_by_lang.en").value
    || get(cfg, "tts.engine")
    || "kokoro";
  const deEngine = get(cfg, "tts.engine_by_lang.de")
    || document.getElementById("tts.engine_by_lang.de").value
    || get(cfg, "tts.engine")
    || "kokoro";
  document.getElementById("tts.engine_by_lang.en").value = enEngine;
  document.getElementById("tts.engine_by_lang.de").value = deEngine;
  fillVoiceSelect(
    "tts.voice_by_lang.en",
    enEngine,
    get(cfg, "tts.voice_by_lang.en")
      || document.getElementById("tts.voice_by_lang.en").value
      || get(cfg, "tts.voice")
      || ""
  );
  fillVoiceSelect(
    "tts.voice_by_lang.de",
    deEngine,
    get(cfg, "tts.voice_by_lang.de")
      || document.getElementById("tts.voice_by_lang.de").value
      || get(cfg, "tts.voice")
      || ""
  );
}

async function refreshTtsModels(cfg){
  const data = await (await fetch("/api/models/tts")).json();
  KOKORO_VOICES = data.kokoro.voices || [];
  PIPER_VOICES = (data.piper || []).map(p => p.voice);
  fillLangVoiceSelects(cfg || {});
  const rows = [
    { id: "kokoro", label: "kokoro (onnx + voices)", downloaded: data.kokoro.downloaded },
    ...(data.piper || []).map(p => ({ id: p.id, label: p.label || p.voice || p.id, downloaded: p.downloaded }))
  ];
  renderModelRows(
    document.getElementById("tts-models"),
    rows,
    (row) => {
      const deEngine = document.getElementById("tts.engine_by_lang.de");
      const deVoice = document.getElementById("tts.voice_by_lang.de");
      if (row.id === "kokoro") {
        deEngine.value = "kokoro";
        fillVoiceSelect("tts.voice_by_lang.de", "kokoro", deVoice.value);
      } else {
        const voice = row.id.replace(/^piper:/, "");
        deEngine.value = "piper";
        fillVoiceSelect("tts.voice_by_lang.de", "piper", voice);
      }
      setStatus("assigned to German column");
    },
    async (row) => {
      setStatus("downloading " + row.id);
      await fetch("/api/models/tts/download", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({id: row.id})});
    }
  );
}

async function refreshNdiSources(){
  const sel = document.getElementById("ndi_sources");
  sel.innerHTML = '<option value="">— refresh —</option>';
  try {
    const sources = await (await fetch("/api/vision/sources")).json();
    for (const name of sources){
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    setStatus(sources.length ? (`${sources.length} NDI sources`) : "no NDI sources (is TD sending?)");
  } catch (e) {
    setStatus("NDI source list failed (install --extra vision?)");
  }
}

async function loadConfig(){
  const cfg = await (await fetch("/api/config")).json();
  for (const p of PATHS){
    const el = document.getElementById(p);
    if (!el) continue;
    const val = get(cfg, p);
    el.value = val == null ? "" : val;
  }
  setMonitorLocal(!!cfg.tts.monitor_local);
  document.getElementById("tts.match_input_language").checked = !!cfg.tts.match_input_language;
  const v = cfg.vision || {};
  document.getElementById("vision.enabled").checked = !!v.enabled;
  document.getElementById("vision.inject_context").checked = v.inject_context !== false;
  document.getElementById("vision.store_in_history").checked = !!v.store_in_history;
  await loadDevices("input", "audio.input_device_id", cfg.audio.input_device_id);
  await loadDevices("output", "tts.output_device_id", cfg.tts.output_device_id);
  await refreshLlmModels();
  await refreshWhisperModels();
  await refreshTtsModels(cfg);
  setStatus("config loaded");
  return cfg;
}

async function collectConfig(){
  const cfg = await (await fetch("/api/config")).json();
  for (const p of PATHS) set(cfg, p, document.getElementById(p).value);
  set(cfg, "audio.input_device_id", document.getElementById("audio.input_device_id").value);
  set(cfg, "tts.output_device_id", document.getElementById("tts.output_device_id").value);
  cfg.tts.monitor_local = document.getElementById("tts.monitor_local").checked;
  cfg.tts.match_input_language = document.getElementById("tts.match_input_language").checked;
  cfg.tts.engine_by_lang = cfg.tts.engine_by_lang || {};
  cfg.tts.voice_by_lang = cfg.tts.voice_by_lang || {};
  cfg.tts.engine_by_lang.en = document.getElementById("tts.engine_by_lang.en").value || "kokoro";
  cfg.tts.engine_by_lang.de = document.getElementById("tts.engine_by_lang.de").value || "kokoro";
  cfg.tts.voice_by_lang.en = document.getElementById("tts.voice_by_lang.en").value || cfg.tts.voice;
  cfg.tts.voice_by_lang.de = document.getElementById("tts.voice_by_lang.de").value || cfg.tts.voice;
  // Legacy single fields track English (fallback when match is off).
  cfg.tts.engine = cfg.tts.engine_by_lang.en;
  cfg.tts.voice = cfg.tts.voice_by_lang.en;
  const lang = document.getElementById("stt.language").value;
  cfg.stt.language = lang === "" ? null : lang;
  cfg.vision = cfg.vision || {};
  cfg.vision.enabled = document.getElementById("vision.enabled").checked;
  cfg.vision.inject_context = document.getElementById("vision.inject_context").checked;
  cfg.vision.store_in_history = document.getElementById("vision.store_in_history").checked;
  return cfg;
}

function setMonitorLocal(on){
  document.getElementById("tts.monitor_local").checked = !!on;
  document.getElementById("direct_playback").checked = !!on;
}

async function saveMonitorLocal(on){
  setMonitorLocal(on);
  const cfg = await collectConfig();
  cfg.tts.monitor_local = !!on;
  const res = await fetch("/api/config", {method:"PUT", headers:{"Content-Type":"application/json"}, body:JSON.stringify(cfg)});
  setStatus(res.ok ? (on ? "direct playback on" : "direct playback off") : "save failed");
}

document.getElementById("direct_playback").onchange = (ev) => saveMonitorLocal(ev.target.checked);
document.getElementById("tts.monitor_local").onchange = (ev) => saveMonitorLocal(ev.target.checked);

document.getElementById("save").onclick = async () => {
  const cfg = await collectConfig();
  const res = await fetch("/api/config", {method:"PUT", headers:{"Content-Type":"application/json"}, body:JSON.stringify(cfg)});
  setStatus(res.ok ? "saved live" : "save failed");
};
document.getElementById("tts.engine_by_lang.en").onchange = () => {
  fillVoiceSelect("tts.voice_by_lang.en", document.getElementById("tts.engine_by_lang.en").value, document.getElementById("tts.voice_by_lang.en").value);
};
document.getElementById("tts.engine_by_lang.de").onchange = () => {
  fillVoiceSelect("tts.voice_by_lang.de", document.getElementById("tts.engine_by_lang.de").value, document.getElementById("tts.voice_by_lang.de").value);
};
document.getElementById("refresh_llm").onclick = () => refreshLlmModels();
document.getElementById("pull_btn").onclick = async () => {
  const name = document.getElementById("pull_name").value.trim() || document.getElementById("llm.model").value.trim();
  if (!name) return;
  setStatus("pulling " + name);
  await fetch("/api/models/llm/pull", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({name})});
};
document.getElementById("send_say").onclick = async () => {
  const text = document.getElementById("test_text").value.trim();
  if (!text) return;
  await fetch("/api/debug/say", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({text})});
  setStatus("queued say");
};
document.getElementById("send_speak").onclick = async () => {
  const text = document.getElementById("test_text").value.trim();
  if (!text) return;
  await fetch("/api/debug/speak", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({text})});
  setStatus("queued speak");
};
document.getElementById("interrupt").onclick = async () => {
  await fetch("/api/debug/interrupt", {method:"POST"});
  setStatus("interrupt");
};
document.getElementById("reset").onclick = async () => {
  await fetch("/api/session/reset", {method:"POST"});
  setStatus("session reset");
};
async function queueVision(kind){
  const path = kind === "comment" ? "/api/vision/comment" : "/api/vision/analyze";
  await fetch(path, {method:"POST"});
  setStatus(kind === "comment" ? "vision comment queued" : "vision analyze queued");
}
document.getElementById("vision_analyze").onclick = () => queueVision("analyze");
document.getElementById("vision_comment").onclick = () => queueVision("comment");
document.getElementById("vision_analyze_footer").onclick = () => queueVision("analyze");
document.getElementById("vision_comment_footer").onclick = () => queueVision("comment");
document.getElementById("refresh_ndi").onclick = () => refreshNdiSources();
document.getElementById("ndi_sources").onchange = (ev) => {
  const v = ev.target.value;
  if (v) document.getElementById("vision.ndi_source").value = v;
};

function setMeter(barId, valId, level01, text){
  const pct = Math.max(0, Math.min(100, Math.round(level01 * 100)));
  document.querySelector(`#${barId} > i`).style.width = pct + "%";
  document.getElementById(valId).textContent = text != null ? text : (pct + "%");
}
function applyMetrics(m){
  setMeter("bar-in", "val-in", m.input_level || 0);
  setMeter("bar-out", "val-out", m.output_level || 0);
  if (m.gpu_available){
    setMeter("bar-gpu", "val-gpu", (m.gpu_util || 0) / 100, Math.round(m.gpu_util || 0) + "%");
    const used = m.vram_used_mb || 0, total = m.vram_total_mb || 1;
    setMeter("bar-vram", "val-vram", used / total, Math.round(used) + "/" + Math.round(total) + " MB");
    document.getElementById("gpu-name").textContent = m.gpu_name || "";
  } else {
    setMeter("bar-gpu", "val-gpu", 0, "n/a");
    setMeter("bar-vram", "val-vram", 0, "n/a");
    document.getElementById("gpu-name").textContent = "nvidia-smi unavailable";
  }
}

function setActivityFlags(flags){
  document.getElementById("flag-mic").classList.toggle("on", !!flags.mic_on);
  document.getElementById("flag-stt").classList.toggle("on", !!flags.stt);
  document.getElementById("flag-think").classList.toggle("on", !!flags.thinking);
  document.getElementById("flag-tts").classList.toggle("on", !!flags.synthesizing);
  document.getElementById("flag-speak").classList.toggle("on", !!flags.ai_speaking);
}

function activityPayload(msg){
  return {
    mic_on: msg.mic_on,
    stt: msg.stt,
    thinking: msg.thinking,
    synthesizing: msg.synthesizing,
    ai_speaking: msg.ai_speaking,
  };
}

function connectWs(){
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "metrics") {
      applyMetrics(msg);
      if ("mic_on" in msg || "stt" in msg || "thinking" in msg || "synthesizing" in msg || "ai_speaking" in msg) {
        setActivityFlags(activityPayload(msg));
      }
      return;
    }
    if (msg.state) document.getElementById("state").textContent = msg.state;
    if ("user_text" in msg) document.getElementById("user_text").textContent = msg.user_text || "";
    if ("language" in msg) {
      const lang = msg.language || "";
      document.getElementById("user_lang").textContent = lang ? `[${lang}]` : "";
    }
    if ("ai_text" in msg) document.getElementById("ai_text").textContent = msg.ai_text || "";
    if ("vision_text" in msg) document.getElementById("vision_text").textContent = msg.vision_text || "";
    if ("mic_on" in msg || "stt" in msg || "thinking" in msg || "synthesizing" in msg || "ai_speaking" in msg) {
      setActivityFlags(activityPayload(msg));
    }
    if (msg.type) {
      const pct = msg.progress == null ? "" : ` ${Math.round(msg.progress*100)}%`;
      setStatus(`${msg.type}: ${msg.status || ""}${pct}`);
      if (msg.status === "done") {
        refreshLlmModels();
        refreshWhisperModels();
        refreshTtsModels();
      }
    }
  };
  ws.onclose = () => setTimeout(connectWs, 1500);
}
loadConfig();
connectWs();
</script>
</body>
</html>
"""


class DashboardHub:
    """Fan-out pipeline state to websocket clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload)
        async with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


class TextBody(BaseModel):
    text: str = Field(min_length=1)


class PullBody(BaseModel):
    name: str = Field(min_length=1)


class WhisperDownloadBody(BaseModel):
    size: str = Field(min_length=1)


class TtsDownloadBody(BaseModel):
    id: str = Field(min_length=1)


def create_dashboard(store: ConfigStore, pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="Luzifer Dashboard", version="0.1.0")
    hub = DashboardHub()
    osc_server = TouchDesignerOscServer(on_command=_make_osc_handler(pipeline))
    app.state.osc_server = osc_server
    app.state.hub = hub
    app.state.bg_tasks = set()

    async def _on_state(state: str, user_text: str, ai_text: str) -> None:
        await hub.broadcast(
            {
                "state": state,
                "user_text": user_text,
                "ai_text": ai_text,
                "vision_text": pipeline.observations.text,
                "language": pipeline.last_language,
                "mic_on": pipeline.mic_on,
                "stt": pipeline.stt_active,
                "thinking": pipeline.thinking,
                "synthesizing": pipeline.synthesizing,
                "ai_speaking": pipeline.ai_speaking,
            }
        )

    pipeline.on_state(_on_state)

    def _track(task: asyncio.Task[Any]) -> None:
        app.state.bg_tasks.add(task)
        task.add_done_callback(app.state.bg_tasks.discard)

    async def _progress(payload: dict[str, Any]) -> None:
        await hub.broadcast(payload)

    @app.on_event("startup")
    async def _startup() -> None:
        settings = await store.get()
        await osc_server.start(settings.osc.listen_host, settings.osc.listen_port)

        async def _metrics_loop() -> None:
            gpu_every = 0
            while True:
                pipeline.metrics.tick(speaking=pipeline.state == "speaking")
                if gpu_every % 4 == 0:  # ~2s at 500ms tick
                    await poll_gpu(pipeline.metrics)
                gpu_every += 1
                payload = pipeline.metrics.snapshot()
                payload["mic_on"] = pipeline.mic_on
                payload["stt"] = pipeline.stt_active
                payload["thinking"] = pipeline.thinking
                payload["synthesizing"] = pipeline.synthesizing
                payload["ai_speaking"] = pipeline.ai_speaking
                payload["state"] = pipeline.state
                await hub.broadcast(payload)
                await asyncio.sleep(0.5)

        _track(asyncio.create_task(_metrics_loop(), name="metrics"))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await osc_server.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DASHBOARD_HTML

    @app.get("/api/metrics")
    async def get_metrics() -> dict[str, Any]:
        return pipeline.metrics.snapshot()

    @app.get("/api/config", response_model=AppSettings)
    async def get_config() -> AppSettings:
        return await store.get()

    @app.put("/api/config", response_model=AppSettings)
    async def put_config(body: AppSettings) -> AppSettings:
        updated = await store.update(body)
        pipeline.osc.reconfigure(updated.osc.host, updated.osc.port)
        await osc_server.reconfigure(updated.osc.listen_host, updated.osc.listen_port)
        # Warm EN/DE TTS targets after live config changes (Kokoro + optional Piper).
        _track(asyncio.create_task(pipeline.tts.preload(updated.tts)))
        return updated

    @app.get("/api/audio/devices")
    async def list_devices(kind: Literal["input", "output"] = "input") -> list[dict[str, Any]]:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            result: list[dict[str, Any]] = []
            for idx, dev in enumerate(devices):
                channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
                if int(dev.get(channels_key, 0)) <= 0:
                    continue
                result.append(
                    {
                        "index": idx,
                        "name": str(dev.get("name", f"device-{idx}")),
                        "channels": int(dev.get(channels_key, 0)),
                        "sample_rate": float(dev.get("default_samplerate", 16000)),
                    }
                )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Device query failed: {}", exc)
            return []

    @app.get("/api/models/llm")
    async def models_llm() -> list[dict[str, Any]]:
        settings = await store.get()
        return await model_mgr.list_llm_models(settings.llm.base_url)

    @app.post("/api/models/llm/pull")
    async def models_llm_pull(body: PullBody) -> dict[str, str]:
        settings = await store.get()

        async def _job() -> None:
            try:
                await model_mgr.pull_llm_model(settings.llm.base_url, body.name, on_progress=_progress)
            except Exception as exc:  # noqa: BLE001
                await _progress({"type": "llm_pull", "name": body.name, "status": f"error: {exc}", "progress": None})

        _track(asyncio.create_task(_job()))
        return {"status": "started"}

    @app.get("/api/models/whisper")
    async def models_whisper() -> list[dict[str, Any]]:
        return model_mgr.list_whisper_models()

    @app.post("/api/models/whisper/download")
    async def models_whisper_download(body: WhisperDownloadBody) -> dict[str, str]:
        async def _job() -> None:
            try:
                await model_mgr.download_whisper(body.size, on_progress=_progress)
            except Exception as exc:  # noqa: BLE001
                await _progress(
                    {"type": "whisper_download", "size": body.size, "status": f"error: {exc}", "progress": None}
                )

        _track(asyncio.create_task(_job()))
        return {"status": "started"}

    @app.get("/api/models/tts")
    async def models_tts() -> dict[str, Any]:
        return model_mgr.list_tts_models()

    @app.post("/api/models/tts/download")
    async def models_tts_download(body: TtsDownloadBody) -> dict[str, str]:
        async def _job() -> None:
            try:
                await model_mgr.download_tts(body.id, on_progress=_progress)
            except Exception as exc:  # noqa: BLE001
                await _progress({"type": "tts_download", "id": body.id, "status": f"error: {exc}", "progress": None})

        _track(asyncio.create_task(_job()))
        return {"status": "started"}

    @app.post("/api/debug/say")
    async def debug_say(body: TextBody) -> dict[str, str]:
        await pipeline.say(body.text)
        return {"status": "queued"}

    @app.post("/api/debug/speak")
    async def debug_speak(body: TextBody) -> dict[str, str]:
        await pipeline.speak(body.text)
        return {"status": "queued"}

    @app.post("/api/debug/interrupt")
    async def debug_interrupt() -> dict[str, str]:
        await pipeline.interrupt()
        return {"status": "ok"}

    @app.post("/api/session/reset")
    async def session_reset() -> dict[str, str]:
        await pipeline.reset_session()
        return {"status": "ok"}

    @app.get("/api/vision/sources")
    async def vision_sources() -> list[str]:
        return await asyncio.to_thread(list_ndi_sources, 5.0)

    @app.post("/api/vision/analyze")
    async def vision_analyze() -> dict[str, str]:
        await pipeline.analyze_vision()
        return {"status": "queued"}

    @app.post("/api/vision/comment")
    async def vision_comment() -> dict[str, str]:
        await pipeline.comment_vision()
        return {"status": "queued"}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        await ws.send_text(
            json.dumps(
                {
                    "state": pipeline.state,
                    "user_text": pipeline.last_user_text,
                    "ai_text": pipeline.last_ai_text,
                    "vision_text": pipeline.observations.text,
                    "language": pipeline.last_language,
                    "mic_on": pipeline.mic_on,
                    "stt": pipeline.stt_active,
                    "thinking": pipeline.thinking,
                    "synthesizing": pipeline.synthesizing,
                    "ai_speaking": pipeline.ai_speaking,
                }
            )
        )
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            await hub.disconnect(ws)

    return app


def _make_osc_handler(pipeline: Pipeline):
    async def _on_command(address: str, value: Any) -> None:
        logger.info("OSC in {} {!r}", address, value)
        if address == ADDR_SAY:
            text = str(value or "").strip()
            if text:
                await pipeline.say(text)
        elif address == ADDR_SPEAK:
            text = str(value or "").strip()
            if text:
                await pipeline.speak(text)
        elif address == ADDR_RESET:
            await pipeline.reset_session()
        elif address == ADDR_INTERRUPT:
            await pipeline.interrupt()
        elif address == ADDR_LISTEN:
            enabled = True
            if isinstance(value, (int, float)):
                enabled = bool(int(value))
            elif isinstance(value, str):
                enabled = value.strip() not in {"0", "false", "False", ""}
            await pipeline.set_listen(enabled)
        elif address == ADDR_VISION:
            fire = True
            if isinstance(value, (int, float)):
                fire = bool(int(value))
            elif isinstance(value, str):
                fire = value.strip() not in {"0", "false", "False", ""}
            if fire:
                await pipeline.request_vision_from_osc()

    return _on_command


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
