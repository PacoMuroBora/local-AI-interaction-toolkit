"""FastAPI live configuration dashboard (embedded in the same process)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger

from luzifer.config import AppSettings, ConfigStore
from luzifer.pipeline import Pipeline

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Luzifer Dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #0f1115; color: #e8eaed; }
    header { padding: 1rem 1.5rem; border-bottom: 1px solid #2a2f3a; display:flex; gap:1rem; align-items:center; }
    .badge { padding: .25rem .6rem; border-radius: 999px; background:#1f6feb33; color:#79b8ff; font-size:.85rem; }
    main { display:grid; grid-template-columns: 1fr 1fr; gap:1rem; padding:1rem 1.5rem 2rem; }
    section { background:#161a22; border:1px solid #2a2f3a; border-radius:12px; padding:1rem; }
    h2 { margin:0 0 .75rem; font-size:1rem; color:#9aa0a6; text-transform:uppercase; letter-spacing:.04em; }
    label { display:block; font-size:.8rem; color:#9aa0a6; margin:.55rem 0 .2rem; }
    input, textarea, select { width:100%; box-sizing:border-box; background:#0f1115; color:#e8eaed;
      border:1px solid #2a2f3a; border-radius:8px; padding:.55rem .65rem; }
    textarea { min-height:140px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    button { margin-top:1rem; background:#1f6feb; color:white; border:0; border-radius:8px;
      padding:.7rem 1rem; font-weight:600; cursor:pointer; }
    button:hover { background:#388bfd; }
    .live { white-space:pre-wrap; background:#0f1115; border-radius:8px; padding:.75rem; min-height:4rem; border:1px solid #2a2f3a; }
    @media (max-width: 960px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <strong>Luzifer</strong>
    <span id="state" class="badge">connecting…</span>
    <span id="status" style="color:#9aa0a6;font-size:.85rem"></span>
  </header>
  <main>
    <section>
      <h2>System Prompt</h2>
      <textarea id="system_prompt"></textarea>
      <h2 style="margin-top:1rem">LLM</h2>
      <label>Model</label><input id="llm.model" />
      <label>Base URL</label><input id="llm.base_url" />
      <label>Temperature</label><input id="llm.temperature" type="number" step="0.05" />
      <label>Top-P</label><input id="llm.top_p" type="number" step="0.05" />
      <label>Max Tokens</label><input id="llm.max_tokens" type="number" step="1" />
    </section>
    <section>
      <h2>Audio / VAD</h2>
      <label>Input Device</label><select id="audio.input_device_id"></select>
      <label>VAD Threshold</label><input id="audio.vad_threshold" type="number" step="0.05" />
      <label>Silence Duration (ms)</label><input id="audio.silence_duration_ms" type="number" step="50" />
      <label>Speaking Tail Pad (ms)</label><input id="audio.speaking_tail_pad_ms" type="number" step="50" />
      <h2 style="margin-top:1rem">TTS</h2>
      <label>Engine</label>
      <select id="tts.engine"><option value="kokoro">kokoro</option><option value="piper">piper</option></select>
      <label>Voice</label><input id="tts.voice" />
      <label>Speed</label><input id="tts.speed" type="number" step="0.05" />
      <h2 style="margin-top:1rem">OSC → TouchDesigner</h2>
      <label>Host</label><input id="osc.host" />
      <label>Port</label><input id="osc.port" type="number" />
      <button id="save">Save Live Config</button>
    </section>
    <section style="grid-column:1 / -1">
      <h2>Live Transcript</h2>
      <div class="live"><strong>User:</strong> <span id="user_text"></span></div>
      <div class="live" style="margin-top:.5rem"><strong>AI:</strong> <span id="ai_text"></span></div>
    </section>
  </main>
<script>
const PATHS = [
  "system_prompt","llm.model","llm.base_url","llm.temperature","llm.top_p","llm.max_tokens",
  "audio.vad_threshold","audio.silence_duration_ms","audio.speaking_tail_pad_ms",
  "tts.engine","tts.voice","tts.speed","osc.host","osc.port"
];
function get(obj, path){ return path.split(".").reduce((o,k)=>o?.[k], obj); }
function set(obj, path, value){
  const parts = path.split(".");
  let cur = obj;
  for (let i=0;i<parts.length-1;i++) cur = cur[parts[i]];
  const key = parts[parts.length-1];
  const el = document.getElementById(path);
  if (el && el.type === "number") value = Number(value);
  if (path === "audio.input_device_id") value = value === "" || value === "null" ? null : Number(value);
  cur[key] = value;
}
async function loadDevices(selected){
  const res = await fetch("/api/audio/devices");
  const devices = await res.json();
  const sel = document.getElementById("audio.input_device_id");
  sel.innerHTML = '<option value="null">Default</option>';
  for (const d of devices){
    const opt = document.createElement("option");
    opt.value = d.index;
    opt.textContent = `${d.index}: ${d.name}`;
    sel.appendChild(opt);
  }
  sel.value = selected == null ? "null" : String(selected);
}
async function loadConfig(){
  const cfg = await (await fetch("/api/config")).json();
  for (const p of PATHS){
    const el = document.getElementById(p);
    if (el) el.value = get(cfg, p) ?? "";
  }
  await loadDevices(cfg.audio.input_device_id);
  document.getElementById("status").textContent = "config loaded";
  return cfg;
}
document.getElementById("save").onclick = async () => {
  const cfg = await (await fetch("/api/config")).json();
  for (const p of PATHS) set(cfg, p, document.getElementById(p).value);
  set(cfg, "audio.input_device_id", document.getElementById("audio.input_device_id").value);
  const res = await fetch("/api/config", {method:"PUT", headers:{"Content-Type":"application/json"}, body:JSON.stringify(cfg)});
  document.getElementById("status").textContent = res.ok ? "saved live" : "save failed";
};
function connectWs(){
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    document.getElementById("state").textContent = msg.state;
    document.getElementById("user_text").textContent = msg.user_text || "";
    document.getElementById("ai_text").textContent = msg.ai_text || "";
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


def create_dashboard(store: ConfigStore, pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="Luzifer Dashboard", version="0.1.0")
    hub = DashboardHub()

    async def _on_state(state: str, user_text: str, ai_text: str) -> None:
        await hub.broadcast({"state": state, "user_text": user_text, "ai_text": ai_text})

    pipeline.on_state(_on_state)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DASHBOARD_HTML

    @app.get("/api/config", response_model=AppSettings)
    async def get_config() -> AppSettings:
        return await store.get()

    @app.put("/api/config", response_model=AppSettings)
    async def put_config(body: AppSettings) -> AppSettings:
        updated = await store.update(body)
        pipeline.osc.reconfigure(updated.osc.host, updated.osc.port)
        return updated

    @app.get("/api/audio/devices")
    async def list_devices() -> list[dict[str, Any]]:
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            result: list[dict[str, Any]] = []
            for idx, dev in enumerate(devices):
                if int(dev.get("max_input_channels", 0)) <= 0:
                    continue
                result.append(
                    {
                        "index": idx,
                        "name": str(dev.get("name", f"device-{idx}")),
                        "channels": int(dev.get("max_input_channels", 0)),
                        "sample_rate": float(dev.get("default_samplerate", 16000)),
                    }
                )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Device query failed: {}", exc)
            return []

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        await ws.send_text(
            json.dumps(
                {
                    "state": pipeline.state,
                    "user_text": pipeline.last_user_text,
                    "ai_text": pipeline.last_ai_text,
                }
            )
        )
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            await hub.disconnect(ws)

    return app


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
