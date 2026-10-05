"""Model listing and download helpers (Ollama / Whisper / TTS assets)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from luzifer.config import ROOT_DIR

ProgressCb = Callable[[dict[str, Any]], Awaitable[None] | None]

CURATED_LLM = [
    "llama3.2",
    "llama3.2:1b",
    "qwen2.5:3b",
    "gemma3:4b",
    "phi4-mini",
]

WHISPER_SIZES = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v2",
    "large-v3",
    "distil-large-v3",
]

KOKORO_VOICES = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
]

KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

PIPER_VOICES = {
    "en_US-lessac-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    "en_GB-alan-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
    },
    "de_DE-thorsten-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
    },
}


def models_dir() -> Path:
    path = ROOT_DIR / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def whisper_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def whisper_downloaded(size: str) -> bool:
    """Heuristic: faster-whisper stores snapshots under hub with model id in the name."""
    cache = whisper_cache_dir()
    if not cache.exists():
        return False
    needle = size.replace(".", "-").lower()
    aliases = {
        "tiny": "whisper-tiny",
        "tiny.en": "whisper-tiny.en",
        "base": "whisper-base",
        "base.en": "whisper-base.en",
        "small": "whisper-small",
        "small.en": "whisper-small.en",
        "medium": "whisper-medium",
        "medium.en": "whisper-medium.en",
        "large-v2": "whisper-large-v2",
        "large-v3": "whisper-large-v3",
        "distil-large-v3": "distil-whisper-large-v3",
    }
    target = aliases.get(size, f"whisper-{needle}")
    for entry in cache.iterdir():
        name = entry.name.lower()
        if target.lower() in name and entry.is_dir():
            return True
    return False


async def _emit(cb: ProgressCb | None, payload: dict[str, Any]) -> None:
    if cb is None:
        return
    result = cb(payload)
    if asyncio.iscoroutine(result):
        await result


async def list_llm_models(base_url: str) -> list[dict[str, Any]]:
    downloaded: set[str] = set()
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=10.0) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            for model in response.json().get("models", []):
                name = str(model.get("name", "")).strip()
                if name:
                    downloaded.add(name)
                    # also bare tag without :latest
                    if name.endswith(":latest"):
                        downloaded.add(name[: -len(":latest")])
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_llm_models failed: {}", exc)

    names = sorted(set(CURATED_LLM) | downloaded)
    result: list[dict[str, Any]] = []
    for name in names:
        is_down = name in downloaded or f"{name}:latest" in downloaded
        result.append({"name": name, "downloaded": is_down, "curated": name in CURATED_LLM})
    return result


async def pull_llm_model(base_url: str, name: str, on_progress: ProgressCb | None = None) -> None:
    await _emit(on_progress, {"type": "llm_pull", "name": name, "status": "starting", "progress": 0})
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=None) as client:
        async with client.stream("POST", "/api/pull", json={"name": name, "stream": True}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total = data.get("total") or 0
                completed = data.get("completed") or 0
                progress = float(completed) / float(total) if total else None
                await _emit(
                    on_progress,
                    {
                        "type": "llm_pull",
                        "name": name,
                        "status": data.get("status", "pulling"),
                        "progress": progress,
                        "raw": data,
                    },
                )
    await _emit(on_progress, {"type": "llm_pull", "name": name, "status": "done", "progress": 1.0})


def list_whisper_models() -> list[dict[str, Any]]:
    return [{"size": size, "downloaded": whisper_downloaded(size)} for size in WHISPER_SIZES]


async def download_whisper(size: str, on_progress: ProgressCb | None = None) -> None:
    await _emit(on_progress, {"type": "whisper_download", "size": size, "status": "starting", "progress": 0})

    def _load() -> None:
        from faster_whisper import WhisperModel

        WhisperModel(size, device="cpu", compute_type="int8")

    try:
        await asyncio.to_thread(_load)
        await _emit(on_progress, {"type": "whisper_download", "size": size, "status": "done", "progress": 1.0})
    except Exception as exc:  # noqa: BLE001
        logger.error("Whisper download failed: {}", exc)
        await _emit(
            on_progress,
            {"type": "whisper_download", "size": size, "status": f"error: {exc}", "progress": None},
        )
        raise


def list_tts_models() -> dict[str, Any]:
    md = models_dir()
    kokoro_files = []
    for filename in KOKORO_FILES:
        path = md / filename
        kokoro_files.append({"id": filename, "path": str(path), "downloaded": path.exists()})
    piper = []
    for voice, urls in PIPER_VOICES.items():
        onnx = md / f"{voice}.onnx"
        meta = md / f"{voice}.onnx.json"
        piper.append(
            {
                "id": f"piper:{voice}",
                "voice": voice,
                "downloaded": onnx.exists() and meta.exists(),
                "onnx": str(onnx),
                "json": str(meta),
            }
        )
    return {
        "kokoro": {
            "id": "kokoro",
            "downloaded": all(f["downloaded"] for f in kokoro_files),
            "files": kokoro_files,
            "voices": KOKORO_VOICES,
        },
        "piper": piper,
    }


async def _download_file(url: str, dest: Path, on_progress: ProgressCb | None, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            done = 0
            with tmp.open("wb") as fh:
                async for chunk in response.aiter_bytes(1024 * 256):
                    fh.write(chunk)
                    done += len(chunk)
                    progress = float(done) / float(total) if total else None
                    await _emit(
                        on_progress,
                        {
                            "type": "tts_download",
                            "id": label,
                            "status": f"downloading {dest.name}",
                            "progress": progress,
                        },
                    )
    tmp.replace(dest)


async def download_tts(asset_id: str, on_progress: ProgressCb | None = None) -> None:
    md = models_dir()
    await _emit(on_progress, {"type": "tts_download", "id": asset_id, "status": "starting", "progress": 0})

    if asset_id == "kokoro":
        for filename, url in KOKORO_FILES.items():
            await _download_file(url, md / filename, on_progress, asset_id)
        await _emit(on_progress, {"type": "tts_download", "id": asset_id, "status": "done", "progress": 1.0})
        return

    if asset_id.startswith("piper:"):
        voice = asset_id.split(":", 1)[1]
        urls = PIPER_VOICES.get(voice)
        if not urls:
            raise ValueError(f"Unknown piper voice: {voice}")
        await _download_file(urls["onnx"], md / f"{voice}.onnx", on_progress, asset_id)
        await _download_file(urls["json"], md / f"{voice}.onnx.json", on_progress, asset_id)
        await _emit(on_progress, {"type": "tts_download", "id": asset_id, "status": "done", "progress": 1.0})
        return

    raise ValueError(f"Unknown tts asset id: {asset_id}")
