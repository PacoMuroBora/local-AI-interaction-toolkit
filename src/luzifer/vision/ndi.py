"""NDI still-frame grabber (optional cyndilib + NDI runtime)."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from loguru import logger

_CYNDILIB_WARNED = False
# NDI find/recv is not safe across concurrent Finder instances on Windows.
_NDI_LOCK = threading.RLock()


def _import_cyndilib() -> tuple[Any, Any, Any, Any, Any] | None:
    global _CYNDILIB_WARNED
    try:
        from cyndilib.finder import Finder
        from cyndilib.receiver import Receiver
        from cyndilib.video_frame import VideoFrameSync
        from cyndilib.wrapper.ndi_recv import RecvBandwidth, RecvColorFormat
    except Exception as exc:  # noqa: BLE001
        if not _CYNDILIB_WARNED:
            logger.warning(
                "Vision NDI unavailable (install NDI runtime + `uv sync --extra vision`): {}",
                exc,
            )
            _CYNDILIB_WARNED = True
        return None
    return Finder, Receiver, VideoFrameSync, RecvBandwidth, RecvColorFormat


def list_ndi_sources(timeout_s: float = 5.0) -> list[str]:
    """Discover NDI source names. Empty list if cyndilib/runtime missing."""
    mods = _import_cyndilib()
    if mods is None:
        return []
    Finder, *_rest = mods
    seen: dict[str, None] = {}
    with _NDI_LOCK:
        try:
            with Finder() as finder:
                deadline = time.time() + max(0.5, timeout_s)
                while time.time() < deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    finder.wait_for_sources(min(1.0, remaining))
                    for full in finder.get_source_names() or []:
                        if full:
                            seen.setdefault(full, None)
                    for source in finder:
                        full = getattr(source, "name", "") or ""
                        stream = getattr(source, "stream_name", "") or ""
                        if full:
                            seen.setdefault(full, None)
                        if stream:
                            seen.setdefault(stream, None)
                    time.sleep(0.15)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NDI source list failed: {}", exc)
    return list(seen.keys())


class NdiGrabber:
    """Long-lived NDI receiver. All public methods take the process-wide NDI lock."""

    def __init__(self) -> None:
        self._finder: Any | None = None
        self._receiver: Any | None = None
        self._video_frame: Any | None = None
        self._source_name: str = ""
        self._available = _import_cyndilib() is not None

    @property
    def available(self) -> bool:
        return self._available

    def close(self) -> None:
        with _NDI_LOCK:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        receiver = self._receiver
        self._receiver = None
        self._video_frame = None
        if receiver is not None:
            try:
                if hasattr(receiver, "disconnect"):
                    receiver.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.debug("NDI receiver disconnect: {}", exc)
        if self._finder is not None:
            try:
                self._finder.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.debug("NDI finder close: {}", exc)
        self._finder = None
        self._source_name = ""

    def _ensure_connected(self, source_name: str) -> bool:
        mods = _import_cyndilib()
        if mods is None:
            return False
        Finder, Receiver, VideoFrameSync, RecvBandwidth, RecvColorFormat = mods

        if (
            self._receiver is not None
            and self._source_name == source_name
            and self._receiver.is_connected()
        ):
            return True

        self._close_unlocked()
        try:
            finder = Finder()
            finder.__enter__()
            want = (source_name or "").strip()
            source = self._find_source(finder, want, timeout_s=6.0)
            if source is None and not want:
                names = finder.get_source_names() or []
                for n in names:
                    if "Remote Connection" in n:
                        continue
                    source = finder.get_source(n)
                    break
                if source is None and names:
                    source = finder.get_source(names[0])
            if source is None:
                logger.warning(
                    "NDI source not found: {!r} (seen: {})",
                    want,
                    finder.get_source_names(),
                )
                finder.__exit__(None, None, None)
                return False

            receiver = Receiver(
                color_format=RecvColorFormat.RGBX_RGBA,
                bandwidth=RecvBandwidth.lowest,
                recv_name="luzifer-vision",
            )
            vf = VideoFrameSync()
            receiver.frame_sync.set_video_frame(vf)
            receiver.set_source(source)
            for _ in range(40):
                if receiver.is_connected():
                    break
                time.sleep(0.25)
            if not receiver.is_connected():
                logger.warning("NDI connect timeout for {}", getattr(source, "name", want))
                finder.__exit__(None, None, None)
                return False

            self._finder = finder
            self._receiver = receiver
            self._video_frame = vf
            self._source_name = want or (getattr(source, "stream_name", None) or source.name)
            logger.info("NDI connected to {}", getattr(source, "name", self._source_name))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("NDI connect failed: {}", exc)
            self._close_unlocked()
            return False

    @staticmethod
    def _find_source(finder: Any, want: str, *, timeout_s: float) -> Any | None:
        """Poll until ``want`` matches full name or stream_name (substring OK)."""
        deadline = time.time() + max(0.5, timeout_s)
        best = None
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            finder.wait_for_sources(min(1.0, remaining))
            for candidate in finder:
                full = getattr(candidate, "name", "") or ""
                stream = getattr(candidate, "stream_name", "") or ""
                if not want:
                    continue
                if full == want or stream == want:
                    return candidate
                if want in full or want in stream:
                    if stream == want or full.endswith(f"({want})"):
                        return candidate
                    best = best or candidate
            if best is not None:
                return best
            time.sleep(0.15)
        return None

    def latest_rgb(self, source_name: str) -> np.ndarray | None:
        """Return HxWx3 uint8 RGB frame, or None. Copies pixels off the NDI buffer."""
        with _NDI_LOCK:
            if not self._ensure_connected(source_name):
                return None
            assert self._receiver is not None and self._video_frame is not None
            try:
                vf = self._video_frame
                # First captures after connect are often empty — retry briefly.
                resolution = (0, 0)
                for _ in range(30):
                    self._receiver.frame_sync.capture_video()
                    resolution = vf.get_resolution()
                    size = vf.get_data_size()
                    if min(resolution) > 0 and size > 0:
                        break
                    time.sleep(0.05)
                else:
                    return None
                # NDI resolution is (xres, yres)
                xres, yres = int(resolution[0]), int(resolution[1])
                # Copy immediately — FrameSync buffers are transient.
                arr = np.array(vf.get_array(), dtype=np.uint8, copy=True)
                if arr.ndim == 1:
                    if arr.size >= yres * xres * 4 and arr.size % (yres * xres) == 0:
                        channels = arr.size // (yres * xres)
                        arr = arr[: yres * xres * channels].reshape((yres, xres, channels))
                    else:
                        return None
                elif arr.ndim == 2:
                    arr = arr.reshape((yres, xres, -1))
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    return np.ascontiguousarray(arr[..., :3])
                return None
            except Exception as exc:  # noqa: BLE001
                logger.warning("NDI frame grab failed: {}", exc)
                return None
