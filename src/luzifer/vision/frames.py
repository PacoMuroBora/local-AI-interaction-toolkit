"""Frame helpers: JPEG encode and change detection."""

from __future__ import annotations

import io

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment, misc]


def downscale_gray(rgb: np.ndarray, edge: int = 64) -> np.ndarray:
    """64px grayscale for MAE change gating. Expects HxWx3 uint8."""
    if Image is None:
        # Crude nearest-neighbour fallback without Pillow
        h, w = rgb.shape[:2]
        scale = edge / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        ys = (np.linspace(0, h - 1, nh)).astype(np.int32)
        xs = (np.linspace(0, w - 1, nw)).astype(np.int32)
        sampled = rgb[ys][:, xs]
        return (
            0.299 * sampled[..., 0] + 0.587 * sampled[..., 1] + 0.114 * sampled[..., 2]
        ).astype(np.float32)

    img = Image.fromarray(rgb, mode="RGB")
    img.thumbnail((edge, edge), Image.Resampling.BILINEAR)
    return np.asarray(img.convert("L"), dtype=np.float32)


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 999.0
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def frame_changed(
    rgb: np.ndarray,
    prev_small: np.ndarray | None,
    threshold: float,
) -> tuple[bool, np.ndarray]:
    """Return (changed, new_small). threshold<=0 always reports changed."""
    small = downscale_gray(rgb, 64)
    if threshold <= 0 or prev_small is None:
        return True, small
    return mean_abs_diff(small, prev_small) >= threshold, small


def jpeg_bytes(rgb: np.ndarray, max_edge: int = 512, quality: int = 70) -> bytes:
    """Encode RGB uint8 to JPEG, long edge capped at max_edge."""
    if Image is None:
        raise RuntimeError("Pillow is required for vision JPEG encoding (uv sync --extra vision)")
    img = Image.fromarray(rgb, mode="RGB")
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
