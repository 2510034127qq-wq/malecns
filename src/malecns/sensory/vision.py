"""Compound-eye sampling of MuJoCo cameras attached to the fly head."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_WIDTH = 32
_HEIGHT = 32
_RENDERERS: dict[int, tuple[object, object]] = {}


def _renderers(model) -> tuple[object, object]:
    import mujoco

    key = id(model)
    cached = _RENDERERS.get(key)
    if cached is not None:
        return cached
    pair = (
        mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH),
        mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH),
    )
    _RENDERERS[key] = pair
    return pair


def render_eyes(model, data) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    left_r, right_r = _renderers(model)
    left_r.update_scene(data, camera="eye_L")
    right_r.update_scene(data, camera="eye_R")
    left = np.asarray(left_r.render(), dtype=np.uint8)
    right = np.asarray(right_r.render(), dtype=np.uint8)
    return left, right


def sample_image_uv(
    image: NDArray[np.uint8],
    u: float,
    v: float,
    *,
    channel: int | None = None,
) -> float:
    """Bilinear sample of a rendered eye image. u,v in [0, 1]. channel None = luminance."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        if channel is None:
            plane = img.mean(axis=2)
        else:
            plane = img[:, :, int(np.clip(channel, 0, img.shape[2] - 1))]
    else:
        plane = img
    plane = plane / 255.0
    h, w = plane.shape
    x = float(np.clip(u, 0.0, 1.0)) * (w - 1)
    y = float(np.clip(v, 0.0, 1.0)) * (h - 1)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    return float(
        plane[y0, x0] * (1 - dx) * (1 - dy)
        + plane[y0, x1] * dx * (1 - dy)
        + plane[y1, x0] * (1 - dx) * dy
        + plane[y1, x1] * dx * dy
    )


def compound_eye_image(samples: NDArray[np.float64], n_side: int | None = None) -> NDArray[np.uint8]:
    """Turn 1-D ommatidium samples into a small 2-D image for the Rerun workbench."""
    vec = np.asarray(samples, dtype=np.float64).reshape(-1)
    if n_side is None:
        n_side = int(np.sqrt(max(vec.size, 1)))
    n_side = max(n_side, 1)
    need = n_side * n_side
    if vec.size < need:
        padded = np.zeros(need, dtype=np.float64)
        padded[: vec.size] = vec
        vec = padded
    grid = vec[:need].reshape(n_side, n_side)
    pix = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(pix[:, :, None], 3, axis=2)


def compound_eye_sample(image: NDArray[np.uint8], n_side: int = 7) -> NDArray[np.float64]:
    """Average luminance on an n_side x n_side grid (ommatidium-like samples)."""
    img = np.asarray(image, dtype=np.float64) / 255.0
    lum = img.mean(axis=2) if img.ndim == 3 else img
    h, w = lum.shape
    ys = np.linspace(0, h, n_side + 1, dtype=int)
    xs = np.linspace(0, w, n_side + 1, dtype=int)
    samples = []
    for i in range(n_side):
        for j in range(n_side):
            patch = lum[ys[i] : max(ys[i + 1], ys[i] + 1), xs[j] : max(xs[j + 1], xs[j] + 1)]
            samples.append(float(patch.mean()) if patch.size else 0.0)
    return np.asarray(samples, dtype=np.float64)
