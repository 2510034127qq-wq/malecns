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
