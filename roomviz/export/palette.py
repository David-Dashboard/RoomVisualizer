"""Stable, readable colours for object labels and structural surfaces."""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np

SURFACE_COLORS: dict[str, tuple[int, int, int]] = {
    "wall": (196, 202, 214),
    "floor": (168, 150, 128),
    "ceiling": (222, 226, 234),
}


def label_color(label: str) -> tuple[int, int, int]:
    """Deterministic, well-separated colour for a class name.

    Hashing the label (rather than counting instances) keeps colours stable
    across runs and across scenes, so "chair" is always the same hue.  Hue
    comes from the hash while saturation and lightness stay in a narrow band,
    which keeps every colour legible against both light and dark backgrounds.
    """
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    saturation = 0.45 + (digest[1] / 255.0) * 0.30
    lightness = 0.48 + (digest[2] / 255.0) * 0.16
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return int(r * 255), int(g * 255), int(b * 255)


def label_color_array(label: str) -> np.ndarray:
    return np.array(label_color(label), dtype=np.uint8)


def surface_color(kind: str) -> tuple[int, int, int]:
    return SURFACE_COLORS.get(kind, (180, 180, 180))
