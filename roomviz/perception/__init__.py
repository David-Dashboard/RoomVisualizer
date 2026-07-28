"""Perception backends.

Importing this package registers every built-in backend.  The neural backends
import torch lazily inside their constructors, so this stays cheap and works
without the model extras installed.
"""

from . import depth, precomputed, segment  # noqa: F401  (registration side effect)
from .base import (
    available_backends,
    build_depth_backend,
    build_segmentation_backend,
    resolve_device,
)
from .labels import classify, structure_kind

__all__ = [
    "available_backends",
    "build_depth_backend",
    "build_segmentation_backend",
    "classify",
    "resolve_device",
    "structure_kind",
]
