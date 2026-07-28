"""Backend protocols and registry for the perception stage.

Depth and segmentation are pluggable so that the geometry half of the pipeline
can be exercised with ground-truth inputs (see :mod:`roomviz.perception.precomputed`)
and so that swapping in a different checkpoint is a one-line change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..config import PipelineConfig
from ..types import DepthMap, Frame, Segmentation


class DepthBackend(Protocol):
    name: str

    def predict(self, frame: Frame) -> DepthMap: ...


class SegmentationBackend(Protocol):
    name: str

    def predict(self, frame: Frame) -> Segmentation: ...


_DEPTH_BACKENDS: dict[str, Callable[[PipelineConfig], DepthBackend]] = {}
_SEG_BACKENDS: dict[str, Callable[[PipelineConfig], SegmentationBackend]] = {}


def register_depth(name: str) -> Callable[[Callable], Callable]:
    def deco(factory: Callable[[PipelineConfig], DepthBackend]):
        _DEPTH_BACKENDS[name] = factory
        return factory

    return deco


def register_segmentation(name: str) -> Callable[[Callable], Callable]:
    def deco(factory: Callable[[PipelineConfig], SegmentationBackend]):
        _SEG_BACKENDS[name] = factory
        return factory

    return deco


def build_depth_backend(cfg: PipelineConfig) -> DepthBackend:
    try:
        factory = _DEPTH_BACKENDS[cfg.depth_backend]
    except KeyError:
        raise ValueError(
            f"unknown depth backend {cfg.depth_backend!r}; "
            f"available: {sorted(_DEPTH_BACKENDS)}"
        ) from None
    return factory(cfg)


def build_segmentation_backend(cfg: PipelineConfig) -> SegmentationBackend:
    try:
        factory = _SEG_BACKENDS[cfg.seg_backend]
    except KeyError:
        raise ValueError(
            f"unknown segmentation backend {cfg.seg_backend!r}; "
            f"available: {sorted(_SEG_BACKENDS)}"
        ) from None
    return factory(cfg)


def available_backends() -> dict[str, list[str]]:
    return {"depth": sorted(_DEPTH_BACKENDS), "segmentation": sorted(_SEG_BACKENDS)}


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string, falling back gracefully."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
