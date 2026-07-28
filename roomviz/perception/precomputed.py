"""Backends that read perception results from disk or memory.

Two uses:

1. **Real captures that already carry depth.**  An iPhone/iPad LiDAR scan, a
   RealSense recording or a rendered dataset gives you far better depth than
   any monocular network.  ``--depth-backend file --depth-dir <dir>`` feeds
   that straight into the reconstruction.
2. **Testing.**  The in-memory ``memory`` backends let the geometry, fusion
   and export stages be exercised against ground truth without downloading
   any weights.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ..config import PipelineConfig
from ..types import DepthMap, Frame, Segment, Segmentation
from .base import register_depth, register_segmentation
from .labels import classify

log = logging.getLogger(__name__)

_DEPTH_SUFFIXES = (".npy", ".exr", ".tiff", ".tif", ".png")
_SEG_SUFFIXES = (".npy", ".png")


def _find_for_frame(directory: Path, index: int, suffixes: tuple[str, ...]) -> Path:
    """Locate the sidecar for an *original media* frame index.

    Sidecars are numbered against the source video or image folder, not
    against the sampled keyframes, so callers must pass ``frame.source_index``.
    """
    for suffix in suffixes:
        candidate = directory / f"{index:06d}{suffix}"
        if candidate.exists():
            return candidate
        candidate = directory / f"{index}{suffix}"
        if candidate.exists():
            return candidate

    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in suffixes)
    if index < len(files):
        return files[index]
    raise FileNotFoundError(f"no sidecar for frame {index} in {directory}")


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
    arr = cv2.imread(str(path), flags)
    if arr is None:
        raise ValueError(f"could not read {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


class FileDepthBackend:
    """Depth loaded from ``.npy`` / 16-bit PNG / TIFF sidecar files."""

    name = "file"

    def __init__(self, cfg: PipelineConfig):
        if not cfg.depth_dir:
            raise ValueError("--depth-dir is required for the 'file' depth backend")
        self.cfg = cfg
        self.dir = Path(cfg.depth_dir)
        if not self.dir.is_dir():
            raise NotADirectoryError(self.dir)

    def predict(self, frame: Frame) -> DepthMap:
        path = _find_for_frame(self.dir, frame.source_index, _DEPTH_SUFFIXES)
        raw = _load_array(path)
        depth = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.integer):
            # Integer depth is conventionally millimetres; depth_scale converts
            # stored units to metres.
            depth *= float(self.cfg.depth_scale)
        if depth.shape != (frame.height, frame.width):
            depth = cv2.resize(
                depth, (frame.width, frame.height), interpolation=cv2.INTER_NEAREST
            )
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth < self.cfg.depth_min) | (depth > self.cfg.depth_trunc)] = 0.0
        return DepthMap(depth=depth, metric=True)


class FileSegmentationBackend:
    """Segment id maps from ``.npy``/PNG plus a ``labels.json`` sidecar.

    ``labels.json`` maps segment id -> label name, e.g.::

        {"1": "wall", "2": "floor", "3": "chair"}
    """

    name = "file"

    def __init__(self, cfg: PipelineConfig):
        if not cfg.seg_dir:
            raise ValueError("--seg-dir is required for the 'file' segmentation backend")
        self.cfg = cfg
        self.dir = Path(cfg.seg_dir)
        if not self.dir.is_dir():
            raise NotADirectoryError(self.dir)
        labels_path = self.dir / "labels.json"
        self.labels: dict[int, str] = {}
        if labels_path.exists():
            self.labels = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}
        else:
            log.warning("%s has no labels.json; segments will be unlabelled", self.dir)

    def predict(self, frame: Frame) -> Segmentation:
        path = _find_for_frame(self.dir, frame.source_index, _SEG_SUFFIXES)
        ids = _load_array(path).astype(np.int32)
        if ids.shape != (frame.height, frame.width):
            ids = cv2.resize(
                ids, (frame.width, frame.height), interpolation=cv2.INTER_NEAREST
            )
        return _segmentation_from_ids(ids, self.labels)


class MemorySegmentationBackend:
    """Segmentation supplied programmatically via ``cfg.extra['segmentations']``."""

    name = "memory"

    def __init__(self, cfg: PipelineConfig):
        self.frames: list[Segmentation] = cfg.extra["segmentations"]

    def predict(self, frame: Frame) -> Segmentation:
        return self.frames[frame.index]


class MemoryDepthBackend:
    """Depth supplied programmatically via ``cfg.extra['depths']``."""

    name = "memory"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.frames: list[np.ndarray] = cfg.extra["depths"]

    def predict(self, frame: Frame) -> DepthMap:
        depth = np.asarray(self.frames[frame.index], dtype=np.float32).copy()
        depth[(depth < self.cfg.depth_min) | (depth > self.cfg.depth_trunc)] = 0.0
        return DepthMap(depth=depth, metric=True)


def _segmentation_from_ids(ids: np.ndarray, labels: dict[int, str]) -> Segmentation:
    segments: list[Segment] = []
    for sid in np.unique(ids):
        sid = int(sid)
        if sid < 0:
            continue
        label = labels.get(sid, f"segment_{sid}")
        role, kind = classify(label)
        segments.append(
            Segment(segment_id=sid, label=label, role=role, structure_kind=kind)
        )
    return Segmentation(ids=ids.astype(np.int32), segments=segments)


@register_depth("file")
def _build_file_depth(cfg: PipelineConfig) -> FileDepthBackend:
    return FileDepthBackend(cfg)


@register_segmentation("file")
def _build_file_seg(cfg: PipelineConfig) -> FileSegmentationBackend:
    return FileSegmentationBackend(cfg)


@register_depth("memory")
def _build_memory_depth(cfg: PipelineConfig) -> MemoryDepthBackend:
    return MemoryDepthBackend(cfg)


@register_segmentation("memory")
def _build_memory_seg(cfg: PipelineConfig) -> MemorySegmentationBackend:
    return MemorySegmentationBackend(cfg)
