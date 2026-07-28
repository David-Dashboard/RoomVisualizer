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
from .labels import classify, is_thing

log = logging.getLogger(__name__)

_DEPTH_SUFFIXES = (".npy", ".exr", ".tiff", ".tif", ".png")
_SEG_SUFFIXES = (".npy", ".png")


def check_sidecar_numbering(directory: Path, suffixes: tuple[str, ...]) -> None:
    """Refuse a sidecar set that is numbered from 1 instead of 0.

    A 1-indexed export is the single most damaging mistake here and the least
    visible: every frame after the first finds a file with the *right name*
    holding the *previous* frame's data, so nothing warns and the geometry is
    quietly wrong.  It is also trivially detectable - index 0 missing while
    index 1 and the following run contiguously.
    """
    numbered = {}
    for path in directory.iterdir():
        if path.suffix.lower() not in suffixes:
            continue
        try:
            numbered[int(path.stem)] = path
        except ValueError:
            continue
    if not numbered or 0 in numbered:
        return
    if 1 in numbered and len(numbered) >= 2 and max(numbered) == len(numbered):
        raise ValueError(
            f"{directory} appears to be numbered from 1 ({min(numbered)}..."
            f"{max(numbered)}) but sidecars are matched to source frame indices, "
            f"which start at 0. Every frame would be paired with the previous "
            f"frame's data. Renumber from 000000, or pass --frame-stride to "
            f"match your export."
        )


def _find_for_frame(
    directory: Path,
    index: int,
    suffixes: tuple[str, ...],
    expected_count: int | None = None,
) -> Path:
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
        # Nothing is numbered for this frame, so fall back to sorted position.
        # That is a guess: a stray file shifts every pairing by one and fuses
        # each frame's colour with another's depth, which looks entirely
        # plausible in the output.  Only say so when the guess is actually in
        # play -- warning on a directory that is simply named differently but
        # ordered correctly is noise, and noise is what gets warnings ignored.
        risky = expected_count is not None and len(files) != expected_count
        if risky:
            log.warning(
                "%s has no sidecar named %06d and holds %d files for %d frames; "
                "falling back to sorted position -> %s. Number the sidecars by "
                "source frame index to make this exact.",
                directory, index, len(files), expected_count, files[index].name,
            )
        else:
            log.debug(
                "%s: no file named %06d; using sorted position -> %s",
                directory, index, files[index].name,
            )
        return files[index]
    raise FileNotFoundError(f"no sidecar for frame {index} in {directory}")


def _warn_on_aspect_change(kind: str, got: tuple[int, int], frame: Frame) -> None:
    """Warn when a sidecar is being stretched, not merely rescaled.

    The comparison is against the *original capture* size, not the working
    frame size.  The loader snaps the working size to a multiple of the model
    patch size, so a 640x480 sidecar against a 644x476 frame differs in aspect
    by 1.5% through no fault of the user - and warning about it fires on the
    documented quickstart, twice per frame, which is how people learn to
    ignore the identical warning when it correctly catches transposed depth.
    """
    reference = frame.original_size or (frame.width, frame.height)
    got_ratio = got[1] / max(got[0], 1)
    want_ratio = reference[0] / max(reference[1], 1)
    if abs(got_ratio - want_ratio) > 0.02 * want_ratio:
        log.warning(
            "%s sidecar is %dx%d but the capture is %dx%d - a different aspect "
            "ratio, so it is being stretched; the reconstruction will be wrong",
            kind, got[1], got[0], reference[0], reference[1],
        )


def _parse_labels(path: Path) -> dict[int, object]:
    """Read and validate a ``labels.json`` sidecar.

    Everything here is user-authored, usually by an export script, so each
    failure mode gets a message naming the file and the offending key rather
    than a traceback from deep inside the loader.
    """
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must be an object mapping segment id -> label, "
            f"got {type(raw).__name__}"
        )

    labels: dict[int, object] = {}
    for key, value in raw.items():
        try:
            segment_id = int(str(key).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{path}: segment id {key!r} is not an integer") from None
        if segment_id in labels:
            # "12", " 12" and "012" all parse to the same id; silently keeping
            # the last would relabel a segment for no visible reason.
            raise ValueError(f"{path}: segment id {segment_id} appears more than once")

        if isinstance(value, str):
            labels[segment_id] = value
            continue
        if isinstance(value, dict):
            label = value.get("label")
            if not isinstance(label, str):
                raise ValueError(
                    f"{path}: segment {segment_id} has a non-string 'label' "
                    f"({label!r})"
                )
            thing = value.get("thing")
            if thing is not None and not isinstance(thing, bool):
                # A JSON string "false" is truthy in Python, so accepting one
                # would mean the exact opposite of what the file says.
                raise ValueError(
                    f"{path}: segment {segment_id} has a non-boolean 'thing' "
                    f"({thing!r}); use true or false"
                )
            labels[segment_id] = {"label": label, "thing": thing}
            continue
        raise ValueError(
            f"{path}: segment {segment_id} must map to a string or an object, "
            f"got {type(value).__name__}"
        )
    return labels


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
        check_sidecar_numbering(self.dir, _DEPTH_SUFFIXES)
        self._expected_count: int | None = None
        self._warned_float_scale = False

    def predict(self, frame: Frame) -> DepthMap:
        path = _find_for_frame(
            self.dir, frame.source_index, _DEPTH_SUFFIXES, self._expected_count
        )
        raw = _load_array(path)
        depth = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.integer):
            # Integer depth is conventionally millimetres; depth_scale converts
            # stored units to metres.
            depth *= float(self.cfg.depth_scale)
        elif self.cfg.depth_scale != 0.001 and not self._warned_float_scale:
            # Float sidecars are assumed to be metres already.  Silently
            # ignoring an explicitly-passed scale is how a millimetre float
            # export becomes a five-kilometre room with no error.
            log.warning(
                "--depth-scale %g is ignored for floating-point sidecars (%s is "
                "%s); float depth is read as metres. Convert to metres, or "
                "store integer units.",
                self.cfg.depth_scale, path.name, raw.dtype,
            )
            self._warned_float_scale = True
        if depth.shape != (frame.height, frame.width):
            _warn_on_aspect_change("depth", depth.shape, frame)
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
        check_sidecar_numbering(self.dir, _SEG_SUFFIXES)
        self._expected_count: int | None = None
        labels_path = self.dir / "labels.json"
        self.labels: dict[int, object] = {}
        if labels_path.exists():
            self.labels = _parse_labels(labels_path)
        else:
            log.warning("%s has no labels.json; segments will be unlabelled", self.dir)

    def predict(self, frame: Frame) -> Segmentation:
        path = _find_for_frame(
            self.dir, frame.source_index, _SEG_SUFFIXES, self._expected_count
        )
        ids = _load_array(path).astype(np.int32)
        if ids.shape != (frame.height, frame.width):
            _warn_on_aspect_change("segmentation", ids.shape, frame)
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


def _segmentation_from_ids(ids: np.ndarray, labels: dict[int, object]) -> Segmentation:
    segments: list[Segment] = []
    for sid in np.unique(ids):
        sid = int(sid)
        if sid < 0:
            continue
        entry = labels.get(sid, f"segment_{sid}")
        # A labels.json value may be a plain name, or an object carrying the
        # thing/stuff flag: {"3": {"label": "sofa", "thing": true}}.
        if isinstance(entry, dict):
            label = str(entry.get("label", f"segment_{sid}"))
            thing = entry.get("thing")
        else:
            label, thing = str(entry), None
        role, kind = classify(label)
        segments.append(
            Segment(
                segment_id=sid,
                label=label,
                role=role,
                structure_kind=kind,
                is_thing=thing if thing is not None else is_thing(label),
            )
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
