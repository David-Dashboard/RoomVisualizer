"""Turn images, videos or directories of images into a list of keyframes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import PipelineConfig
from ..types import Frame

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}


@dataclass
class MediaSource:
    path: Path
    kind: str  # "image" | "video" | "image_dir"
    frame_count: int
    fps: float = 0.0


def classify(path: Path) -> MediaSource:
    """Work out what kind of input we were handed."""
    if path.is_dir():
        images = sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"no images found in directory: {path}")
        return MediaSource(path=path, kind="image_dir", frame_count=len(images))

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return MediaSource(path=path, kind="image", frame_count=1)
    if suffix in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"could not open video: {path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        cap.release()
        return MediaSource(path=path, kind="video", frame_count=count, fps=fps)

    # Unknown extension: let OpenCV have a go at it as a video, then as an image.
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        cap.release()
        if count > 1:
            return MediaSource(path=path, kind="video", frame_count=count, fps=fps)
    if cv2.imread(str(path)) is not None:
        return MediaSource(path=path, kind="image", frame_count=1)
    raise ValueError(f"unsupported media file: {path}")


def resize_frame(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is at most ``max_side``.

    Dimensions are snapped to multiples of 14, which keeps ViT-based depth
    backbones (patch size 14) from silently resampling the image.
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    new_w = max(14, int(round(w * scale / 14)) * 14)
    new_h = max(14, int(round(h * scale / 14)) * 14)
    if (new_w, new_h) == (w, h):
        return rgb
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(rgb, (new_w, new_h), interpolation=interp)


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian - a cheap, standard blur score."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _read_image(path: Path) -> np.ndarray:
    # imdecode handles non-ASCII paths that imread trips over on some builds.
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_video_frames(source: MediaSource, cfg: PipelineConfig) -> list[Frame]:
    cap = cv2.VideoCapture(str(source.path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {source.path}")

    total = source.frame_count
    stride = cfg.frame_stride
    if stride <= 0:
        # Spread the sampled keyframes evenly over the whole clip.
        stride = max(1, int(np.ceil(total / max(1, cfg.max_frames)))) if total > 0 else 1

    fps = source.fps or 30.0
    frames: list[Frame] = []
    raw_index = 0
    kept = 0
    while len(frames) < cfg.max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if raw_index % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            original_size = (rgb.shape[1], rgb.shape[0])
            rgb = resize_frame(rgb, cfg.max_side)
            if cfg.min_sharpness > 0 and sharpness(rgb) < cfg.min_sharpness:
                log.debug("dropping blurry frame %d", raw_index)
            else:
                frames.append(
                    Frame(
                        index=kept,
                        rgb=rgb,
                        timestamp=raw_index / fps,
                        source=str(source.path),
                        source_index=raw_index,
                        original_size=original_size,
                    )
                )
                kept += 1
        raw_index += 1
    cap.release()

    if not frames:
        raise ValueError(f"no usable frames decoded from {source.path}")
    return frames


def load_frames(path: str | os.PathLike[str], cfg: PipelineConfig) -> list[Frame]:
    """Load and subsample keyframes from an image, video or image directory."""
    source = classify(Path(path))
    log.info("input: %s (%s, %d frames)", source.path, source.kind, source.frame_count)

    if source.kind == "video":
        frames = _load_video_frames(source, cfg)
    elif source.kind == "image":
        full = _read_image(source.path)
        frames = [
            Frame(
                index=0,
                rgb=resize_frame(full, cfg.max_side),
                source=str(source.path),
                original_size=(full.shape[1], full.shape[0]),
            )
        ]
    else:  # image_dir
        paths = sorted(
            p for p in source.path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        step = max(1, int(np.ceil(len(paths) / max(1, cfg.max_frames))))
        frames = []
        for i, p in enumerate(paths[::step][: cfg.max_frames]):
            full = _read_image(p)
            frames.append(
                Frame(
                    index=i,
                    rgb=resize_frame(full, cfg.max_side),
                    timestamp=float(i),
                    source=str(p),
                    source_index=i * step,
                    original_size=(full.shape[1], full.shape[0]),
                )
            )

    log.info("using %d keyframe(s) at %dx%d", len(frames), frames[0].width, frames[0].height)
    return frames
