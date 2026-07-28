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
    """Resize so the longest side is at most ``max_side``.

    Dimensions are snapped to multiples of 14, which keeps ViT-based depth
    backbones (patch size 14) from silently resampling the image.  That snap
    means the aspect ratio can shift by up to a percent or so, and a small
    image may be scaled *up* slightly - intrinsics are rescaled per axis to
    match (see `resolve_intrinsics`), so geometry is unaffected.
    """
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        # Already within the cap: leave it exactly alone.  Snapping to a patch
        # multiple here would *upscale* 640x480 to 644x476 - changing the
        # aspect ratio by 1.5%, for no benefit, on input the user never asked
        # to be touched.  Model processors resample internally anyway.
        return rgb
    scale = max_side / float(max(h, w))
    # Round the long side *down* to a patch multiple so the cap is honoured -
    # rounding to nearest could exceed it (1920x1080 at max_side 768 gave 770).
    # The short side rounds to *nearest*, to stay as close to the true aspect
    # ratio as a patch multiple allows - but rounding up can carry it past the
    # cap too.  On an exactly square image the "short" side is also at the cap,
    # so it always did: 900x900 at max_side 512 came out 518x504, breaking both
    # the documented cap and the squareness, for any cap with `max_side % 14 >
    # 7` (768, the default, is one).  Stepping back one patch is exactly the
    # floor the long side already takes, so a square image stays square.
    def _short(value: float) -> int:
        rounded = max(14, int(round(value / 14)) * 14)
        return max(14, rounded - 14) if rounded > max_side else rounded

    if w >= h:
        new_w = max(14, int(w * scale // 14) * 14)
        new_h = _short(h * scale)
    else:
        new_h = max(14, int(h * scale // 14) * 14)
        new_w = _short(w * scale)
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


MAX_TRACKABLE_GAP_SECONDS = 0.5
"""Longest gap between keyframes that frame-to-frame tracking can bridge.

Odometry matches ORB features between *consecutive keyframes*, so what matters
is not how many frames were sampled but how much the view changed between the
ones that were.  Half a second of ordinary handheld panning still leaves a
large shared field of view; several seconds of walking does not, and the
tracker then has nothing to match.  This is a guide rather than a limit -
a slow orbit survives a longer gap than a walk through a doorway - so it warns
rather than refusing.
"""


def _warn_if_keyframes_are_too_far_apart(
    stride: int, fps: float, total: int, cfg: PipelineConfig
) -> None:
    """Say so when the sampling makes the camera untrackable.

    ``max_frames`` spreads its budget over the whole clip, so the longer the
    video the further apart the keyframes: a 60-second walkthrough at the
    default 24 frames is one keyframe every 2.5 seconds.  Consecutive
    keyframes then share almost no features, every pose link fails or solves
    badly, and the failure surfaces far downstream as a scene where nothing
    merges across frames - hundreds of duplicated objects and no floor - with
    nothing pointing back at the sampling.
    """
    if total <= 0 or fps <= 0 or stride <= 1:
        return
    gap = stride / fps
    if gap <= MAX_TRACKABLE_GAP_SECONDS:
        return
    suggested = max(cfg.max_frames, int(np.ceil(total / (MAX_TRACKABLE_GAP_SECONDS * fps))))
    log.warning(
        "Sampling one keyframe every %.1f s (%.0f s of video into %d frames). "
        "Camera tracking matches features between consecutive keyframes, and "
        "much beyond %.1f s they no longer overlap - poses drift or fail, and "
        "the same object reconstructs several times over. Pass --max-frames %d "
        "for this clip, or film a shorter, slower pass.",
        gap, total / fps, cfg.max_frames, MAX_TRACKABLE_GAP_SECONDS, suggested,
    )


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
    _warn_if_keyframes_are_too_far_apart(stride, fps, total, cfg)
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
