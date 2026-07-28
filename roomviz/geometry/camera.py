"""Camera intrinsics and depth back-projection."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..types import CameraIntrinsics, DepthMap

log = logging.getLogger(__name__)


def intrinsics_from_exif(path: str | Path, width: int, height: int) -> CameraIntrinsics | None:
    """Recover intrinsics from EXIF focal-length metadata, if present.

    Uses ``FocalLengthIn35mmFilm``, which already accounts for sensor crop.
    A 35mm frame is 36 x 24 mm, and the 36 mm side maps to the image's **long**
    side regardless of how the camera was held - so the focal length in pixels
    is ``max(width, height) * f35 / 36``.

    Dividing the width by 36 unconditionally would make the same camera and
    lens report different focal lengths in landscape and portrait, which is
    physically impossible for square pixels: an upright phone photo would come
    out roughly a third too wide in every reconstructed dimension.
    """
    try:
        from PIL import ExifTags, Image
    except ImportError:  # pragma: no cover
        return None

    # HEIC carries EXIF like any other format, but Pillow cannot open the file
    # at all without the plugin - and a silent `return None` here costs the
    # field of view, not just the format.
    from .._heif import enable_heif

    enable_heif()

    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None

    tag_by_name = {v: k for k, v in ExifTags.TAGS.items()}
    f35 = exif.get(tag_by_name.get("FocalLengthIn35mmFilm", -1))
    if not f35:
        return None

    # The 36 mm reference side is the long side of the frame, whichever way up
    # the camera was held.
    f_px = max(width, height) * float(f35) / 36.0
    log.info(
        "intrinsics from EXIF: 35mm-equivalent f=%smm at %dx%d -> f=%.1fpx",
        f35, width, height, f_px,
    )
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=f_px,
        fy=f_px,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
    )


def resolve_intrinsics(
    cfg: PipelineConfig,
    width: int,
    height: int,
    source_path: str | None = None,
    original_size: tuple[int, int] | None = None,
) -> CameraIntrinsics:
    """Pick intrinsics: explicit config > EXIF > assumed field of view.

    Intrinsics are always derived at the *original* capture resolution and then
    rescaled to the working resolution.  This matters because the working
    resolution is snapped to a multiple of the model's patch size, which can
    shift the aspect ratio by a percent or two - enough that assuming square
    pixels at the working resolution would skew every back-projected point.
    """
    source_width, source_height = original_size or (width, height)
    provenance = "assumed_default"

    if cfg.intrinsics is not None:
        provenance = "explicit_intrinsics"
        fx, fy, cx, cy = cfg.intrinsics
        base = CameraIntrinsics(
            width=source_width, height=source_height, fx=fx, fy=fy, cx=cx, cy=cy
        )
    else:
        base = None
        if cfg.hfov_explicit:
            # An explicitly supplied --hfov outranks EXIF: the user is telling
            # us something about the capture that the file metadata may have
            # lost (a crop, a digital zoom, a re-encode).
            provenance = "hfov_flag"
            base = CameraIntrinsics.from_hfov(source_width, source_height, cfg.hfov_deg)
        else:
            if source_path:
                base = intrinsics_from_exif(source_path, source_width, source_height)
                if base is not None:
                    provenance = "exif"
            if base is None:
                log.warning(
                    "No camera information: assuming a %.0f degree horizontal "
                    "field of view. Every reported dimension scales with this "
                    "guess -- a wrong FOV silently shrinks or inflates the whole "
                    "room. Pass --hfov or --intrinsics for measurable output.",
                    cfg.hfov_deg,
                )
                base = CameraIntrinsics.from_hfov(
                    source_width, source_height, cfg.hfov_deg
                )

    resolved = base if (base.width, base.height) == (width, height) else base.scaled_to(width, height)
    resolved.provenance = provenance
    return resolved


def pixel_rays(intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel normalised camera-frame ray directions (x/z and y/z)."""
    u = np.arange(intr.width, dtype=np.float32)
    v = np.arange(intr.height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    return (uu - intr.cx) / intr.fx, (vv - intr.cy) / intr.fy


def backproject(
    depth: DepthMap, intr: CameraIntrinsics, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth map into camera-frame 3D points.

    Returns ``(points, pixel_index)`` where ``points`` is ``(N, 3)`` float32 in
    the OpenCV camera frame and ``pixel_index`` is the flat index of each point
    in the original image, so per-pixel attributes (colour, segment id) can be
    gathered with ``attr.reshape(-1)[pixel_index]``.
    """
    z = depth.depth
    valid = depth.valid
    if mask is not None:
        valid = valid & mask

    rx, ry = pixel_rays(intr)
    flat_valid = valid.reshape(-1)
    idx = np.flatnonzero(flat_valid)
    if idx.size == 0:
        return np.zeros((0, 3), np.float32), idx

    zz = z.reshape(-1)[idx]
    xx = rx.reshape(-1)[idx] * zz
    yy = ry.reshape(-1)[idx] * zz
    return np.stack([xx, yy, zz], axis=1).astype(np.float32), idx


def project(points: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to pixels.  Returns ``(uv, depth)``."""
    z = points[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = points[:, 0] / safe * intr.fx + intr.cx
    v = points[:, 1] / safe * intr.fy + intr.cy
    return np.stack([u, v], axis=1), z


def depth_edge_mask(depth: DepthMap, rel_threshold: float) -> np.ndarray:
    """Mask out pixels sitting on a depth discontinuity.

    Back-projecting across an occlusion boundary smears "flying pixels" through
    empty space between the foreground and the background, which is the single
    ugliest artefact in monocular reconstructions.  We flag any pixel whose
    local depth range exceeds ``rel_threshold`` times its own depth.
    """
    if rel_threshold <= 0:
        return np.ones_like(depth.depth, dtype=bool)

    import cv2

    z = depth.depth.astype(np.float32)
    valid = depth.valid
    # Morphological gradient over a 3x3 window == local max minus local min.
    filled = np.where(valid, z, np.nan)
    filled = np.nan_to_num(filled, nan=0.0)
    kernel = np.ones((3, 3), np.uint8)
    local_max = cv2.dilate(filled, kernel)
    # Erode only over valid pixels so invalid zeros do not fake a discontinuity.
    big = np.where(valid, z, np.float32(1e6))
    local_min = cv2.erode(big, kernel)
    spread = local_max - local_min
    keep = spread <= (rel_threshold * np.maximum(z, 1e-3))
    return keep & valid
