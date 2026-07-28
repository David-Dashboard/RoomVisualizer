"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PipelineConfig:
    # ---- input handling -------------------------------------------------
    max_frames: int = 24
    """Upper bound on keyframes taken from a video."""

    frame_stride: int = 0
    """Sample every Nth video frame.  0 selects a stride automatically so the
    clip yields at most ``max_frames`` frames."""

    min_sharpness: float = 0.0
    """Drop frames whose variance-of-Laplacian is below this.  0 disables the
    blur filter."""

    max_side: int = 768
    """Longest image side fed to the perception models (and hence the
    resolution of the reconstruction)."""

    # ---- camera ---------------------------------------------------------
    hfov_deg: float = 60.0
    """Assumed horizontal field of view when intrinsics are unknown."""

    intrinsics: tuple[float, float, float, float] | None = None
    """Explicit ``(fx, fy, cx, cy)`` at the *original* input resolution."""

    # ---- perception backends --------------------------------------------
    depth_backend: str = "depth-anything-v2"
    depth_model: str = "depth-anything/Depth-Anything-V2-metric-indoor-small-hf"
    seg_backend: str = "mask2former"
    seg_model: str = "facebook/mask2former-swin-base-ade-panoptic"
    device: str = "auto"

    models_dir: str | None = None
    """Weight cache directory.  ``depth_model``/``seg_model`` may also be plain
    filesystem paths to a downloaded checkpoint."""

    offline: bool = False
    """Never hit the network; resolve checkpoints from the local cache only."""

    depth_dir: str | None = None
    """Directory of precomputed depth sidecars (``file`` depth backend)."""

    seg_dir: str | None = None
    """Directory of precomputed segment-id maps (``file`` segmentation backend)."""

    depth_scale: float = 0.001
    """Multiplier converting integer sidecar depth to metres (mm -> m)."""

    # ---- reconstruction --------------------------------------------------
    depth_trunc: float = 12.0
    """Ignore depth beyond this many metres (kills sky/window blowouts)."""

    depth_min: float = 0.15
    voxel_size: float = 0.025
    """Voxel size (m) for point-cloud downsampling and fusion."""

    edge_discard: float = 0.06
    """Drop points on strong depth discontinuities; fraction of local depth."""

    # ---- odometry / fusion ----------------------------------------------
    estimate_poses: bool = True
    """Estimate inter-frame camera motion for video input."""

    min_matches: int = 30
    """Minimum feature correspondences required to trust a pose estimate."""

    association_iou: float = 0.15
    """Minimum voxel IoU to merge two per-frame instances into one object."""

    object_split_gap: float = 0.12
    """Empty gap (m) that must separate two objects sharing a segmentation
    label before they are split into separate instances.  Panoptic "stuff"
    classes emit one mask per class, so three paintings on a wall arrive as a
    single segment; splitting them happens in 3D, and this is the smallest gap
    treated as a genuine separation rather than a hole in the sampling."""

    min_object_points: int = 120
    """Discard objects thinner than this after fusion."""

    # ---- structure extraction -------------------------------------------
    plane_threshold: float = 0.04
    """RANSAC inlier distance (m) for plane fitting."""

    plane_min_inliers: int = 800
    max_walls: int = 12
    align_gravity: bool = True

    # ---- output ----------------------------------------------------------
    export_ply: bool = True
    export_glb: bool = True
    export_objects: bool = True
    export_viewer: bool = True
    point_budget: int = 600_000
    """Cap on points written to the viewer scene."""

    seed: int = 0

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
