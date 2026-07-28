"""Core data structures shared across the RoomVisualizer pipeline.

Coordinate conventions used throughout the package:

* **Camera frame** follows OpenCV: ``+x`` right, ``+y`` down, ``+z`` forward
  (into the scene).  Depth maps store the ``z`` coordinate in metres.
* **World frame** is Y-up / right-handed, matching glTF.  The gravity
  alignment step in :mod:`roomviz.geometry.align` is what takes us from the
  first camera's frame into the world frame, so before that step "world"
  simply means "frame 0's camera frame".
* **Poses** are 4x4 camera-to-world matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Semantic role of a segment.  ``structure`` segments participate in plane
# fitting and room layout; ``object`` segments become individual 3D entities.
ROLE_OBJECT = "object"
ROLE_STRUCTURE = "structure"
ROLE_IGNORE = "ignore"


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for a single image resolution."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    provenance: str = "unknown"
    """Where these came from: ``explicit_intrinsics``, ``hfov_flag``, ``exif``
    or ``assumed_default``.  Every reported dimension scales with the focal
    length, so a consumer must be able to tell a measured camera from a
    guessed one."""

    @classmethod
    def from_hfov(
        cls, width: int, height: int, hfov_deg: float = 60.0
    ) -> CameraIntrinsics:
        """Build intrinsics from an assumed horizontal field of view.

        Pixel *centres* sit at integer coordinates ``0 .. width - 1`` (this is
        the convention :func:`roomviz.geometry.camera.pixel_rays` samples), so
        the optical centre of the image is at ``(width - 1) / 2``.  The focal
        length still divides the full ``width`` of image extent, which spans
        ``-0.5 .. width - 0.5`` under the same convention.
        """
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(
            width=width,
            height=height,
            fx=float(f),
            fy=float(f),
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
        )

    def scaled_to(self, width: int, height: int) -> CameraIntrinsics:
        """Return intrinsics for the same camera at a different resolution.

        The principal point maps through pixel *edges* rather than centres -
        ``(c + 0.5) * s - 0.5`` - because a resize preserves the image
        rectangle, not the integer grid.  Scaling ``c`` directly would drift
        the optical centre by half a pixel per resize.
        """
        sx = width / self.width
        sy = height / self.height
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "provenance": self.provenance,
        }


@dataclass
class Frame:
    """A single RGB frame pulled out of an image or video."""

    index: int
    """Position within the sampled keyframe list."""

    rgb: np.ndarray  # (H, W, 3) uint8
    timestamp: float = 0.0
    source: str = ""

    source_index: int = 0
    """Index in the *original* media: the video frame number, or the position
    in a sorted image directory.  Keyframe sampling means this differs from
    ``index``, and sidecar files (precomputed depth, masks) are numbered
    against the original media, so lookups must use this."""

    original_size: tuple[int, int] | None = None
    """``(width, height)`` before any resize.  Intrinsics are derived at the
    original resolution and then rescaled, so that a resize which changes the
    aspect ratio slightly still yields correct, independent ``fx``/``fy``."""

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass
class DepthMap:
    """Metric depth in metres, plus a validity mask."""

    depth: np.ndarray  # (H, W) float32, metres, 0 == invalid
    metric: bool = True

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.depth) & (self.depth > 0)


@dataclass
class Segment:
    """One segmented region within a single frame."""

    segment_id: int
    label: str
    role: str  # ROLE_OBJECT / ROLE_STRUCTURE / ROLE_IGNORE
    score: float = 1.0
    # Structural sub-type for ROLE_STRUCTURE segments: wall / floor / ceiling.
    structure_kind: str | None = None

    is_thing: bool | None = None
    """Whether the segmenter instance-separates this class.  ``True`` means the
    mask covers exactly one object and must not be split geometrically;
    ``None`` means unknown, which is treated conservatively."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "label": self.label,
            "role": self.role,
            "score": self.score,
            "structure_kind": self.structure_kind,
            "is_thing": self.is_thing,
        }


@dataclass
class Segmentation:
    """Per-pixel segment ids for a frame, together with segment metadata."""

    ids: np.ndarray  # (H, W) int32, -1 == unlabelled
    segments: list[Segment] = field(default_factory=list)

    def by_id(self) -> dict[int, Segment]:
        return {s.segment_id: s for s in self.segments}

    def mask_for(self, segment_id: int) -> np.ndarray:
        return self.ids == segment_id


@dataclass
class Observation:
    """Everything the perception stage produced for one frame."""

    frame: Frame
    depth: DepthMap
    segmentation: Segmentation
    intrinsics: CameraIntrinsics
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))  # camera->world


@dataclass
class ObjectInstance:
    """A single 3D object accumulated across one or more frames."""

    instance_id: int
    label: str
    points: np.ndarray  # (N, 3) float32, world coordinates
    colors: np.ndarray  # (N, 3) uint8
    score: float = 1.0
    observations: int = 1
    frame_indices: list[int] = field(default_factory=list)

    split_gap: float = 0.0
    """The empty distance (m) at which this instance was parted from its
    siblings during the per-frame 3D split, or 0 if it was never split.

    Carried so that a merge can be undone at the same distance that justified
    it: a mask split at 30 cm must be rejoinable at 30 cm.  0 means "use the
    caller's default gap"."""

    sources: list[tuple[int, int]] = field(default_factory=list)
    """``(frame_index, segment_id)`` pairs this instance was built from.

    This is the provenance used to decide whether two instances may be merged.
    The segmenter's per-frame decision is authoritative: two clusters carrying
    *different* segment ids in the *same* frame are distinct objects and must
    never be combined, however close together they sit.  Clusters sharing a
    segment id came from one mask that the 3D split broke apart, so they may be
    rejoined if later evidence connects them.
    """

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.aabb
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "score": round(float(self.score), 4),
            "observations": self.observations,
            "frames": self.frame_indices,
            "point_count": int(self.points.shape[0]),
            "centroid": [round(float(v), 4) for v in self.centroid],
            "aabb_min": [round(float(v), 4) for v in lo],
            "aabb_max": [round(float(v), 4) for v in hi],
            "size": [round(float(v), 4) for v in (hi - lo)],
        }


@dataclass
class PlaneSurface:
    """A planar structural surface: a wall, the floor or the ceiling."""

    surface_id: int
    kind: str  # "wall" | "floor" | "ceiling"
    normal: np.ndarray  # (3,) unit normal, world frame
    offset: float  # plane is {p : normal . p + offset = 0}
    quad: np.ndarray  # (4, 3) corners of the bounded surface, world frame
    inlier_count: int = 0
    area: float = 0.0

    @property
    def extents(self) -> tuple[float, float]:
        """``(width, height)`` of the bounded surface, in metres.

        The quad's corners are ordered around the rectangle, so adjacent edges
        give the two side lengths.  For a wall this is the number a user
        actually wants -- "how long is this wall and how tall is it" -- which
        an area and four raw corner coordinates do not answer.
        """
        import numpy as _np

        quad = _np.asarray(self.quad, dtype=float)
        a = float(_np.linalg.norm(quad[1] - quad[0]))
        b = float(_np.linalg.norm(quad[2] - quad[1]))
        if self.kind == "wall":
            # Report the horizontal span first, then the vertical one.
            vertical = abs(quad[1][1] - quad[0][1]) > abs(quad[2][1] - quad[1][1])
            return (b, a) if vertical else (a, b)
        return (max(a, b), min(a, b))

    def to_dict(self) -> dict[str, Any]:
        width, height = self.extents
        return {
            "surface_id": self.surface_id,
            "kind": self.kind,
            "width": round(width, 4),
            "height": round(height, 4),
            "normal": [round(float(v), 4) for v in self.normal],
            "offset": round(float(self.offset), 4),
            "quad": [[round(float(v), 4) for v in c] for c in self.quad],
            "inlier_count": self.inlier_count,
            "area": round(float(self.area), 4),
        }


@dataclass
class Scene:
    """The fully reconstructed scene: the deliverable of the pipeline."""

    points: np.ndarray  # (N, 3) float32 world coordinates
    colors: np.ndarray  # (N, 3) uint8
    objects: list[ObjectInstance] = field(default_factory=list)
    surfaces: list[PlaneSurface] = field(default_factory=list)
    poses: list[np.ndarray] = field(default_factory=list)
    intrinsics: CameraIntrinsics | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.points.size == 0:
            return np.zeros(3), np.zeros(3)
        return self.points.min(axis=0), self.points.max(axis=0)
