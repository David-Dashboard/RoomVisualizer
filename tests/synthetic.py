"""A tiny ray-traced synthetic room used to test the pipeline end to end.

Everything downstream of the neural networks - back-projection, odometry,
clustering, cross-frame association, plane fitting, alignment and export - can
be checked against exact ground truth by rendering a room whose dimensions we
chose ourselves.  The renderer only needs axis-aligned boxes, so it is a few
dozen lines of slab intersection.

Camera convention matches the rest of the package (OpenCV: ``+x`` right,
``+y`` down, ``+z`` forward), and rays are built with ``z = 1`` so the ray
parameter *is* the camera-frame depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roomviz.types import CameraIntrinsics, Frame, Segment, Segmentation

# Segment ids.  Walls share the "wall" label but get distinct ids, exactly as a
# panoptic model would emit them.
FLOOR_ID = 1
CEILING_ID = 2
WALL_IDS = (3, 4, 5, 6)
FIRST_OBJECT_ID = 10


@dataclass
class Box:
    """An axis-aligned object box."""

    lo: np.ndarray
    hi: np.ndarray
    label: str
    color: tuple[int, int, int]

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo


@dataclass
class Room:
    width: float = 5.0  # x extent
    height: float = 2.7  # y extent (floor at y=0)
    depth: float = 4.0  # z extent
    boxes: list[Box] = field(default_factory=list)

    @property
    def lo(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.width, self.height, self.depth])


def default_room() -> Room:
    """A room with four distinctly sized, well-separated pieces of furniture."""
    return Room(
        width=5.0,
        height=2.7,
        depth=4.0,
        boxes=[
            Box(np.array([0.8, 0.0, 1.6]), np.array([2.0, 0.75, 2.6]),
                "table", (150, 105, 70)),
            Box(np.array([2.4, 0.0, 1.5]), np.array([3.0, 0.95, 2.1]),
                "chair", (90, 130, 170)),
            Box(np.array([3.3, 0.0, 2.5]), np.array([4.7, 0.62, 3.7]),
                "sofa", (170, 90, 110)),
            Box(np.array([0.25, 0.0, 3.0]), np.array([0.85, 1.85, 3.6]),
                "bookcase", (120, 150, 110)),
        ],
    )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Camera-to-world pose looking from ``eye`` at ``target`` (OpenCV axes)."""
    up = np.array([0.0, 1.0, 0.0]) if up is None else up
    z = target - eye
    z = z / np.linalg.norm(z)
    down = -up
    y = down - z * float(down @ z)  # orthogonalise "down" against the view axis
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)  # right-handed: x = y * z
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = eye
    return pose


def _hash3(cells: np.ndarray) -> np.ndarray:
    """Cheap integer hash of 3D cell coordinates -> pseudo-random [0, 1)."""
    x = (cells[:, 0] * np.int64(73856093)) ^ (cells[:, 1] * np.int64(19349663))
    x = x ^ (cells[:, 2] * np.int64(83492791))
    x = (x ^ (x >> np.int64(13))) * np.int64(1274126177)
    return ((x ^ (x >> np.int64(16))) & np.int64(0xFFFF)).astype(np.float64) / 65535.0


TEXTURE_AMPLITUDE = 0.45
"""Default peak-to-peak contrast of the surface texture, as a fraction of the
base albedo.  Everything ORB has to work with comes from this; see
``tests/test_harness_cliffs.py`` for the measured amplitude at which the
harness stops reconstructing."""


def _texture(
    points: np.ndarray,
    base: np.ndarray,
    cell: float = 0.045,
    amplitude: float = TEXTURE_AMPLITUDE,
) -> np.ndarray:
    """Deterministic random-dot texture keyed to world position.

    Real photos give ORB plenty of corners to match; flat shading gives it
    none.  A *random* per-cell pattern is what is needed rather than a smooth
    or periodic one: smooth gradients localise corners poorly and periodic
    patterns produce ambiguous matches, both of which wreck pose estimation.
    Keying on world position (not screen position) keeps the texture
    view-consistent, so it behaves like real surface detail.
    """
    cells = np.floor(points / cell).astype(np.int64)
    modulation = 1.0 + amplitude * (_hash3(cells)[:, None] - 0.5)
    return np.clip(base * modulation, 0, 255)


def _ray_box_enter(
    origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Slab test: nearest entry ``t`` into a box.  Returns ``(t, hit_mask)``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t0 = (lo - origin) * inv
        t1 = (hi - origin) * inv
    tmin = np.minimum(t0, t1).max(axis=1)
    tmax = np.maximum(t0, t1).min(axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 1e-6)
    return tmin, hit


def _ray_room_exit(
    origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exit ``t`` and face axis/sign for a ray starting inside the room."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t0 = (lo - origin) * inv
        t1 = (hi - origin) * inv
    t_far = np.maximum(t0, t1)
    t_exit = t_far.min(axis=1)
    axis = t_far.argmin(axis=1)
    return t_exit, axis


def render(
    room: Room,
    pose: np.ndarray,
    intr: CameraIntrinsics,
    texture_amplitude: float = TEXTURE_AMPLITUDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one view.  Returns ``(rgb uint8, depth float32, segment ids)``."""
    h, w = intr.height, intr.width
    u = (np.arange(w, dtype=np.float64) - intr.cx) / intr.fx
    v = (np.arange(h, dtype=np.float64) - intr.cy) / intr.fy
    uu, vv = np.meshgrid(u, v)
    # z = 1 so the ray parameter is exactly the camera-frame depth.
    dirs_cam = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1)
    dirs = dirs_cam @ pose[:3, :3].T
    origin = pose[:3, 3]

    n = dirs.shape[0]
    best_t = np.full(n, np.inf)
    seg = np.full(n, -1, np.int32)
    base = np.zeros((n, 3), np.float64)

    # --- room shell -------------------------------------------------------
    t_exit, axis = _ray_room_exit(origin, dirs, room.lo, room.hi)
    best_t = t_exit
    positive = dirs[np.arange(n), axis] > 0
    seg = np.where(axis == 1, np.where(positive, CEILING_ID, FLOOR_ID), 0).astype(np.int32)
    # The four vertical faces: -x, +x, -z, +z.
    wall_index = np.where(axis == 0, np.where(positive, 1, 0), np.where(positive, 3, 2))
    is_wall = axis != 1
    seg[is_wall] = np.array(WALL_IDS, np.int32)[wall_index[is_wall]]

    shell_colors = {
        FLOOR_ID: np.array([150, 130, 110], float),
        CEILING_ID: np.array([225, 228, 232], float),
        WALL_IDS[0]: np.array([205, 200, 190], float),
        WALL_IDS[1]: np.array([198, 205, 200], float),
        WALL_IDS[2]: np.array([210, 202, 195], float),
        WALL_IDS[3]: np.array([200, 198, 208], float),
    }
    for sid, colour in shell_colors.items():
        mask = seg == sid
        if mask.any():
            base[mask] = colour

    # --- furniture --------------------------------------------------------
    for i, box in enumerate(room.boxes):
        t, hit = _ray_box_enter(origin, dirs, box.lo, box.hi)
        closer = hit & (t < best_t)
        best_t[closer] = t[closer]
        seg[closer] = FIRST_OBJECT_ID + i
        base[closer] = np.array(box.color, float)

    points = origin + dirs * best_t[:, None]
    rgb = (
        _texture(points, base, amplitude=texture_amplitude)
        .astype(np.uint8)
        .reshape(h, w, 3)
    )
    depth = best_t.astype(np.float32).reshape(h, w)
    return rgb, depth, seg.reshape(h, w)


def labels_for(room: Room) -> dict[int, str]:
    labels = {FLOOR_ID: "floor", CEILING_ID: "ceiling"}
    for wid in WALL_IDS:
        labels[wid] = "wall"
    for i, box in enumerate(room.boxes):
        labels[FIRST_OBJECT_ID + i] = box.label
    return labels


def segmentation_for(
    ids: np.ndarray, room: Room, thing_flags: dict[int, bool] | None = None
) -> Segmentation:
    from roomviz.perception.labels import classify, is_thing

    labels = labels_for(room)
    segments = []
    for sid in np.unique(ids):
        sid = int(sid)
        if sid < 0:
            continue
        label = labels[sid]
        role, kind = classify(label)
        segments.append(
            Segment(
                segment_id=sid, label=label, role=role, structure_kind=kind,
                is_thing=thing_flags.get(sid) if thing_flags else is_thing(label),
            )
        )
    return Segmentation(ids=ids.astype(np.int32), segments=segments)


PATH_SPAN = 1.4
"""Default horizontal travel (m) of the camera sweep.  Parallax - and hence
everything odometry can recover - scales with this."""

FRAME_COUNT = 6
"""Default number of views in the sweep."""


def orbit_poses(
    room: Room, count: int = FRAME_COUNT, span: float = PATH_SPAN
) -> list[np.ndarray]:
    """A camera sweep along the near wall, converging on the room centre.

    The path is chosen so every piece of furniture stays fully inside the
    frame in every view - which is what lets the tests assert on true object
    dimensions rather than on whatever happened to be visible - while still
    translating enough to give odometry real parallax to work with.

    ``count`` and ``span`` are exposed so the suite can measure how much of
    either the pipeline actually needs, rather than assuming the shipped
    values sit anywhere in particular relative to the failure point.
    """
    poses = []
    for i in range(count):
        s = i / max(1, count - 1)
        eye = np.array(
            [1.8 + span * s, 1.55 + 0.05 * np.sin(s * 5.0), 0.15 + 0.15 * s]
        )
        target = np.array([2.4 + 0.25 * (s - 0.5), 0.75, 2.7])
        poses.append(look_at(eye, target))
    return poses


def render_sequence(
    room: Room | None = None,
    poses: list[np.ndarray] | None = None,
    width: int = 224,
    height: int = 168,
    hfov: float = 70.0,
    texture_amplitude: float = TEXTURE_AMPLITUDE,
    frame_count: int = FRAME_COUNT,
    path_span: float = PATH_SPAN,
) -> tuple[Room, CameraIntrinsics, list[np.ndarray], list[Frame], list[np.ndarray], list[Segmentation]]:
    """Render a whole sequence.

    Returns ``(room, intrinsics, poses, frames, depths, segmentations)``.
    """
    room = room or default_room()
    intr = CameraIntrinsics.from_hfov(width, height, hfov)
    if poses is None:
        poses = orbit_poses(room, count=frame_count, span=path_span)

    frames, depths, segmentations = [], [], []
    for i, pose in enumerate(poses):
        rgb, depth, ids = render(room, pose, intr, texture_amplitude=texture_amplitude)
        frames.append(Frame(index=i, rgb=rgb, timestamp=float(i), source="synthetic"))
        depths.append(depth)
        segmentations.append(segmentation_for(ids, room))
    return room, intr, poses, frames, depths, segmentations
