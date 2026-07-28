"""glTF/GLB export.

The scene is written as a set of *named* nodes so a viewer (ours, Blender, or
anything else that reads glTF) can list and toggle individual objects:

* ``cloud`` - the full coloured point cloud
* ``object__<id>__<label>`` - one point cloud per segmented object
* ``box__<id>__<label>`` - that object's axis-aligned bounding box
* ``surface__<kind>__<id>`` - wall, floor and ceiling quads

Double underscores rather than slashes because three.js sanitises glTF node
names by stripping ``/``, ``.``, ``:`` and brackets; underscores survive
intact, so the viewer can recover the node's role from its name alone.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from ..types import ObjectInstance, PlaneSurface, Scene
from .palette import label_color, surface_color

log = logging.getLogger(__name__)


def _require_trimesh():
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise ImportError("GLB export needs trimesh: pip install trimesh") from exc
    return trimesh


def safe_label(label: str, max_length: int = 48) -> str:
    """First synonym of a class name, reduced to name-safe characters.

    Length-capped because this doubles as a filename component: a checkpoint
    with a pathological class name would otherwise raise ENAMETOOLONG at the
    very end of a run, after the rest of the output had already been written.
    """
    first = label.split(";")[0].strip() or "object"
    return re.sub(r"[^A-Za-z0-9-]+", "_", first)[:max_length] or "object"


def object_node_name(inst: ObjectInstance) -> str:
    return f"object__{inst.instance_id}__{safe_label(inst.label)}"


def box_node_name(inst: ObjectInstance) -> str:
    return f"box__{inst.instance_id}__{safe_label(inst.label)}"


def surface_node_name(surface: PlaneSurface) -> str:
    return f"surface__{surface.kind}__{surface.surface_id}"


def _box_mesh(trimesh, lo: np.ndarray, hi: np.ndarray, color: tuple[int, int, int]):
    """A hollow-looking wireframe box built from thin edge cylinders.

    glTF has no line primitive that survives every importer, so the box is real
    geometry: 12 thin boxes, one per edge.  Thickness scales with the object so
    a mug's box is not a solid blob.
    """
    size = np.maximum(hi - lo, 1e-3)
    thickness = float(np.clip(size.min() * 0.02, 0.004, 0.02))
    # Inset the edge centre-lines by half the bar thickness so the drawn box
    # sits *inside* the reported AABB rather than overshooting it - otherwise
    # measuring the GLB gives a different answer from `scene.json`.
    inset = thickness / 2.0
    lo = lo + inset
    hi = np.maximum(hi - inset, lo)
    corners = np.array(
        [
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ]
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    parts = []
    for i, j in edges:
        a, b = corners[i], corners[j]
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        extents = np.full(3, thickness)
        axis = int(np.argmax(np.abs(direction)))
        extents[axis] = length
        transform = np.eye(4)
        transform[:3, 3] = (a + b) / 2.0
        parts.append(trimesh.creation.box(extents=extents, transform=transform))

    if not parts:
        return None
    mesh = trimesh.util.concatenate(parts)
    mesh.visual.face_colors = np.tile(np.array([*color, 255], np.uint8), (len(mesh.faces), 1))
    return mesh


def _quad_mesh(trimesh, quad: np.ndarray, color: tuple[int, int, int], alpha: int = 190):
    """Two triangles spanning a planar quad.

    Single-sided on purpose: the viewer renders surfaces with
    ``THREE.DoubleSide``, and emitting reversed duplicates as well would draw
    every quad twice.  Coincident transparent triangles blend twice over
    (turning 0.75 opacity into ~0.94) and z-fight against each other.
    """
    vertices = np.asarray(quad, dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = np.tile(
        np.array([*color, alpha], np.uint8), (len(mesh.faces), 1)
    )
    return mesh


def build_trimesh_scene(scene: Scene, include_cloud: bool = True):
    """Assemble the scene as a :class:`trimesh.Scene`."""
    trimesh = _require_trimesh()
    geometry: dict[str, object] = {}

    if include_cloud and scene.points.shape[0]:
        geometry["cloud"] = trimesh.PointCloud(
            vertices=scene.points.astype(np.float64), colors=scene.colors
        )

    for inst in scene.objects:
        color = label_color(inst.label)
        geometry[object_node_name(inst)] = trimesh.PointCloud(
            vertices=inst.points.astype(np.float64), colors=inst.colors
        )
        lo, hi = inst.aabb
        box = _box_mesh(trimesh, lo, hi, color)
        if box is not None:
            geometry[box_node_name(inst)] = box

    for surface in scene.surfaces:
        geometry[surface_node_name(surface)] = _quad_mesh(
            trimesh, surface.quad, surface_color(surface.kind)
        )

    return trimesh.Scene(geometry=geometry)


def write_glb(path: str | Path, scene: Scene, include_cloud: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tm_scene = build_trimesh_scene(scene, include_cloud=include_cloud)
    data = tm_scene.export(file_type="glb")
    path.write_bytes(data)
    log.info("wrote %s (%.1f MB)", path, len(data) / 1e6)
    return path
