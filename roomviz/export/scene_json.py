"""Machine-readable scene description.

``scene.json`` is the structured half of the output: everything the pipeline
inferred about the room, in a form other tools can consume without parsing
geometry files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..types import Scene
from .gltf import box_node_name, object_node_name, safe_label, surface_node_name
from .palette import label_color, surface_color


def _room_dimensions(scene: Scene) -> dict[str, Any]:
    """Floor area and room height, where the structure supports it."""
    out: dict[str, Any] = {}
    lo, hi = scene.bounds
    out["bounds_min"] = [round(float(v), 4) for v in lo]
    out["bounds_max"] = [round(float(v), 4) for v in hi]
    out["extent"] = [round(float(v), 4) for v in (hi - lo)]

    floors = [s for s in scene.surfaces if s.kind == "floor"]
    ceilings = [s for s in scene.surfaces if s.kind == "ceiling"]
    if floors:
        out["floor_area"] = round(float(floors[0].area), 3)
    if floors and ceilings:
        # Both planes are horizontal after alignment, so their height gap is
        # just the difference of the plane offsets along the up axis.
        floor_y = float(np.mean(floors[0].quad[:, 1]))
        ceiling_y = float(np.mean(ceilings[0].quad[:, 1]))
        out["room_height"] = round(abs(ceiling_y - floor_y), 3)
    return out


def build_scene_dict(scene: Scene, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    counts = Counter(o.label.split(";")[0] for o in scene.objects)
    payload: dict[str, Any] = {
        "format": "roomviz-scene",
        "version": 1,
        "up_axis": "+Y",
        "units": "metres",
        "summary": {
            "point_count": int(scene.points.shape[0]),
            "object_count": len(scene.objects),
            "surface_count": len(scene.surfaces),
            "frame_count": int(scene.meta.get("frames", len(scene.poses))),
            "labels": dict(counts.most_common()),
            "surfaces_by_kind": dict(Counter(s.kind for s in scene.surfaces)),
        },
        "room": _room_dimensions(scene),
        "objects": [],
        "surfaces": [],
        "cameras": [],
    }

    for inst in scene.objects:
        entry = inst.to_dict()
        entry["color"] = list(label_color(inst.label))
        entry["node"] = object_node_name(inst)
        entry["box_node"] = box_node_name(inst)
        if cfg is None or cfg.export_objects:
            # Only claim a file that this run actually writes; a dangling
            # reference is worse than none.
            entry["cloud_file"] = (
                f"objects/{inst.instance_id:03d}_{safe_label(inst.label)}.ply"
            )
        payload["objects"].append(entry)

    for surface in scene.surfaces:
        entry = surface.to_dict()
        entry["color"] = list(surface_color(surface.kind))
        entry["node"] = surface_node_name(surface)
        payload["surfaces"].append(entry)

    for i, pose in enumerate(scene.poses):
        payload["cameras"].append(
            {
                "frame": i,
                "position": [round(float(v), 4) for v in pose[:3, 3]],
                "matrix": [round(float(v), 6) for v in np.asarray(pose).reshape(-1)],
            }
        )

    if scene.intrinsics is not None:
        payload["intrinsics"] = scene.intrinsics.to_dict()
    if cfg is not None:
        payload["config"] = cfg.to_dict()
        payload["config"].pop("extra", None)
    payload["meta"] = {k: v for k, v in scene.meta.items()}
    return payload


def write_scene_json(
    path: str | Path, scene: Scene, cfg: PipelineConfig | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_scene_dict(scene, cfg), indent=2))
    return path
