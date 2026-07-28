"""Fuse per-frame observations into a single 3D scene.

The pipeline here is:

1. Back-project every frame's depth into world space using its camera pose.
2. Split the resulting points into *object* points (grouped per segment, then
   split into physically separate 3D clusters) and *structural* points.
3. Associate per-frame object clusters across frames into global instances.
4. Gravity-align the whole scene, then fit walls / floor / ceiling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import PipelineConfig
from ..geometry.align import alignment_transform, apply_transform
from ..geometry.camera import backproject, depth_edge_mask
from ..geometry.planes import extract_surfaces
from ..geometry.pointcloud import (
    adaptive_voxel,
    largest_clusters,
    remove_statistical_outliers,
    voxel_downsample,
    voxel_iou,
)
from ..types import ROLE_OBJECT, ROLE_STRUCTURE, ObjectInstance, Observation, Scene

log = logging.getLogger(__name__)


@dataclass
class FrameDetection:
    """One object cluster seen in one frame."""

    frame_index: int
    label: str
    score: float
    points: np.ndarray
    colors: np.ndarray


def _frame_world_points(
    obs: Observation, cfg: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-space points, colours and per-point segment ids for one frame."""
    keep = depth_edge_mask(obs.depth, cfg.edge_discard)
    points_cam, pixel_idx = backproject(obs.depth, obs.intrinsics, mask=keep)
    if points_cam.shape[0] == 0:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint8),
            np.zeros(0, np.int32),
        )

    points_world = apply_transform(obs.pose, points_cam)
    colors = obs.frame.rgb.reshape(-1, 3)[pixel_idx]
    seg_ids = obs.segmentation.ids.reshape(-1)[pixel_idx]
    return points_world, colors, seg_ids


def _detections_for_frame(
    obs: Observation,
    points: np.ndarray,
    colors: np.ndarray,
    seg_ids: np.ndarray,
    cfg: PipelineConfig,
) -> list[FrameDetection]:
    """Split a frame's object segments into physically separate clusters.

    Panoptic "stuff" classes hand back one mask for every instance of a class
    (all the paintings in one mask), so splitting happens in 3D rather than in
    the image: genuinely separate objects are separated in space, while an
    object split in two by an occluder is still contiguous in 3D.
    """
    detections: list[FrameDetection] = []
    for segment in obs.segmentation.segments:
        if segment.role != ROLE_OBJECT:
            continue
        mask = seg_ids == segment.segment_id
        count = int(mask.sum())
        if count < cfg.min_object_points // 2:
            continue

        seg_points = points[mask]
        seg_colors = colors[mask]

        inliers = remove_statistical_outliers(seg_points)
        seg_points, seg_colors = seg_points[inliers], seg_colors[inliers]
        if seg_points.shape[0] < cfg.min_object_points // 2:
            continue

        # Split only across gaps wider than a real object separation, and
        # never narrower than this segment's own sampling density allows.
        cluster_voxel = adaptive_voxel(seg_points, cfg.object_split_gap)
        for cluster in largest_clusters(
            seg_points, voxel=cluster_voxel, min_fraction=0.08
        ):
            if cluster.size < cfg.min_object_points // 2:
                continue
            detections.append(
                FrameDetection(
                    frame_index=obs.frame.index,
                    label=segment.label,
                    score=segment.score,
                    points=seg_points[cluster],
                    colors=seg_colors[cluster],
                )
            )
    return detections


def _associate(
    detections: list[FrameDetection], cfg: PipelineConfig
) -> list[ObjectInstance]:
    """Merge per-frame detections into global instances.

    Greedy nearest-match on voxel IoU, restricted to detections that agree on
    the class label.  Detections from the same frame are never merged: the
    segmenter already decided they were distinct.

    Detections are consumed in frame order (largest first within a frame) so
    each instance grows through consecutive views.  Order matters because pose
    error accumulates along the trajectory: neighbouring frames overlap almost
    perfectly, whereas the first and last frames of a sweep may have drifted
    far enough apart to fall under the IoU threshold and split one object in
    two.
    """
    instances: list[ObjectInstance] = []
    voxel = cfg.voxel_size * 3.0

    ordered = sorted(detections, key=lambda d: (d.frame_index, -d.points.shape[0]))
    for det in ordered:
        best_idx, best_iou = -1, 0.0
        for i, inst in enumerate(instances):
            if inst.label != det.label:
                continue
            if det.frame_index in inst.frame_indices:
                continue
            iou = voxel_iou(det.points, inst.points, voxel)
            if iou > best_iou:
                best_idx, best_iou = i, iou

        if best_idx >= 0 and best_iou >= cfg.association_iou:
            inst = instances[best_idx]
            inst.points = np.vstack([inst.points, det.points])
            inst.colors = np.vstack([inst.colors, det.colors])
            inst.observations += 1
            inst.frame_indices.append(det.frame_index)
            inst.score = max(inst.score, det.score)
        else:
            instances.append(
                ObjectInstance(
                    instance_id=len(instances),
                    label=det.label,
                    points=det.points,
                    colors=det.colors,
                    score=det.score,
                    observations=1,
                    frame_indices=[det.frame_index],
                )
            )

    return instances


def _aabb_iou(a: ObjectInstance, b: ObjectInstance) -> float:
    """Volumetric IoU of two instances' axis-aligned bounding boxes."""
    a_lo, a_hi = a.aabb
    b_lo, b_hi = b.aabb
    lo = np.maximum(a_lo, b_lo)
    hi = np.minimum(a_hi, b_hi)
    overlap = np.prod(np.maximum(hi - lo, 0.0))
    if overlap <= 0:
        return 0.0
    volume_a = np.prod(np.maximum(a_hi - a_lo, 1e-6))
    volume_b = np.prod(np.maximum(b_hi - b_lo, 1e-6))
    return float(overlap / (volume_a + volume_b - overlap))


def _absorb(target: ObjectInstance, other: ObjectInstance) -> None:
    target.points = np.vstack([target.points, other.points])
    target.colors = np.vstack([target.colors, other.colors])
    target.observations += other.observations
    target.frame_indices.extend(other.frame_indices)
    target.score = max(target.score, other.score)


def _min_distance(a: np.ndarray, b: np.ndarray, sample: int = 4000) -> float:
    """Smallest distance between two point sets."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    if b.shape[0] > sample:
        b = b[rng.choice(b.shape[0], sample, replace=False)]
    return float(cKDTree(a).query(b, k=1, workers=-1)[0].min())


def _merge_duplicates(
    instances: list[ObjectInstance],
    gap: float,
    iou_threshold: float = 0.5,
    fragment_ratio: float = 0.4,
) -> list[ObjectInstance]:
    """Fold together same-label instances that describe one physical object.

    Two distinct failure modes need cleaning up, and they need different rules:

    * **Drift duplicates** - one object recovered twice because camera drift
      pushed its early and late observations below the association threshold.
      Both copies are of comparable size and occupy the same volume, so a
      bounding-box IoU test catches them.  The threshold is deliberately high:
      two chairs side by side share very little of their bounding volume.

    * **Fragments** - a self-occluding object (a sofa seen along its length)
      can appear in an early frame as two disconnected surface patches, and the
      smaller patch may never rejoin the main body.  A fragment is recognised
      by being much smaller than its neighbour *and* touching it, using the
      same gap that governed the split in the first place.  Requiring the size
      asymmetry is what keeps two genuinely adjacent objects apart.
    """
    merged: list[ObjectInstance] = []
    for inst in sorted(instances, key=lambda i: -i.points.shape[0]):
        for target in merged:
            if target.label != inst.label:
                continue
            same_volume = _aabb_iou(target, inst) >= iou_threshold
            is_fragment = (
                inst.points.shape[0] <= fragment_ratio * target.points.shape[0]
                and _min_distance(target.points, inst.points) <= gap
            )
            if same_volume or is_fragment:
                _absorb(target, inst)
                break
        else:
            merged.append(inst)

    if len(merged) != len(instances):
        log.info("merged %d duplicate instances", len(instances) - len(merged))
    return merged


def _finalise_instances(
    instances: list[ObjectInstance], cfg: PipelineConfig
) -> list[ObjectInstance]:
    """Downsample, drop the wisps, and renumber."""
    instances = _merge_duplicates(instances, gap=cfg.object_split_gap)
    kept: list[ObjectInstance] = []
    for inst in instances:
        points, colors, _ = voxel_downsample(inst.points, inst.colors, cfg.voxel_size)
        if points.shape[0] < cfg.min_object_points:
            continue
        inst.points, inst.colors = points, colors if colors is not None else inst.colors
        kept.append(inst)

    kept.sort(key=lambda i: -i.points.shape[0])
    for new_id, inst in enumerate(kept):
        inst.instance_id = new_id
        inst.frame_indices = sorted(set(inst.frame_indices))
    return kept


def fuse(observations: list[Observation], cfg: PipelineConfig) -> Scene:
    """Build a :class:`Scene` from per-frame perception output."""
    if not observations:
        raise ValueError("no observations to fuse")

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    structure_points: list[np.ndarray] = []
    structure_kinds: list[np.ndarray] = []
    detections: list[FrameDetection] = []

    for obs in observations:
        points, colors, seg_ids = _frame_world_points(obs, cfg)
        if points.shape[0] == 0:
            log.warning("frame %d produced no valid depth", obs.frame.index)
            continue

        all_points.append(points)
        all_colors.append(colors)
        detections.extend(_detections_for_frame(obs, points, colors, seg_ids, cfg))

        for segment in obs.segmentation.segments:
            if segment.role != ROLE_STRUCTURE or segment.structure_kind is None:
                continue
            mask = seg_ids == segment.segment_id
            if not mask.any():
                continue
            structure_points.append(points[mask])
            structure_kinds.append(
                np.full(int(mask.sum()), segment.structure_kind, dtype=object)
            )

    if not all_points:
        raise ValueError("every frame produced empty geometry; check depth input")

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    struct = (
        np.vstack(structure_points)
        if structure_points
        else np.zeros((0, 3), np.float32)
    )
    kinds = (
        np.concatenate(structure_kinds) if structure_kinds else np.zeros(0, dtype=object)
    )

    log.info(
        "fused %d raw points (%d structural) and %d detections",
        points.shape[0],
        struct.shape[0],
        len(detections),
    )

    instances = _finalise_instances(_associate(detections, cfg), cfg)

    # ---- gravity / Manhattan alignment ---------------------------------
    transform = np.eye(4)
    if cfg.align_gravity:
        # The camera of frame 0 defines the pre-alignment frame; in OpenCV
        # convention its up axis is -y, rotated by its own pose.
        camera_up = observations[0].pose[:3, :3] @ np.array([0.0, -1.0, 0.0])
        basis = struct if struct.shape[0] >= 500 else points
        basis_kinds = kinds if struct.shape[0] >= 500 else None
        transform = alignment_transform(basis, basis_kinds, camera_up=camera_up)
        points = apply_transform(transform, points)
        struct = apply_transform(transform, struct)
        for inst in instances:
            inst.points = apply_transform(transform, inst.points)

    poses = [transform @ obs.pose for obs in observations]

    # ---- structural surfaces --------------------------------------------
    up = np.array([0.0, 1.0, 0.0]) if cfg.align_gravity else np.array([0.0, -1.0, 0.0])
    surfaces: list = []
    if struct.shape[0] >= 3:
        struct_ds, _, inverse = voxel_downsample(struct, None, cfg.voxel_size)
        # Carry the semantic kind through the downsample.  Every output voxel
        # has at least one source point, and a voxel is far smaller than a
        # wall or floor patch, so picking an arbitrary member is safe.
        kinds_ds = np.empty(struct_ds.shape[0], dtype=object)
        kinds_ds[inverse] = kinds
        surfaces, _ = extract_surfaces(
            struct_ds,
            up=up,
            labels=list(kinds_ds),
            threshold=cfg.plane_threshold,
            min_inliers=max(50, int(cfg.plane_min_inliers)),
            max_planes=cfg.max_walls + 2,
            seed=cfg.seed,
        )

    # ---- global point cloud ---------------------------------------------
    points, colors, _ = voxel_downsample(points, colors, cfg.voxel_size)
    if points.shape[0] > cfg.point_budget:
        rng = np.random.default_rng(cfg.seed)
        pick = rng.choice(points.shape[0], cfg.point_budget, replace=False)
        points, colors = points[pick], colors[pick]
        log.info("subsampled scene cloud to the %d point budget", cfg.point_budget)

    scene = Scene(
        points=points,
        colors=colors,
        objects=instances,
        surfaces=surfaces,
        poses=poses,
        intrinsics=observations[0].intrinsics,
        meta={
            "frames": len(observations),
            "aligned": bool(cfg.align_gravity),
            "voxel_size": cfg.voxel_size,
        },
    )
    lo, hi = scene.bounds
    log.info(
        "scene: %d points, %d objects, %d surfaces, extent %s m",
        points.shape[0],
        len(instances),
        len(surfaces),
        np.round(hi - lo, 2).tolist(),
    )
    return scene
