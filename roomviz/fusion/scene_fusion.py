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
    sets_are_connected,
    voxel_downsample,
    voxel_overlap,
)
from ..perception.labels import is_split_candidate
from ..types import ROLE_OBJECT, ROLE_STRUCTURE, ObjectInstance, Observation, Scene

log = logging.getLogger(__name__)


@dataclass
class FrameDetection:
    """One object cluster seen in one frame."""

    frame_index: int
    segment_id: int
    """Which 2D segment this cluster came from.  Two detections in one frame
    with different segment ids are distinct objects by the segmenter's own
    reckoning; two with the same id are pieces of one mask that the 3D split
    separated."""

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
    the image: genuinely separate objects are separated in space.

    A split here is provisional.  An object that occludes itself - a sofa seen
    along its length - can present as two disconnected patches in one view and
    as one contiguous patch in the next, so every cluster records the segment
    it came from and :func:`_merge_instances` may rejoin them later.
    """
    detections: list[FrameDetection] = []
    floor = max(1, cfg.min_object_points // 2)
    for segment in obs.segmentation.segments:
        if segment.role != ROLE_OBJECT:
            continue
        mask = seg_ids == segment.segment_id
        if int(mask.sum()) < floor:
            continue

        seg_points = points[mask]
        seg_colors = colors[mask]

        inliers = remove_statistical_outliers(seg_points)
        seg_points, seg_colors = seg_points[inliers], seg_colors[inliers]
        if seg_points.shape[0] < floor:
            continue

        # A "thing" mask covers exactly one object by the segmenter's own
        # reckoning, so it is never split - an object an occluder cut into two
        # visible patches must stay one object.  Only stuff (or unknown)
        # classes, which can hold several objects in one mask, are split.
        if not is_split_candidate(segment.label, segment.is_thing):
            clusters = [np.arange(seg_points.shape[0])]
        else:
            # 26-connectivity links points up to two voxels apart, so a voxel
            # of `gap / 2` makes the realised split distance `gap`.  Using
            # `gap` itself splits at roughly twice the documented distance, and
            # the grid's origin then leaks into the result: whether two objects
            # separate would depend on where they sit in world coordinates.
            cluster_voxel = adaptive_voxel(seg_points, cfg.object_split_gap / 2.0)
            clusters = largest_clusters(
                seg_points, voxel=cluster_voxel, min_points=floor
            )
        for cluster in clusters:
            detections.append(
                FrameDetection(
                    frame_index=obs.frame.index,
                    segment_id=segment.segment_id,
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

    Greedy nearest-match on voxel overlap, restricted to detections that agree
    on the class label.  Two detections in the same frame carrying *different*
    segment ids are never merged - the segmenter has already ruled they are
    distinct objects - but two carrying the same id may be, since those were
    separated by our own 3D split rather than by the segmenter.

    Detections are consumed in frame order (largest first within a frame) so
    each instance grows through consecutive views.  Order matters because pose
    error accumulates along the trajectory: neighbouring frames overlap almost
    perfectly, whereas the first and last frames of a sweep may have drifted
    far enough apart to fall below the threshold and split one object in two.
    """
    instances: list[ObjectInstance] = []
    voxel = cfg.voxel_size * 3.0

    ordered = sorted(detections, key=lambda d: (d.frame_index, -d.points.shape[0]))
    for det in ordered:
        best_idx, best_score = -1, 0.0
        for i, inst in enumerate(instances):
            if inst.label != det.label:
                continue
            if _conflicts(inst.sources, [(det.frame_index, det.segment_id)]):
                continue
            score = voxel_overlap(det.points, inst.points, voxel)
            if score > best_score:
                best_idx, best_score = i, score

        if best_idx >= 0 and best_score >= cfg.association_iou:
            inst = instances[best_idx]
            inst.points = np.vstack([inst.points, det.points])
            inst.colors = np.vstack([inst.colors, det.colors])
            inst.observations += 1
            inst.frame_indices.append(det.frame_index)
            inst.sources.append((det.frame_index, det.segment_id))
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
                    sources=[(det.frame_index, det.segment_id)],
                )
            )

    return instances


def _conflicts(
    a_sources: list[tuple[int, int]], b_sources: list[tuple[int, int]]
) -> bool:
    """Whether two provenances prove their instances are different objects.

    They do when some frame saw both under *different* segment ids: within a
    single frame the segmenter assigns one id per object, so two ids means two
    objects, no matter how close together they ended up in 3D.  This is what
    keeps a row of identical stools from collapsing into one, and it does not
    depend on how many frames happened to see each one.
    """
    by_frame: dict[int, set[int]] = {}
    for frame, segment in a_sources:
        by_frame.setdefault(frame, set()).add(segment)
    for frame, segment in b_sources:
        seen = by_frame.get(frame)
        if seen is not None and seen - {segment}:
            return True
    return False


def _aabb_iou(a: ObjectInstance, b: ObjectInstance, thickness: float = 1e-3) -> float:
    """Volumetric IoU of two instances' axis-aligned bounding boxes.

    Extents are floored at ``thickness`` so that a flat object - a painting, a
    television, a door - still has a comparable volume.  Without it a perfectly
    planar instance has zero volume, every IoU involving it is zero, and two
    copies of the same painting can never be recognised as duplicates.
    """
    def inflated(box: ObjectInstance) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = box.aabb
        centre = (lo + hi) / 2.0
        extent = np.maximum(hi - lo, thickness)
        return centre - extent / 2.0, centre + extent / 2.0

    a_lo, a_hi = inflated(a)
    b_lo, b_hi = inflated(b)
    overlap = float(
        np.prod(np.maximum(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0.0))
    )
    volume_a = float(np.prod(a_hi - a_lo))
    volume_b = float(np.prod(b_hi - b_lo))
    union = volume_a + volume_b - overlap
    if union <= 0:
        return 0.0
    return max(0.0, overlap / union)


def _absorb(target: ObjectInstance, other: ObjectInstance) -> None:
    target.points = np.vstack([target.points, other.points])
    target.colors = np.vstack([target.colors, other.colors])
    target.observations += other.observations
    target.frame_indices.extend(other.frame_indices)
    target.sources.extend(other.sources)
    target.score = max(target.score, other.score)


def _merge_instances(
    instances: list[ObjectInstance], gap: float, iou_threshold: float = 0.5
) -> list[ObjectInstance]:
    """Fold together instances that describe one physical object.

    Two failure modes need cleaning up:

    * **Drift duplicates** - one object recovered twice because camera drift
      pushed its early and late observations below the association threshold.
      Both copies occupy the same volume, so bounding-box IoU catches them.
    * **Split siblings** - a self-occluding object that presented as two
      disconnected patches.  These are recognised by touching each other within
      the same gap that governed the split.

    Provenance is what makes the second rule safe.  Merging on proximity alone
    would fuse two chairs standing side by side; merging only when no frame saw
    the two under different segment ids cannot, because any frame that saw both
    chairs gave them separate ids.  Sizes are deliberately *not* consulted: the
    point count of an instance tracks how many frames happened to see it, not
    how big it is, so a size-ratio rule silently destroys real objects whenever
    two neighbours differ in visibility.

    Merging is transitive and iterated to a fixed point, so a chain of pieces
    collapses to one object regardless of the order they are considered in.
    """
    pool = sorted(instances, key=lambda i: -i.points.shape[0])
    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(len(pool)):
            if pool[i] is None:
                continue
            for j in range(i + 1, len(pool)):
                if pool[j] is None:
                    continue
                a, b = pool[i], pool[j]
                if a.label != b.label:
                    continue
                if _conflicts(a.sources, b.sources):
                    continue
                same_volume = _aabb_iou(a, b) >= iou_threshold
                touching = sets_are_connected(a.points, b.points, gap)
                if same_volume or touching:
                    _absorb(a, b)
                    pool[j] = None
                    merged_any = True
        pool = [p for p in pool if p is not None]

    if len(pool) != len(instances):
        log.info("merged %d instance(s) into their parent object", len(instances) - len(pool))
    return pool


def _finalise_instances(
    instances: list[ObjectInstance], cfg: PipelineConfig
) -> list[ObjectInstance]:
    """Downsample, drop the wisps, and renumber."""
    instances = _merge_instances(instances, gap=cfg.object_split_gap)
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
