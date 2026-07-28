"""Unit tests for the fusion stage's association and merge rules."""

from __future__ import annotations

import numpy as np
import pytest

from roomviz.config import PipelineConfig
from roomviz.fusion.odometry import invert_rigid
from roomviz.fusion.scene_fusion import (
    FrameDetection,
    _aabb_iou,
    _associate,
    _conflicts,
    _merge_instances,
)
from roomviz.geometry.pointcloud import (
    adaptive_voxel,
    median_spacing,
    voxel_overlap,
)
from roomviz.types import ObjectInstance


def block(centre, size=(0.4, 0.4, 0.4), n=600, seed=0):
    rng = np.random.default_rng(seed)
    half = np.array(size) / 2
    return (rng.uniform(-half, half, (n, 3)) + np.array(centre)).astype(np.float32)


def detection(frame, label, points, segment_id=None):
    colors = np.full((points.shape[0], 3), 128, np.uint8)
    # Default: one segment id per label, i.e. the segmenter saw one object.
    if segment_id is None:
        segment_id = abs(hash(label)) % 1000
    return FrameDetection(
        frame_index=frame,
        segment_id=segment_id,
        label=label,
        score=1.0,
        points=points,
        colors=colors,
    )


def instance(instance_id, label, points, sources=None):
    return ObjectInstance(
        instance_id=instance_id,
        label=label,
        points=points,
        colors=np.full((points.shape[0], 3), 128, np.uint8),
        frame_indices=[0],
        sources=list(sources) if sources is not None else [(0, instance_id)],
    )


# --------------------------------------------------------------------------
# association
# --------------------------------------------------------------------------

def test_same_object_across_frames_becomes_one_instance():
    cfg = PipelineConfig()
    points = block((1.0, 0.5, 2.0))
    detections = [detection(i, "chair", points + i * 0.01) for i in range(5)]
    instances = _associate(detections, cfg)
    assert len(instances) == 1
    assert instances[0].observations == 5
    assert sorted(instances[0].frame_indices) == [0, 1, 2, 3, 4]


def test_distant_objects_stay_separate():
    cfg = PipelineConfig()
    detections = [
        detection(0, "chair", block((0.0, 0.5, 2.0), seed=1), segment_id=1),
        detection(0, "chair", block((3.0, 0.5, 2.0), seed=2), segment_id=2),
        detection(1, "chair", block((0.0, 0.5, 2.0), seed=3), segment_id=1),
        detection(1, "chair", block((3.0, 0.5, 2.0), seed=4), segment_id=2),
    ]
    instances = _associate(detections, cfg)
    assert len(instances) == 2
    assert all(i.observations == 2 for i in instances)


def test_different_labels_never_merge():
    cfg = PipelineConfig()
    points = block((1.0, 0.5, 2.0))
    instances = _associate(
        [detection(0, "chair", points), detection(1, "table", points)], cfg
    )
    assert len(instances) == 2


def test_different_segments_in_one_frame_are_never_merged():
    """The segmenter already decided these are distinct objects."""
    cfg = PipelineConfig(association_iou=0.0)
    points = block((1.0, 0.5, 2.0))
    instances = _associate(
        [
            detection(0, "chair", points, segment_id=1),
            detection(0, "chair", points + 0.001, segment_id=2),
        ],
        cfg,
    )
    assert len(instances) == 2


# --------------------------------------------------------------------------
# duplicate and fragment merging
# --------------------------------------------------------------------------

def test_aabb_iou_extremes():
    a = instance(0, "x", block((0.0, 0.0, 0.0)))
    assert _aabb_iou(a, a) == pytest.approx(1.0)
    b = instance(1, "x", block((9.0, 0.0, 0.0)))
    assert _aabb_iou(a, b) == 0.0


def test_aabb_iou_is_a_union_not_a_maximum():
    """Partial overlap must divide by the union.

    Identical and disjoint boxes score the same under intersection-over-union,
    over-maximum and over-minimum alike, so only a partial overlap tells them
    apart -- and the difference straddles the 0.5 merge threshold.
    """
    def box(lo, hi):
        corners = np.array([[x, y, z] for x in lo[:1] + hi[:1]
                            for y in lo[1:2] + hi[1:2]
                            for z in lo[2:3] + hi[2:3]], np.float32)
        return instance(0, "x", corners)

    # 1.0 x 1.0 x 1.0 boxes offset by 0.4 in x: overlap 0.6, union 1.4.
    a = box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    b = box([0.4, 0.0, 0.0], [1.4, 1.0, 1.0])
    iou = _aabb_iou(a, b, thickness=0.0)
    assert iou == pytest.approx(0.6 / 1.4, rel=1e-3)     # union: ~0.43
    assert iou < 0.5                                     # so they must NOT merge
    # over-maximum would give 0.6 here and wrongly merge them.
    assert len(_merge_instances([a, b], gap=0.01)) == 2


def test_drift_duplicates_are_merged():
    a = instance(0, "sofa", block((1.0, 0.3, 2.0), size=(1.4, 0.6, 1.2), n=2000, seed=1),
                 sources=[(0, 7)])
    b = instance(1, "sofa", block((1.05, 0.3, 2.05), size=(1.4, 0.6, 1.2), n=1800, seed=2),
                 sources=[(5, 7)])
    merged = _merge_instances([a, b], gap=0.12)
    assert len(merged) == 1
    assert merged[0].points.shape[0] == 3800


def test_touching_split_sibling_is_rejoined():
    """Two patches of one segment that touch are one object."""
    body = instance(0, "sofa", block((1.0, 0.3, 2.0), size=(1.4, 0.6, 1.2), n=2000, seed=1),
                    sources=[(0, 7)])
    sibling = instance(
        1, "sofa", block((1.0, 0.65, 2.0), size=(1.2, 0.06, 1.0), n=200, seed=2),
        sources=[(0, 7)],
    )
    merged = _merge_instances([body, sibling], gap=0.12)
    assert len(merged) == 1
    assert merged[0].points.shape[0] == 2200


def test_equal_sized_split_siblings_are_rejoined():
    """A sofa occluded mid-span yields two halves of the *same* size.

    A size-asymmetry rule cannot rejoin these; provenance can, because both
    halves carry the same segment id.
    """
    left = instance(0, "sofa", block((0.5, 0.3, 2.0), size=(1.0, 0.6, 1.0), n=1400, seed=1),
                    sources=[(0, 7)])
    right = instance(1, "sofa", block((1.55, 0.3, 2.0), size=(1.0, 0.6, 1.0), n=1400, seed=2),
                     sources=[(0, 7)])
    merged = _merge_instances([left, right], gap=0.12)
    assert len(merged) == 1


def test_adjacent_distinct_objects_are_not_merged():
    """Two chairs pushed together must stay two chairs."""
    a = instance(0, "chair", block((0.0, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=1000, seed=1),
                 sources=[(0, 1), (1, 1)])
    b = instance(1, "chair", block((0.55, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=950, seed=2),
                 sources=[(0, 2), (1, 2)])
    merged = _merge_instances([a, b], gap=0.12)
    assert len(merged) == 2


def test_visibility_asymmetry_does_not_destroy_a_neighbour():
    """Point count tracks how many frames saw an object, not how big it is.

    A rule keyed on relative point count merges a rarely-seen chair into its
    well-seen neighbour; provenance is immune to that.
    """
    seen_often = instance(
        0, "chair", block((0.0, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=5000, seed=1),
        sources=[(f, 1) for f in range(6)],
    )
    seen_once = instance(
        1, "chair", block((0.58, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=400, seed=2),
        sources=[(0, 2)],
    )
    assert len(_merge_instances([seen_often, seen_once], gap=0.12)) == 2


def test_merging_does_not_cascade_along_a_row():
    """Absorbing must not grow one instance into its distant neighbours."""
    stools = [
        instance(
            i, "stool",
            block((0.6 * i, 0.4, 0.0), size=(0.5, 0.8, 0.5), n=n, seed=i + 1),
            sources=[(f, i) for f in range(3)],
        )
        for i, n in enumerate([1000, 400, 390, 380, 370])
    ]
    merged = _merge_instances(stools, gap=0.12)
    assert len(merged) == 5
    widths = [float(m.aabb[1][0] - m.aabb[0][0]) for m in merged]
    assert max(widths) < 0.6, widths


def test_distant_sibling_is_not_absorbed():
    body = instance(0, "sofa", block((1.0, 0.3, 2.0), size=(1.4, 0.6, 1.2), n=2000, seed=1),
                    sources=[(0, 7)])
    far = instance(1, "sofa", block((6.0, 0.3, 2.0), size=(0.3, 0.2, 0.3), n=150, seed=2),
                   sources=[(0, 7)])
    assert len(_merge_instances([body, far], gap=0.12)) == 2


def test_conflicts_detects_same_frame_different_segments():
    assert _conflicts([(0, 1)], [(0, 2)]) is True
    assert _conflicts([(0, 1)], [(0, 1)]) is False
    assert _conflicts([(0, 1)], [(1, 2)]) is False
    assert _conflicts([(0, 1), (1, 1)], [(2, 5), (1, 9)]) is True


def test_planar_object_duplicates_are_still_comparable():
    """A painting has zero thickness; IoU must not collapse to zero."""
    flat = np.column_stack([
        np.random.default_rng(3).uniform(0, 1.2, 800),
        np.random.default_rng(4).uniform(0, 0.8, 800),
        np.zeros(800),
    ]).astype(np.float32)
    a = instance(0, "painting", flat, sources=[(0, 3)])
    b = instance(1, "painting", flat.copy(), sources=[(4, 3)])
    assert _aabb_iou(a, b) > 0.9
    assert len(_merge_instances([a, b], gap=0.12)) == 1


# --------------------------------------------------------------------------
# density-aware clustering scale
# --------------------------------------------------------------------------

def test_median_spacing_matches_a_known_grid():
    grid = np.stack(
        np.meshgrid(*[np.arange(10) * 0.05] * 3, indexing="ij"), -1
    ).reshape(-1, 3).astype(np.float32)
    assert median_spacing(grid) == pytest.approx(0.05, abs=1e-4)


def test_adaptive_voxel_never_shrinks_below_the_base():
    dense = block((0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2), n=4000)
    assert adaptive_voxel(dense, base=0.12) == pytest.approx(0.12)


def test_adaptive_voxel_grows_with_sparse_sampling():
    sparse = np.stack(
        np.meshgrid(*[np.arange(6) * 0.4] * 3, indexing="ij"), -1
    ).reshape(-1, 3).astype(np.float32)
    assert adaptive_voxel(sparse, base=0.12) > 0.5


def test_median_spacing_handles_degenerate_input():
    assert median_spacing(np.zeros((1, 3), np.float32)) == 0.0
    assert median_spacing(np.zeros((0, 3), np.float32)) == 0.0


# --------------------------------------------------------------------------
# rigid transform helper
# --------------------------------------------------------------------------

def test_invert_rigid_is_a_true_inverse():
    rng = np.random.default_rng(0)
    for _ in range(10):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        transform = np.eye(4)
        transform[:3, :3] = q
        transform[:3, 3] = rng.normal(size=3)
        assert np.allclose(invert_rigid(transform) @ transform, np.eye(4), atol=1e-10)


def test_association_threshold_is_enforced():
    """Provenance cannot separate objects that were never co-visible.

    Two same-label objects seen in disjoint frame sets carry no conflicting
    evidence, so nothing but the overlap threshold stops them being fused.
    """
    cfg = PipelineConfig(association_iou=0.35)
    # The two must *touch* slightly: a pair with no overlap at all never
    # becomes a candidate, so it would stay separate whether or not the
    # threshold is applied, and would not test anything.
    near = block((0.00, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=1500, seed=1)
    far = block((0.38, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=1500, seed=3)
    overlap = voxel_overlap(near, far, PipelineConfig().voxel_size * 3.0)
    assert 0.0 < overlap < cfg.association_iou, overlap   # the case must be live

    detections = [
        detection(0, "chair", near, segment_id=1),
        detection(1, "chair", near + 0.005, segment_id=1),
        detection(2, "chair", far, segment_id=1),
        detection(3, "chair", far + 0.005, segment_id=1),
    ]
    instances = _associate(detections, cfg)
    assert len(instances) == 2, [
        (i.frame_indices, i.centroid.round(2).tolist()) for i in instances
    ]


def test_association_threshold_admits_a_genuine_match():
    """The same threshold must not reject a real overlap."""
    cfg = PipelineConfig(association_iou=0.15)
    base = block((1.0, 0.5, 2.0), n=1500)
    detections = [
        detection(0, "chair", base, segment_id=1),
        detection(1, "chair", base + 0.02, segment_id=1),
    ]
    assert len(_associate(detections, cfg)) == 1


def test_association_is_robust_to_input_order():
    """A drifting object must fuse regardless of the order detections arrive.

    Consecutive views overlap well while the first and last barely do, so an
    implementation that consumed detections as given would split the object
    when handed them scrambled.  (Ascending and descending frame order are
    equivalent -- both walk the chain -- so only a scrambled order tests this.)
    """
    cfg = PipelineConfig(association_iou=0.35)
    detections = [
        detection(i, "sofa", block((0.35 * i, 0.5, 2.0), size=(0.8, 0.5, 0.5), n=1200,
                                   seed=i + 1), segment_id=1)
        for i in range(6)
    ]
    ends = voxel_overlap(detections[0].points, detections[-1].points,
                         cfg.voxel_size * 3.0)
    assert ends < cfg.association_iou, ends   # the ends really are far apart

    for order in ([0, 5, 1, 4, 2, 3], [3, 0, 5, 2, 1, 4], [5, 4, 3, 2, 1, 0]):
        scrambled = [detections[i] for i in order]
        assert len(_associate(scrambled, cfg)) == 1, order


def test_object_aabb_is_exactly_the_point_extent():
    """`aabb` must report the measured extent, not a padded or scaled one.

    Every dimension in `scene.json` derives from this, and end-to-end
    tolerances are metres-scale, so a few percent of inflation here would pass
    every downstream assertion while quietly overstating every object.
    """
    points = np.array(
        [[-1.0, 2.0, 0.5], [3.0, -4.0, 0.5], [0.0, 0.0, 7.0]], np.float32
    )
    inst = instance(0, "x", points)
    lo, hi = inst.aabb
    assert lo.tolist() == [-1.0, -4.0, 0.5]
    assert hi.tolist() == [3.0, 2.0, 7.0]
    assert np.allclose(inst.centroid, points.mean(axis=0))
