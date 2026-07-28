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
    _containment,
    _merge_instances,
    _voxel_set,
    fuse,
)
from roomviz.geometry.pointcloud import (
    adaptive_voxel,
    median_spacing,
    voxel_overlap,
)
from roomviz.perception.labels import THING_CLASSES, split_policy
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
    voxel = PipelineConfig().voxel_size * 3.0
    overlap = voxel_overlap(near, far, voxel)
    assert 0.0 < overlap < cfg.association_iou, overlap   # the case must be live
    # These two are the same size, so the fragment-size guard `_associate`
    # applies is inactive and the score it sees is exactly `voxel_overlap`.
    assert _containment(_voxel_set(near, voxel), _voxel_set(far, voxel)) == (
        pytest.approx(overlap)
    )

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


# --------------------------------------------------------------------------
# splitting one mask into the objects it holds
#
# The scenes below are rendered rather than hand-built because the question
# they ask - "is the space between these two pieces empty, or is something
# standing in it?" - is only meaningful for a real view of a real room.  Ground
# truth poses are used throughout: odometry is tested elsewhere and would only
# add noise to the measurement.
# --------------------------------------------------------------------------

SPLIT_WIDTH, SPLIT_HEIGHT, SPLIT_HFOV = 224, 168, 95.0


def _split_config() -> PipelineConfig:
    return PipelineConfig(
        voxel_size=0.03,
        min_object_points=120,
        plane_min_inliers=600,
        max_side=max(SPLIT_WIDTH, SPLIT_HEIGHT),
    )


def _observations(room, poses, shared_ids, labels, thing_flags, cfg):
    """Render a room and package it as Observations.

    ``shared_ids`` maps a rendered box's segment id onto the id the segmenter
    is pretending to report, so several boxes can arrive in one mask.
    """
    from synthetic import render

    from roomviz.perception.labels import classify
    from roomviz.types import (
        CameraIntrinsics,
        DepthMap,
        Frame,
        Observation,
        Segment,
        Segmentation,
    )

    intr = CameraIntrinsics.from_hfov(SPLIT_WIDTH, SPLIT_HEIGHT, SPLIT_HFOV)
    observations = []
    for i, pose in enumerate(poses):
        rgb, depth, ids = render(room, pose, intr)
        ids = ids.copy()
        for source, target in shared_ids.items():
            ids[ids == source] = target
        segments = []
        for sid in np.unique(ids):
            sid = int(sid)
            if sid < 0:
                continue
            role, kind = classify(labels[sid])
            segments.append(
                Segment(
                    segment_id=sid,
                    label=labels[sid],
                    role=role,
                    structure_kind=kind,
                    is_thing=thing_flags.get(sid),
                )
            )
        observations.append(
            Observation(
                frame=Frame(index=i, rgb=rgb, timestamp=float(i), source="test"),
                depth=DepthMap(depth=depth),
                segmentation=Segmentation(ids=ids.astype(np.int32), segments=segments),
                intrinsics=intr,
                pose=np.linalg.inv(poses[0]) @ pose,
            )
        )
    return observations


def _wall_poses(count=5):
    from synthetic import look_at

    poses = []
    for i in range(count):
        s = i / max(1, count - 1)
        poses.append(
            look_at(
                np.array([1.6 + 1.6 * s, 1.5 + 0.05 * np.sin(s * 5.0), 0.3 + 0.2 * s]),
                np.array([2.4 + 0.3 * (s - 0.5), 1.4, 3.9]),
            )
        )
    return poses


def _shell_labels(extra):
    labels = {1: "floor", 2: "ceiling", 3: "wall", 4: "wall", 5: "wall", 6: "wall"}
    labels.update(extra)
    return labels


def _extents(scene, label):
    return sorted(
        (float(o.aabb[1][0] - o.aabb[0][0]) for o in scene.objects if o.label == label),
        reverse=True,
    )


@pytest.mark.parametrize("flag", [True, None, False], ids=["thing", "unknown", "stuff"])
def test_three_paintings_in_one_mask_stay_three_objects(flag):
    """The motivating example: one mask, three objects, whatever the label says.

    Splitting used to be gated on a hand-typed word list, and `painting` is in
    it, so all three came back as a single 3.59 m object.  The answer must not
    depend on that list, so the same scene is run with the segmenter calling
    the mask a thing, calling it stuff, and saying nothing at all.
    """
    from synthetic import FIRST_OBJECT_ID, Box, Room

    spans = [(0.60, 1.50), (1.95, 2.85), (3.30, 4.20)]
    room = Room(
        width=5.0, height=2.7, depth=4.0,
        boxes=[
            Box(np.array([x0, 1.15, 3.88]), np.array([x1, 1.90, 3.99]),
                "painting", (60, 70, 90))
            for x0, x1 in spans
        ],
    )
    shared = FIRST_OBJECT_ID
    cfg = _split_config()
    scene = fuse(
        _observations(
            room,
            _wall_poses(),
            {FIRST_OBJECT_ID + i: shared for i in range(3)},
            _shell_labels({shared: "painting"}),
            {shared: flag},
            cfg,
        ),
        cfg,
    )
    widths = _extents(scene, "painting")
    assert len(widths) == 3, widths
    # Each painting is 0.90 m wide and the three together span 3.60 m, so a
    # merged pair or triple could not pass this.  Measured spread on this
    # harness is 0.89-0.90 m.
    assert max(widths) < 1.1, widths


def test_an_unknown_class_is_not_shattered_by_its_own_folds():
    """A pleated curtain is one object arriving as several 3D components.

    `curtain` is in no list, so it used to be split with no restraint at all
    and came back as eight separate objects.  An unknown class has to sit
    between the two extremes: split, but not at the first hole in the sampling.
    """
    from synthetic import FIRST_OBJECT_ID, Box, Room

    boxes, x = [], 1.00
    for _ in range(8):
        boxes.append(Box(np.array([x, 0.30, 3.80]), np.array([x + 0.22, 2.35, 3.97]),
                         "curtain", (140, 120, 160)))
        x += 0.22 + 0.14
    shared = FIRST_OBJECT_ID
    cfg = _split_config()
    scene = fuse(
        _observations(
            Room(width=5.0, height=2.7, depth=4.0, boxes=boxes),
            _wall_poses(),
            {FIRST_OBJECT_ID + i: shared for i in range(len(boxes))},
            _shell_labels({shared: "curtain"}),
            {shared: None},   # the segmenter said nothing about this class
            cfg,
        ),
        cfg,
    )
    widths = _extents(scene, "curtain")
    assert len(widths) == 1, widths


def test_an_occluder_does_not_cut_one_object_into_two():
    """A bookcase 1.5 m in front of a sofa hides a 0.9 m band of it.

    No camera position in this sweep ever sees that band, so the sofa arrives
    as two patches nearly a metre apart - further apart than the paintings
    above, which *are* separate objects.  Distance alone cannot tell the two
    cases apart; what does is that the sofa's gap has the bookcase standing in
    it and the paintings' gap has only the wall behind them.
    """
    from synthetic import FIRST_OBJECT_ID, Box, Room, look_at

    room = Room(
        width=5.0, height=2.7, depth=4.0,
        boxes=[
            Box(np.array([0.60, 0.0, 3.00]), np.array([4.20, 0.72, 3.80]),
                "sofa", (170, 90, 110)),
            Box(np.array([2.15, 0.0, 1.60]), np.array([2.75, 1.85, 2.20]),
                "bookcase", (120, 150, 110)),
        ],
    )
    poses = [
        look_at(
            np.array([2.2 + 0.5 * (i / 4 - 0.5), 1.25, 0.30 + 0.10 * (i / 4)]),
            np.array([2.45, 0.6, 3.4]),
        )
        for i in range(5)
    ]
    cfg = _split_config()
    scene = fuse(
        _observations(
            room, poses, {},
            _shell_labels({FIRST_OBJECT_ID: "sofa", FIRST_OBJECT_ID + 1: "bookcase"}),
            {FIRST_OBJECT_ID: True, FIRST_OBJECT_ID + 1: True},
            cfg,
        ),
        cfg,
    )
    widths = _extents(scene, "sofa")
    assert len(widths) == 1, widths
    # True width 3.60 m; measured 3.56 m on this harness.  A tolerance loose
    # enough to pass either half alone (1.6 m and 1.1 m) would assert nothing.
    assert widths[0] == pytest.approx(3.60, abs=0.15), widths


def test_a_dozen_objects_in_one_mask_all_survive():
    """The case the 3D split exists for; splitting must not have got shyer."""
    from synthetic import FIRST_OBJECT_ID, Box, Room, look_at

    boxes = []
    for i in range(13):
        column, row = i % 5, i // 5
        lo = np.array([0.45 + column * 0.90, 0.0, 1.30 + row * 0.85])
        boxes.append(Box(lo, lo + np.array([0.34, 0.45, 0.34]), "clutter", (120, 140, 120)))
    shared = FIRST_OBJECT_ID
    poses = [
        look_at(np.array([1.4 + 2.0 * (i / 4), 2.05, 0.35]), np.array([2.5, 0.2, 2.3]))
        for i in range(5)
    ]
    cfg = _split_config()
    scene = fuse(
        _observations(
            Room(width=5.0, height=2.7, depth=4.0, boxes=boxes),
            poses,
            {FIRST_OBJECT_ID + i: shared for i in range(13)},
            _shell_labels({shared: "clutter"}),
            {shared: False},
            cfg,
        ),
        cfg,
    )
    widths = _extents(scene, "clutter")
    assert len(widths) == 13, widths
    assert max(widths) < 0.5, widths   # 0.34 m boxes; a merged pair spans 1.2 m


def test_split_policy_cannot_veto_a_split():
    """Membership of the word list may only move the threshold, not gate it.

    A curated list of a hundred names is a list someone will forget to update,
    so nothing that decides object counts may branch on it.
    """
    listed = split_policy("painting", None)
    unlisted = split_policy("kumquat stand", None)
    for policy in (listed, unlisted):
        assert np.isfinite(policy.gap_scale) and policy.gap_scale > 0
    # Being in the list buys reluctance, and only a bounded amount of it.
    assert 1.0 <= unlisted.gap_scale <= listed.gap_scale <= 3.0 * unlisted.gap_scale
    assert "painting" in THING_CLASSES and "kumquat stand" not in THING_CLASSES
    # The segmenter's own flag outranks the list in both directions.
    assert split_policy("painting", False).gap_scale < listed.gap_scale
    assert split_policy("kumquat stand", True).gap_scale > unlisted.gap_scale


# --------------------------------------------------------------------------
# merging: what the absence of evidence is worth
# --------------------------------------------------------------------------

def test_neighbouring_objects_survive_a_dropped_mask():
    """Two chairs 6 cm apart, seen in disjoint halves of the sequence.

    No frame saw both, so no frame contradicts merging them - and proximity
    plus the absence of a contradiction used to fuse them into one 1.06 m
    object.  Absence of evidence is not evidence: a merge needs a frame that
    positively saw the two under *one* segment id, and there is none.
    """
    a = instance(0, "chair", block((0.0, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=1200, seed=1),
                 sources=[(f, 1) for f in range(3)])
    b = instance(1, "chair", block((0.56, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=1100, seed=2),
                 sources=[(f, 2) for f in range(3, 6)])
    merged = _merge_instances([a, b], gap=0.12)
    assert len(merged) == 2, [float(m.aabb[1][0] - m.aabb[0][0]) for m in merged]


def test_a_row_of_stools_survives_a_camera_pan():
    """Five stools, each caught in its own pair of frames as the camera pans.

    Neighbours are 10 cm apart, so proximity alone chains the whole row
    together: the row used to come back as two objects of 1.70 m and 1.09 m.
    """
    stools = [
        instance(
            i, "stool",
            block((0.6 * i, 0.4, 0.0), size=(0.5, 0.8, 0.5), n=n, seed=i + 1),
            sources=[(2 * i, i), (2 * i + 1, i)],
        )
        for i, n in enumerate([1000, 400, 390, 380, 370])
    ]
    merged = _merge_instances(stools, gap=0.12)
    assert len(merged) == 5, [float(m.aabb[1][0] - m.aabb[0][0]) for m in merged]
    assert max(float(m.aabb[1][0] - m.aabb[0][0]) for m in merged) < 0.6


def test_pieces_of_one_mask_stay_apart_when_they_are_far_apart():
    """Sharing a mask licenses a merge; it does not compel one.

    Thirteen objects that arrived in a single stuff mask all carry the same
    provenance, so only the distance between them keeps them apart.
    """
    pieces = [
        instance(i, "clutter",
                 block((0.5 * i, 0.2, 0.0), size=(0.3, 0.4, 0.3), n=500, seed=i + 1),
                 sources=[(f, 9) for f in range(4)])
        for i in range(13)
    ]
    assert len(_merge_instances(pieces, gap=0.12)) == 13


def _random_instances(seed: int) -> list[ObjectInstance]:
    """A random scene that exercises every branch of the merge predicate."""
    rng = np.random.default_rng(seed)
    instances = []
    for i in range(int(rng.integers(4, 10))):
        centre = rng.uniform(-0.8, 0.8, 3)
        size = rng.uniform(0.2, 0.7, 3)
        # Ties in the point count are deliberate: the old implementation sorted
        # by count, so ties were exactly where list order leaked into the result.
        n = 600 if rng.random() < 0.5 else int(rng.integers(200, 2000))
        points = (rng.uniform(-size / 2, size / 2, (n, 3)) + centre).astype(np.float32)
        frames = sorted(
            rng.choice(4, size=int(rng.integers(1, 4)), replace=False).tolist()
        )
        instances.append(
            instance(i, "x", points, sources=[(int(f), int(rng.integers(0, 3)))
                                              for f in frames])
        )
    return instances


def _partition(originals, merged) -> frozenset:
    """The merge result as a set of frozensets of original indices."""
    lookup = {tuple(o.points[0].tolist()): i for i, o in enumerate(originals)}
    return frozenset(
        frozenset(lookup[t] for t in (tuple(p) for p in m.points.tolist())
                  if t in lookup)
        for m in merged
    )


def _clone(instances):
    return [
        ObjectInstance(
            instance_id=o.instance_id, label=o.label, points=o.points.copy(),
            colors=o.colors.copy(), frame_indices=list(o.frame_indices),
            sources=list(o.sources), split_gap=o.split_gap,
        )
        for o in instances
    ]


def test_merge_partition_is_invariant_to_input_order():
    """The partition is a function of the instances, not of the list.

    The docstring of `_merge_instances` claimed order-independence while the
    implementation absorbed into a running instance, so each merge changed the
    bounding box and the provenance the *next* comparison saw.  Measured
    against that implementation, this generator produces an order-dependent
    partition in 28 of 400 scenes - 4 of them inside the 60 seeds below, which
    is why 60 is enough here.  A wider sweep belongs in a benchmark, not in a
    unit test that runs on every commit.
    """
    scenes = 0
    for seed in range(60):
        base = _random_instances(seed)
        reference = _partition(base, _merge_instances(_clone(base), gap=0.12))
        if len(reference) < len(base):
            scenes += 1          # this scene actually merged something
        for permutation in range(4):
            order = list(range(len(base)))
            np.random.default_rng(seed * 131 + permutation).shuffle(order)
            shuffled = _clone([base[i] for i in order])
            assert _partition(base, _merge_instances(shuffled, gap=0.12)) == reference, (
                seed, permutation, sorted(map(sorted, reference))
            )
    # If nothing ever merged, the invariance above would be vacuous.
    assert scenes >= 20, scenes


def test_a_sliver_does_not_outscore_a_real_match():
    """Intersection-over-minimum saturates; a size guard is what stops it.

    A handful of voxels fully inside a large instance scores a perfect 1.0 -
    the top of the scale, beating any partial match anywhere else - so
    `_associate` hands every stray fragment to whichever object contains it,
    however great the size disparity.
    """
    voxel = 0.075
    body = _voxel_set(block((0.0, 0.0, 0.0), size=(1.6, 0.8, 0.8), n=8000, seed=1), voxel)
    # Centred inside a single voxel so that the sliver really is contained.
    sliver = _voxel_set(
        block((voxel / 2, voxel / 2, voxel / 2), size=(0.02, 0.02, 0.02), n=60, seed=2),
        voxel,
    )
    assert sliver < body                       # fully contained, so IoU-min == 1
    assert len(sliver) < 0.05 * len(body)
    assert _containment(sliver, body) < 0.5
    # Without the guard this is exactly 1.0 -- the saturation being complained of.
    assert _containment(sliver, body, min_fragment_ratio=0.0) == pytest.approx(1.0)

    # A candidate that covers enough of the model is scored exactly as before.
    half = _voxel_set(block((-0.4, 0.0, 0.0), size=(0.8, 0.8, 0.8), n=4000, seed=3), voxel)
    assert len(half) > 0.05 * len(body)
    assert _containment(half, body) == pytest.approx(
        _containment(half, body, min_fragment_ratio=0.0)
    )


def test_merging_sixty_instances_is_not_quadratic_in_points():
    """Merging used to run a KD-tree query per pair over the raw clouds.

    The instances below all share one mask and all sit close enough that the
    bounding-box prefilter cannot discard the pair, so every one of the 1770
    pairs reaches the proximity test - the worst case, not a lucky one.  A
    KD-tree over the raw clouds takes 21 s on this input; the voxel probe takes
    1.1 s.  The bound is set between the two.
    """
    import time

    rng = np.random.default_rng(7)
    instances = [
        instance(i, "x",
                 (rng.uniform(-0.3, 0.3, (12000, 3))
                  + rng.uniform(-2.0, 2.0, 3)).astype(np.float32),
                 sources=[(0, 5)])
        for i in range(60)
    ]
    start = time.perf_counter()
    _merge_instances(instances, gap=0.12)
    assert time.perf_counter() - start < 8.0


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
