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
    _merge_duplicates,
)
from roomviz.geometry.pointcloud import adaptive_voxel, median_spacing
from roomviz.types import ObjectInstance


def block(centre, size=(0.4, 0.4, 0.4), n=600, seed=0):
    rng = np.random.default_rng(seed)
    half = np.array(size) / 2
    return (rng.uniform(-half, half, (n, 3)) + np.array(centre)).astype(np.float32)


def detection(frame, label, points, seed=0):
    colors = np.full((points.shape[0], 3), 128, np.uint8)
    return FrameDetection(
        frame_index=frame, label=label, score=1.0, points=points, colors=colors
    )


def instance(instance_id, label, points):
    return ObjectInstance(
        instance_id=instance_id,
        label=label,
        points=points,
        colors=np.full((points.shape[0], 3), 128, np.uint8),
        frame_indices=[0],
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
        detection(0, "chair", block((0.0, 0.5, 2.0), seed=1)),
        detection(0, "chair", block((3.0, 0.5, 2.0), seed=2)),
        detection(1, "chair", block((0.0, 0.5, 2.0), seed=3)),
        detection(1, "chair", block((3.0, 0.5, 2.0), seed=4)),
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


def test_two_detections_in_one_frame_are_never_merged():
    """The segmenter already decided these are distinct objects."""
    cfg = PipelineConfig(association_iou=0.0)
    points = block((1.0, 0.5, 2.0))
    instances = _associate(
        [detection(0, "chair", points), detection(0, "chair", points + 0.001)], cfg
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


def test_drift_duplicates_are_merged():
    a = instance(0, "sofa", block((1.0, 0.3, 2.0), size=(1.4, 0.6, 1.2), n=2000, seed=1))
    b = instance(1, "sofa", block((1.05, 0.3, 2.05), size=(1.4, 0.6, 1.2), n=1800, seed=2))
    merged = _merge_duplicates([a, b], gap=0.12)
    assert len(merged) == 1
    assert merged[0].points.shape[0] == 3800


def test_touching_fragment_is_absorbed():
    """A small patch adjacent to a much larger same-label body is part of it."""
    body = instance(0, "sofa", block((1.0, 0.3, 2.0), size=(1.4, 0.6, 1.2), n=2000, seed=1))
    fragment = instance(
        1, "sofa", block((1.0, 0.65, 2.0), size=(1.2, 0.06, 1.0), n=200, seed=2)
    )
    merged = _merge_duplicates([body, fragment], gap=0.12)
    assert len(merged) == 1
    assert merged[0].points.shape[0] == 2200


def test_similar_sized_neighbours_are_not_merged():
    """Two chairs pushed together must stay two chairs."""
    a = instance(0, "chair", block((0.0, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=1000, seed=1))
    b = instance(1, "chair", block((0.55, 0.5, 0.0), size=(0.5, 0.9, 0.5), n=950, seed=2))
    merged = _merge_duplicates([a, b], gap=0.12)
    assert len(merged) == 2


def test_distant_fragment_is_not_absorbed():
    body = instance(0, "sofa", block((1.0, 0.3, 2.0), size=(1.4, 0.6, 1.2), n=2000, seed=1))
    far = instance(1, "sofa", block((6.0, 0.3, 2.0), size=(0.3, 0.2, 0.3), n=150, seed=2))
    assert len(_merge_duplicates([body, far], gap=0.12)) == 2


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
