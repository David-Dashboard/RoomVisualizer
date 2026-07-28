"""Unit tests for the geometry primitives."""

from __future__ import annotations

import numpy as np
import pytest

from roomviz.geometry.align import (
    alignment_transform,
    apply_transform,
    rotation_between,
)
from roomviz.geometry.camera import backproject, depth_edge_mask, project
from roomviz.geometry.planes import extract_surfaces, fit_plane_lsq, plane_quad, ransac_plane
from roomviz.geometry.pointcloud import (
    cluster_connected,
    largest_clusters,
    voxel_downsample,
    voxel_iou,
)
from roomviz.types import CameraIntrinsics, DepthMap


def test_backproject_project_roundtrip():
    intr = CameraIntrinsics.from_hfov(64, 48, 60.0)
    rng = np.random.default_rng(0)
    depth = DepthMap(depth=rng.uniform(1.0, 4.0, (48, 64)).astype(np.float32))

    points, idx = backproject(depth, intr)
    assert points.shape[0] == 48 * 64
    uv, z = project(points, intr)

    expected_v, expected_u = np.divmod(idx, 64)
    assert np.allclose(uv[:, 0], expected_u, atol=1e-3)
    assert np.allclose(uv[:, 1], expected_v, atol=1e-3)
    assert np.allclose(z, depth.depth.reshape(-1)[idx], atol=1e-5)


def test_backproject_respects_invalid_depth():
    intr = CameraIntrinsics.from_hfov(8, 8, 60.0)
    depth = np.ones((8, 8), np.float32)
    depth[0, :] = 0.0  # invalid row
    points, idx = backproject(DepthMap(depth=depth), intr)
    assert points.shape[0] == 56
    assert (idx >= 8).all()


def test_intrinsics_scaling_preserves_ray_directions():
    intr = CameraIntrinsics.from_hfov(640, 480, 65.0)
    small = intr.scaled_to(320, 240)
    # A pixel at the same relative position must unproject along the same ray.
    for u, v in [(0, 0), (640, 480), (100, 350)]:
        big_ray = ((u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy)
        small_ray = ((u / 2 - small.cx) / small.fx, (v / 2 - small.cy) / small.fy)
        assert big_ray == pytest.approx(small_ray, abs=1e-9)


def test_depth_edge_mask_flags_discontinuities():
    depth = np.full((16, 16), 2.0, np.float32)
    depth[:, 8:] = 4.0  # a step edge down the middle
    keep = depth_edge_mask(DepthMap(depth=depth), 0.06)
    assert not keep[:, 7].any() and not keep[:, 8].any()
    assert keep[:, 0].all() and keep[:, 15].all()


def test_depth_edge_mask_disabled():
    depth = np.full((8, 8), 2.0, np.float32)
    depth[:, 4:] = 9.0
    assert depth_edge_mask(DepthMap(depth=depth), 0.0).all()


def test_fit_plane_lsq_recovers_known_plane():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-2, 2, (500, 2))
    # Plane: z = 0.3x - 0.2y + 1.5
    points = np.stack([xy[:, 0], xy[:, 1], 0.3 * xy[:, 0] - 0.2 * xy[:, 1] + 1.5], 1)
    normal, offset = fit_plane_lsq(points)
    residual = points @ normal + offset
    assert np.abs(residual).max() < 1e-6


def test_ransac_plane_ignores_outliers():
    rng = np.random.default_rng(2)
    inliers = np.column_stack(
        [rng.uniform(-1, 1, 800), rng.uniform(-1, 1, 800), rng.normal(0, 0.005, 800) + 2.0]
    )
    outliers = rng.uniform(-3, 3, (300, 3))
    points = np.vstack([inliers, outliers]).astype(np.float32)

    normal, offset, mask = ransac_plane(points, threshold=0.03, rng=rng)
    assert abs(abs(float(normal @ np.array([0.0, 0.0, 1.0]))) - 1.0) < 0.02
    assert abs(abs(offset) - 2.0) < 0.02
    # Every true inlier should be recovered, without swallowing all the noise.
    assert mask[:800].sum() > 780
    assert mask[800:].sum() < 120


def test_plane_quad_spans_the_points():
    rng = np.random.default_rng(3)
    pts = np.column_stack(
        [rng.uniform(0, 3, 2000), rng.uniform(0, 2, 2000), np.full(2000, 1.0)]
    )
    quad, area = plane_quad(pts, np.array([0.0, 0.0, 1.0]), -1.0, percentile=0.0)
    assert area == pytest.approx(6.0, rel=0.05)
    assert np.allclose(quad[:, 2], 1.0, atol=1e-5)


def test_rotation_between_handles_antiparallel():
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, -1.0, 0.0])
    rotation = rotation_between(a, b)
    assert np.allclose(rotation @ a, b, atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


def test_rotation_between_is_a_rotation():
    rng = np.random.default_rng(4)
    for _ in range(20):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        rotation = rotation_between(a, b)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(
            rotation @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9
        )


def test_voxel_downsample_averages_within_cells():
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 5.0, 5.0]], np.float32)
    colors = np.array([[0, 0, 0], [100, 100, 100], [255, 255, 255]], np.uint8)
    out_points, out_colors, inverse = voxel_downsample(points, colors, 0.1)
    assert out_points.shape[0] == 2
    assert inverse[0] == inverse[1] and inverse[0] != inverse[2]
    merged = out_points[inverse[0]]
    assert merged[0] == pytest.approx(0.005, abs=1e-6)
    assert out_colors[inverse[0]][0] == 50


def test_cluster_connected_separates_distant_groups():
    a = np.random.default_rng(5).uniform(0, 0.2, (200, 3))
    b = a + np.array([5.0, 0.0, 0.0])
    labels = cluster_connected(np.vstack([a, b]).astype(np.float32), 0.05)
    assert len(np.unique(labels)) == 2
    assert len(np.unique(labels[:200])) == 1
    assert labels[0] != labels[-1]


def test_largest_clusters_drops_noise():
    rng = np.random.default_rng(6)
    main = rng.uniform(0, 0.3, (500, 3))
    speck = np.array([[9.0, 9.0, 9.0]])
    clusters = largest_clusters(np.vstack([main, speck]).astype(np.float32), 0.05)
    assert len(clusters) == 1
    assert clusters[0].size == 500


def test_voxel_iou_bounds():
    a = np.random.default_rng(7).uniform(0, 1, (500, 3)).astype(np.float32)
    assert voxel_iou(a, a, 0.05) == pytest.approx(1.0)
    assert voxel_iou(a, a + 10.0, 0.05) == 0.0


def test_alignment_puts_floor_at_origin_and_up_on_y():
    rng = np.random.default_rng(8)
    # A floor and a wall expressed in a tilted frame.
    floor = np.column_stack(
        [rng.uniform(-2, 2, 3000), np.zeros(3000), rng.uniform(-2, 2, 3000)]
    )
    wall = np.column_stack(
        [np.full(3000, -2.0), rng.uniform(0, 2.5, 3000), rng.uniform(-2, 2, 3000)]
    )
    points = np.vstack([floor, wall]).astype(np.float32)
    kinds = np.array(["floor"] * 3000 + ["wall"] * 3000, dtype=object)

    tilt = rotation_between(np.array([0.0, 1.0, 0.0]), np.array([0.2, 0.95, 0.1]))
    tilted = (points @ tilt.T).astype(np.float32)

    transform = alignment_transform(tilted, kinds, camera_up=tilt @ np.array([0.0, 1.0, 0.0]))
    out = apply_transform(transform, tilted)

    assert np.abs(out[:3000, 1]).max() < 0.02  # floor flattened onto y = 0
    assert out[3000:, 1].min() > -0.05  # wall sits above the floor
    # The wall should be axis-aligned after the Manhattan step.
    wall_x = out[3000:, 0]
    assert wall_x.std() < 0.05


def test_extract_surfaces_finds_room_shell():
    rng = np.random.default_rng(9)
    n = 4000

    def slab(fixed_axis, value, a_range, b_range):
        pts = np.zeros((n, 3))
        axes = [i for i in range(3) if i != fixed_axis]
        pts[:, fixed_axis] = value + rng.normal(0, 0.004, n)
        pts[:, axes[0]] = rng.uniform(*a_range, n)
        pts[:, axes[1]] = rng.uniform(*b_range, n)
        return pts

    floor = slab(1, 0.0, (0, 4), (0, 3))
    ceiling = slab(1, 2.6, (0, 4), (0, 3))
    wall_a = slab(0, 0.0, (0, 2.6), (0, 3))
    wall_b = slab(2, 3.0, (0, 4), (0, 2.6))
    points = np.vstack([floor, ceiling, wall_a, wall_b]).astype(np.float32)
    kinds = ["floor"] * n + ["ceiling"] * n + ["wall"] * 2 * n

    surfaces, assignment = extract_surfaces(
        points, up=np.array([0.0, 1.0, 0.0]), labels=kinds,
        threshold=0.02, min_inliers=500, max_planes=8,
    )
    kinds_found = sorted(s.kind for s in surfaces)
    assert kinds_found == ["ceiling", "floor", "wall", "wall"]

    ceiling_surface = next(s for s in surfaces if s.kind == "ceiling")
    assert abs(abs(ceiling_surface.offset) - 2.6) < 0.02
    assert ceiling_surface.area == pytest.approx(12.0, rel=0.05)
    assert (assignment >= 0).mean() > 0.95
