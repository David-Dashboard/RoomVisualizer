"""Unit tests for the geometry primitives."""

from __future__ import annotations

import numpy as np
import pytest

from roomviz.geometry.align import (
    alignment_transform,
    apply_transform,
    estimate_up,
    rotation_between,
)
from roomviz.geometry.camera import backproject, depth_edge_mask, pixel_rays, project
from roomviz.geometry.planes import (
    classify_plane,
    extract_surfaces,
    fit_plane_lsq,
    plane_quad,
    ransac_plane,
)
from roomviz.geometry.pointcloud import (
    cluster_connected,
    largest_clusters,
    voxel_downsample,
    voxel_iou,
    voxel_overlap,
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
    # A point at the same position *in the image rectangle* must unproject
    # along the same ray.  Pixel centres map through edges: u' = (u+0.5)*s-0.5.
    for u, v in [(0, 0), (639, 479), (100, 350)]:
        big_ray = ((u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy)
        us, vs = (u + 0.5) * 0.5 - 0.5, (v + 0.5) * 0.5 - 0.5
        small_ray = ((us - small.cx) / small.fx, (vs - small.cy) / small.fy)
        assert big_ray == pytest.approx(small_ray, abs=1e-9)


def test_principal_point_is_the_image_centre():
    """The optical axis must land midway between the first and last pixel."""
    intr = CameraIntrinsics.from_hfov(64, 48, 70.0)
    rx, ry = pixel_rays(intr)
    # Rays must be symmetric about the centre, and the centre ray must be zero.
    assert rx[0, 0] == pytest.approx(-rx[0, -1], abs=1e-12)
    assert ry[0, 0] == pytest.approx(-ry[-1, 0], abs=1e-12)
    # The realised field of view spans the full image rectangle.
    half = np.arctan((intr.width / 2.0) / intr.fx)
    assert np.rad2deg(2 * half) == pytest.approx(70.0, abs=1e-9)


def test_non_square_pixels_are_honoured():
    """fx and fy must be used independently, not assumed equal.

    Every scene built by `from_hfov` has fx == fy, so a swap of the two is
    invisible there; a resize that changes the aspect ratio makes them differ.
    """
    intr = CameraIntrinsics(width=64, height=48, fx=100.0, fy=50.0, cx=31.5, cy=23.5)
    depth = DepthMap(depth=np.full((48, 64), 2.0, np.float32))
    points, idx = backproject(depth, intr)
    uv, _ = project(points, intr)
    expected_v, expected_u = np.divmod(idx, 64)
    assert np.allclose(uv[:, 0], expected_u, atol=1e-6)
    assert np.allclose(uv[:, 1], expected_v, atol=1e-6)
    # A point off-centre in x must not have the same offset as one off-centre
    # in y by the same pixel count -- that is what a swapped fx/fy would give.
    corner = points[idx == (10 * 64 + 10)][0]
    assert abs(corner[0]) == pytest.approx(2.0 * 21.5 / 100.0, abs=1e-6)
    assert abs(corner[1]) == pytest.approx(2.0 * 13.5 / 50.0, abs=1e-6)


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


def test_rotation_between_survives_the_near_antiparallel_band():
    """Rodrigues is ill-conditioned just short of 180 degrees.

    This is the level-camera case: the up axis fitted from a floor plane is
    almost exactly antiparallel to world up.  Getting it wrong here returns a
    near-identity matrix instead of a flip, and the whole room comes out
    upside down -- so the entire band is swept, not just the exact endpoint.
    """
    up = np.array([0.0, 1.0, 0.0])
    for tilt_deg in (3.0, 0.5, 0.05, 0.01, 1e-3, 1e-5, 1e-9, 0.0):
        t = np.deg2rad(tilt_deg)
        # float32 as well as float64: `fit_plane_lsq` on a float32 cloud
        # returns a float32 normal, so float32 is what the pipeline actually
        # feeds this function -- and it is the dtype in which `1 + cos` is
        # unresolvable, so testing only float64 hides the failure entirely.
        for dtype in (np.float64, np.float32):
            down = np.array([np.sin(t), -np.cos(t), 0.0], dtype=dtype)
            rotation = rotation_between(down, up.astype(dtype))

            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9), (tilt_deg, dtype)
            assert np.abs(rotation @ rotation.T - np.eye(3)).max() < 1e-9, (tilt_deg, dtype)
            # It must actually flip: the result has to point up, not stay down.
            assert float((rotation @ down.astype(np.float64))[1]) > 0.99, (tilt_deg, dtype)
            assert np.linalg.norm(
                rotation @ down.astype(np.float64) - up
            ) < 1e-7, (tilt_deg, dtype)


def test_rotation_between_fuzz_near_antiparallel():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(3000):
        a = rng.normal(size=3)
        # Concentrate samples in the ill-conditioned region, and run each one
        # through both dtypes.
        b = -a + rng.normal(size=3) * 10 ** rng.uniform(-18, -1)
        for dtype in (np.float64, np.float32):
            # Compare against the vectors the function was actually handed.
            # Measuring a float32 call against float64 ground truth measures
            # the input rounding (float32 eps ~ 1.2e-7), not the algorithm.
            given_a = a.astype(dtype).astype(np.float64)
            given_b = b.astype(dtype).astype(np.float64)
            rotation = rotation_between(a.astype(dtype), b.astype(dtype))
            assert np.abs(rotation @ rotation.T - np.eye(3)).max() < 1e-9
            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
            worst = max(
                worst,
                np.linalg.norm(
                    rotation @ (given_a / np.linalg.norm(given_a))
                    - given_b / np.linalg.norm(given_b)
                ),
            )
    # Bounded by the sqrt(eps) cutoff below which a half turn is substituted.
    assert worst < 2e-8, f"worst mapping error {worst:.2e}"


def test_rotation_between_is_scale_and_dtype_invariant():
    """Magnitude and dtype must not change the answer."""
    a = np.array([0.3, -0.9, 0.2])
    b = np.array([-0.4, 0.1, 0.8])
    reference = rotation_between(a, b)
    for scale in (1e-13, 1e-3, 1.0, 1e3, 1e200):
        assert np.allclose(rotation_between(a * scale, b), reference, atol=1e-9), scale
        assert np.allclose(rotation_between(a, b * scale), reference, atol=1e-9), scale
    assert np.allclose(
        rotation_between(a.astype(np.float32), b.astype(np.float32)),
        reference, atol=1e-6,
    )


def test_rotation_between_is_continuous_through_the_band():
    """Two nearly identical inputs must not give wildly different rotations.

    Substituting a half turn about a *fixed* helper axis made the result jump
    by 180 degrees of yaw for a 2e-6 rad change of input.
    """
    a = np.array([0.0, 1.0, 0.0])
    previous = None
    for delta in np.linspace(1.0e-4, 2.0e-4, 25):
        b = -a + np.array([delta, 0.0, 0.0])
        rotation = rotation_between(a, b)
        if previous is not None:
            assert np.linalg.norm(rotation - previous) < 1e-3
        previous = rotation


def test_estimate_up_refuses_rank_deficient_walls():
    """Two opposite walls do not determine up; the prior must win.

    Parallel wall normals span a line, so the null space is a plane and the
    smallest singular vector is arbitrary within it -- previously this returned
    an axis 88 degrees from true up and the floor never got flattened.
    """
    rng = np.random.default_rng(12)
    n = 3000
    wall_a = np.column_stack(
        [np.zeros(n), rng.uniform(0, 2.5, n), rng.uniform(-2, 2, n)]
    )
    wall_b = np.column_stack(
        [np.full(n, 4.0), rng.uniform(0, 2.5, n), rng.uniform(-2, 2, n)]
    )
    points = np.vstack([wall_a, wall_b]).astype(np.float32)
    kinds = np.array(["wall"] * (2 * n), dtype=object)

    prior = np.array([0.0, -1.0, 0.0])
    up = estimate_up(points, kinds, fallback=prior)
    assert np.allclose(up, prior, atol=1e-9)


def test_estimate_up_uses_perpendicular_walls():
    rng = np.random.default_rng(13)
    n = 3000
    wall_a = np.column_stack(
        [np.zeros(n), rng.uniform(0, 2.5, n), rng.uniform(-2, 2, n)]
    )
    wall_b = np.column_stack(
        [np.full(n, 4.0), rng.uniform(0, 2.5, n), rng.uniform(-2, 2, n)]
    )
    wall_c = np.column_stack(
        [rng.uniform(0, 4, n), rng.uniform(0, 2.5, n), np.full(n, -2.0)]
    )
    points = np.vstack([wall_a, wall_b, wall_c]).astype(np.float32)
    kinds = np.array(["wall"] * (3 * n), dtype=object)

    up = estimate_up(points, kinds, fallback=np.array([0.0, -1.0, 0.0]))
    assert abs(abs(float(up @ np.array([0.0, 1.0, 0.0]))) - 1.0) < 0.01


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
    clusters = largest_clusters(
        np.vstack([main, speck]).astype(np.float32), 0.05, min_points=10
    )
    assert len(clusters) == 1
    assert clusters[0].size == 500


def test_largest_clusters_keeps_every_object_however_many_there_are():
    """The noise threshold must be absolute, not a fraction of the input.

    A relative cutoff scales with how many objects share the segment: thirteen
    equal objects each hold 7.7% of it, so an 8% rule discards all of them and
    the fallback collapses the lot into one.  That silently deleted twelve real
    objects, so the count is swept well past the old cliff.
    """
    rng = np.random.default_rng(7)
    for count in (2, 10, 12, 13, 20, 40):
        blobs = [
            rng.uniform(0, 0.2, (300, 3)) + np.array([2.0 * i, 0.0, 0.0])
            for i in range(count)
        ]
        points = np.vstack(blobs).astype(np.float32)
        clusters = largest_clusters(points, voxel=0.05, min_points=50)
        assert len(clusters) == count, f"{count} objects -> {len(clusters)} clusters"
        assert all(c.size == 300 for c in clusters)


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


def _closed_room_points(seed: int = 9, n: int = 4000):
    """A complete 4 x 2.6 x 3 m box: floor, ceiling and all four walls."""
    rng = np.random.default_rng(seed)

    def slab(fixed_axis, value, a_range, b_range):
        pts = np.zeros((n, 3))
        axes = [i for i in range(3) if i != fixed_axis]
        pts[:, fixed_axis] = value + rng.normal(0, 0.004, n)
        pts[:, axes[0]] = rng.uniform(*a_range, n)
        pts[:, axes[1]] = rng.uniform(*b_range, n)
        return pts

    parts = [
        slab(1, 0.0, (0, 4), (0, 3)),
        slab(1, 2.6, (0, 4), (0, 3)),
        slab(0, 0.0, (0, 2.6), (0, 3)),
        slab(0, 4.0, (0, 2.6), (0, 3)),
        slab(2, 0.0, (0, 4), (0, 2.6)),
        slab(2, 3.0, (0, 4), (0, 2.6)),
    ]
    kinds = ["floor"] * n + ["ceiling"] * n + ["wall"] * 4 * n
    return np.vstack(parts).astype(np.float32), kinds, np.array([2.0, 1.3, 1.5])


def test_horizontal_plane_normals_are_oriented_upwards():
    """The half of the orientation rule in `extract_surfaces` that is real.

    Floor and ceiling normals must both point along +up, whichever side of the
    plane the points were seen from.  This is what makes `classify_plane`'s
    height comparison and the exported `normal` field mean anything.
    """
    points, kinds, _ = _closed_room_points()
    surfaces, _ = extract_surfaces(
        points, up=np.array([0.0, 1.0, 0.0]), labels=kinds,
        threshold=0.02, min_inliers=500, max_planes=10,
    )
    horizontals = [s for s in surfaces if s.kind in ("floor", "ceiling")]
    assert len(horizontals) == 2
    for surface in horizontals:
        assert float(surface.normal[1]) > 0.99, (surface.kind, surface.normal)


def test_wall_normals_point_into_the_room():
    """Every wall normal points at the room, not away from it.

    This was a real defect, found by writing the test for a README claim that
    had none.  Both halves of the orientation step were one condition guarded
    by `abs(normal @ up) >= HORIZONTAL_COS`, which is true only for horizontal
    planes, so a wall kept whatever sign the SVD produced: three of the four
    walls below pointed out of the room, and two of three did on the
    end-to-end scene.  A wall has no "up" to agree with, so it is the bulk of
    the scene that decides which side is inside.
    """
    points, kinds, interior = _closed_room_points()
    surfaces, _ = extract_surfaces(
        points, up=np.array([0.0, 1.0, 0.0]), labels=kinds,
        threshold=0.02, min_inliers=500, max_planes=10,
    )
    walls = [s for s in surfaces if s.kind == "wall"]
    assert len(walls) == 4
    for wall in walls:
        signed = float(np.asarray(wall.normal, float) @ interior + wall.offset)
        assert signed > 0, (
            f"wall with normal {np.round(wall.normal, 3)} at offset "
            f"{wall.offset:.3f} points away from the room interior"
        )


def test_voxel_iou_partial_overlap():
    """Half-overlapping sets must score strictly between 0 and 1.

    Testing only the identical and disjoint cases is worthless: those two give
    the same answer under intersection-over-union, intersection-over-minimum
    and intersection-over-maximum alike, so any of them passes.
    """
    grid = np.stack(np.meshgrid(np.arange(10), np.arange(4), np.arange(4), indexing="ij"), -1)
    grid = grid.reshape(-1, 3).astype(np.float32) * 0.1
    left = grid[grid[:, 0] < 0.6]        # 6 slabs
    right = grid[grid[:, 0] >= 0.3]      # 7 slabs, 3 shared
    iou = voxel_iou(left, right, 0.05)
    assert 0.2 < iou < 0.4, iou
    # 3 shared of 10 occupied slabs total.
    assert iou == pytest.approx(3 / 10, rel=0.15)


def test_voxel_overlap_does_not_decay_as_the_model_grows():
    """Intersection-over-minimum is stable; IoU is not.

    A single view matched against an accumulated model must keep scoring well
    as the model grows, or a long object splits partway through a sweep.
    """
    rng = np.random.default_rng(21)
    view = rng.uniform([0.0, 0, 0], [1.0, 0.5, 0.5], (8000, 3)).astype(np.float32)
    scores_iou, scores_overlap = [], []
    for extent in (1.0, 2.0, 4.0, 8.0):
        # Point count scales with extent so the model's *density* is constant;
        # otherwise the model simply gets sparser and that, not the metric,
        # would be what drives the score down.
        model = rng.uniform(
            [0.0, 0, 0], [extent, 0.5, 0.5], (int(8000 * extent), 3)
        ).astype(np.float32)
        scores_iou.append(voxel_iou(view, model, 0.05))
        scores_overlap.append(voxel_overlap(view, model, 0.05))

    assert scores_iou[-1] < 0.4 * scores_iou[0], scores_iou      # IoU collapses
    assert min(scores_overlap) > 0.9, scores_overlap             # overlap holds


def test_merge_similar_folds_a_split_wall():
    """Sequential RANSAC can cut one wall into parallel slabs; they must rejoin."""
    rng = np.random.default_rng(22)
    n = 3000
    # One physical wall at x = 2, recovered as two offset sheets (what pose
    # drift does to a wall seen from both ends of a sweep).  The separation is
    # deliberately wider than the RANSAC threshold -- otherwise a single fit
    # swallows both sheets and the merge path is never reached -- but inside
    # the merge tolerance.
    a = np.column_stack([np.full(n, 2.00) + rng.normal(0, 0.004, n),
                         rng.uniform(0, 2.5, n), rng.uniform(-2, 0, n)])
    b = np.column_stack([np.full(n, 2.07) + rng.normal(0, 0.004, n),
                         rng.uniform(0, 2.5, n), rng.uniform(0, 2, n)])
    floor = np.column_stack([rng.uniform(0, 2, n), np.zeros(n) + rng.normal(0, 0.004, n),
                             rng.uniform(-2, 2, n)])
    points = np.vstack([a, b, floor]).astype(np.float32)
    kinds = ["wall"] * (2 * n) + ["floor"] * n

    surfaces, _ = extract_surfaces(
        points, up=np.array([0.0, 1.0, 0.0]), labels=kinds,
        threshold=0.02, min_inliers=500, max_planes=8,
    )
    walls = [s for s in surfaces if s.kind == "wall"]
    assert len(walls) == 1, [(s.kind, round(s.offset, 3)) for s in surfaces]
    # The merged wall must span both sheets, not just one.
    assert walls[0].inlier_count > 1.5 * n
    assert walls[0].quad[:, 2].max() - walls[0].quad[:, 2].min() > 3.0


def test_classify_plane_without_labels_uses_height():
    """The unlabelled branch exists for when segmentation gives no structure.

    Every end-to-end scene supplies labels, so this fallback is only reachable
    from here -- and swapping its two outcomes puts the ceiling on the floor.
    """
    up = np.array([0.0, 1.0, 0.0])
    horizontal = np.array([0.0, 1.0, 0.0])
    span = (0.0, 2.7)

    assert classify_plane(horizontal, 0.0, up, None, 0.05, span) == "floor"
    assert classify_plane(horizontal, -2.7, up, None, 2.65, span) == "ceiling"
    # An empty label list must behave the same as no labels at all.
    assert classify_plane(horizontal, 0.0, up, [], 0.05, span) == "floor"
    # A vertical plane is a wall regardless of height.
    assert classify_plane(np.array([1.0, 0.0, 0.0]), 0.0, up, None, 1.3, span) == "wall"


def test_classify_plane_labels_break_the_floor_ceiling_tie():
    up = np.array([0.0, 1.0, 0.0])
    horizontal = np.array([0.0, 1.0, 0.0])
    span = (0.0, 2.7)
    # A low plane the segmenter calls ceiling is trusted over the height rule.
    assert classify_plane(horizontal, 0.0, up, ["ceiling"] * 5, 0.05, span) == "ceiling"


PLANE_NOISE_SIGMA = 0.006


def _noisy_plane(seed: int, n_inliers: int = 1500, n_outliers: int = 400):
    """A z = 2 plane with Gaussian thickness, buried in uniform outliers."""
    rng = np.random.default_rng(seed)
    inliers = np.column_stack(
        [
            rng.uniform(-1, 1, n_inliers),
            rng.uniform(-1, 1, n_inliers),
            rng.normal(0, PLANE_NOISE_SIGMA, n_inliers) + 2.0,
        ]
    )
    outliers = rng.uniform(-3, 3, (n_outliers, 3))
    return np.vstack([inliers, outliers]).astype(np.float32)


def _plane_error(normal: np.ndarray, offset: float) -> tuple[float, float]:
    """(tilt from the true +z normal in radians, |offset| error in metres)."""
    flip = 1.0 if normal[2] > 0 else -1.0
    tilt = float(np.arccos(np.clip(abs(float(normal[2] * flip)), -1.0, 1.0)))
    return tilt, abs(abs(offset) - 2.0)


def test_ransac_plane_is_accurate_to_far_better_than_the_sample_that_won_it():
    """The refit, not the winning triplet, is what sets the accuracy.

    The old version of this test asserted that the returned plane equals the
    least-squares fit of the returned inliers -- a tautology, and one the
    production code was changed to satisfy (commit 4722218), so it asserted a
    property it had itself caused.  What actually matters is that the answer
    is *good*, and three random points from a 6 mm-thick plane are not.

    Measured on this fixture (12 seeds, sigma = 6 mm, 21% outliers), taking
    the winning minimal sample verbatim gives a worst-case tilt of 15.1 mrad
    and a 4.9 mm offset error; refitting once brings that to 1.5 mrad /
    0.30 mm and twice to 1.0 mrad / 0.33 mm.  The bounds below sit between
    the two regimes, so deleting the refit fails and keeping it passes with
    roughly 3x headroom.
    """
    for seed in range(6):
        points = _noisy_plane(seed)
        normal, offset, mask = ransac_plane(
            points, threshold=0.03, rng=np.random.default_rng(100 + seed)
        )
        tilt, offset_error = _plane_error(normal, offset)
        assert tilt < 3e-3, f"seed {seed}: plane tilted {tilt * 1e3:.2f} mrad off true"
        assert offset_error < 1e-3, (
            f"seed {seed}: offset off by {offset_error * 1e3:.2f} mm"
        )
        # The true inliers must all be recovered, and the outliers left out.
        assert mask[:1500].all()
        assert mask[1500:].mean() < 0.15


def test_ransac_plane_residual_sits_at_the_noise_floor():
    """The returned plane must fit its own inliers as well as least squares can.

    That is the observable consequence of refitting: the RMS residual of the
    returned plane over the inliers it returns drops to the injected noise
    level.  Measured worst case over 6 seeds: 6.13 mm with the refit (sigma is
    6.0 mm, so 1.02x), 10.96 mm without it (1.83x).
    """
    for seed in range(6):
        points = _noisy_plane(seed)
        normal, offset, mask = ransac_plane(
            points, threshold=0.03, rng=np.random.default_rng(100 + seed)
        )
        residual = points[mask] @ normal + offset
        rms = float(np.sqrt((residual**2).mean()))
        assert rms < 1.2 * PLANE_NOISE_SIGMA, (
            f"seed {seed}: rms residual {rms * 1e3:.2f} mm against a "
            f"{PLANE_NOISE_SIGMA * 1e3:.1f} mm noise floor"
        )
        # And least squares over the same inliers cannot do materially better.
        best_normal, best_offset = fit_plane_lsq(points[mask])
        best = points[mask] @ best_normal + best_offset
        assert rms <= float(np.sqrt((best**2).mean())) * 1.001


def test_ransac_plane_answer_does_not_depend_on_which_triplet_won():
    """Iterating the refit to a fixed point makes the result seed-independent.

    This is the property that a single refit does not have.  Measured on one
    fixed cloud over 20 different RANSAC seeds: the shipped two-refit code
    returns bit-identical planes (spread 0.0), one refit spreads by 2.2e-4,
    and none at all by far more.  A consumer comparing two runs of the same
    scene should not see the plane move because a different random triplet
    happened to score best.
    """
    points = _noisy_plane(0)
    planes = []
    for seed in range(8):
        normal, offset, _ = ransac_plane(
            points, threshold=0.03, rng=np.random.default_rng(seed)
        )
        if normal[2] < 0:
            normal, offset = -normal, -offset
        planes.append(np.append(normal, offset))
    planes = np.array(planes)
    spread = float(np.abs(planes - planes.mean(axis=0)).max())
    assert spread < 1e-5, f"plane moved by {spread:.2e} across RANSAC seeds"


@pytest.mark.parametrize("n_outliers", [0, 400, 1500, 3000])
def test_ransac_plane_is_stable_across_outlier_fraction(n_outliers):
    """Accuracy must not degrade as the plane becomes a minority of the cloud.

    Measured worst case over 4 seeds each: tilt 0.49 mrad at 0% outliers
    rising only to 1.38 mrad at 67%, offset error under 0.34 mm throughout,
    and every true inlier recovered at every fraction.
    """
    for seed in range(4):
        points = _noisy_plane(seed, n_outliers=n_outliers)
        normal, offset, mask = ransac_plane(
            points, threshold=0.03, rng=np.random.default_rng(200 + seed)
        )
        tilt, offset_error = _plane_error(normal, offset)
        assert tilt < 3e-3, (n_outliers, seed, tilt)
        assert offset_error < 1e-3, (n_outliers, seed, offset_error)
        assert mask[:1500].mean() > 0.99, (n_outliers, seed)


def test_plane_quad_corners_bound_the_points_they_were_fitted_to():
    """The quad is what the viewer draws and what `width`/`height` report.

    Checking only its area and its plane leaves the corner arithmetic free:
    the same area can be placed anywhere on the plane.
    """
    rng = np.random.default_rng(41)
    pts = np.column_stack(
        [rng.uniform(1.0, 4.0, 3000), rng.uniform(-2.0, 0.5, 3000), np.full(3000, 1.0)]
    )
    quad, area = plane_quad(pts, np.array([0.0, 0.0, 1.0]), -1.0, percentile=0.0)

    assert area == pytest.approx(3.0 * 2.5, rel=0.05)
    assert np.allclose(quad[:, 2], 1.0, atol=1e-5)
    # Every corner is at an extreme of the point set, and the point set fits
    # inside the corners.
    assert quad[:, 0].min() == pytest.approx(pts[:, 0].min(), abs=0.02)
    assert quad[:, 0].max() == pytest.approx(pts[:, 0].max(), abs=0.02)
    assert quad[:, 1].min() == pytest.approx(pts[:, 1].min(), abs=0.02)
    assert quad[:, 1].max() == pytest.approx(pts[:, 1].max(), abs=0.02)
    # The corners run around the rectangle rather than criss-crossing it, so
    # adjacent edges are perpendicular and opposite edges equal.
    edges = quad[[1, 2, 3, 0]] - quad
    for i in range(4):
        assert abs(float(edges[i] @ edges[(i + 1) % 4])) < 1e-3
    assert np.linalg.norm(edges[0]) == pytest.approx(np.linalg.norm(edges[2]), rel=1e-4)


def test_plane_quad_percentile_trims_stragglers():
    """A handful of outliers must not stretch a wall across the whole room."""
    rng = np.random.default_rng(42)
    bulk = np.column_stack(
        [rng.uniform(0, 2, 4000), rng.uniform(0, 2, 4000), np.zeros(4000)]
    )
    strays = np.array([[50.0, 0.0, 0.0], [-50.0, 0.0, 0.0]])
    points = np.vstack([bulk, strays])
    _, trimmed = plane_quad(points, np.array([0.0, 0.0, 1.0]), 0.0, percentile=1.0)
    _, untrimmed = plane_quad(points, np.array([0.0, 0.0, 1.0]), 0.0, percentile=0.0)
    assert trimmed == pytest.approx(4.0, rel=0.1)
    assert untrimmed > 100.0


def test_ransac_plane_on_degenerate_input_falls_back_to_least_squares():
    """Collinear points give every random triplet a zero-length normal.

    There is no hypothesis to score, so the only sane answer is the
    least-squares plane through everything - and the inlier mask must still
    describe that plane rather than being empty or full of nonsense.
    """
    t = np.linspace(-1.0, 1.0, 200)
    collinear = np.column_stack([t, 2.0 * t, 3.0 * t]).astype(np.float32)
    normal, offset, mask = ransac_plane(collinear, threshold=0.01,
                                        rng=np.random.default_rng(0))
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)
    # Every point lies on the returned plane, so every point is an inlier.
    assert np.abs(collinear @ normal + offset).max() < 1e-5
    assert mask.all()


def test_ransac_plane_on_fewer_than_three_points_returns_no_inliers():
    for count in (0, 1, 2):
        normal, offset, mask = ransac_plane(
            np.zeros((count, 3), np.float32), threshold=0.01
        )
        assert mask.shape == (count,)
        assert not mask.any()
        assert np.linalg.norm(normal) == pytest.approx(1.0)


def test_backproject_with_a_single_valid_pixel():
    """The empty-input guard must not swallow a one-pixel cloud as well."""
    intr = CameraIntrinsics.from_hfov(8, 8, 60.0)
    depth = np.zeros((8, 8), np.float32)
    depth[3, 5] = 2.0
    points, idx = backproject(DepthMap(depth=depth), intr)
    assert points.shape == (1, 3)
    assert idx.tolist() == [3 * 8 + 5]
    assert points[0, 2] == pytest.approx(2.0)
    assert points[0, 0] == pytest.approx((5 - intr.cx) / intr.fx * 2.0)


def test_backproject_of_an_entirely_invalid_depth_map_is_empty():
    intr = CameraIntrinsics.from_hfov(8, 8, 60.0)
    points, idx = backproject(DepthMap(depth=np.zeros((8, 8), np.float32)), intr)
    assert points.shape == (0, 3)
    assert idx.size == 0


def test_ransac_plane_degenerate_fallback_returns_the_right_offset():
    """The collinear fallback must return the plane the points are actually on.

    A line *through the origin* has offset 0, which hides sign and arithmetic
    errors in the fallback's own inlier test; this one is deliberately offset
    from the origin so `points @ normal + offset` is not trivially symmetric.
    """
    t = np.linspace(-1.0, 1.0, 200)
    direction = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    origin = np.array([4.0, -1.5, 2.25])
    collinear = (origin + t[:, None] * direction).astype(np.float32)

    normal, offset, mask = ransac_plane(
        collinear, threshold=0.001, rng=np.random.default_rng(0)
    )
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)
    assert abs(offset) > 0.5, "the fixture must not sit on a plane through the origin"
    assert np.abs(collinear @ normal + offset).max() < 1e-4
    assert mask.all()
    # The returned plane contains the line, so its normal is perpendicular to it.
    assert abs(float(normal @ direction)) < 1e-5


def test_classify_plane_at_exactly_the_horizontal_threshold():
    """HORIZONTAL_COS is inclusive: a normal exactly on it is horizontal.

    Nothing else in the suite lands on the boundary, so `>=` and `>` are
    indistinguishable everywhere else.
    """
    from roomviz.geometry.planes import HORIZONTAL_COS, VERTICAL_COS

    up = np.array([0.0, 1.0, 0.0])
    span = (0.0, 2.7)
    on_threshold = np.array(
        [np.sqrt(1.0 - HORIZONTAL_COS**2), HORIZONTAL_COS, 0.0]
    )
    assert float(on_threshold @ up) == pytest.approx(HORIZONTAL_COS)
    assert classify_plane(on_threshold, 0.0, up, None, 0.05, span) == "floor"
    assert classify_plane(on_threshold, -2.7, up, None, 2.65, span) == "ceiling"

    # Just below it is slanted, not horizontal: with no labels it is dropped.
    below = np.array([np.sqrt(1.0 - 0.84**2), 0.84, 0.0])
    assert classify_plane(below, 0.0, up, None, 0.05, span) is None

    # And VERTICAL_COS is inclusive at the other end.
    on_vertical = np.array([np.sqrt(1.0 - VERTICAL_COS**2), VERTICAL_COS, 0.0])
    assert classify_plane(on_vertical, 0.0, up, None, 1.3, span) == "wall"


def test_extract_surfaces_at_exactly_the_minimum_point_count():
    """`min_inliers` is the floor, not the first rejected value.

    A cloud with exactly `min_inliers` points must still be fitted; one point
    fewer must be refused rather than fitted from too little evidence.
    """
    rng = np.random.default_rng(51)
    n = 600
    points = np.column_stack(
        [rng.uniform(0, 3, n), rng.normal(0, 0.003, n), rng.uniform(0, 3, n)]
    ).astype(np.float32)
    kinds = ["floor"] * n

    fitted, _ = extract_surfaces(
        points, up=np.array([0.0, 1.0, 0.0]), labels=kinds,
        threshold=0.02, min_inliers=n, max_planes=4,
    )
    assert len(fitted) == 1 and fitted[0].kind == "floor"

    refused, assignment = extract_surfaces(
        points, up=np.array([0.0, 1.0, 0.0]), labels=kinds,
        threshold=0.02, min_inliers=n + 1, max_planes=4,
    )
    assert refused == []
    assert (assignment == -1).all()


def test_median_spacing_needs_two_points_and_no_more():
    """The guard is `n < 2`, so two points already have a spacing."""
    from roomviz.geometry.pointcloud import median_spacing

    assert median_spacing(np.zeros((0, 3), np.float32)) == 0.0
    assert median_spacing(np.zeros((1, 3), np.float32)) == 0.0
    pair = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], np.float32)
    assert median_spacing(pair) == pytest.approx(0.25, abs=1e-6)


def test_sets_are_connected_is_inclusive_at_exactly_the_gap():
    """Two objects exactly `gap` apart are touching, by definition of the gap."""
    from roomviz.geometry.pointcloud import sets_are_connected

    # 0.125 is exact in binary floating point, so "exactly the gap" really is.
    a = np.zeros((1, 3), np.float32)
    b = np.array([[0.125, 0.0, 0.0]], np.float32)
    assert sets_are_connected(a, b, gap=0.125) is True
    assert sets_are_connected(a, b, gap=0.124) is False
    # Empty input is never connected, whatever the gap.
    assert sets_are_connected(np.zeros((0, 3), np.float32), b, gap=100.0) is False


def test_statistical_outlier_removal_keeps_the_bulk_and_drops_the_stragglers():
    """`std_ratio` is a real threshold, not decoration.

    A dense blob plus a few points parked far away: the blob must survive
    intact at the shipped 2.0, and widening the ratio must let the stragglers
    back in - which is what pins the constant.
    """
    from roomviz.geometry.pointcloud import remove_statistical_outliers

    rng = np.random.default_rng(52)
    blob = rng.normal(0, 0.02, (600, 3))
    strays = rng.normal(0, 0.02, (6, 3)) + np.array([1.5, 0.0, 0.0])
    points = np.vstack([blob, strays]).astype(np.float32)

    keep = remove_statistical_outliers(points, k=12, std_ratio=2.0)
    assert keep[:600].mean() > 0.98, keep[:600].mean()
    assert not keep[600:].any()
    # A very wide ratio keeps everything; a very narrow one cannot keep all.
    assert remove_statistical_outliers(points, k=12, std_ratio=50.0).all()
    assert not remove_statistical_outliers(points, k=12, std_ratio=0.05).all()
    # Too few points to judge: keep them all rather than delete the cloud.
    assert remove_statistical_outliers(np.zeros((5, 3), np.float32), k=12).all()
