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
        down = np.array([np.sin(t), -np.cos(t), 0.0])
        rotation = rotation_between(down, up)

        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6), tilt_deg
        assert np.abs(rotation @ rotation.T - np.eye(3)).max() < 1e-6, tilt_deg
        # It must actually flip: the result has to point up, not stay down.
        assert float((rotation @ down)[1]) > 0.99, tilt_deg
        assert np.linalg.norm(rotation @ down - up) < 2e-3, tilt_deg


def test_rotation_between_fuzz_near_antiparallel():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(3000):
        a = rng.normal(size=3)
        # Concentrate samples in the ill-conditioned region.
        b = -a + rng.normal(size=3) * 10 ** rng.uniform(-14, -1)
        rotation = rotation_between(a, b)
        assert np.abs(rotation @ rotation.T - np.eye(3)).max() < 1e-5
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-5)
        worst = max(
            worst,
            np.linalg.norm(
                rotation @ (a / np.linalg.norm(a)) - b / np.linalg.norm(b)
            ),
        )
    assert worst < 2e-3, f"worst mapping error {worst:.2e}"


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


def test_ransac_plane_refits_on_its_inliers():
    """The winning minimal sample is biased; the refit is what removes that.

    Skipping it leaves the plane defined by three random points, which is
    still accurate enough to pass loose end-to-end checks -- so the refit is
    pinned exactly here instead.
    """
    rng = np.random.default_rng(31)
    inliers = np.column_stack(
        [rng.uniform(-1, 1, 1500), rng.uniform(-1, 1, 1500), rng.normal(0, 0.006, 1500) + 2.0]
    )
    outliers = rng.uniform(-3, 3, (400, 3))
    points = np.vstack([inliers, outliers]).astype(np.float32)

    normal, offset, mask = ransac_plane(points, threshold=0.03, rng=rng)
    # The returned plane must be the least-squares fit of its own inliers.
    refit_normal, refit_offset = fit_plane_lsq(points[mask])
    if float(refit_normal @ normal) < 0:
        refit_normal, refit_offset = -refit_normal, -refit_offset
    assert np.allclose(normal, refit_normal, atol=1e-6)
    assert offset == pytest.approx(refit_offset, abs=1e-6)
