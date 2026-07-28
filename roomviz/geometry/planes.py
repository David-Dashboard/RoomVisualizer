"""Plane extraction: turning structural points into walls, floor and ceiling."""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from ..types import PlaneSurface

log = logging.getLogger(__name__)

# A surface counts as horizontal when its normal is within ~32 degrees of the
# up axis, and as vertical when its normal is within ~20 degrees of horizontal.
HORIZONTAL_COS = 0.85
VERTICAL_COS = 0.35


def fit_plane_lsq(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through points: returns ``(normal, offset)``.

    The plane is ``{p : normal . p + offset = 0}`` with a unit normal.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    # Smallest singular vector of the centred cloud is the plane normal.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    offset = float(-normal @ centroid)
    return normal, offset


def ransac_plane(
    points: np.ndarray,
    threshold: float,
    iterations: int = 320,
    rng: np.random.Generator | None = None,
    max_scoring_points: int = 40_000,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit the dominant plane.  Returns ``(normal, offset, inlier_mask)``."""
    n = points.shape[0]
    if n < 3:
        return np.array([0.0, 1.0, 0.0]), 0.0, np.zeros(n, bool)

    rng = rng or np.random.default_rng(0)

    # Score hypotheses against a subsample; the final refit uses every point.
    if n > max_scoring_points:
        score_idx = rng.choice(n, max_scoring_points, replace=False)
    else:
        score_idx = np.arange(n)
    scoring = points[score_idx]

    triplets = rng.integers(0, n, size=(iterations, 3))
    a = points[triplets[:, 0]]
    b = points[triplets[:, 1]]
    c = points[triplets[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-8
    if not np.any(ok):
        normal, offset = fit_plane_lsq(points)
        return normal, offset, np.abs(points @ normal + offset) <= threshold

    normals = normals[ok] / lengths[ok, None]
    offsets = -np.einsum("ij,ij->i", normals, a[ok])

    best_count = -1
    best: tuple[np.ndarray, float] | None = None
    # Chunk the hypothesis scoring so peak memory stays bounded.
    chunk = max(1, int(4_000_000 / max(1, scoring.shape[0])))
    for start in range(0, normals.shape[0], chunk):
        nrm = normals[start : start + chunk]
        off = offsets[start : start + chunk]
        dist = np.abs(scoring @ nrm.T + off[None, :])
        counts = (dist <= threshold).sum(axis=0)
        k = int(np.argmax(counts))
        if counts[k] > best_count:
            best_count = int(counts[k])
            best = (nrm[k], float(off[k]))

    assert best is not None
    normal, offset = best
    inliers = np.abs(points @ normal + offset) <= threshold

    # Refit on the full inlier set, then re-select: one round is enough to
    # remove the bias of the random minimal sample.
    if inliers.sum() >= 3:
        normal, offset = fit_plane_lsq(points[inliers])
        inliers = np.abs(points @ normal + offset) <= threshold
    return normal, offset, inliers


def plane_quad(
    points: np.ndarray, normal: np.ndarray, offset: float, percentile: float = 1.0
) -> tuple[np.ndarray, float]:
    """Bound a set of coplanar points with a rectangle on their plane.

    Returns ``(corners, area)``.  Extents are taken at the ``percentile`` /
    ``100 - percentile`` quantiles so a handful of stragglers do not stretch a
    wall across the whole room.
    """
    # Build an orthonormal basis on the plane.
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(normal, u)
    v /= np.linalg.norm(v) + 1e-12

    origin = -normal * offset  # closest point on the plane to the world origin
    rel = points - origin
    a = rel @ u
    b = rel @ v
    lo_a, hi_a = np.percentile(a, [percentile, 100.0 - percentile])
    lo_b, hi_b = np.percentile(b, [percentile, 100.0 - percentile])

    corners = np.array(
        [
            origin + u * lo_a + v * lo_b,
            origin + u * hi_a + v * lo_b,
            origin + u * hi_a + v * hi_b,
            origin + u * lo_a + v * hi_b,
        ],
        dtype=np.float32,
    )
    area = float(max(hi_a - lo_a, 0.0) * max(hi_b - lo_b, 0.0))
    return corners, area


def classify_plane(
    normal: np.ndarray,
    offset: float,
    up: np.ndarray,
    inlier_labels: list[str] | None,
    centroid_height: float,
    scene_height_range: tuple[float, float],
) -> str | None:
    """Name a plane ``wall`` / ``floor`` / ``ceiling``, or ``None`` to drop it.

    Geometry decides first (a normal either points up or it does not), with
    the semantic labels used to break the floor/ceiling tie and to veto planes
    that the segmentation says are not structural at all.
    """
    vertical_alignment = abs(float(normal @ up))
    vote = Counter(inlier_labels or []).most_common(1)
    majority = vote[0][0] if vote else None

    if vertical_alignment >= HORIZONTAL_COS:
        if majority in ("floor", "ceiling"):
            return majority
        # No usable label: the lower horizontal plane is the floor.
        lo, hi = scene_height_range
        midpoint = (lo + hi) / 2.0
        return "floor" if centroid_height <= midpoint else "ceiling"

    if vertical_alignment <= VERTICAL_COS:
        return "wall"

    # Slanted: only trust it if the labels agree it is structural.
    if majority in ("wall", "ceiling", "floor"):
        return majority
    return None


def extract_surfaces(
    points: np.ndarray,
    up: np.ndarray,
    labels: list[str] | None = None,
    threshold: float = 0.04,
    min_inliers: int = 800,
    max_planes: int = 12,
    seed: int = 0,
) -> tuple[list[PlaneSurface], np.ndarray]:
    """Sequential RANSAC over structural points.

    Returns ``(surfaces, assignment)`` where ``assignment[i]`` is the index of
    the surface point ``i`` was assigned to, or ``-1``.
    """
    n = points.shape[0]
    assignment = np.full(n, -1, np.int64)
    if n < max(3, min_inliers):
        log.warning("only %d structural points; skipping plane extraction", n)
        return [], assignment

    rng = np.random.default_rng(seed)
    heights = points @ up
    height_range = (float(heights.min()), float(heights.max()))

    remaining = np.arange(n)
    surfaces: list[PlaneSurface] = []

    while len(surfaces) < max_planes and remaining.size >= min_inliers:
        subset = points[remaining]
        normal, offset, inliers = ransac_plane(subset, threshold, rng=rng)
        count = int(inliers.sum())
        if count < min_inliers:
            break

        inlier_global = remaining[inliers]
        inlier_points = points[inlier_global]
        inlier_labels = [labels[i] for i in inlier_global] if labels else None

        # Orient every normal consistently: walls point towards the interior
        # (the side the bulk of the scene is on), horizontals point up.
        if float(normal @ up) < 0 and abs(float(normal @ up)) >= HORIZONTAL_COS:
            normal, offset = -normal, -offset

        centroid_height = float(inlier_points.mean(axis=0) @ up)
        kind = classify_plane(
            normal, offset, up, inlier_labels, centroid_height, height_range
        )

        if kind is not None:
            quad, area = plane_quad(inlier_points, normal, offset)
            surface = PlaneSurface(
                surface_id=len(surfaces),
                kind=kind,
                normal=normal.astype(np.float32),
                offset=float(offset),
                quad=quad,
                inlier_count=count,
                area=area,
            )
            assignment[inlier_global] = len(surfaces)
            surfaces.append(surface)
            log.debug(
                "plane %d: %s, %d inliers, %.2f m^2, normal %s",
                surface.surface_id,
                kind,
                count,
                area,
                np.round(normal, 3),
            )

        remaining = remaining[~inliers]

    # Sequential RANSAC readily splits one real wall into two nearly identical
    # planes, so fold those back together before anything else.
    surfaces = _merge_similar(surfaces, assignment, points, threshold)

    # Keep the biggest floor and ceiling only; multiple parallel "floors" are
    # nearly always a table top or a step misread as ground.
    surfaces = _dedupe_horizontals(surfaces, assignment)
    log.info(
        "extracted %d surfaces (%s)",
        len(surfaces),
        ", ".join(f"{k}x{v}" for k, v in Counter(s.kind for s in surfaces).items()) or "none",
    )
    return surfaces, assignment


def _merge_similar(
    surfaces: list[PlaneSurface],
    assignment: np.ndarray,
    points: np.ndarray,
    threshold: float,
    angle_cos: float = 0.985,
    offset_factor: float = 5.0,
) -> list[PlaneSurface]:
    """Merge planes that describe the same physical surface.

    Two planes are the same surface when their normals are nearly parallel and
    the perpendicular distance between them is within a few RANSAC thresholds.
    Merged surfaces are refit over the union of their points, which also tidies
    up the bounding quad.
    """
    merged: list[PlaneSurface] = []
    groups: list[list[int]] = []

    for surface in surfaces:
        for group_index, representative in enumerate(merged):
            if representative.kind != surface.kind:
                continue
            alignment = float(representative.normal @ surface.normal)
            flip = -1.0 if alignment < 0 else 1.0
            if abs(alignment) < angle_cos:
                continue
            if abs(representative.offset - flip * surface.offset) > offset_factor * threshold:
                continue
            groups[group_index].append(surface.surface_id)
            break
        else:
            merged.append(surface)
            groups.append([surface.surface_id])

    if len(merged) == len(surfaces):
        return surfaces

    lookup = np.full(len(surfaces), -1, np.int64)
    for new_id, members in enumerate(groups):
        for old_id in members:
            lookup[old_id] = new_id
    assigned = assignment >= 0
    assignment[assigned] = lookup[assignment[assigned]]

    for new_id, surface in enumerate(merged):
        surface.surface_id = new_id
        member_points = points[assignment == new_id]
        if member_points.shape[0] >= 3:
            normal, offset = fit_plane_lsq(member_points)
            if float(normal @ surface.normal) < 0:
                normal, offset = -normal, -offset
            surface.normal = normal.astype(np.float32)
            surface.offset = float(offset)
            surface.quad, surface.area = plane_quad(member_points, normal, offset)
            surface.inlier_count = int(member_points.shape[0])

    log.debug("merged %d planes into %d", len(surfaces), len(merged))
    return merged


def _dedupe_horizontals(
    surfaces: list[PlaneSurface], assignment: np.ndarray
) -> list[PlaneSurface]:
    """Keep one floor and one ceiling (the largest), plus every wall.

    ``assignment`` is renumbered in place to match the surviving surface ids.
    """
    keep: list[PlaneSurface] = []
    for kind in ("floor", "ceiling"):
        matching = [s for s in surfaces if s.kind == kind]
        if matching:
            keep.append(max(matching, key=lambda s: s.inlier_count))
    keep.extend(s for s in surfaces if s.kind == "wall")
    keep.sort(key=lambda s: (s.kind != "floor", s.kind != "ceiling", -s.inlier_count))

    # Build an old-id -> new-id lookup, then renumber in one pass.
    if surfaces:
        lookup = np.full(len(surfaces), -1, np.int64)
        for new_id, surface in enumerate(keep):
            lookup[surface.surface_id] = new_id
        assigned = assignment >= 0
        assignment[assigned] = lookup[assignment[assigned]]

    for new_id, surface in enumerate(keep):
        surface.surface_id = new_id
    return keep
