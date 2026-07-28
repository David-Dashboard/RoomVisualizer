"""Gravity and Manhattan alignment.

Monocular reconstructions come out in whatever frame the first camera happened
to be in, so a room arrives tilted.  Two corrections make the output far easier
to look at and to measure:

* **Gravity alignment** rotates the scene so the floor normal is ``+Y`` and
  puts the floor at ``y = 0`` - so object heights are heights above the floor.
* **Manhattan alignment** spins the scene about ``Y`` so the dominant wall is
  axis-aligned, which squares the room up in the viewer.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

WORLD_UP = np.array([0.0, 1.0, 0.0])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix taking unit vector ``a`` onto ``b``."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        # Parallel or antiparallel: identity, or a 180-degree flip about any
        # axis perpendicular to a.
        if c > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(a @ helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)

    kmat = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )
    return np.eye(3) + kmat + kmat @ kmat * (1.0 / (1.0 + c))


def estimate_up(
    points: np.ndarray,
    kinds: np.ndarray | None = None,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the up direction in the current frame.

    Preference order: fit a plane to points labelled ``floor``/``ceiling``;
    else derive the axis orthogonal to all wall normals; else fall back to the
    camera's ``-y`` (OpenCV cameras look along ``+z`` with ``+y`` pointing down,
    so ``-y`` is up for a level camera).
    """
    default = fallback if fallback is not None else np.array([0.0, -1.0, 0.0])

    if kinds is not None and points.shape[0] >= 50:
        from .planes import fit_plane_lsq, ransac_plane

        horizontal = np.isin(kinds, ("floor", "ceiling"))
        if horizontal.sum() >= 50:
            normal, _, inliers = ransac_plane(points[horizontal], threshold=0.05)
            if inliers.sum() >= 30:
                normal, _ = fit_plane_lsq(points[horizontal][inliers])
                # Point it the same way as the rough prior.
                if float(normal @ default) < 0:
                    normal = -normal
                log.info("up axis from floor/ceiling plane: %s", np.round(normal, 3))
                return normal / np.linalg.norm(normal)

        wall = kinds == "wall"
        if wall.sum() >= 200:
            sampled = _wall_normal_samples(points[wall])
            if sampled is not None and sampled[0].shape[0] >= 2:
                normals, weights = sampled
                # Up is the direction least represented among wall normals.
                # Weighting rows by sqrt(count) makes the SVD a weighted fit.
                weighted = normals * np.sqrt(weights)[:, None]
                _, _, vt = np.linalg.svd(weighted, full_matrices=False)
                candidate = vt[-1]
                if float(candidate @ default) < 0:
                    candidate = -candidate
                log.info("up axis from wall normals: %s", np.round(candidate, 3))
                return candidate / np.linalg.norm(candidate)

    log.info("up axis falling back to camera up: %s", np.round(default, 3))
    return default / np.linalg.norm(default)


def _wall_normal_samples(
    wall_points: np.ndarray, planes: int = 6, seed: int = 0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Dominant plane normals within the wall points, with inlier counts.

    The counts matter: a small fragment carries far less evidence about the
    room's heading than the wall spanning half the cloud, so callers weight by
    them rather than treating every fitted plane alike.
    """
    from .planes import ransac_plane

    rng = np.random.default_rng(seed)
    remaining = wall_points
    normals: list[np.ndarray] = []
    weights: list[float] = []
    for _ in range(planes):
        if remaining.shape[0] < 200:
            break
        normal, _, inliers = ransac_plane(remaining, threshold=0.05, rng=rng)
        count = int(inliers.sum())
        if count < 150:
            break
        normals.append(normal)
        weights.append(float(count))
        remaining = remaining[~inliers]
    if not normals:
        return None
    return np.array(normals), np.array(weights)


def manhattan_yaw(points: np.ndarray, kinds: np.ndarray | None) -> float:
    """Yaw (radians about ``+Y``) that squares the dominant walls to an axis.

    A rectangular room's wall normals all agree modulo 90 degrees, so the
    headings are folded into a quarter turn and averaged.  Two details make
    this robust: only genuinely vertical planes are considered (a slanted
    fragment says nothing about the room's heading), and each plane is weighted
    by its inlier count so one small misfit cannot rotate the whole room.
    """
    if kinds is None:
        return 0.0
    wall = kinds == "wall"
    if wall.sum() < 200:
        return 0.0
    sampled = _wall_normal_samples(points[wall])
    if sampled is None:
        return 0.0
    normals, weights = sampled

    # Keep only near-vertical surfaces, then project their normals onto the
    # floor plane to get a heading.
    horizontal = normals.copy()
    horizontal[:, 1] = 0.0
    lengths = np.linalg.norm(horizontal, axis=1)
    vertical_enough = lengths > 0.9
    if not vertical_enough.any():
        return 0.0
    horizontal = horizontal[vertical_enough]
    weights = weights[vertical_enough]

    angles = np.arctan2(horizontal[:, 2], horizontal[:, 0]) % (np.pi / 2)
    # Scaling by 4 maps the [0, pi/2) period onto a full turn, where a plain
    # weighted circular mean is valid.
    scaled = angles * 4.0
    mean = np.arctan2(
        float(np.sum(weights * np.sin(scaled))),
        float(np.sum(weights * np.cos(scaled))),
    ) / 4.0
    # Rotating by +mean maps a normal at heading `mean` onto heading 0: the
    # yaw matrix below takes a heading `theta` to `theta - yaw`.
    return float(mean)


def alignment_transform(
    points: np.ndarray,
    kinds: np.ndarray | None = None,
    camera_up: np.ndarray | None = None,
    manhattan: bool = True,
) -> np.ndarray:
    """Build the 4x4 transform taking the current frame into the world frame.

    The result puts up at ``+Y``, the floor at ``y = 0`` and (optionally) the
    dominant wall parallel to an axis.
    """
    transform = np.eye(4)
    if points.shape[0] < 10:
        return transform

    up = estimate_up(points, kinds, fallback=camera_up)
    rotation = rotation_between(up, WORLD_UP)
    rotated = points @ rotation.T

    if manhattan:
        yaw = manhattan_yaw(rotated, kinds)
        if abs(yaw) > 1e-4:
            c, s = np.cos(yaw), np.sin(yaw)
            yaw_matrix = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            rotation = yaw_matrix @ rotation
            rotated = points @ rotation.T
            log.info("manhattan yaw: %.1f degrees", np.rad2deg(yaw))

    # Put the floor at y = 0.  The 1st percentile is a robust "lowest surface"
    # that ignores stray points punched through the floor by depth noise.
    floor_y = float(np.percentile(rotated[:, 1], 1.0))
    if kinds is not None:
        floor_mask = kinds == "floor"
        if floor_mask.sum() >= 50:
            floor_y = float(np.median(rotated[floor_mask, 1]))

    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([0.0, -floor_y, 0.0])
    return transform


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an ``(N, 3)`` array."""
    if points.shape[0] == 0:
        return points
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
