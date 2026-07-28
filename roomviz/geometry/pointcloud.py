"""Point-cloud utilities: voxel downsampling, outlier removal, clustering.

Implemented on plain numpy so the package has no hard dependency on Open3D or
PCL; everything here is O(N) hashing or a KD-tree query.
"""

from __future__ import annotations

import numpy as np


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    """Integer voxel coordinates for each point."""
    if points.size == 0:
        return np.zeros((0, 3), np.int64)
    return np.floor(points / voxel).astype(np.int64)


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray | None = None, voxel: float = 0.02
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Average points within each voxel.

    Returns ``(points, colors, inverse)`` where ``inverse[i]`` is the index of
    the output point that input point ``i`` collapsed into.
    """
    if points.shape[0] == 0:
        return points, colors, np.zeros(0, np.int64)

    keys = voxel_keys(points, voxel)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n_out = counts.shape[0]

    sums = np.zeros((n_out, 3), np.float64)
    np.add.at(sums, inverse, points)
    out_points = (sums / counts[:, None]).astype(np.float32)

    out_colors = None
    if colors is not None:
        csums = np.zeros((n_out, 3), np.float64)
        np.add.at(csums, inverse, colors.astype(np.float64))
        out_colors = np.clip(csums / counts[:, None], 0, 255).astype(np.uint8)

    return out_points, out_colors, inverse


def remove_statistical_outliers(
    points: np.ndarray, k: int = 12, std_ratio: float = 2.0
) -> np.ndarray:
    """Boolean keep-mask rejecting points far from their k nearest neighbours."""
    n = points.shape[0]
    if n <= k + 1:
        return np.ones(n, bool)

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    # +1 because the first neighbour of a point is itself.
    dists, _ = tree.query(points, k=k + 1, workers=-1)
    mean_dist = dists[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist <= threshold


def cluster_connected(points: np.ndarray, voxel: float) -> np.ndarray:
    """Label points by 26-connected voxel connectivity.

    Cheap alternative to DBSCAN that is entirely sufficient for splitting a
    segmentation mask that covers several physically separate objects (three
    paintings sharing one "painting" stuff mask, say).  Returns an integer
    label per point.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, np.int64)

    keys = voxel_keys(points, voxel)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    m = unique.shape[0]

    # Union-find over occupied voxels, linked through the 13 forward neighbours
    # (the other 13 are covered by symmetry).
    parent = np.arange(m, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    lookup = {tuple(k): i for i, k in enumerate(unique.tolist())}
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0)
    ]
    for i, key in enumerate(unique.tolist()):
        for dx, dy, dz in offsets:
            j = lookup.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if j is not None:
                union(i, j)

    roots = np.array([find(i) for i in range(m)], dtype=np.int64)
    _, compact = np.unique(roots, return_inverse=True)
    return compact.reshape(-1)[inverse]


def largest_clusters(
    points: np.ndarray, voxel: float, min_fraction: float = 0.1
) -> list[np.ndarray]:
    """Split points into connected clusters, keeping the substantial ones.

    Returns a list of index arrays.  Clusters holding less than
    ``min_fraction`` of the points are dropped as noise; if that would discard
    everything, the single largest cluster is returned.
    """
    labels = cluster_connected(points, voxel)
    if labels.size == 0:
        return []
    counts = np.bincount(labels)
    threshold = max(1, int(min_fraction * points.shape[0]))
    keep = [np.flatnonzero(labels == i) for i in np.argsort(-counts) if counts[i] >= threshold]
    if not keep:
        keep = [np.flatnonzero(labels == int(np.argmax(counts)))]
    return keep


def median_spacing(points: np.ndarray, sample: int = 2000, seed: int = 0) -> float:
    """Median nearest-neighbour distance: how densely the cloud is sampled."""
    n = points.shape[0]
    if n < 2:
        return 0.0

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    probe = points if n <= sample else points[rng.choice(n, sample, replace=False)]
    tree = cKDTree(points)
    dists, _ = tree.query(probe, k=2, workers=-1)
    return float(np.median(dists[:, 1]))


def adaptive_voxel(points: np.ndarray, base: float, factor: float = 2.5) -> float:
    """A connectivity voxel that adapts to how densely a cloud is sampled.

    Voxel-connected clustering only behaves if the voxel is comfortably larger
    than the spacing between neighbouring points.  A fixed voxel works at high
    resolution and silently fragments objects at low resolution - the far end
    of a large object is sampled sparsely and breaks off as its own cluster.
    Scaling with the observed spacing keeps the behaviour stable across input
    resolutions and distances.
    """
    spacing = median_spacing(points)
    return float(max(base, spacing * factor))


def voxel_iou(a: np.ndarray, b: np.ndarray, voxel: float) -> float:
    """Intersection-over-union of two point sets, measured on occupied voxels."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    ka = {tuple(k) for k in voxel_keys(a, voxel).tolist()}
    kb = {tuple(k) for k in voxel_keys(b, voxel).tolist()}
    inter = len(ka & kb)
    if inter == 0:
        return 0.0
    return inter / float(len(ka | kb))


def oriented_bbox_2d(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimum-area-ish 2D oriented box via PCA.

    Returns ``(corners, center, angle)`` with corners ordered counter-clockwise.
    """
    center = xy.mean(axis=0)
    centred = xy - center
    if centred.shape[0] < 3:
        axes = np.eye(2)
    else:
        cov = np.cov(centred.T)
        _, axes = np.linalg.eigh(cov)
        axes = axes[:, ::-1].T  # rows = principal axes, major first
    local = centred @ axes.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    corners_local = np.array(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
    )
    corners = corners_local @ axes + center
    angle = float(np.arctan2(axes[0, 1], axes[0, 0]))
    return corners, center, angle
