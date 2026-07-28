"""Fuse per-frame observations into a single 3D scene.

The pipeline here is:

1. Back-project every frame's depth into world space using its camera pose.
2. Split the resulting points into *object* points (grouped per segment, then
   split into physically separate 3D clusters) and *structural* points.
3. Associate per-frame object clusters across frames into global instances.
4. Gravity-align the whole scene, then fit walls / floor / ceiling.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import PipelineConfig
from ..geometry.align import alignment_transform, apply_transform
from ..geometry.camera import backproject, depth_edge_mask
from ..geometry.planes import extract_surfaces
from ..geometry.pointcloud import (
    adaptive_voxel,
    largest_clusters,
    remove_statistical_outliers,
    voxel_downsample,
    voxel_keys,
)
from ..perception.labels import split_policy
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

    split_gap: float = 0.0
    """The empty distance that separated this cluster from its siblings.

    A split is provisional, and the distance that justified it is the same
    distance that should undo it: two pieces of one object that were parted at
    30 cm must be allowed to rejoin at 30 cm, or the split becomes permanent
    for exactly the objects it was least confident about."""


def _frame_world_points(
    obs: Observation, cfg: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """World points, colours, segment ids and source pixel index for one frame.

    The pixel index is kept because the split step needs to look back at the
    image: whether a gap between two pieces of one mask is empty space or an
    occluder's shadow is a question about what the camera saw *between* them,
    and that is only answerable in the frame the pieces came from.
    """
    keep = depth_edge_mask(obs.depth, cfg.edge_discard)
    points_cam, pixel_idx = backproject(obs.depth, obs.intrinsics, mask=keep)
    if points_cam.shape[0] == 0:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint8),
            np.zeros(0, np.int32),
            np.zeros(0, np.int64),
        )

    points_world = apply_transform(obs.pose, points_cam)
    colors = obs.frame.rgb.reshape(-1, 3)[pixel_idx]
    seg_ids = obs.segmentation.ids.reshape(-1)[pixel_idx]
    return points_world, colors, seg_ids, pixel_idx


OCCLUDER_MARGIN = 0.10
"""How much nearer than both pieces a surface between them has to be to count
as the occluder that separated them.

A fraction of the pieces' own depth rather than an absolute distance, because
depth error and the depth quantisation of a monocular model both scale with
range.  10% at 3 m is 30 cm; the bookcase that cuts a sofa in half in the
synthetic harness sits 1.5 m in front of it."""

_BRIDGE_SAMPLES = 24
"""Pixels probed along the line joining two pieces.  The line is a few hundred
pixels long at most, so this samples it every handful of pixels."""


def _is_occlusion_shadow(
    obs: Observation,
    seg_id: int,
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
    margin: float = OCCLUDER_MARGIN,
) -> bool:
    """Whether a nearer surface stands between two pieces of one mask.

    Splitting a mask in 3D cannot, on geometry alone, tell three paintings hung
    45 cm apart from the two halves of a sofa that a bookcase cut in two - the
    sofa's halves are further apart than the paintings are.  The image answers
    it directly: walk the straight line between the two pieces and look at what
    the camera saw there.  Wall *behind* the paintings means the gap is real
    and they are separate objects; a bookcase *in front of* the sofa means the
    gap is that bookcase's shadow and the two halves are one object.

    Only pixels belonging to some other segment count, and they have to be
    nearer than *both* pieces - a background surface, or the object's own far
    side, proves nothing.
    """
    depth = obs.depth.depth
    height, width = depth.shape
    ids = obs.segmentation.ids.reshape(-1)
    flat = depth.reshape(-1)

    near_a = float(np.median(flat[pixels_a]))
    near_b = float(np.median(flat[pixels_b]))
    if not (np.isfinite(near_a) and np.isfinite(near_b)) or min(near_a, near_b) <= 0:
        return False
    threshold = min(near_a, near_b) * (1.0 - margin)

    centre_a = np.array(
        [float(np.median(pixels_a // width)), float(np.median(pixels_a % width))]
    )
    centre_b = np.array(
        [float(np.median(pixels_b // width)), float(np.median(pixels_b % width))]
    )

    t = np.linspace(0.0, 1.0, _BRIDGE_SAMPLES + 2)[1:-1]
    probe = centre_a[None, :] + t[:, None] * (centre_b - centre_a)[None, :]
    rows = np.clip(np.rint(probe[:, 0]).astype(np.int64), 0, height - 1)
    cols = np.clip(np.rint(probe[:, 1]).astype(np.int64), 0, width - 1)
    flat_idx = rows * width + cols

    other = (ids[flat_idx] != seg_id) & (flat[flat_idx] > 0)
    if not other.any():
        return False
    nearer = other & (flat[flat_idx] < threshold)
    # A majority of the *usable* probes, so that a couple of stray pixels at
    # either end of the line cannot decide it either way.
    return bool(nearer.sum() * 2 > other.sum())


def _bridge_shadowed_clusters(
    obs: Observation,
    segment_id: int,
    clusters: list[np.ndarray],
    seg_pixels: np.ndarray,
) -> list[np.ndarray]:
    """Rejoin clusters whose separation is an occluder's shadow.

    Returns the clusters with any occlusion-separated pair unioned.  This runs
    before the split is published, so an object that a chair or a bookcase cut
    into two visible patches never becomes two detections in the first place -
    which matters because the later proximity rejoin cannot reach across a
    shadow a metre wide.
    """
    count = len(clusters)
    if count < 2:
        return clusters

    parent = list(range(count))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in itertools.combinations(range(count), 2):
        if find(i) == find(j):
            continue
        if _is_occlusion_shadow(
            obs, segment_id, seg_pixels[clusters[i]], seg_pixels[clusters[j]]
        ):
            parent[find(j)] = find(i)

    grouped: dict[int, list[np.ndarray]] = {}
    for i in range(count):
        grouped.setdefault(find(i), []).append(clusters[i])
    joined = [np.concatenate(parts) for parts in grouped.values()]
    joined.sort(key=lambda c: -c.shape[0])
    return joined


def _detections_for_frame(
    obs: Observation,
    points: np.ndarray,
    colors: np.ndarray,
    seg_ids: np.ndarray,
    pixel_idx: np.ndarray,
    cfg: PipelineConfig,
) -> list[FrameDetection]:
    """Split a frame's object segments into physically separate clusters.

    Panoptic "stuff" classes hand back one mask for every instance of a class
    (all the paintings in one mask), so splitting happens in 3D rather than in
    the image: genuinely separate objects are separated in space.

    *Every* object mask is a split candidate.  What the segmenter's thing/stuff
    flag buys is not permission but a distance: see
    :func:`roomviz.perception.labels.split_policy`.  A mask the model calls one
    object has to show a wider gap before we overrule it; a stuff mask is split
    at the configured gap; an unknown class sits between the two.  Making this a
    gate instead - "things are never split" - meant that whether three paintings
    in one mask came out as three objects or one depended on whether the word
    "painting" appeared in a hand-typed list.

    Distance alone cannot carry that decision, because an occluder's shadow is
    routinely wider than the space between two genuinely separate objects.  So
    every candidate split is then checked against the image it came from: a pair
    of pieces with a *nearer* surface between them is one object seen past an
    obstruction, and is rejoined immediately
    (:func:`_is_occlusion_shadow`).

    What survives that is still provisional.  An object that occludes itself -
    a sofa seen along its length - can present as two disconnected patches in
    one view and as one contiguous patch in the next, so every cluster records
    the segment it came from *and* the gap that parted it, and
    :func:`_merge_instances` may rejoin them later at that same distance.
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
        seg_pixels = pixel_idx[mask]

        inliers = remove_statistical_outliers(seg_points)
        seg_points, seg_colors = seg_points[inliers], seg_colors[inliers]
        seg_pixels = seg_pixels[inliers]
        if seg_points.shape[0] < floor:
            continue

        policy = split_policy(segment.label, segment.is_thing)
        gap = cfg.object_split_gap * policy.gap_scale
        # 26-connectivity links points up to two voxels apart, so a voxel of
        # `gap / 2` makes the realised split distance `gap`.  Using `gap`
        # itself splits at roughly twice the documented distance, and the
        # grid's origin then leaks into the result: whether two objects
        # separate would depend on where they sit in world coordinates.
        cluster_voxel = adaptive_voxel(seg_points, gap / 2.0)
        clusters = largest_clusters(seg_points, voxel=cluster_voxel, min_points=floor)
        clusters = _bridge_shadowed_clusters(
            obs, segment.segment_id, clusters, seg_pixels
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
                    # Only a split that actually happened licenses a rejoin at
                    # the wider distance; a mask that came out in one piece
                    # must not widen the merge radius for its neighbours.
                    split_gap=gap if len(clusters) > 1 else 0.0,
                )
            )
    return detections


MIN_FRAGMENT_RATIO = 0.05
"""Smallest share of an instance's occupied volume that a candidate view has to
cover before containment counts as a full match.

:func:`roomviz.geometry.pointcloud.voxel_overlap` divides by the *smaller* of
the two voxel sets, which is right for matching a partial view against a model
grown from many views but saturates at 1.0 for anything fully contained: a
three-voxel sliver scores a perfect 1.0 against a 600-voxel sofa and outranks a
genuine 0.8 match elsewhere.  Flooring the denominator at this share of the
larger set turns that sliver's score into ``3 / 30 = 0.1`` while leaving any
candidate that covers at least this much of the model scored exactly as before.
"""


def _voxel_set(points: np.ndarray, voxel: float) -> set[tuple[int, int, int]]:
    """Occupied voxel coordinates of a point cloud, as a hashable set.

    ``np.unique`` collapses the cloud in C first, so the Python-level cost is
    the number of occupied voxels rather than the number of points.
    """
    if points.shape[0] == 0:
        return set()
    unique = np.unique(voxel_keys(points, voxel), axis=0)
    return {(int(a), int(b), int(c)) for a, b, c in unique.tolist()}


def _containment(
    a: set, b: set, min_fragment_ratio: float = MIN_FRAGMENT_RATIO
) -> float:
    """Intersection over the smaller voxel set, guarded against size disparity.

    See :data:`MIN_FRAGMENT_RATIO`.  With ``min_fragment_ratio=0`` this is
    exactly :func:`roomviz.geometry.pointcloud.voxel_overlap`.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b) if len(a) < len(b) else len(b & a)
    if inter == 0:
        return 0.0
    small, large = min(len(a), len(b)), max(len(a), len(b))
    return inter / max(float(small), min_fragment_ratio * large)


def _associate(
    detections: list[FrameDetection], cfg: PipelineConfig
) -> list[ObjectInstance]:
    """Merge per-frame detections into global instances.

    Greedy nearest-match on voxel containment, restricted to detections that
    agree on the class label.  Two detections in the same frame carrying
    *different* segment ids are never merged - the segmenter has already ruled
    they are distinct objects - but two carrying the same id may be, since
    those were separated by our own 3D split rather than by the segmenter.

    Detections are consumed in frame order (largest first within a frame) so
    each instance grows through consecutive views.  Order matters because pose
    error accumulates along the trajectory: neighbouring frames overlap almost
    perfectly, whereas the first and last frames of a sweep may have drifted
    far enough apart to fall below the threshold and split one object in two.
    """
    instances: list[ObjectInstance] = []
    voxel = cfg.voxel_size * 3.0
    # Voxel occupancy of each instance, kept in step with its points.  The
    # occupancy of a union is the union of the occupancies, so this stays exact
    # while turning the rescan of every instance's full cloud into a set union.
    occupancy: list[set] = []

    ordered = sorted(detections, key=lambda d: (d.frame_index, -d.points.shape[0]))
    for det in ordered:
        det_voxels = _voxel_set(det.points, voxel)
        best_idx, best_score = -1, 0.0
        for i, inst in enumerate(instances):
            if inst.label != det.label:
                continue
            if _conflicts(inst.sources, [(det.frame_index, det.segment_id)]):
                continue
            score = _containment(det_voxels, occupancy[i])
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
            inst.split_gap = max(inst.split_gap, det.split_gap)
            occupancy[best_idx] |= det_voxels
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
                    split_gap=det.split_gap,
                )
            )
            occupancy.append(det_voxels)

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
    target.split_gap = max(target.split_gap, other.split_gap)


def _shares_source(
    a_sources: list[tuple[int, int]], b_sources: list[tuple[int, int]]
) -> bool:
    """Whether some frame saw both instances under the *same* segment id.

    This is positive evidence that the two are pieces of one mask that our own
    3D split parted, which is the only thing that ever creates split siblings:
    :func:`_detections_for_frame` stamps every cluster it cuts out of a mask
    with that mask's ``(frame, segment_id)``.  Two objects the segmenter told
    apart never share such a pair.
    """
    return bool(set(a_sources) & set(b_sources))


_NEIGHBOURHOOD = tuple(itertools.product((-1, 0, 1), repeat=3))


@dataclass
class _MergeNode:
    """An instance prepared for merging: identity, bounds and voxel occupancy."""

    inst: ObjectInstance
    key: tuple
    lo: np.ndarray
    hi: np.ndarray
    voxels: set
    dilated: set | None = field(default=None, repr=False)

    def halo(self) -> set:
        """This instance's voxels grown by one cell in all 26 directions."""
        if self.dilated is None:
            self.dilated = {
                (x + dx, y + dy, z + dz)
                for (x, y, z) in self.voxels
                for dx, dy, dz in _NEIGHBOURHOOD
            }
        return self.dilated


def _identity(inst: ObjectInstance) -> tuple:
    """A total, content-derived ordering key for an instance.

    Merging has to produce the same partition however the caller ordered the
    list, so every tie-break must come from the data rather than from a
    position in a Python list.  The digest is over the raw point bytes, which
    makes the key total in practice; two instances that collide on all of this
    are interchangeable as far as every merge predicate is concerned.
    """
    lo, hi = inst.aabb
    return (
        inst.label,
        int(inst.points.shape[0]),
        tuple(np.round(lo, 6).tolist()),
        tuple(np.round(hi, 6).tolist()),
        tuple(sorted(inst.sources)),
        hashlib.blake2b(
            np.ascontiguousarray(inst.points, dtype=np.float32).tobytes(),
            digest_size=8,
        ).hexdigest(),
    )


def _frame_ids(sources: list[tuple[int, int]]) -> dict[int, set[int]]:
    by_frame: dict[int, set[int]] = {}
    for frame, segment in sources:
        by_frame.setdefault(frame, set()).add(segment)
    return by_frame


def _groups_conflict(a: dict[int, set[int]], b: dict[int, set[int]]) -> bool:
    """``_conflicts`` lifted to two accumulated ``frame -> segment ids`` maps."""
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    for frame, ids in small.items():
        other = large.get(frame)
        if other is not None and len(ids | other) > 1:
            return True
    return False


def _merge_instances(
    instances: list[ObjectInstance], gap: float, iou_threshold: float = 0.5
) -> list[ObjectInstance]:
    """Fold together instances that describe one physical object.

    Two failure modes need cleaning up:

    * **Drift duplicates** - one object recovered twice because camera drift
      pushed its early and late observations below the association threshold.
      Both copies occupy the same volume, so bounding-box IoU catches them.
    * **Split siblings** - a self-occluding object that presented as two
      disconnected patches.  These are recognised by *sharing a mask*: some
      frame saw both under one segment id, which only happens when our own 3D
      split cut them out of that mask.  They then have to be close, within the
      gap that parted them (:attr:`ObjectInstance.split_gap`, falling back to
      ``gap``).

    Positive provenance is what makes the second rule safe, and mere absence of
    contradiction is not enough.  "No frame saw these two under different ids"
    holds for any pair that was never co-visible - one dropped mask, or a
    camera that panned past two chairs in turn - and proximity alone then fuses
    two chairs 6 cm apart into one 1.06 m object, and a row of five stools into
    two.  Requiring a shared ``(frame, segment_id)`` cannot make that mistake:
    the segmenter gave the two chairs different ids in every frame that saw
    them, and in the frames that saw only one there is no shared pair to find.

    Sizes are deliberately not consulted: the point count of an instance tracks
    how many frames happened to see it, not how big it is, so a size-ratio rule
    silently destroys real objects whenever two neighbours differ in
    visibility.

    **The partition does not depend on the order of ``instances``.**  That is a
    property of the construction, not a hope: the pairwise predicate above is
    symmetric and is evaluated only on the *original* instances, edges are
    processed in an order derived from instance content (see :func:`_identity`)
    rather than from list position, and the groups are the connected components
    of that edge set under union-find.  Conflicts are additionally re-checked
    per group, so an edge is dropped if it would transitively join two
    instances the segmenter told apart; that check too is a function of the
    group contents alone.
    """
    n = len(instances)
    if n < 2:
        return list(instances)

    nodes: list[_MergeNode] = []
    for inst in instances:
        lo, hi = inst.aabb
        reach = max(gap, float(inst.split_gap))
        # 26-connectivity spans two voxels, so a voxel of `reach / 2` makes the
        # realised proximity threshold `reach` - the same convention
        # `_detections_for_frame` used to split in the first place, though not
        # necessarily the same voxel, since the split adapts its grid to how
        # densely that one segment was sampled.  Working on voxels rather than
        # raw points is what takes the pairwise test from a KD-tree query over
        # every point to a set probe.
        nodes.append(
            _MergeNode(
                inst=inst,
                key=_identity(inst),
                lo=lo,
                hi=hi,
                voxels=_voxel_set(inst.points, reach / 2.0),
            )
        )

    # ---- candidate pairs -------------------------------------------------
    # Only same-label pairs can ever merge, and within a label only pairs whose
    # bounding boxes come within `reach` of each other: anything further apart
    # has zero AABB IoU and cannot be touching.  Enumerating the pairs is still
    # quadratic, but it is a vectorised box test rather than a KD-tree built
    # over one instance's full cloud per pair, which is what took 21 s for 60
    # instances of 12k points.
    by_label: dict[str, list[int]] = {}
    for i, node in enumerate(nodes):
        by_label.setdefault(node.inst.label, []).append(i)

    edges: list[tuple[float, tuple, tuple, int, int]] = []
    for indices in by_label.values():
        if len(indices) < 2:
            continue
        lows = np.array([nodes[i].lo for i in indices])
        highs = np.array([nodes[i].hi for i in indices])
        reaches = np.array(
            [max(gap, float(nodes[i].inst.split_gap)) for i in indices]
        )
        # near[p, q] is True when the boxes overlap once inflated by the larger
        # of the two reaches.
        margin = np.maximum(reaches[:, None], reaches[None, :])[:, :, None]
        near = (
            (lows[:, None, :] - margin <= highs[None, :, :])
            & (highs[:, None, :] + margin >= lows[None, :, :])
        ).all(axis=2)
        for p, q in zip(*np.nonzero(np.triu(near, k=1)), strict=True):
            i, j = indices[int(p)], indices[int(q)]
            a, b = nodes[i], nodes[j]
            if _conflicts(a.inst.sources, b.inst.sources):
                continue
            iou = _aabb_iou(a.inst, b.inst)
            if iou < iou_threshold:
                if not _shares_source(a.inst.sources, b.inst.sources):
                    continue
                small, large = (a, b) if len(a.voxels) <= len(b.voxels) else (b, a)
                if large.halo().isdisjoint(small.voxels):
                    continue
            lo_key, hi_key = sorted((a.key, b.key))
            edges.append((iou, lo_key, hi_key, i, j))

    if not edges:
        return list(instances)

    # ---- union-find over content-ordered edges ---------------------------
    # Strongest evidence first (a high box IoU is a duplicate; a sibling edge
    # may score 0), then by the instances' own identity keys so that the order
    # is a function of the data and nothing else.
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))

    parent = list(range(n))
    frame_ids = [_frame_ids(inst.sources) for inst in instances]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, _, _, i, j in edges:
        ra, rb = find(i), find(j)
        if ra == rb:
            continue
        # A pairwise merge is not allowed to smuggle in a pair the segmenter
        # told apart: two groups may only join if no frame saw any member of
        # one and any member of the other under different segment ids.
        if _groups_conflict(frame_ids[ra], frame_ids[rb]):
            continue
        keep, gone = (ra, rb) if nodes[ra].key <= nodes[rb].key else (rb, ra)
        parent[gone] = keep
        merged = frame_ids[keep]
        for frame, ids in frame_ids[gone].items():
            merged.setdefault(frame, set()).update(ids)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged_instances: list[ObjectInstance] = []
    for members in groups.values():
        members.sort(key=lambda i: nodes[i].key)
        target = instances[members[0]]
        for other in members[1:]:
            _absorb(target, instances[other])
        merged_instances.append(target)

    merged_instances.sort(key=lambda inst: _identity(inst))
    if len(merged_instances) != n:
        log.info(
            "merged %d instance(s) into their parent object",
            n - len(merged_instances),
        )
    return merged_instances


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


def _warn_if_implausible(scene: Scene, intrinsics) -> None:
    """Shout when the reconstruction is not a plausible room.

    Everything scales with the focal length, so a wrong field of view produces
    a self-consistent, confidently-reported, entirely wrong room - a 1.5 m
    ceiling and a 36 cm sofa - with nothing else to give it away.  The scale is
    unknowable from the data, but implausibility is not.
    """
    if scene.points.shape[0] == 0:
        return
    lo, hi = scene.bounds
    height = float(hi[1] - lo[1])
    guessed = getattr(intrinsics, "provenance", "") == "assumed_default"
    if 1.9 <= height <= 6.0:
        return
    log.warning(
        "The reconstructed room is %.2f m from floor to ceiling, which is not a "
        "plausible interior. Every dimension scales with the assumed field of "
        "view%s, so the whole scene is likely mis-scaled by roughly %.1fx.",
        height,
        " (which was guessed, not measured)" if guessed else "",
        2.5 / max(height, 1e-3),
    )


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
        points, colors, seg_ids, pixel_idx = _frame_world_points(obs, cfg)
        if points.shape[0] == 0:
            log.warning("frame %d produced no valid depth", obs.frame.index)
            continue

        all_points.append(points)
        all_colors.append(colors)
        detections.extend(
            _detections_for_frame(obs, points, colors, seg_ids, pixel_idx, cfg)
        )

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
            "metric_depth": all(o.depth.metric for o in observations),
            "source_indices": [o.frame.source_index for o in observations],
            "voxel_size": cfg.voxel_size,
        },
    )
    _warn_if_implausible(scene, observations[0].intrinsics)
    lo, hi = scene.bounds
    log.info(
        "scene: %d points, %d objects, %d surfaces, extent %s m",
        points.shape[0],
        len(instances),
        len(surfaces),
        np.round(hi - lo, 2).tolist(),
    )
    return scene
