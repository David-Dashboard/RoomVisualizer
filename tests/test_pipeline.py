"""End-to-end tests against the synthetic room.

These run the real pipeline - back-projection, odometry, clustering,
association, alignment, plane fitting and export - with ground-truth depth and
segmentation standing in for the neural backends.  Everything asserted here is
checked against the room's known dimensions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from synthetic import render_sequence

from roomviz.config import PipelineConfig
from roomviz.fusion.odometry import estimate_trajectory
from roomviz.fusion.scene_fusion import fuse
from roomviz.types import (
    CameraIntrinsics,
    DepthMap,
    Frame,
    Observation,
    Segment,
    Segmentation,
)

WIDTH, HEIGHT, HFOV = 320, 240, 95.0


def build_config(**overrides) -> PipelineConfig:
    cfg = PipelineConfig(
        voxel_size=0.03,
        min_object_points=150,
        plane_min_inliers=600,
        max_side=max(WIDTH, HEIGHT),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture(scope="module")
def rendered():
    return render_sequence(width=WIDTH, height=HEIGHT, hfov=HFOV)


@pytest.fixture(scope="module")
def scene(rendered):
    room, intr, _, frames, depths, segs = rendered
    cfg = build_config()
    depth_maps = [DepthMap(depth=d) for d in depths]
    poses = estimate_trajectory(frames, depth_maps, intr, cfg)
    observations = [
        Observation(frame=f, depth=d, segmentation=s, intrinsics=intr, pose=p)
        for f, d, s, p in zip(frames, depth_maps, segs, poses, strict=True)
    ]
    return fuse(observations, cfg), room, cfg


# --------------------------------------------------------------------------
# odometry
# --------------------------------------------------------------------------

def test_odometry_tracks_the_camera(rendered):
    _, intr, gt_poses, frames, depths, _ = rendered
    poses = estimate_trajectory(
        frames, [DepthMap(depth=d) for d in depths], intr, build_config()
    )
    assert len(poses) == len(gt_poses)

    total_motion = np.linalg.norm(
        gt_poses[-1][:3, 3] - gt_poses[0][:3, 3]
    )
    for i, (estimated, truth) in enumerate(zip(poses, gt_poses, strict=True)):
        relative_truth = np.linalg.inv(gt_poses[0]) @ truth
        translation_error = np.linalg.norm(estimated[:3, 3] - relative_truth[:3, 3])
        rotation = estimated[:3, :3].T @ relative_truth[:3, :3]
        angle = np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        # Frame-to-frame ORB+PnP with no bundle adjustment, so drift grows
        # along the trajectory.  This bound is EMPIRICAL, not derived.
        #
        # Measured on this harness (95 deg HFOV, 320x240, the 6-pose sweep in
        # synthetic.orbit_poses) at commit fe0251f: worst-case translation
        # error 0.036 m, worst-case rotation error 0.60 deg.  0.08 m and
        # 1.5 deg are roughly 2.2x and 2.5x headroom on those.
        #
        # What this bound does and does not catch, all measured on the same
        # trajectory (scratch harness tf_odom_claims.py):
        #   composing the relative pose on the wrong side -> 0.084 m: caught,
        #       but with only a 5% margin.
        #   a 5% error in the translation scale           -> 0.036 m: NOT caught.
        #   intrinsics 10% too long                       -> 0.043 m: NOT caught.
        # The last two are why `test_odometry_recovers_the_path_length_to_scale`
        # exists; a drift bound is simply the wrong instrument for them.  (A
        # previous version of this comment asserted that all three "stay under
        # 12 cm" and that the tight bound was what caught them.  Neither claim
        # was measured and neither is true.)
        assert translation_error < 0.08, f"frame {i} drifted {translation_error:.3f} m"
        assert angle < 1.5, f"frame {i} rotation error {angle:.2f} deg"
    assert total_motion > 1.0  # the trajectory is long enough for this to mean something


def test_odometry_disabled_returns_identity(rendered):
    _, intr, _, frames, depths, _ = rendered
    poses = estimate_trajectory(
        frames, [DepthMap(depth=d) for d in depths], intr,
        build_config(estimate_poses=False),
    )
    assert all(np.allclose(p, np.eye(4)) for p in poses)


# --------------------------------------------------------------------------
# objects
# --------------------------------------------------------------------------

def test_finds_every_object_exactly_once(scene):
    reconstructed, room, _ = scene
    found = sorted(o.label for o in reconstructed.objects)
    assert found == sorted(b.label for b in room.boxes)


def test_objects_are_seen_in_multiple_frames(scene):
    reconstructed, _, _ = scene
    for inst in reconstructed.objects:
        assert inst.observations >= 4, f"{inst.label} only fused {inst.observations} views"


def test_object_footprints_match_ground_truth(scene):
    """Horizontal extent is fully observable for every object in this scene."""
    reconstructed, room, _ = scene
    truth = {b.label: b for b in room.boxes}
    for inst in reconstructed.objects:
        box = truth[inst.label]
        lo, hi = inst.aabb
        size = hi - lo
        for axis, name in ((0, "x"), (2, "z")):
            assert abs(size[axis] - box.size[axis]) < 0.15, (
                f"{inst.label} {name}-extent {size[axis]:.2f} m "
                f"vs ground truth {box.size[axis]:.2f} m"
            )


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform mapping ``source`` onto ``target``."""
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    # Guard against a reflection sneaking in when the points are near-planar.
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ correction @ u.T
    return rotation, target_centre - rotation @ source_centre


def _register_to_room(reconstructed, room):
    """Rigid transform from the reconstruction's frame into room coordinates.

    Gravity alignment fixes the up axis and puts the floor at y = 0, but the
    horizontal origin and heading are arbitrary (Manhattan alignment is only
    defined up to a quarter turn).  Comparing against the room's own
    coordinates therefore requires registering the two frames first.  The three
    fully-visible objects are used as correspondences; the bookcase is left out
    because its occluded base biases its centre.
    """
    truth = {b.label: b for b in room.boxes}
    source, target = [], []
    for label in ("table", "chair", "sofa"):
        inst = next(o for o in reconstructed.objects if o.label == label)
        lo, hi = inst.aabb
        source.append((lo + hi) / 2)
        target.append((truth[label].lo + truth[label].hi) / 2)
    return _kabsch(np.array(source), np.array(target))


def test_objects_do_not_hallucinate_geometry(scene):
    """Reconstructed points must lie inside the true object, within tolerance.

    Under-reporting an occluded surface is acceptable; inventing geometry that
    is not there is not.
    """
    reconstructed, room, _ = scene
    rotation, translation = _register_to_room(reconstructed, room)
    truth = {b.label: b for b in room.boxes}

    for inst in reconstructed.objects:
        box = truth[inst.label]
        registered = inst.points @ rotation.T + translation
        assert (registered.min(axis=0) > box.lo - 0.12).all(), (
            f"{inst.label} extends outside its true box at "
            f"{np.round(registered.min(axis=0), 3)} (box starts at {box.lo})"
        )
        assert (registered.max(axis=0) < box.hi + 0.12).all(), (
            f"{inst.label} extends outside its true box at "
            f"{np.round(registered.max(axis=0), 3)} (box ends at {box.hi})"
        )


def test_object_layout_is_metrically_correct(scene):
    """Distances between objects are frame-independent, so check them directly."""
    reconstructed, room, _ = scene
    truth = {b.label: b for b in room.boxes}

    def centre(inst):
        lo, hi = inst.aabb
        return (lo + hi) / 2

    objects = reconstructed.objects
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            measured = np.linalg.norm(centre(a) - centre(b))
            expected = np.linalg.norm(
                (truth[a.label].lo + truth[a.label].hi) / 2
                - (truth[b.label].lo + truth[b.label].hi) / 2
            )
            assert abs(measured - expected) < 0.2, (
                f"{a.label}-{b.label} distance {measured:.2f} m "
                f"vs ground truth {expected:.2f} m"
            )


def test_objects_rest_on_the_floor(scene):
    reconstructed, _, _ = scene
    for inst in reconstructed.objects:
        lo, _ = inst.aabb
        assert lo[1] < 0.55, f"{inst.label} floats at y={lo[1]:.2f}"


def test_unoccluded_object_heights_match(scene):
    """The table, chair and sofa are fully visible, so heights should match.

    The bookcase is deliberately left out: the table occludes its lower half
    from every camera position, so its true height is not observable.
    """
    reconstructed, room, _ = scene
    truth = {b.label: b for b in room.boxes}
    for label in ("table", "chair", "sofa"):
        inst = next(o for o in reconstructed.objects if o.label == label)
        lo, hi = inst.aabb
        assert abs((hi[1] - lo[1]) - truth[label].size[1]) < 0.15


def test_occluded_object_reports_what_it_saw(scene):
    """The bookcase's visible top is recovered even though its base is hidden."""
    reconstructed, room, _ = scene
    box = next(b for b in room.boxes if b.label == "bookcase")
    inst = next(o for o in reconstructed.objects if o.label == "bookcase")
    _, hi = inst.aabb
    assert abs(hi[1] - box.hi[1]) < 0.15


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_floor_and_ceiling_recovered(scene):
    reconstructed, room, _ = scene
    kinds = [s.kind for s in reconstructed.surfaces]
    assert kinds.count("floor") == 1
    assert kinds.count("ceiling") == 1

    floor = next(s for s in reconstructed.surfaces if s.kind == "floor")
    ceiling = next(s for s in reconstructed.surfaces if s.kind == "ceiling")
    assert abs(float(np.mean(floor.quad[:, 1]))) < 0.05
    assert abs(float(np.mean(ceiling.quad[:, 1])) - room.height) < 0.08


def test_walls_are_vertical_and_axis_aligned(scene):
    reconstructed, _, _ = scene
    walls = [s for s in reconstructed.surfaces if s.kind == "wall"]
    assert len(walls) >= 3
    for wall in walls:
        # Vertical: the normal must be perpendicular to up.
        assert abs(float(wall.normal[1])) < 0.2
        # Manhattan-aligned: the normal points along x or z, not diagonally.
        horizontal = np.abs([wall.normal[0], wall.normal[2]])
        assert horizontal.max() > 0.93, f"wall normal {wall.normal} is not axis aligned"


def test_scene_is_gravity_aligned(scene):
    reconstructed, room, _ = scene
    low, high = reconstructed.bounds
    assert abs(low[1]) < 0.1, "floor should sit at y = 0"
    assert abs(high[1] - room.height) < 0.1, "ceiling should sit at the room height"


def test_room_width_matches(scene):
    reconstructed, room, _ = scene
    low, high = reconstructed.bounds
    assert abs((high[0] - low[0]) - room.width) < 0.25


def test_alignment_can_be_disabled(rendered):
    room, intr, _, frames, depths, segs = rendered
    cfg = build_config(align_gravity=False)
    depth_maps = [DepthMap(depth=d) for d in depths]
    poses = estimate_trajectory(frames, depth_maps, intr, cfg)
    observations = [
        Observation(frame=f, depth=d, segmentation=s, intrinsics=intr, pose=p)
        for f, d, s, p in zip(frames, depth_maps, segs, poses, strict=True)
    ]
    result = fuse(observations, cfg)
    assert result.meta["aligned"] is False
    assert np.allclose(result.poses[0], np.eye(4))


# --------------------------------------------------------------------------
# single-image path
# --------------------------------------------------------------------------

def test_single_frame_reconstruction(rendered):
    _, intr, poses, frames, depths, segs = rendered
    cfg = build_config()
    observation = Observation(
        frame=frames[0],
        depth=DepthMap(depth=depths[0]),
        segmentation=segs[0],
        intrinsics=intr,
        pose=np.eye(4),
    )
    result = fuse([observation], cfg)
    assert len(result.objects) == 4
    assert any(s.kind == "floor" for s in result.surfaces)
    assert result.points.shape[0] > 1000


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def test_export_writes_every_artefact(scene, tmp_path):
    from roomviz.pipeline import export_scene

    reconstructed, _, cfg = scene
    files = export_scene(reconstructed, tmp_path, cfg)

    assert (tmp_path / "scene.json").exists()
    assert (tmp_path / "scene.ply").exists()
    assert (tmp_path / "scene.glb").exists()
    assert (tmp_path / "viewer.html").exists()
    assert (tmp_path / "vendor" / "three.module.js").exists()
    assert (tmp_path / "vendor" / "jsm" / "loaders" / "GLTFLoader.js").exists()

    object_files = sorted((tmp_path / "objects").glob("*.ply"))
    assert len(object_files) == len(reconstructed.objects)
    assert files["scene_glb"].stat().st_size > 1000


def test_scene_json_describes_the_room(scene, tmp_path):
    from roomviz.export.scene_json import write_scene_json

    reconstructed, room, cfg = scene
    path = write_scene_json(tmp_path / "scene.json", reconstructed, cfg)
    payload = json.loads(path.read_text())

    assert payload["units"] == "metres"
    assert payload["up_axis"] == "+Y"
    assert payload["summary"]["object_count"] == 4
    assert set(payload["summary"]["labels"]) == {b.label for b in room.boxes}
    # Measured error is ~1 cm; a 10 cm tolerance would let a 3% inflation pass.
    assert payload["room"]["room_height"] == pytest.approx(room.height, abs=0.05)

    for entry in payload["objects"]:
        assert len(entry["color"]) == 3
        assert entry["node"].startswith("object__")
        assert entry["point_count"] > 0
    for entry in payload["surfaces"]:
        assert entry["kind"] in {"wall", "floor", "ceiling"}
        assert len(entry["quad"]) == 4
    assert len(payload["cameras"]) == len(reconstructed.poses)


def test_ply_roundtrip(tmp_path):
    from roomviz.export.ply import write_ply

    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], np.float32)
    colors = np.array([[10, 20, 30], [40, 50, 60]], np.uint8)
    path = write_ply(tmp_path / "cloud.ply", points, colors)

    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    assert b"element vertex 2" in header
    values = np.frombuffer(
        body,
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("r", "u1"), ("g", "u1"), ("b", "u1")]),
    )
    assert np.allclose(np.stack([values["x"], values["y"], values["z"]], 1), points)
    assert values["r"].tolist() == [10, 40]


def test_ply_rejects_mismatched_colors(tmp_path):
    from roomviz.export.ply import write_ply

    with pytest.raises(ValueError):
        write_ply(tmp_path / "bad.ply", np.zeros((4, 3)), np.zeros((2, 3), np.uint8))


def test_glb_nodes_are_named_for_the_viewer(scene, tmp_path):
    import trimesh

    from roomviz.export.gltf import write_glb

    reconstructed, _, _ = scene
    path = write_glb(tmp_path / "scene.glb", reconstructed)
    loaded = trimesh.load(str(path), force="scene")

    names = set(loaded.geometry.keys()) | set(loaded.graph.nodes)
    assert any(n == "cloud" or n.endswith("cloud") for n in names)
    assert sum(1 for n in names if n.startswith("object__")) == len(reconstructed.objects)
    assert sum(1 for n in names if n.startswith("box__")) == len(reconstructed.objects)
    assert sum(1 for n in names if n.startswith("surface__")) == len(reconstructed.surfaces)


# --------------------------------------------------------------------------
# no over-claiming, no silent under-reporting
# --------------------------------------------------------------------------

def test_scene_never_claims_more_room_than_exists(scene):
    """Reported bounds must not exceed the true room.

    Only the width is fully observable from this trajectory: the camera starts
    near the z=0 wall looking away from it, so the near wall is never seen and
    the recovered depth is legitimately short.  What must never happen is the
    reverse - reporting a room *larger* than it is.
    """
    reconstructed, room, _ = scene
    low, high = reconstructed.bounds
    extent = high - low
    assert extent[0] <= room.width + 0.25, extent
    assert extent[1] <= room.height + 0.15, extent
    assert extent[2] <= room.depth + 0.25, extent
    # And the observable part must actually be recovered, not collapsed.
    assert extent[0] > room.width - 0.25, extent
    assert extent[2] > 2.0, extent


def test_objects_are_not_silently_shrunk(scene):
    """Containment alone is one-sided: a shrunken object would still pass.

    For the three fully-visible objects the reconstruction must *fill* most of
    the true box, not merely stay inside it.
    """
    reconstructed, room, _ = scene
    rotation, translation = _register_to_room(reconstructed, room)
    truth = {b.label: b for b in room.boxes}

    for label in ("table", "chair", "sofa"):
        inst = next(o for o in reconstructed.objects if o.label == label)
        registered = inst.points @ rotation.T + translation
        lo = np.maximum(registered.min(axis=0), truth[label].lo)
        hi = np.minimum(registered.max(axis=0), truth[label].hi)
        covered = np.prod(np.maximum(hi - lo, 0.0))
        true_volume = np.prod(truth[label].size)
        assert covered / true_volume > 0.7, (
            f"{label} fills only {covered / true_volume:.0%} of its true volume"
        )


def test_observed_floor_area_is_labelled_as_observed(scene, tmp_path):
    """The floor patch is not the room's floor area, and must not claim to be.

    Filming half a room halves this number with no other symptom, so two
    captures of the same room legitimately disagree.  The field name and an
    explicit coverage figure are what stop it being read as gross area.
    """
    from roomviz.export.scene_json import build_scene_dict

    reconstructed, room, cfg = scene
    payload = build_scene_dict(reconstructed, cfg)

    assert "floor_area" not in payload["room"], "the bare name invites misreading"
    observed = payload["room"]["observed_floor_area"]
    true_area = room.width * room.depth
    # The camera cannot see the whole floor from this trajectory, so the
    # observed patch must be a strict under-estimate -- that is the point.
    assert observed < true_area
    assert payload["room"]["floor_area_is_observed_only"] is True
    assert 0.0 < payload["room"]["floor_coverage"] <= 1.0


def test_caveats_are_machine_readable(rendered):
    """Warnings that only reach stderr are invisible to a batch pipeline."""
    from roomviz.export.scene_json import build_scene_dict

    _, intr, _, frames, depths, segs = rendered
    cfg = build_config()
    depth_maps = [DepthMap(depth=d) for d in depths]
    poses = estimate_trajectory(frames, depth_maps, intr, cfg)
    observations = [
        Observation(frame=f, depth=d, segmentation=s, intrinsics=intr, pose=p)
        for f, d, s, p in zip(frames, depth_maps, segs, poses, strict=True)
    ]
    good = build_scene_dict(fuse(observations, cfg), cfg)
    assert isinstance(good["caveats"], list)
    assert not any(c.startswith("implausible_height") for c in good["caveats"])
    assert not any(c.startswith("camera_assumed") for c in good["caveats"])

    # A guessed camera must be declared in the file, not just on stderr.
    intr.provenance = "assumed_default"
    guessed = build_scene_dict(fuse(observations, cfg), cfg)
    assert any(c.startswith("camera_assumed") for c in guessed["caveats"])

    # So must an unaligned scene.
    unaligned_cfg = build_config(align_gravity=False)
    unaligned = build_scene_dict(fuse(observations, unaligned_cfg), unaligned_cfg)
    assert any(c.startswith("not_gravity_aligned") for c in unaligned["caveats"])
    assert unaligned["up_axis"] != "+Y"


def test_default_config_still_reconstructs(rendered):
    """The shipped defaults must work, not just the tuned test config.

    `build_config` overrides voxel size, the object-point floor and the plane
    inlier floor; without this the suite would never exercise what a user
    actually gets from `PipelineConfig()`.
    """
    _, intr, _, frames, depths, segs = rendered
    cfg = PipelineConfig()
    depth_maps = [DepthMap(depth=d) for d in depths]
    poses = estimate_trajectory(frames, depth_maps, intr, cfg)
    result = fuse(
        [
            Observation(frame=f, depth=d, segmentation=s, intrinsics=intr, pose=p)
            for f, d, s, p in zip(frames, depth_maps, segs, poses, strict=True)
        ],
        cfg,
    )
    assert sorted(o.label for o in result.objects) == ["bookcase", "chair", "sofa", "table"]
    assert any(s.kind == "floor" for s in result.surfaces)


def test_survives_depth_noise(rendered):
    """1% multiplicative depth noise is mild next to any real depth sensor."""
    room, intr, _, frames, depths, segs = rendered
    rng = np.random.default_rng(5)
    noisy = [
        DepthMap(depth=(d * (1.0 + rng.normal(0, 0.01, d.shape))).astype(np.float32))
        for d in depths
    ]
    cfg = build_config()
    poses = estimate_trajectory(frames, noisy, intr, cfg)
    result = fuse(
        [
            Observation(frame=f, depth=d, segmentation=s, intrinsics=intr, pose=p)
            for f, d, s, p in zip(frames, noisy, segs, poses, strict=True)
        ],
        cfg,
    )
    assert sorted(o.label for o in result.objects) == ["bookcase", "chair", "sofa", "table"]
    truth = {b.label: b for b in room.boxes}
    for inst in result.objects:
        lo, hi = inst.aabb
        size = hi - lo
        for axis in (0, 2):
            assert abs(size[axis] - truth[inst.label].size[axis]) < 0.25, inst.label


@pytest.fixture(scope="module")
def rendered_at_default_fov():
    """The same room shot at the package's *default* 60 degree HFOV."""
    return render_sequence(width=WIDTH, height=HEIGHT, hfov=60.0)


def test_survives_a_narrow_field_of_view(rendered_at_default_fov):
    """A 60-degree lens - the package default - sees much less of the room.

    Objects survive: every one is still found and its horizontal extent is
    still recovered.  Structure does not: the ceiling leaves the frame, so
    only the floor is asserted here.  And the *odometry* does not either - see
    `test_odometry_meets_its_own_bound_at_the_default_field_of_view`, which is
    an xfail recording that the pipeline misses its own drift bound at its own
    default.  This test deliberately does not assert drift; that assertion
    lives in the xfail so the failure is visible rather than omitted.
    """
    room, intr, _, frames, depths, segs = rendered_at_default_fov
    cfg = build_config()
    depth_maps = [DepthMap(depth=d) for d in depths]
    poses = estimate_trajectory(frames, depth_maps, intr, cfg)
    result = fuse(
        [
            Observation(frame=f, depth=d, segmentation=s, intrinsics=intr, pose=p)
            for f, d, s, p in zip(frames, depth_maps, segs, poses, strict=True)
        ],
        cfg,
    )
    assert sorted(o.label for o in result.objects) == ["bookcase", "chair", "sofa", "table"]
    assert any(s.kind == "floor" for s in result.surfaces)
    truth = {b.label: b for b in room.boxes}
    for inst in result.objects:
        lo, hi = inst.aabb
        assert abs((hi - lo)[0] - truth[inst.label].size[0]) < 0.25, inst.label


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEASURED FAILURE, not a flake. At the package default 60 degree HFOV "
        "the worst frame-to-frame odometry drift on this trajectory is 0.119 m "
        "against the 0.08 m bound asserted by test_odometry_tracks_the_camera, "
        "and the worst rotation error is 1.69 deg against 1.5 deg. The harness "
        "hides this by shooting at 95 deg, where the same numbers are 0.036 m "
        "and 0.60 deg. Full sweep (320x240, 6-pose orbit, commit fe0251f): "
        "50 deg 0.120 m, 55 deg 0.101, 60 deg 0.119, 65 deg 0.120, 70 deg 0.078, "
        "75 deg 0.054, 80 deg 0.041, 85 deg 0.039, 95 deg 0.036. Drift roughly "
        "triples between 70 and 65 degrees. Remove the xfail when odometry "
        "meets its bound at the default FOV, or when the bound is honestly "
        "widened to cover it."
    ),
)
def test_odometry_meets_its_own_bound_at_the_default_field_of_view(
    rendered_at_default_fov,
):
    _, intr, gt_poses, frames, depths, _ = rendered_at_default_fov
    poses = estimate_trajectory(
        frames, [DepthMap(depth=d) for d in depths], intr, build_config()
    )
    for i, (estimated, truth) in enumerate(zip(poses, gt_poses, strict=True)):
        relative_truth = np.linalg.inv(gt_poses[0]) @ truth
        translation_error = np.linalg.norm(estimated[:3, 3] - relative_truth[:3, 3])
        rotation = estimated[:3, :3].T @ relative_truth[:3, :3]
        angle = np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        assert translation_error < 0.08, f"frame {i} drifted {translation_error:.3f} m"
        assert angle < 1.5, f"frame {i} rotation error {angle:.2f} deg"


def test_odometry_recovers_the_path_length_to_scale(rendered):
    """Metric scale, which the drift bound above cannot see.

    A pure scale error in the recovered translations barely shows up as
    absolute drift: the clean run already under-reports the path by 2.45%, so
    a +5% scale error lands at +2.42% on the other side of unity and the
    symmetric 0.08 m drift bound never notices it (measured: 0.036 m clean,
    0.036 m with the 5% error).  Comparing the *length* of the recovered path
    against the truth is the instrument that does see it.

    Measured on this harness at commit fe0251f, and bit-for-bit repeatable
    across five runs: recovered/true path length = 0.97547 (true path 1.4175 m,
    recovered 1.3827 m).  The band below is that value plus or minus 0.025.
    A 5% translation-scale error gives 1.0242
    and fails it; intrinsics 10% short give 0.9141 and fail it; composing the
    relative pose on the wrong side gives 0.9420 and fails it.  Intrinsics 10%
    *long* give 0.9817 and are NOT caught - PnP absorbs most of that into
    rotation on this trajectory.
    """
    _, intr, gt_poses, frames, depths, _ = rendered
    poses = estimate_trajectory(
        frames, [DepthMap(depth=d) for d in depths], intr, build_config()
    )

    def length(chain):
        return sum(
            float(np.linalg.norm(chain[i][:3, 3] - chain[i - 1][:3, 3]))
            for i in range(1, len(chain))
        )

    ratio = length(poses) / length(gt_poses)
    assert 0.9505 < ratio < 1.0005, (
        f"recovered path is {ratio:.4f} of the true 1.4175 m; the odometry "
        "chain has picked up a scale error"
    )


def test_objects_sharing_one_segment_id_are_separated(rendered):
    """Panoptic stuff masks hold several objects; all of them must survive.

    Deleting the 3D split entirely used to leave the suite green, because the
    synthetic room gives every box its own segment id and so never exercises
    the case the split exists for.
    """
    from synthetic import FIRST_OBJECT_ID

    _, intr, _, frames, depths, segs = rendered
    cfg = build_config()

    merged_segs = []
    for seg in segs:
        ids = seg.ids.copy()
        object_ids = [s.segment_id for s in seg.segments if s.role == "object"]
        for sid in object_ids:
            ids[ids == sid] = FIRST_OBJECT_ID
        segments = [s for s in seg.segments if s.role != "object"]
        shared = next(s for s in seg.segments if s.role == "object")
        segments.append(
            Segment(
                segment_id=FIRST_OBJECT_ID,
                label="clutter",
                role="object",
                structure_kind=None,
                is_thing=False,   # stuff: one mask covering several objects
            )
        )
        merged_segs.append(Segmentation(ids=ids, segments=segments))
        del shared

    poses = estimate_trajectory(frames, [DepthMap(depth=d) for d in depths], intr, cfg)
    result = fuse(
        [
            Observation(frame=f, depth=DepthMap(depth=d), segmentation=s,
                        intrinsics=intr, pose=p)
            for f, d, s, p in zip(frames, depths, merged_segs, poses, strict=True)
        ],
        cfg,
    )
    # All four pieces of furniture arrived in one mask; all four must come out.
    assert len(result.objects) == 4, [
        (o.label, o.points.shape[0]) for o in result.objects
    ]


# --------------------------------------------------------------------------
# claims the README makes that nothing was checking
# --------------------------------------------------------------------------

def _one_mask_over_everything(segs, *, is_thing: bool):
    """Relabel every object segment into a single mask with the given flag."""
    from synthetic import FIRST_OBJECT_ID

    out = []
    for seg in segs:
        ids = seg.ids.copy()
        for sid in [s.segment_id for s in seg.segments if s.role == "object"]:
            ids[ids == sid] = FIRST_OBJECT_ID
        segments = [s for s in seg.segments if s.role != "object"]
        segments.append(
            Segment(
                segment_id=FIRST_OBJECT_ID,
                label="clutter",
                role="object",
                structure_kind=None,
                is_thing=is_thing,
            )
        )
        out.append(Segmentation(ids=ids, segments=segments))
    return out


def _fuse_with_segmentations(rendered, segmentations, cfg):
    _, intr, _, frames, depths, _ = rendered
    depth_maps = [DepthMap(depth=d) for d in depths]
    poses = estimate_trajectory(frames, depth_maps, intr, cfg)
    return fuse(
        [
            Observation(frame=f, depth=d, segmentation=s, intrinsics=intr, pose=p)
            for f, d, s, p in zip(frames, depth_maps, segmentations, poses, strict=True)
        ],
        cfg,
    )


PAIR_WIDTH, PAIR_HEIGHT = 200, 120
PAIR_DEPTH, PAIR_BACKGROUND = 2.0, 6.0


def _two_pieces_of_one_mask(gap_pixels: int, occluder: bool = False):
    """One segment id covering two rectangles, separated by ``gap_pixels``.

    Built by hand rather than rendered because the whole point is a *precisely
    known* gap: at 90 degrees over 200 px, fx is 100, so one pixel at 2 m is
    exactly 2 cm of world space.
    """
    intr = CameraIntrinsics.from_hfov(PAIR_WIDTH, PAIR_HEIGHT, 90.0)
    depth = np.full((PAIR_HEIGHT, PAIR_WIDTH), PAIR_BACKGROUND, np.float32)
    ids = np.full((PAIR_HEIGHT, PAIR_WIDTH), 1, np.int32)  # background wall

    piece_width, mid = 40, PAIR_WIDTH // 2
    for x0 in (mid - gap_pixels // 2 - piece_width, mid + gap_pixels // 2):
        depth[30:90, x0 : x0 + piece_width] = PAIR_DEPTH
        ids[30:90, x0 : x0 + piece_width] = 7

    if occluder:
        # Something nearer standing in the gap: the two pieces are one object
        # seen past an obstruction.
        depth[30:90, mid - gap_pixels // 2 : mid + gap_pixels // 2] = PAIR_DEPTH - 0.5

    return intr, depth, ids


def _fuse_one_mask(gap_pixels: int, thing: bool | None, occluder: bool = False) -> int:
    intr, depth, ids = _two_pieces_of_one_mask(gap_pixels, occluder)
    segments = [
        Segment(segment_id=1, label="wall", role="structure", structure_kind="wall"),
        Segment(segment_id=7, label="widget", role="object", is_thing=thing),
    ]
    cfg = build_config(
        voxel_size=0.02,
        min_object_points=50,
        plane_min_inliers=100_000,   # no plane fitting; only the objects matter
        align_gravity=False,         # world == camera frame, so the gap is exact
        object_split_gap=0.12,       # the shipped default
        max_side=PAIR_WIDTH,
    )
    observation = Observation(
        frame=Frame(index=0, rgb=np.zeros((PAIR_HEIGHT, PAIR_WIDTH, 3), np.uint8)),
        depth=DepthMap(depth=depth),
        segmentation=Segmentation(ids=ids, segments=segments),
        intrinsics=intr,
        pose=np.eye(4),
    )
    return len(fuse([observation], cfg).objects)


@pytest.mark.parametrize(
    "gap_metres,gap_pixels,thing,stuff,unknown",
    [
        # gap    px   thing stuff unknown   (measured at commit a0b1423)
        (0.12,    6,      1,    2,      1),
        (0.16,    8,      1,    2,      2),
        (0.20,   10,      2,    2,      2),
    ],
)
def test_the_thing_stuff_flag_sets_the_split_distance_not_a_veto(
    gap_metres, gap_pixels, thing, stuff, unknown
):
    """README: "The segmenter's instance decision sets a threshold, not a veto."

    A "thing" mask is split too, but only across a wider gap than a "stuff"
    mask needs, with unknown classes in between.  Nothing was checking that
    ordering from outside the fusion module, and the ordering is the whole
    claim: if all three thresholds were equal the flag would carry no
    information at all, and if `thing` were still a veto the middle row here
    would read 1/2/2 forever.

    Measured on this fixture at `object_split_gap` 0.12 m, the shipped default
    (the realised world gap runs a little wider than the nominal pixel gap
    because the occlusion-edge filter eats the patch borders):

        gap    thing  stuff  unknown
        0.08     1      2       1
        0.12     1      2       1
        0.16     1      2       2
        0.20     2      2       2
        0.24+    2      2       2
    """
    assert _fuse_one_mask(gap_pixels, thing=True) == thing, "thing"
    assert _fuse_one_mask(gap_pixels, thing=False) == stuff, "stuff"
    assert _fuse_one_mask(gap_pixels, thing=None) == unknown, "unknown"


def test_pieces_with_a_nearer_surface_between_them_are_one_object():
    """README: "if the pixels between two pieces belong to a nearer surface,
    they are one object seen past an obstruction, and are rejoined."

    A 0.60 m gap is five times `object_split_gap` and splits every class when
    it is empty (measured: 2 objects for thing, stuff and unknown alike).  Fill
    it with a surface 0.5 m nearer and all three collapse to one, which is the
    only thing separating "two chairs" from "one sofa behind a pillar".
    """
    for thing in (True, False, None):
        assert _fuse_one_mask(30, thing=thing, occluder=False) == 2, thing
        assert _fuse_one_mask(30, thing=thing, occluder=True) == 1, thing


def test_occlusion_edge_points_are_discarded(rendered):
    """README: "Pixels sitting on a large local depth jump are dropped."

    `test_depth_edge_mask_flags_discontinuities` checks the mask function in
    isolation; nothing checked that the pipeline applies it.  A hard synthetic
    step has no flying pixels to drop, so the depth map here has the ramp that
    a real (interpolated, upsampled) depth map puts across an occlusion
    boundary - the pixels that back-project into empty space.

    Gravity alignment is off so world coordinates are camera coordinates and
    the assertion can be made directly on z.
    """
    _, intr, _, frames, _, _ = rendered
    height, width = frames[0].rgb.shape[:2]

    near, far = 1.0, 4.0
    depth = np.full((height, width), near, np.float32)
    depth[:, width // 2 + 3 :] = far
    # Three columns of interpolated "flying pixels" bridging the jump.
    for k in range(3):
        depth[:, width // 2 + k] = near + (far - near) * (k + 1) / 4.0

    segmentation = Segmentation(
        ids=np.zeros((height, width), np.int32),
        segments=[Segment(segment_id=0, label="wall", role="structure", structure_kind="wall")],
    )
    observation = Observation(
        frame=frames[0],
        depth=DepthMap(depth=depth),
        segmentation=segmentation,
        intrinsics=intr,
        pose=np.eye(4),
    )

    kept = fuse([observation], build_config(align_gravity=False, edge_discard=0.06))
    dropped_off = fuse([observation], build_config(align_gravity=False, edge_discard=0.0))

    bridging = lambda scene: int(  # noqa: E731
        ((scene.points[:, 2] > near + 0.2) & (scene.points[:, 2] < far - 0.2)).sum()
    )
    # Control first: with the filter disabled the flying pixels are there.
    assert bridging(dropped_off) > 100, "the fixture produced no flying pixels to drop"
    assert bridging(kept) == 0, (
        f"{bridging(kept)} points survive in the empty space between "
        f"{near} m and {far} m"
    )
    # And the real surfaces on both sides are kept, not thrown away with them.
    assert (np.abs(kept.points[:, 2] - near) < 0.05).sum() > 500
    assert (np.abs(kept.points[:, 2] - far) < 0.05).sum() > 500


def test_camera_poses_are_camera_to_world(scene):
    """README/scene.json: poses are camera-to-world, row-major.

    Nothing checked the *direction*.  Inverting the matrix, or transposing it,
    leaves it a perfectly valid rigid transform, and every other assertion in
    the suite is insensitive to which way round it is.  The check that is not:
    ``inv(pose)`` must carry the reconstructed scene into the camera's own
    frame, where the points that camera saw are in front of it (z > 0) and
    land inside the image rectangle.  A world-to-camera matrix put here sends
    them behind the camera instead.
    """
    reconstructed, _, _ = scene
    points = reconstructed.points
    intr = reconstructed.intrinsics
    assert intr is not None

    for i, pose in enumerate(reconstructed.poses):
        rotation, translation = pose[:3, :3], pose[:3, 3]
        # It is a rigid transform at all.
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6), i
        assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-6), i

        camera_frame = (points - translation) @ rotation  # inv(pose) applied
        in_front = camera_frame[:, 2] > 0
        assert in_front.mean() > 0.75, (
            f"camera {i}: only {in_front.mean():.0%} of the scene is in front "
            "of it; the pose looks inverted"
        )
        visible = camera_frame[in_front]
        u = visible[:, 0] / visible[:, 2] * intr.fx + intr.cx
        v = visible[:, 1] / visible[:, 2] * intr.fy + intr.cy
        inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
        assert inside.mean() > 0.5, (
            f"camera {i}: only {inside.mean():.0%} of what is in front of it "
            "projects into the image"
        )

    # The camera centres are the translation column, and they trace the sweep.
    centres = np.array([p[:3, 3] for p in reconstructed.poses])
    travel = float(np.linalg.norm(centres[-1] - centres[0]))
    assert travel > 1.0, f"cameras barely moved ({travel:.2f} m)"


def test_camera_block_in_scene_json_matches_the_scene_poses(scene, tmp_path):
    """The exported matrix must be the pose itself, row-major, not a transpose."""
    from roomviz.export.scene_json import build_scene_dict

    reconstructed, _, cfg = scene
    payload = build_scene_dict(reconstructed, cfg)
    assert len(payload["cameras"]) == len(reconstructed.poses)
    for entry, pose in zip(payload["cameras"], reconstructed.poses, strict=True):
        exported = np.array(entry["matrix_row_major"]).reshape(4, 4)
        assert np.allclose(exported, pose, atol=1e-5)
        assert np.allclose(entry["position"], pose[:3, 3], atol=1e-4)
        # A transposed rotation would still be orthonormal, so compare against
        # the transpose explicitly - it must NOT match unless the pose happens
        # to be symmetric, which it is not for this trajectory.
        assert not np.allclose(exported[:3, :3], pose[:3, :3].T, atol=1e-3)


def test_observed_floor_area_tracks_how_much_floor_was_filmed(rendered):
    """README: "film half a room and it halves, with no other symptom".

    The existing floor-area test checks the *naming* and that the number is
    below the true area.  It does not check the semantics that make the name
    necessary - that the number tracks how much floor was in shot rather than
    how big the room is.

    Here the same six views of the same room are reconstructed twice, the
    second time with the bottom 35% of every depth map thrown away, which is
    where the near floor is.  Measured: 14.008 m^2 of observed floor becomes
    5.163 m^2 - a 2.7x difference - while the reported room extent moves from
    5.02 x 3.14 m to 5.02 x 2.52 m and the room height not at all.  That is
    exactly the "no other symptom" the README warns about, and it is why the
    field is not called `floor_area`.
    """
    from roomviz.export.scene_json import build_scene_dict

    _, intr, _, frames, depths, segs = rendered
    cfg = build_config()
    depth_maps = [DepthMap(depth=d) for d in depths]
    poses = estimate_trajectory(frames, depth_maps, intr, cfg)

    def build(crop: float | None):
        observations = []
        for f, d, s, p in zip(frames, depths, segs, poses, strict=True):
            depth = d.copy()
            if crop is not None:
                depth[int(depth.shape[0] * crop) :, :] = 0.0
            observations.append(
                Observation(
                    frame=f, depth=DepthMap(depth=depth), segmentation=s,
                    intrinsics=intr, pose=p,
                )
            )
        return build_scene_dict(fuse(observations, cfg), cfg)

    full = build(None)
    partial = build(0.65)

    assert partial["room"]["observed_floor_area"] < 0.5 * full["room"]["observed_floor_area"]
    # ... while the room itself is reported as very nearly the same size.
    assert partial["room"]["room_height"] == pytest.approx(
        full["room"]["room_height"], abs=0.02
    )
    assert partial["room"]["extent"][0] == pytest.approx(full["room"]["extent"][0], abs=0.05)

    for payload in (full, partial):
        assert payload["room"]["floor_area_is_observed_only"] is True
        # It is the floor *plane's* area, not the reconstruction's footprint,
        # and floor_coverage is exactly the ratio of the two.
        floor_area = payload["room"]["observed_floor_area"]
        extent = payload["room"]["extent"]
        assert floor_area <= extent[0] * extent[2] + 1e-6
        assert payload["room"]["floor_coverage"] == pytest.approx(
            min(1.0, floor_area / (extent[0] * extent[2])), abs=0.002
        )

    # Only the partial capture is flagged; a false alarm on the full one would
    # make the flag useless.
    assert any(c.startswith("partial_floor") for c in partial["caveats"])
    assert not any(c.startswith("partial_floor") for c in full["caveats"])
