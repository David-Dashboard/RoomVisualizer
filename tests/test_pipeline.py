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
from roomviz.types import DepthMap, Observation

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
        # Frame-to-frame ORB+PnP with no bundle adjustment: allow drift to grow,
        # but it must stay a small fraction of the distance travelled.
        assert translation_error < 0.12, f"frame {i} drifted {translation_error:.3f} m"
        assert angle < 3.0, f"frame {i} rotation error {angle:.2f} deg"
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
    assert payload["room"]["room_height"] == pytest.approx(room.height, abs=0.1)

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
