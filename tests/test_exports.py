"""Export-layer behaviour that end-to-end runs never reach.

Every end-to-end scene is a good one: a floor, a ceiling, a plausible height,
metric depth and gravity alignment.  So none of `scene.json`'s caveats fire,
none of the "do not reference an artefact this run did not write" branches are
taken, and none of the output-hygiene code in `pipeline.py` runs.  Those are
exactly the paths that matter when a reconstruction goes wrong, which is when
someone reads the JSON.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from roomviz.config import PipelineConfig
from roomviz.export.scene_json import build_scene_dict, write_scene_json
from roomviz.pipeline import (
    _clear_previous_object_clouds,
    _remove_stale,
    object_cloud_name,
)
from roomviz.types import CameraIntrinsics, ObjectInstance, PlaneSurface, Scene


def _quad(y: float, size: float = 3.0) -> np.ndarray:
    return np.array(
        [[0.0, y, 0.0], [size, y, 0.0], [size, y, size], [0.0, y, size]], np.float32
    )


def _surface(kind: str, y: float, area: float = 9.0, inliers: int = 1000) -> PlaneSurface:
    return PlaneSurface(
        surface_id=0,
        kind=kind,
        normal=np.array([0.0, 1.0, 0.0], np.float32),
        offset=-y,
        quad=_quad(y),
        inlier_count=inliers,
        area=area,
    )


def _scene(surfaces=None, height=2.6, objects=None, meta=None, provenance="hfov_flag"):
    rng = np.random.default_rng(0)
    points = rng.uniform([0, 0, 0], [3.0, height, 3.0], (398, 3))
    # Pin the two extreme corners so the scene bounds - and therefore the
    # footprint, the extent and the room height the caveats are computed from -
    # are exactly 3.0 x height x 3.0 rather than whatever the sample reached.
    points = np.vstack([points, [0.0, 0.0, 0.0], [3.0, height, 3.0]]).astype(np.float32)
    intrinsics = CameraIntrinsics.from_hfov(64, 48, 70.0)
    intrinsics.provenance = provenance
    return Scene(
        points=points,
        colors=np.zeros((400, 3), np.uint8),
        objects=objects or [],
        surfaces=surfaces if surfaces is not None else [_surface("floor", 0.0), _surface("ceiling", height)],
        poses=[np.eye(4)],
        intrinsics=intrinsics,
        meta={"frames": 1, **(meta or {})},
    )


def _object(label="chair", instance_id=0):
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.8, 0.4]], np.float32)
    return ObjectInstance(
        instance_id=instance_id,
        label=label,
        points=points,
        colors=np.zeros((2, 3), np.uint8),
        frame_indices=[0],
    )


# --------------------------------------------------------------------------
# caveats
# --------------------------------------------------------------------------

def test_a_good_scene_raises_no_caveats():
    """The control.  A caveat list that always fires is as useless as none."""
    payload = build_scene_dict(_scene())
    assert payload["caveats"] == []
    assert payload["up_axis"] == "+Y"
    assert payload["units"] == "metres"
    assert payload["room"]["room_height"] == pytest.approx(2.6, abs=1e-3)


def test_a_missing_ceiling_is_declared():
    payload = build_scene_dict(_scene(surfaces=[_surface("floor", 0.0)]))
    assert any(c.startswith("no_ceiling") for c in payload["caveats"])
    assert "room_height" not in payload["room"]


def test_a_missing_floor_is_declared():
    payload = build_scene_dict(_scene(surfaces=[_surface("ceiling", 2.6)]))
    assert any(c.startswith("no_floor") for c in payload["caveats"])
    assert "observed_floor_area" not in payload["room"]
    assert "floor_coverage" not in payload["room"]


def test_a_partly_observed_floor_is_declared():
    """Fires below 80% coverage, and not at or above it."""
    # Footprint is 3 x 3 = 9 m^2; a 7.2 m^2 floor patch is exactly 80%.
    at_threshold = build_scene_dict(
        _scene(surfaces=[_surface("floor", 0.0, area=7.2), _surface("ceiling", 2.6)])
    )
    assert not any(c.startswith("partial_floor") for c in at_threshold["caveats"])
    assert at_threshold["room"]["floor_coverage"] == pytest.approx(0.8)

    below = build_scene_dict(
        _scene(surfaces=[_surface("floor", 0.0, area=7.1), _surface("ceiling", 2.6)])
    )
    assert any(c.startswith("partial_floor") for c in below["caveats"])


@pytest.mark.parametrize("height,flagged", [(1.85, True), (1.9, False), (6.0, False), (6.1, True)])
def test_an_implausible_room_height_is_declared(height, flagged):
    """The plausible band is 1.9 m to 6.0 m inclusive; both ends are pinned."""
    payload = build_scene_dict(_scene(height=height))
    fired = any(c.startswith("implausible_height") for c in payload["caveats"])
    assert fired is flagged, payload["room"]["extent"]


def test_an_assumed_camera_is_declared_and_a_measured_one_is_not():
    assumed = build_scene_dict(_scene(provenance="assumed_default"))
    assert any(c.startswith("camera_assumed") for c in assumed["caveats"])
    assert assumed["intrinsics"]["provenance"] == "assumed_default"
    for measured in ("hfov_flag", "exif", "explicit_intrinsics"):
        payload = build_scene_dict(_scene(provenance=measured))
        assert not any(c.startswith("camera_assumed") for c in payload["caveats"])


def test_an_unaligned_scene_is_declared_and_does_not_claim_plus_y():
    payload = build_scene_dict(_scene(meta={"aligned": False}))
    assert any(c.startswith("not_gravity_aligned") for c in payload["caveats"])
    assert payload["gravity_aligned"] is False
    assert payload["up_axis"] != "+Y"


def test_relative_depth_is_declared_and_the_units_string_says_so():
    payload = build_scene_dict(_scene(meta={"metric_depth": False}))
    assert any(c.startswith("relative_depth") for c in payload["caveats"])
    assert payload["metric_depth"] is False
    assert payload["units"] != "metres"
    assert "assumed scale" in payload["units"]


def test_caveats_accumulate_rather_than_replacing_one_another():
    payload = build_scene_dict(
        _scene(
            surfaces=[],
            height=1.2,
            meta={"aligned": False, "metric_depth": False},
            provenance="assumed_default",
        )
    )
    prefixes = {c.split(":")[0] for c in payload["caveats"]}
    assert prefixes == {
        "camera_assumed", "no_ceiling", "no_floor",
        "implausible_height", "not_gravity_aligned", "relative_depth",
    }


# --------------------------------------------------------------------------
# never reference an artefact this run did not write
# --------------------------------------------------------------------------

def test_object_entries_only_reference_files_and_nodes_that_are_written():
    scene = _scene(objects=[_object()])

    both = build_scene_dict(scene, PipelineConfig())
    assert both["objects"][0]["node"] == "object__0__chair"
    assert both["objects"][0]["cloud_file"] == "objects/000_chair.ply"

    no_glb = build_scene_dict(scene, PipelineConfig(export_glb=False))
    assert "node" not in no_glb["objects"][0]
    assert "box_node" not in no_glb["objects"][0]
    assert "cloud_file" in no_glb["objects"][0]

    no_objects = build_scene_dict(scene, PipelineConfig(export_objects=False))
    assert "cloud_file" not in no_objects["objects"][0]
    assert "node" in no_objects["objects"][0]

    surfaces_no_glb = build_scene_dict(_scene(), PipelineConfig(export_glb=False))
    assert "node" not in surfaces_no_glb["surfaces"][0]


def test_the_cloud_file_name_matches_what_the_pipeline_writes():
    """`scene.json` and the file on disk must agree, including the sanitiser."""
    inst = _object(label="chest of drawers;bureau", instance_id=7)
    payload = build_scene_dict(_scene(objects=[inst]), PipelineConfig())
    assert payload["objects"][0]["cloud_file"] == f"objects/{object_cloud_name(inst)}"
    assert object_cloud_name(inst) == "007_chest_of_drawers.ply"


def test_summary_counts_and_labels_come_from_the_scene():
    scene = _scene(objects=[_object("chair", 0), _object("chair", 1), _object("sofa;couch", 2)])
    payload = build_scene_dict(scene)
    assert payload["summary"]["object_count"] == 3
    assert payload["summary"]["surface_count"] == 2
    assert payload["summary"]["labels"] == {"chair": 2, "sofa": 1}
    assert payload["summary"]["surfaces_by_kind"] == {"floor": 1, "ceiling": 1}
    assert payload["summary"]["point_count"] == 400


def test_write_scene_json_creates_parents_and_valid_json(tmp_path):
    path = write_scene_json(tmp_path / "nested" / "deep" / "scene.json", _scene())
    payload = json.loads(path.read_text())
    assert payload["format"] == "roomviz-scene"
    assert payload["version"] == 1


def test_config_is_recorded_without_the_extra_blob():
    payload = build_scene_dict(_scene(), PipelineConfig(extra={"depths": [1, 2, 3]}))
    assert "extra" not in payload["config"]
    assert payload["config"]["voxel_size"] == PipelineConfig().voxel_size


# --------------------------------------------------------------------------
# output hygiene
# --------------------------------------------------------------------------

def test_stale_outputs_this_run_will_not_regenerate_are_removed(tmp_path):
    path = tmp_path / "scene.glb"
    path.write_bytes(b"old")
    _remove_stale(path, will_be_written=False)
    assert not path.exists()


def test_an_output_that_will_be_rewritten_is_left_alone(tmp_path):
    path = tmp_path / "scene.glb"
    path.write_bytes(b"old")
    _remove_stale(path, will_be_written=True)
    assert path.read_bytes() == b"old"


def test_a_symlinked_output_is_never_followed_and_deleted(tmp_path):
    real = tmp_path / "somewhere_important.glb"
    real.write_bytes(b"precious")
    link = tmp_path / "scene.glb"
    link.symlink_to(real)
    _remove_stale(link, will_be_written=False)
    assert real.read_bytes() == b"precious"
    assert link.is_symlink()


def test_only_this_exporters_object_clouds_are_cleared(tmp_path):
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / "000_sofa.ply").write_bytes(b"ours")
    (objects / "012_chest_of_drawers.ply").write_bytes(b"ours")
    (objects / "my_scan.ply").write_bytes(b"theirs")     # not our naming
    (objects / "sofa.ply").write_bytes(b"theirs")        # no index prefix
    (objects / "notes.txt").write_bytes(b"theirs")
    (objects / "0000_x.ply").write_bytes(b"theirs")      # four digits, not three

    _clear_previous_object_clouds(objects)

    assert sorted(p.name for p in objects.iterdir()) == [
        "0000_x.ply", "my_scan.ply", "notes.txt", "sofa.ply"
    ]


def test_a_symlinked_objects_directory_is_left_entirely_alone(tmp_path):
    real = tmp_path / "real_objects"
    real.mkdir()
    (real / "000_sofa.ply").write_bytes(b"precious")
    link = tmp_path / "objects"
    link.symlink_to(real, target_is_directory=True)

    _clear_previous_object_clouds(link)
    assert (real / "000_sofa.ply").read_bytes() == b"precious"


def test_clearing_a_directory_that_does_not_exist_is_not_an_error(tmp_path):
    _clear_previous_object_clouds(tmp_path / "never_created")


def test_a_symlinked_object_cloud_inside_a_real_directory_is_not_deleted(tmp_path):
    objects = tmp_path / "objects"
    objects.mkdir()
    real = tmp_path / "elsewhere.ply"
    real.write_bytes(b"precious")
    (objects / "000_sofa.ply").symlink_to(real)
    _clear_previous_object_clouds(objects)
    assert real.read_bytes() == b"precious"


# --------------------------------------------------------------------------
# glTF geometry
# --------------------------------------------------------------------------

def test_the_drawn_bounding_box_stays_inside_the_reported_aabb():
    """Measuring the GLB must not give a bigger object than `scene.json` does."""
    trimesh = pytest.importorskip("trimesh")
    from roomviz.export.gltf import _box_mesh

    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([1.4, 0.63, 1.16])
    mesh = _box_mesh(trimesh, lo, hi, (10, 20, 30))
    assert mesh is not None
    assert (mesh.vertices.min(axis=0) >= lo - 1e-9).all(), mesh.vertices.min(axis=0)
    assert (mesh.vertices.max(axis=0) <= hi + 1e-9).all(), mesh.vertices.max(axis=0)
    # ... and it still spans nearly the whole box rather than collapsing.
    assert (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0) > 0.9 * (hi - lo)).all()


def test_a_degenerate_object_still_produces_a_box():
    """A perfectly flat object - a painting, a door - must not vanish."""
    trimesh = pytest.importorskip("trimesh")
    from roomviz.export.gltf import _box_mesh

    mesh = _box_mesh(trimesh, np.zeros(3), np.array([1.0, 1.0, 0.0]), (10, 20, 30))
    assert mesh is not None and len(mesh.faces) > 0


# --------------------------------------------------------------------------
# PLY writer
# --------------------------------------------------------------------------

def _read_ply(path):
    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    values = np.frombuffer(
        body,
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("r", "u1"), ("g", "u1"), ("b", "u1")]),
    )
    return header.decode("ascii"), values


def test_ply_defaults_to_a_mid_grey_when_no_colours_are_given(tmp_path):
    from roomviz.export.ply import write_ply

    _, values = _read_ply(write_ply(tmp_path / "grey.ply", np.zeros((3, 3), np.float32)))
    assert values["r"].tolist() == [200, 200, 200]
    assert values["g"].tolist() == [200, 200, 200]


def test_ply_accepts_float64_input_and_stores_float32(tmp_path):
    from roomviz.export.ply import write_ply

    points = np.array([[1.5, -2.25, 3.125]], np.float64)
    header, values = _read_ply(write_ply(tmp_path / "f64.ply", points))
    assert "property float x" in header
    assert [values["x"][0], values["y"][0], values["z"][0]] == [1.5, -2.25, 3.125]


def test_ply_writes_a_well_formed_header_for_an_empty_cloud(tmp_path):
    from roomviz.export.ply import write_ply

    header, values = _read_ply(
        write_ply(tmp_path / "empty.ply", np.zeros((0, 3), np.float32))
    )
    assert "element vertex 0" in header
    assert "format binary_little_endian 1.0" in header
    assert len(values) == 0


def test_ply_creates_missing_parent_directories(tmp_path):
    from roomviz.export.ply import write_ply

    path = write_ply(tmp_path / "a" / "b" / "c.ply", np.zeros((2, 3), np.float32))
    assert path.exists()


# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------

def test_surface_colours_are_the_documented_fixed_palette():
    """These are a hand-picked palette, not a computation; pin them exactly.

    The viewer and `scene.json` both publish them, so a silent change shows up
    in every downstream artefact.
    """
    from roomviz.export.palette import SURFACE_COLORS, surface_color

    assert SURFACE_COLORS == {
        "wall": (196, 202, 214),
        "floor": (168, 150, 128),
        "ceiling": (222, 226, 234),
    }
    for kind, expected in SURFACE_COLORS.items():
        assert surface_color(kind) == expected


def test_label_colours_stay_inside_the_saturation_and_lightness_band():
    """The band is what keeps every label legible on light and dark alike.

    Hue is free (that is what separates classes); saturation and lightness are
    deliberately narrow.  Round-tripping through HLS recovers them, so a change
    to either constant in `label_color` moves a measured value out of band.
    """
    import colorsys

    from roomviz.export.palette import label_color

    labels = [
        "chair", "sofa", "table", "bookcase", "lamp", "person", "rug", "plant",
        "television", "bed", "desk", "cushion", "vase", "clock", "mirror",
    ]
    hues, lightnesses, saturations = [], [], []
    for label in labels:
        r, g, b = (c / 255.0 for c in label_color(label))
        hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
        hues.append(hue)
        lightnesses.append(lightness)
        saturations.append(saturation)
        # 0.48 <= L <= 0.64 and 0.45 <= S <= 0.75 by construction; allow one
        # 8-bit quantisation step (1/255 ~= 0.004) at each end.
        assert 0.47 < lightness < 0.65, (label, lightness)
        assert 0.44 < saturation < 0.76, (label, saturation)
    # Hue really does spread across the wheel rather than clustering.
    assert max(hues) - min(hues) > 0.6
    assert len({round(h, 2) for h in hues}) >= len(labels) - 1


def test_label_colours_of_different_labels_are_visibly_different():
    from roomviz.export.palette import label_color

    labels = ["chair", "sofa", "table", "bookcase", "lamp", "person"]
    colours = [np.array(label_color(x), float) for x in labels]
    for i, a in enumerate(colours):
        for b in colours[i + 1 :]:
            assert np.abs(a - b).max() > 20, (a, b)


# --------------------------------------------------------------------------
# glTF wireframe geometry
# --------------------------------------------------------------------------

def test_the_wireframe_box_reaches_all_eight_corners():
    """Twelve edges over eight corners.  Dropping or duplicating one leaves a
    corner unvisited and the box visibly open."""
    trimesh = pytest.importorskip("trimesh")
    from roomviz.export.gltf import _box_mesh

    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 3.0])
    mesh = _box_mesh(trimesh, lo, hi, (10, 20, 30))
    vertices = np.asarray(mesh.vertices)
    thickness = float(np.clip((hi - lo).min() * 0.02, 0.004, 0.02))
    for corner in np.array(np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]])).T.reshape(-1, 3):
        near = np.linalg.norm(vertices - corner, axis=1).min()
        assert near < 3 * thickness, (corner, near)
    # Twelve bars of 8 vertices each, before any welding.
    assert len(vertices) == 12 * 8


def test_a_scene_with_no_points_still_exports_its_objects_and_surfaces():
    trimesh = pytest.importorskip("trimesh")
    from roomviz.export.gltf import build_trimesh_scene

    scene = _scene(objects=[_object()])
    scene.points = np.zeros((0, 3), np.float32)
    scene.colors = np.zeros((0, 3), np.uint8)
    built = build_trimesh_scene(scene)
    names = set(built.geometry)
    assert "cloud" not in names
    assert "object__0__chair" in names
    assert any(n.startswith("surface__") for n in names)
    assert isinstance(built, trimesh.Scene)


def test_the_cloud_can_be_left_out_on_request():
    pytest.importorskip("trimesh")
    from roomviz.export.gltf import build_trimesh_scene

    scene = _scene(objects=[_object()])
    assert "cloud" in build_trimesh_scene(scene, include_cloud=True).geometry
    assert "cloud" not in build_trimesh_scene(scene, include_cloud=False).geometry


def test_export_scene_creates_the_output_directory_and_honours_toggles(tmp_path):
    from roomviz.pipeline import export_scene

    scene = _scene(objects=[_object()])
    out = tmp_path / "does" / "not" / "exist"
    files = export_scene(
        scene, out, PipelineConfig(export_glb=False, export_viewer=False)
    )
    assert (out / "scene.json").exists()
    assert (out / "scene.ply").exists()
    assert (out / "objects" / "000_chair.ply").exists()
    assert not (out / "scene.glb").exists()
    assert not (out / "viewer.html").exists()
    assert set(files) >= {"scene_json", "cloud_ply", "object_0"}


def test_a_second_export_removes_artefacts_the_new_run_will_not_write(tmp_path):
    """A stale GLB beside a fresh JSON renders the previous reconstruction
    annotated with the current one's numbers."""
    from roomviz.pipeline import export_scene

    scene = _scene(objects=[_object()])
    export_scene(scene, tmp_path, PipelineConfig(export_viewer=False))
    assert (tmp_path / "scene.glb").exists()
    assert (tmp_path / "scene.ply").exists()

    export_scene(
        scene, tmp_path,
        PipelineConfig(export_glb=False, export_ply=False, export_viewer=False),
    )
    assert not (tmp_path / "scene.glb").exists()
    assert not (tmp_path / "scene.ply").exists()
    assert (tmp_path / "scene.json").exists()


def test_object_clouds_from_a_previous_run_do_not_outlive_it(tmp_path):
    from roomviz.pipeline import export_scene

    two = _scene(objects=[_object("chair", 0), _object("sofa", 1)])
    export_scene(two, tmp_path, PipelineConfig(export_glb=False, export_viewer=False))
    assert {p.name for p in (tmp_path / "objects").glob("*.ply")} == {
        "000_chair.ply", "001_sofa.ply"
    }

    one = _scene(objects=[_object("chair", 0)])
    export_scene(one, tmp_path, PipelineConfig(export_glb=False, export_viewer=False))
    assert {p.name for p in (tmp_path / "objects").glob("*.ply")} == {"000_chair.ply"}


def test_surface_quads_carry_their_palette_colour_into_the_glb():
    """Colourless quads render as untinted grey, losing wall/floor/ceiling."""
    pytest.importorskip("trimesh")
    from roomviz.export.gltf import build_trimesh_scene
    from roomviz.export.palette import surface_color

    scene = _scene()
    built = build_trimesh_scene(scene)
    for surface in scene.surfaces:
        mesh = built.geometry[f"surface__{surface.kind}__{surface.surface_id}"]
        colours = np.asarray(mesh.visual.face_colors)
        assert colours.shape[0] == len(mesh.faces) == 2
        assert tuple(int(c) for c in colours[0][:3]) == surface_color(surface.kind)
        assert 0 < int(colours[0][3]) < 255, "surfaces are drawn translucent"


def test_the_cloud_is_included_by_default():
    """`build_trimesh_scene(scene)` with no flag must still write the cloud;
    `write_glb` passes its own default through, so a flipped default here
    silently drops the point cloud from every export."""
    pytest.importorskip("trimesh")
    from roomviz.export.gltf import build_trimesh_scene

    assert "cloud" in build_trimesh_scene(_scene(objects=[_object()])).geometry


def test_a_plane_surface_defaults_to_no_inliers_and_no_area():
    """A surface nobody counted inliers for must not claim to have some."""
    surface = PlaneSurface(
        surface_id=0,
        kind="wall",
        normal=np.array([1.0, 0.0, 0.0], np.float32),
        offset=0.0,
        quad=np.zeros((4, 3), np.float32),
    )
    assert surface.inlier_count == 0
    assert surface.area == 0.0
    assert surface.to_dict()["inlier_count"] == 0


def test_reconstruct_handles_a_single_image_and_records_its_timing(tmp_path):
    """The single-frame path indexes `frames[0]` for EXIF and the source size.

    Every other end-to-end test feeds a video, where an off-by-one in that
    indexing still finds a frame.  With one frame it cannot, and the EXIF
    lookup is only attempted at all when there is exactly one frame.
    """
    from roomviz.pipeline import reconstruct

    rng = np.random.default_rng(3)
    image = tmp_path / "shot.png"
    cv2.imwrite(str(image), rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))

    depth_dir = tmp_path / "depth"
    seg_dir = tmp_path / "seg"
    depth_dir.mkdir()
    seg_dir.mkdir()
    np.save(depth_dir / "000000.npy", np.full((120, 160), 2.5, np.float32))
    ids = np.zeros((120, 160), np.int32)
    ids[80:, :] = 2
    np.save(seg_dir / "000000.npy", ids)
    (seg_dir / "labels.json").write_text(json.dumps({"0": "wall", "2": "floor"}))

    cfg = PipelineConfig(
        depth_backend="file", depth_dir=str(depth_dir),
        seg_backend="file", seg_dir=str(seg_dir),
        max_side=160, hfov_deg=90.0, hfov_explicit=True,
        plane_min_inliers=200, voxel_size=0.02,
    )
    scene, observations = reconstruct(image, cfg)

    assert len(observations) == 1
    assert scene.meta["frames"] == 1
    assert scene.intrinsics.provenance == "hfov_flag"
    # The original capture size is what intrinsics were derived at.
    assert observations[0].frame.original_size == (160, 120)
    # Timing is recorded, so a batch run can spot a stall.
    assert isinstance(scene.meta["elapsed_seconds"], float)
    assert scene.meta["elapsed_seconds"] >= 0.0
    assert scene.meta["input"] == str(image)
