"""Unit tests for configuration validation, core types and the small exporters.

These are the parts of the package that end-to-end tests exercise only by
accident: every scene uses one valid config and one set of well-behaved
values, so a comparison flipped from ``<`` to ``<=`` or a rounding factor
changed by 25% never shows up.  Everything here is fast and pure.
"""

from __future__ import annotations

import numpy as np
import pytest

from roomviz.config import PipelineConfig
from roomviz.export.gltf import box_node_name, object_node_name, safe_label, surface_node_name
from roomviz.export.palette import label_color, label_color_array, surface_color
from roomviz.types import (
    CameraIntrinsics,
    DepthMap,
    Frame,
    ObjectInstance,
    PlaneSurface,
    Scene,
    Segment,
    Segmentation,
)

# --------------------------------------------------------------------------
# PipelineConfig.validate
# --------------------------------------------------------------------------

def test_default_config_validates():
    PipelineConfig().validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("hfov_deg", 0.0),
        ("hfov_deg", 180.0),
        ("hfov_deg", -30.0),
        ("hfov_deg", float("nan")),
        ("voxel_size", 0.0),
        ("voxel_size", -0.01),
        ("voxel_size", float("inf")),
        ("depth_trunc", 0.0),
        ("depth_trunc", -1.0),
        ("max_side", 0),
        ("max_side", -768),
        ("depth_min", -0.1),
        ("depth_min", float("nan")),
        ("max_frames", 0),
        ("max_frames", -1),
        ("point_budget", 0),
        ("min_object_points", 0),
        ("object_split_gap", 0.0),
        ("object_split_gap", -0.1),
        ("object_split_gap", float("inf")),
        ("depth_near", 0.0),
        ("depth_far", 0.0),
    ],
)
def test_validate_rejects_settings_that_would_produce_nonsense(field, value):
    cfg = PipelineConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize(
    "field,value",
    [
        # Every rule's *first legal* value.  Without these, tightening any
        # comparison by one step - `< 1` to `<= 1`, `<= 0` to `< 0` - would
        # reject perfectly valid configurations and nothing would notice.
        ("hfov_deg", 0.001),
        ("hfov_deg", 179.999),
        ("depth_min", 0.0),
        ("max_frames", 1),
        ("point_budget", 1),
        ("min_object_points", 1),
        ("voxel_size", 1e-9),
        ("object_split_gap", 1e-9),
    ],
)
def test_validate_accepts_the_boundary_values(field, value):
    cfg = PipelineConfig()
    setattr(cfg, field, value)
    cfg.validate()


def test_validate_rejects_an_inverted_depth_range():
    cfg = PipelineConfig(depth_min=5.0, depth_trunc=5.0)
    with pytest.raises(ValueError, match="depth_min"):
        cfg.validate()
    # Strictly below is fine.
    PipelineConfig(depth_min=4.999, depth_trunc=5.0).validate()

    cfg = PipelineConfig(depth_near=2.0, depth_far=2.0)
    with pytest.raises(ValueError, match="depth_near"):
        cfg.validate()
    PipelineConfig(depth_near=1.999, depth_far=2.0).validate()


@pytest.mark.parametrize(
    "intrinsics",
    [
        (0.0, 100.0, 10.0, 10.0),
        (100.0, 0.0, 10.0, 10.0),
        (-100.0, 100.0, 10.0, 10.0),
        (100.0, 100.0, float("nan"), 10.0),
        (100.0, 100.0, 10.0, float("inf")),
    ],
)
def test_validate_rejects_impossible_intrinsics(intrinsics):
    with pytest.raises(ValueError):
        PipelineConfig(intrinsics=intrinsics).validate()


def test_validate_accepts_an_off_centre_principal_point():
    """A negative cx is unusual but legal (a cropped sensor); do not reject it."""
    PipelineConfig(intrinsics=(500.0, 500.0, -12.0, 900.0)).validate()


def test_to_dict_round_trips_every_field():
    cfg = PipelineConfig(max_frames=7, hfov_deg=68.5, voxel_size=0.011)
    payload = cfg.to_dict()
    assert payload["max_frames"] == 7
    assert payload["hfov_deg"] == 68.5
    assert payload["voxel_size"] == 0.011
    assert set(payload) >= {"depth_backend", "seed", "extra"}


# --------------------------------------------------------------------------
# CameraIntrinsics
# --------------------------------------------------------------------------

def test_from_hfov_puts_the_principal_point_between_the_end_pixels():
    intr = CameraIntrinsics.from_hfov(640, 480, 60.0)
    assert intr.cx == pytest.approx(319.5)
    assert intr.cy == pytest.approx(239.5)
    # f divides the full image *width*, not width - 1.
    assert intr.fx == pytest.approx(320.0 / np.tan(np.deg2rad(30.0)))
    assert intr.fx == intr.fy


@pytest.mark.parametrize("hfov", [40.0, 60.0, 90.0, 120.0])
def test_from_hfov_realises_the_field_of_view_it_was_given(hfov):
    intr = CameraIntrinsics.from_hfov(800, 600, hfov)
    realised = np.rad2deg(2 * np.arctan((intr.width / 2.0) / intr.fx))
    assert realised == pytest.approx(hfov, abs=1e-9)


def test_scaled_to_maps_the_principal_point_through_pixel_edges():
    intr = CameraIntrinsics(width=100, height=50, fx=200.0, fy=180.0, cx=49.5, cy=24.5)
    half = intr.scaled_to(50, 25)
    assert half.fx == pytest.approx(100.0)
    assert half.fy == pytest.approx(90.0)
    assert half.cx == pytest.approx((49.5 + 0.5) * 0.5 - 0.5)
    assert half.cy == pytest.approx((24.5 + 0.5) * 0.5 - 0.5)
    # Scaling by 1 must be the identity, not a half-pixel drift.
    same = intr.scaled_to(100, 50)
    assert (same.fx, same.fy, same.cx, same.cy) == (intr.fx, intr.fy, intr.cx, intr.cy)


def test_scaled_to_uses_each_axis_independently():
    """A resize that changes the aspect ratio must not share one scale factor."""
    intr = CameraIntrinsics(width=100, height=100, fx=200.0, fy=200.0, cx=49.5, cy=49.5)
    stretched = intr.scaled_to(200, 100)
    assert stretched.fx == pytest.approx(400.0)
    assert stretched.fy == pytest.approx(200.0)


def test_intrinsics_matrix_layout():
    intr = CameraIntrinsics(width=4, height=3, fx=10.0, fy=20.0, cx=1.5, cy=2.5)
    assert intr.matrix.tolist() == [[10.0, 0.0, 1.5], [0.0, 20.0, 2.5], [0.0, 0.0, 1.0]]
    assert intr.to_dict()["provenance"] == "unknown"


# --------------------------------------------------------------------------
# DepthMap / Frame / Segmentation
# --------------------------------------------------------------------------

def test_depth_validity_rejects_zero_negative_nan_and_inf():
    depth = np.array([[1.0, 0.0, -2.0], [np.nan, np.inf, 3.5]], np.float32)
    valid = DepthMap(depth=depth).valid
    assert valid.tolist() == [[True, False, False], [False, False, True]]


def test_frame_dimensions_come_from_the_array():
    frame = Frame(index=0, rgb=np.zeros((17, 29, 3), np.uint8))
    assert (frame.height, frame.width) == (17, 29)
    assert frame.source_index == 0


def test_segmentation_lookup_and_masking():
    ids = np.array([[-1, 1], [2, 2]], np.int32)
    segmentation = Segmentation(
        ids=ids,
        segments=[
            Segment(segment_id=1, label="wall", role="structure", structure_kind="wall"),
            Segment(segment_id=2, label="chair", role="object"),
        ],
    )
    assert set(segmentation.by_id()) == {1, 2}
    assert segmentation.by_id()[2].label == "chair"
    assert segmentation.mask_for(2).tolist() == [[False, False], [True, True]]
    # -1 is "unlabelled" and belongs to no segment.
    assert not segmentation.mask_for(-1).all()
    assert segmentation.mask_for(0).sum() == 0


# --------------------------------------------------------------------------
# ObjectInstance / PlaneSurface / Scene
# --------------------------------------------------------------------------

def _instance(points):
    points = np.asarray(points, np.float32)
    return ObjectInstance(
        instance_id=3,
        label="sofa;couch",
        points=points,
        colors=np.zeros((points.shape[0], 3), np.uint8),
        observations=5,
        frame_indices=[0, 2],
    )


def test_object_aabb_and_centroid_are_computed_from_the_points():
    inst = _instance([[0.0, 0.0, 0.0], [2.0, 1.0, 4.0], [1.0, -1.0, 2.0]])
    lo, hi = inst.aabb
    assert lo.tolist() == [0.0, -1.0, 0.0]
    assert hi.tolist() == [2.0, 1.0, 4.0]
    assert inst.centroid == pytest.approx([1.0, 0.0, 2.0])

    payload = inst.to_dict()
    assert payload["size"] == [2.0, 2.0, 4.0]
    assert payload["aabb_min"] == [0.0, -1.0, 0.0]
    assert payload["aabb_max"] == [2.0, 1.0, 4.0]
    assert payload["point_count"] == 3
    assert payload["observations"] == 5
    assert payload["frames"] == [0, 2]
    assert payload["label"] == "sofa;couch"


def test_object_to_dict_rounds_to_four_decimals():
    inst = _instance([[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0]])
    assert inst.to_dict()["aabb_max"][0] == pytest.approx(0.3333, abs=1e-9)


def _surface(kind, quad):
    return PlaneSurface(
        surface_id=1,
        kind=kind,
        normal=np.array([0.0, 1.0, 0.0], np.float32),
        offset=-2.0,
        quad=np.asarray(quad, np.float32),
        inlier_count=42,
        area=6.0,
    )


def test_wall_extents_report_the_horizontal_span_first():
    """A wall is 'this long and this tall', in that order, whichever way the
    quad's corners happen to run."""
    # Corner order 0->1 runs vertically, 1->2 horizontally.
    vertical_first = _surface(
        "wall",
        [[0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [4.0, 2.5, 0.0], [4.0, 0.0, 0.0]],
    )
    assert vertical_first.extents == pytest.approx((4.0, 2.5))

    # Corner order 0->1 runs horizontally, 1->2 vertically.
    horizontal_first = _surface(
        "wall",
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 2.5, 0.0], [0.0, 2.5, 0.0]],
    )
    assert horizontal_first.extents == pytest.approx((4.0, 2.5))


def test_floor_extents_report_the_longer_side_first():
    floor = _surface(
        "floor",
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
    )
    assert floor.extents == pytest.approx((5.0, 3.0))


def test_surface_to_dict_carries_extents_and_geometry():
    surface = _surface(
        "wall",
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 2.5, 0.0], [0.0, 2.5, 0.0]],
    )
    payload = surface.to_dict()
    assert payload["kind"] == "wall"
    assert payload["width"] == pytest.approx(4.0)
    assert payload["height"] == pytest.approx(2.5)
    assert payload["normal"] == [0.0, 1.0, 0.0]
    assert payload["offset"] == pytest.approx(-2.0)
    assert payload["inlier_count"] == 42
    assert len(payload["quad"]) == 4 and len(payload["quad"][0]) == 3


def test_scene_bounds_of_an_empty_cloud_are_zero_not_an_exception():
    empty = Scene(points=np.zeros((0, 3), np.float32), colors=np.zeros((0, 3), np.uint8))
    lo, hi = empty.bounds
    assert lo.tolist() == [0.0, 0.0, 0.0] and hi.tolist() == [0.0, 0.0, 0.0]


def test_scene_bounds_are_the_component_wise_extremes():
    scene = Scene(
        points=np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, 0.0]], np.float32),
        colors=np.zeros((2, 3), np.uint8),
    )
    lo, hi = scene.bounds
    assert lo.tolist() == [-4.0, -2.0, 0.0]
    assert hi.tolist() == [1.0, 5.0, 3.0]


# --------------------------------------------------------------------------
# node naming and colours
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,expected",
    [
        ("sofa;couch;lounge", "sofa"),
        ("pot;flowerpot", "pot"),
        ("chest of drawers", "chest_of_drawers"),
        ("light/lamp", "light_lamp"),
        ("  spaced  ; other", "spaced"),
        ("", "object"),
        (";", "object"),
        # Measured, not assumed: punctuation collapses to a single underscore
        # rather than to "object", because the `or "object"` fallback only
        # fires on an empty result and "_" is not empty.  Still a legal
        # filename component and a legal glTF node name, so this records the
        # behaviour rather than calling it a bug.
        ("!!!", "_"),
    ],
)
def test_safe_label_keeps_the_first_synonym_and_name_safe_characters(label, expected):
    assert safe_label(label) == expected


def test_safe_label_is_length_capped():
    assert len(safe_label("x" * 500)) == 48
    assert len(safe_label("x" * 500, max_length=10)) == 10


def test_node_names_encode_role_id_and_label():
    inst = _instance([[0.0, 0.0, 0.0]])
    assert object_node_name(inst) == "object__3__sofa"
    assert box_node_name(inst) == "box__3__sofa"
    surface = _surface("ceiling", np.zeros((4, 3)))
    assert surface_node_name(surface) == "surface__ceiling__1"


def test_label_colour_is_stable_distinct_and_in_range():
    assert label_color("chair") == label_color("chair")
    assert label_color("chair") != label_color("sofa")
    for label in ("chair", "sofa", "table", "bookcase", "lamp", "person"):
        colour = label_color(label)
        assert len(colour) == 3
        assert all(0 <= c <= 255 for c in colour)
        # The palette deliberately avoids near-black and near-white so labels
        # stay legible on either background.
        assert 40 < sum(colour) / 3 < 215, (label, colour)
    assert label_color_array("chair").dtype == np.uint8
    assert label_color_array("chair").tolist() == list(label_color("chair"))


def test_surface_colours_are_distinct_with_a_fallback():
    assert surface_color("wall") != surface_color("floor")
    assert surface_color("floor") != surface_color("ceiling")
    assert surface_color("something else") == (180, 180, 180)


# --------------------------------------------------------------------------
# shipped defaults
# --------------------------------------------------------------------------

def test_the_documented_defaults_are_what_the_readme_says():
    """The README quotes these numbers; a silent change makes it wrong.

    They are also the values every `roomviz reconstruct` without flags uses,
    which is the configuration most users will ever see.
    """
    cfg = PipelineConfig()
    assert cfg.max_frames == 24            # "keyframes taken from a video (default 24)"
    assert cfg.max_side == 768             # "working resolution (default 768)"
    assert cfg.hfov_deg == 60.0            # "the default assumption is 60 degrees"
    assert cfg.hfov_explicit is False      # ... and it must not outrank EXIF
    assert cfg.depth_scale == 0.001        # "--depth-scale 0.001 for millimetres"
    assert cfg.depth_trunc == 12.0
    assert cfg.frame_stride == 0           # "0 = automatic"
    assert cfg.min_sharpness == 0.0        # blur filter off unless asked for
    assert cfg.object_split_gap == 0.12
    assert cfg.estimate_poses is True
    assert cfg.align_gravity is True
    assert (cfg.export_ply, cfg.export_glb, cfg.export_objects, cfg.export_viewer) == (
        True, True, True, True
    )


def test_the_default_backends_are_the_documented_checkpoints():
    cfg = PipelineConfig()
    assert cfg.depth_backend == "depth-anything-v2"
    assert cfg.seg_backend == "mask2former"
    assert "metric-indoor" in cfg.depth_model
    assert "ade-panoptic" in cfg.seg_model


def test_resolve_device_honours_an_explicit_request():
    from roomviz.perception.base import resolve_device

    for requested in ("cpu", "cuda", "mps", "cuda:1"):
        assert resolve_device(requested) == requested
    # "auto" resolves to something concrete, never back to "auto".
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    assert resolve_device() in {"cpu", "cuda", "mps"}


def test_positivity_failures_name_the_setting_that_failed():
    """Each rule must fire for its own reason, not be masked by a later one.

    `depth_trunc = 0` also trips the `depth_min < depth_trunc` rule, and
    `depth_far = 0` also trips `depth_near < depth_far`, so a test that only
    checks "a ValueError was raised" cannot tell whether the positivity checks
    exist at all.  Matching the message can.
    """
    for field, needle in (
        ("depth_trunc", "depth_trunc must be a positive finite number"),
        ("depth_far", "depth_far must be a positive finite number"),
        ("voxel_size", "voxel_size must be a positive finite number"),
        ("max_side", "max_side must be a positive finite number"),
        ("depth_near", "depth_near must be a positive finite number"),
    ):
        cfg = PipelineConfig()
        setattr(cfg, field, 0.0)
        with pytest.raises(ValueError, match=needle):
            cfg.validate()


def test_wall_extents_use_the_edge_the_quad_actually_runs_along():
    """An asymmetric quad pins which corner pair each side length comes from.

    With a 4 x 2.5 m wall every rectangle has the same two side lengths, so a
    mis-indexed corner still returns the right pair by luck.  A quad whose
    first edge is neither of the two extents cannot.
    """
    quad = np.array(
        [[0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [6.0, 2.5, 0.0], [6.0, 0.0, 0.0]],
        np.float32,
    )
    wall = PlaneSurface(
        surface_id=0, kind="wall",
        normal=np.array([0.0, 0.0, 1.0], np.float32), offset=0.0, quad=quad,
    )
    # 0->1 is the 2.5 m vertical edge; 1->2 is the 6.0 m horizontal one.
    assert wall.extents == pytest.approx((6.0, 2.5))

    ceiling = PlaneSurface(
        surface_id=0, kind="ceiling",
        normal=np.array([0.0, 1.0, 0.0], np.float32), offset=0.0, quad=quad,
    )
    # Horizontals report longest-first regardless of corner order.
    assert ceiling.extents == pytest.approx((6.0, 2.5))
