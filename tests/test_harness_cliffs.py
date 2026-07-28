"""Where the synthetic harness stops working, measured rather than assumed.

Every other end-to-end test runs the harness at one operating point - 95 degree
HFOV, 0.45 texture amplitude, six views, a 1.4 m camera sweep, no depth noise -
and asserts that the pipeline reconstructs the room.  That says nothing about
how much margin there is.  If a knob is only 10% above the value at which the
reconstruction falls apart, "the suite is green" means almost nothing.

So each test here brackets one axis: a point on the working side and the
measured first failing point, both asserted.  A regression that moves a cliff
towards the operating point fails the "still works" half; an improvement that
moves it away fails the "still broken" half, at which point the recorded number
should be re-measured and updated.  Both halves are the point.

All numbers below were measured on this machine at commit fe0251f with
``scratchpad/tf_cliffs.py`` and are bit-for-bit repeatable (the renderer, ORB,
PnP and RANSAC are all seeded).

These are full pipeline runs and cost roughly 5 s each.  The module is marked
``slow``; ``pytest -m "not slow"`` skips it.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from synthetic import render_sequence

from roomviz.config import PipelineConfig
from roomviz.fusion.odometry import estimate_trajectory
from roomviz.fusion.scene_fusion import fuse
from roomviz.types import DepthMap, Observation

pytestmark = pytest.mark.slow

WIDTH, HEIGHT = 320, 240
EXPECTED_LABELS = ["bookcase", "chair", "sofa", "table"]

# The tolerance test_object_footprints_match_ground_truth uses.  "Works" here
# means exactly what the rest of the suite means by it.
EXTENT_TOLERANCE = 0.15


def _config() -> PipelineConfig:
    return PipelineConfig(
        voxel_size=0.03,
        min_object_points=150,
        plane_min_inliers=600,
        max_side=max(WIDTH, HEIGHT),
    )


@cache
def run_harness(
    hfov: float = 95.0,
    texture_amplitude: float = 0.45,
    frame_count: int = 6,
    path_span: float = 1.4,
    noise: float = 0.0,
):
    """One full reconstruction.  Cached: the same knobs are reused, not rerun."""
    room, intr, gt_poses, frames, depths, segs = render_sequence(
        width=WIDTH,
        height=HEIGHT,
        hfov=hfov,
        texture_amplitude=texture_amplitude,
        frame_count=frame_count,
        path_span=path_span,
    )
    if noise:
        rng = np.random.default_rng(5)
        depth_maps = [
            DepthMap(depth=(d * (1.0 + rng.normal(0, noise, d.shape))).astype(np.float32))
            for d in depths
        ]
    else:
        depth_maps = [DepthMap(depth=d) for d in depths]

    cfg = _config()
    poses = estimate_trajectory(frames, depth_maps, intr, cfg)
    scene = fuse(
        [
            Observation(frame=f, depth=d, segmentation=s, intrinsics=intr, pose=p)
            for f, d, s, p in zip(frames, depth_maps, segs, poses, strict=True)
        ],
        cfg,
    )
    return scene, room, poses, gt_poses


def worst_footprint_error(scene, room) -> float:
    """Largest horizontal extent error over all objects, in metres."""
    truth = {b.label: b for b in room.boxes}
    worst = 0.0
    for inst in scene.objects:
        if inst.label not in truth:
            continue
        lo, hi = inst.aabb
        size = hi - lo
        for axis in (0, 2):
            worst = max(worst, abs(float(size[axis]) - float(truth[inst.label].size[axis])))
    return worst


def reconstructs(scene, room) -> bool:
    """The harness's own definition of a successful run."""
    labels_ok = sorted(o.label for o in scene.objects) == EXPECTED_LABELS
    floor_ok = sum(1 for s in scene.surfaces if s.kind == "floor") == 1
    return labels_ok and floor_ok and worst_footprint_error(scene, room) < EXTENT_TOLERANCE


def describe(scene, room) -> str:
    kinds = [s.kind for s in scene.surfaces]
    return (
        f"labels={sorted(o.label for o in scene.objects)} "
        f"worst_extent_error={worst_footprint_error(scene, room):.3f} "
        f"floor={kinds.count('floor')} ceiling={kinds.count('ceiling')} "
        f"walls={kinds.count('wall')}"
    )


# --------------------------------------------------------------------------
# texture
# --------------------------------------------------------------------------

@pytest.mark.parametrize("amplitude", [0.45, 0.20])
def test_texture_amplitude_above_the_cliff_still_reconstructs(amplitude):
    """Everything ORB has to match on is the texture contrast.

    Measured sweep: 0.45 (the harness value) through 0.20 all reconstruct;
    0.18 does not.  So the harness sits at 2.25x its own texture cliff - close
    enough that "the suite passes" says little about a flatly-lit real room.
    """
    scene, room, _, _ = run_harness(texture_amplitude=amplitude)
    assert reconstructs(scene, room), describe(scene, room)


def test_texture_amplitude_cliff_is_at_0_18():
    """Recorded boundary: at amplitude 0.18 the reconstruction is wrong.

    Measured at 0.18: worst horizontal extent error 0.200 m (against the
    0.15 m tolerance) and worst odometry drift 0.386 m, up from 0.035 m at the
    harness's 0.45.  At 0.15 it degenerates completely (0.771 m extent error,
    1.409 m drift).
    """
    scene, room, poses, gt = run_harness(texture_amplitude=0.18)
    assert not reconstructs(scene, room), (
        "texture cliff has moved: 0.18 amplitude now reconstructs correctly "
        f"({describe(scene, room)}). Re-measure and update this test."
    )
    assert worst_footprint_error(scene, room) > EXTENT_TOLERANCE


# --------------------------------------------------------------------------
# camera path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("span", [1.4, 0.6])
def test_camera_travel_above_the_cliff_still_reconstructs(span):
    """Parallax is what makes the geometry metric; it scales with travel.

    Measured sweep of the sweep's horizontal extent: 1.4 m (harness), 1.0,
    0.8 and 0.6 m all reconstruct; 0.4 m does not.  2.3x margin.
    """
    scene, room, _, _ = run_harness(path_span=span)
    assert reconstructs(scene, room), describe(scene, room)


def test_camera_travel_cliff_is_at_0_4_metres():
    """Recorded boundary: a 0.4 m sweep no longer measures the objects.

    Measured worst horizontal extent error 0.290 m at 0.4 m of travel, 0.286 m
    at 0.2 m, 0.272 m at 0.1 m - and note the odometry drift stays *small*
    (0.028 m) throughout, because with little motion there is little motion to
    get wrong.  Short baselines fail through the reconstruction, not the poses,
    which is why an odometry assertion alone would not have caught this.
    """
    scene, room, _, _ = run_harness(path_span=0.4)
    assert not reconstructs(scene, room), (
        "camera-travel cliff has moved: a 0.4 m sweep now reconstructs "
        f"correctly ({describe(scene, room)}). Re-measure and update this test."
    )


# --------------------------------------------------------------------------
# depth noise
# --------------------------------------------------------------------------

def test_two_percent_depth_noise_still_reconstructs():
    """`test_survives_depth_noise` uses 1%; this records how much is left.

    Measured: 1% gives a 0.083 m worst extent error, 2% gives 0.139 m (just
    inside the 0.15 m tolerance), 3% gives 0.158 m (just outside) and 4% loses
    the bookcase entirely.  The suite's 1% run therefore sits at roughly 2x
    its own noise cliff, and 1% is optimistic for any real depth sensor.
    """
    scene, room, _, _ = run_harness(noise=0.02)
    assert reconstructs(scene, room), describe(scene, room)


def test_depth_noise_cliff_loses_an_object_by_four_percent():
    """Recorded boundary: at 4% multiplicative depth noise the bookcase is gone.

    Measured at 4%: objects come back as ['chair', 'sofa', 'table'], worst
    extent error 0.303 m, no ceiling plane and one wall instead of three.  At
    5% no object survives at all.
    """
    scene, room, _, _ = run_harness(noise=0.04)
    labels = sorted(o.label for o in scene.objects)
    assert labels != EXPECTED_LABELS, (
        "depth-noise cliff has moved: 4% noise now finds every object "
        f"({describe(scene, room)}). Re-measure and update this test."
    )
    assert "bookcase" not in labels


# --------------------------------------------------------------------------
# field of view
# --------------------------------------------------------------------------

def test_ceiling_and_room_height_survive_down_to_85_degrees():
    """The room height depends on seeing the ceiling, and that goes first.

    Measured: the ceiling plane is recovered at 95 and 85 degrees and lost by
    75, where the reported floor-to-ceiling extent drops to 2.55 m against a
    true 2.70 m.  The harness's 95 degrees is barely 1.1x this cliff, so of the
    room-shell numbers the suite asserts, room height has the least margin of
    anything measured here.
    """
    scene, room, _, _ = run_harness(hfov=85.0)
    assert sum(1 for s in scene.surfaces if s.kind == "ceiling") == 1, describe(scene, room)
    low, high = scene.bounds
    assert abs(float(high[1] - low[1]) - room.height) < 0.1


def test_ceiling_is_lost_by_75_degrees():
    """Recorded boundary: no ceiling plane, and the height is 15 cm short.

    Measured at 75 degrees: zero ceiling surfaces and a 2.55 m reported
    floor-to-ceiling extent (true 2.70 m).  Objects are still fine - the
    worst horizontal extent error is 0.052 m - so this is a structure cliff,
    not an object cliff.
    """
    scene, room, _, _ = run_harness(hfov=75.0)
    assert sum(1 for s in scene.surfaces if s.kind == "ceiling") == 0, (
        "ceiling cliff has moved: 75 degrees now recovers a ceiling "
        f"({describe(scene, room)}). Re-measure and update this test."
    )
    # Objects are unaffected: it really is only the shell that is lost.
    assert sorted(o.label for o in scene.objects) == EXPECTED_LABELS
    assert worst_footprint_error(scene, room) < EXTENT_TOLERANCE


def test_three_walls_survive_to_70_degrees_and_not_to_68():
    """`test_walls_are_vertical_and_axis_aligned` needs three walls to mean much.

    Measured wall counts across the sweep: 3 at 95, 85, 75 and 70 degrees;
    2 at 68; 1 at 65, 62 and 60.  So that test's `len(walls) >= 3` holds only
    down to 70 degrees - it would be vacuous at the package's own 60 degree
    default, where a single wall is recovered.
    """
    wide, room, _, _ = run_harness(hfov=70.0)
    narrow, _, _, _ = run_harness(hfov=68.0)
    assert sum(1 for s in wide.surfaces if s.kind == "wall") >= 3, describe(wide, room)
    assert sum(1 for s in narrow.surfaces if s.kind == "wall") < 3, (
        "wall-count cliff has moved: 68 degrees now recovers three walls "
        f"({describe(narrow, room)}). Re-measure and update this test."
    )


# --------------------------------------------------------------------------
# frame count
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frame_count", [3, 2])
def test_there_is_no_frame_count_cliff_down_to_two_views(frame_count):
    """Contrary to expectation, this axis has enormous margin, not little.

    Measured: every count from 8 down to 2 reconstructs all four objects with
    the floor, ceiling and three walls, worst extent error 0.084-0.128 m.  The
    harness's six frames are not near a cliff here.

    What *does* depend on frame count is
    `test_objects_are_seen_in_multiple_frames`, which asserts every object
    fused at least four views and so is arithmetically unsatisfiable below four
    frames.  That is a property of the assertion, not of the pipeline, and it
    is recorded here so the distinction is not lost.
    """
    scene, room, _, _ = run_harness(frame_count=frame_count)
    assert reconstructs(scene, room), describe(scene, room)
    assert max(o.observations for o in scene.objects) <= frame_count
