"""Tests for media loading, label classification and the sidecar backends."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from synthetic import default_room, labels_for, look_at, render

from roomviz.config import PipelineConfig
from roomviz.media.loader import classify as classify_media
from roomviz.media.loader import load_frames, resize_frame, sharpness
from roomviz.perception.base import (
    available_backends,
    build_depth_backend,
    build_segmentation_backend,
)
from roomviz.perception.labels import classify, structure_kind, synonyms
from roomviz.types import ROLE_IGNORE, ROLE_OBJECT, ROLE_STRUCTURE, CameraIntrinsics

# --------------------------------------------------------------------------
# label classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,expected_kind",
    [
        ("wall", "wall"),
        ("floor;flooring", "floor"),
        ("ceiling", "ceiling"),
        ("windowpane;window", "window"),
        ("door;double door", "door"),
        ("rug;carpet;carpeting", "floor"),
        ("stairway;staircase", "stairs"),
    ],
)
def test_structural_labels(label, expected_kind):
    role, kind = classify(label)
    assert role == ROLE_STRUCTURE
    assert kind == expected_kind
    assert structure_kind(label) == expected_kind


@pytest.mark.parametrize(
    "label", ["chair", "sofa;couch;lounge", "table", "lamp", "person;individual"]
)
def test_object_labels(label):
    role, kind = classify(label)
    assert role == ROLE_OBJECT
    assert kind is None


@pytest.mark.parametrize("label", ["sky", "tree", "building;edifice", "mountain;mount"])
def test_ignored_labels(label):
    role, _ = classify(label)
    assert role == ROLE_IGNORE


def test_synonyms_splitting():
    assert synonyms("person;individual;someone") == ["person", "individual", "someone"]
    assert synonyms("") == []


def test_unknown_label_defaults_to_object():
    role, kind = classify("some_novel_class")
    assert role == ROLE_OBJECT and kind is None


# --------------------------------------------------------------------------
# media loading
# --------------------------------------------------------------------------

@pytest.fixture
def demo_media(tmp_path):
    """A short video plus matching ground-truth depth and segment sidecars."""
    room = default_room()
    width, height, count = 160, 120, 12
    intr = CameraIntrinsics.from_hfov(width, height, 95.0)

    depth_dir = tmp_path / "depth"
    seg_dir = tmp_path / "seg"
    depth_dir.mkdir()
    seg_dir.mkdir()

    video = tmp_path / "room.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
    )
    assert writer.isOpened()

    for i in range(count):
        s = i / (count - 1)
        pose = look_at(
            np.array([1.8 + 1.4 * s, 1.55, 0.15 + 0.15 * s]),
            np.array([2.4, 0.75, 2.7]),
        )
        rgb, depth, ids = render(room, pose, intr)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_dir / f"{i:06d}.npy", depth)
        np.save(seg_dir / f"{i:06d}.npy", ids)
    writer.release()

    (seg_dir / "labels.json").write_text(
        json.dumps({str(k): v for k, v in labels_for(room).items()})
    )
    return video, depth_dir, seg_dir, count


def test_classify_media_kinds(tmp_path, demo_media):
    video, _, _, count = demo_media
    source = classify_media(video)
    assert source.kind == "video"
    assert source.frame_count == count

    image = tmp_path / "shot.png"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    assert classify_media(image).kind == "image"
    assert classify_media(tmp_path).kind == "image_dir"


def test_classify_media_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        classify_media(tmp_path / "nope.mp4")


def test_resize_frame_snaps_to_patch_multiple():
    out = resize_frame(np.zeros((480, 640, 3), np.uint8), 768)
    assert out.shape[0] % 14 == 0 and out.shape[1] % 14 == 0
    # Downscaling must respect the cap on the longest side.
    small = resize_frame(np.zeros((1000, 2000, 3), np.uint8), 700)
    assert max(small.shape[:2]) <= 700 + 14


def test_sharpness_ranks_blur():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(sharp, (9, 9), 4)
    assert sharpness(sharp) > sharpness(blurry) * 2


def test_video_keyframes_record_their_source_index(demo_media):
    """Sidecars are numbered against the video, not against the keyframes.

    Keyframe sampling means keyframe 1 can be video frame 3.  Losing that
    mapping silently pairs each frame's colour with another frame's depth,
    so it is pinned down here.
    """
    video, _, _, count = demo_media
    cfg = PipelineConfig(max_frames=4)
    frames = load_frames(video, cfg)

    assert len(frames) <= 4
    assert [f.index for f in frames] == list(range(len(frames)))

    stride = int(np.ceil(count / 4))
    assert [f.source_index for f in frames] == [
        i * stride for i in range(len(frames))
    ]
    assert frames[0].original_size == (160, 120)


def test_image_directory_records_source_index(tmp_path):
    for i in range(6):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), np.full((28, 28, 3), i * 20, np.uint8))
    frames = load_frames(tmp_path, PipelineConfig(max_frames=3))
    assert len(frames) == 3
    assert [f.source_index for f in frames] == [0, 2, 4]


def test_max_frames_is_respected(demo_media):
    video, _, _, _ = demo_media
    assert len(load_frames(video, PipelineConfig(max_frames=3))) <= 3


def test_frame_stride_overrides_automatic_sampling(demo_media):
    video, _, _, _ = demo_media
    frames = load_frames(video, PipelineConfig(max_frames=100, frame_stride=4))
    assert [f.source_index for f in frames] == [0, 4, 8]


# --------------------------------------------------------------------------
# sidecar backends
# --------------------------------------------------------------------------

def test_registry_lists_backends():
    backends = available_backends()
    assert {"file", "memory", "depth-anything-v2"} <= set(backends["depth"])
    assert {"file", "memory", "mask2former"} <= set(backends["segmentation"])


def test_unknown_backend_is_reported_clearly():
    with pytest.raises(ValueError, match="unknown depth backend"):
        build_depth_backend(PipelineConfig(depth_backend="nope"))
    with pytest.raises(ValueError, match="unknown segmentation backend"):
        build_segmentation_backend(PipelineConfig(seg_backend="nope"))


def test_file_backend_requires_a_directory():
    with pytest.raises(ValueError, match="--depth-dir"):
        build_depth_backend(PipelineConfig(depth_backend="file"))
    with pytest.raises(ValueError, match="--seg-dir"):
        build_segmentation_backend(PipelineConfig(seg_backend="file"))


def test_file_backends_pair_sidecars_with_the_right_frame(demo_media):
    """Regression test: depth must match the colour frame it is fused with."""
    video, depth_dir, seg_dir, count = demo_media
    cfg = PipelineConfig(
        max_frames=4,
        depth_backend="file",
        depth_dir=str(depth_dir),
        seg_backend="file",
        seg_dir=str(seg_dir),
    )
    frames = load_frames(video, cfg)
    depth_backend = build_depth_backend(cfg)
    seg_backend = build_segmentation_backend(cfg)

    for frame in frames:
        expected = np.load(depth_dir / f"{frame.source_index:06d}.npy")
        loaded = depth_backend.predict(frame).depth
        assert loaded.shape == (frame.height, frame.width)
        # Nearest-neighbour resize preserves the value distribution exactly.
        assert abs(float(loaded[loaded > 0].mean()) - float(expected.mean())) < 0.05

        segmentation = seg_backend.predict(frame)
        assert segmentation.ids.shape == (frame.height, frame.width)
        labels = {s.label for s in segmentation.segments}
        assert "wall" in labels
        assert {s.role for s in segmentation.segments} <= {
            ROLE_OBJECT, ROLE_STRUCTURE, ROLE_IGNORE
        }


def test_file_depth_scales_integer_units(tmp_path):
    directory = tmp_path / "depth"
    directory.mkdir()
    np.save(directory / "000000.npy", np.full((16, 16), 2500, np.uint16))
    cfg = PipelineConfig(depth_backend="file", depth_dir=str(directory), depth_scale=0.001)
    backend = build_depth_backend(cfg)

    from roomviz.types import Frame

    frame = Frame(index=0, rgb=np.zeros((16, 16, 3), np.uint8), source_index=0)
    assert backend.predict(frame).depth.mean() == pytest.approx(2.5)


def test_file_depth_truncates_out_of_range_values(tmp_path):
    directory = tmp_path / "depth"
    directory.mkdir()
    depth = np.array([[0.05, 2.0], [50.0, 3.0]], np.float32)
    np.save(directory / "000000.npy", depth)
    cfg = PipelineConfig(depth_backend="file", depth_dir=str(directory),
                         depth_min=0.15, depth_trunc=12.0)

    from roomviz.types import Frame

    frame = Frame(index=0, rgb=np.zeros((2, 2, 3), np.uint8), source_index=0)
    out = build_depth_backend(cfg).predict(frame)
    assert out.depth.tolist() == [[0.0, 2.0], [0.0, 3.0]]
    assert out.valid.sum() == 2


def test_missing_sidecar_raises(tmp_path):
    directory = tmp_path / "depth"
    directory.mkdir()
    np.save(directory / "000000.npy", np.ones((4, 4), np.float32))
    cfg = PipelineConfig(depth_backend="file", depth_dir=str(directory))

    from roomviz.types import Frame

    frame = Frame(index=9, rgb=np.zeros((4, 4, 3), np.uint8), source_index=9)
    with pytest.raises(FileNotFoundError):
        build_depth_backend(cfg).predict(frame)


# --------------------------------------------------------------------------
# intrinsics resolution
# --------------------------------------------------------------------------

def _write_jpeg_with_focal(path, width, height, f35):
    from PIL import ExifTags, Image

    tag = {v: k for k, v in ExifTags.TAGS.items()}["FocalLengthIn35mmFilm"]
    image = Image.new("RGB", (width, height), (128, 128, 128))
    exif = Image.Exif()
    exif[tag] = int(f35)
    image.save(path, exif=exif)
    return path


def test_exif_focal_length_is_orientation_independent(tmp_path):
    """The same camera and lens must give the same focal length either way up.

    A 35mm frame is 36x24mm and the 36mm side is the *long* one, so dividing
    the width by 36 unconditionally under-reports a portrait photo by the
    aspect ratio -- roughly a third for a phone, which scales every recovered
    dimension with it.
    """
    from roomviz.geometry.camera import intrinsics_from_exif

    landscape = intrinsics_from_exif(
        _write_jpeg_with_focal(tmp_path / "l.jpg", 4032, 3024, 26), 4032, 3024
    )
    portrait = intrinsics_from_exif(
        _write_jpeg_with_focal(tmp_path / "p.jpg", 3024, 4032, 26), 3024, 4032
    )
    assert landscape is not None and portrait is not None
    assert landscape.fx == pytest.approx(portrait.fx, rel=1e-9)
    assert landscape.fx == pytest.approx(4032 * 26 / 36.0, rel=1e-9)


def test_exif_absent_returns_none(tmp_path):
    from PIL import Image

    from roomviz.geometry.camera import intrinsics_from_exif

    path = tmp_path / "bare.jpg"
    Image.new("RGB", (64, 48)).save(path)
    assert intrinsics_from_exif(path, 64, 48) is None


def test_explicit_intrinsics_win_and_rescale(tmp_path):
    """`--intrinsics` is given at capture resolution and must be rescaled."""
    from roomviz.geometry.camera import resolve_intrinsics

    cfg = PipelineConfig(intrinsics=(1000.0, 1000.0, 640.0, 360.0))
    intr = resolve_intrinsics(cfg, 640, 360, original_size=(1280, 720))
    assert intr.fx == pytest.approx(500.0)
    assert intr.fy == pytest.approx(500.0)
    assert intr.cx == pytest.approx((640.0 + 0.5) * 0.5 - 0.5)


def test_explicit_intrinsics_beat_exif(tmp_path):
    from roomviz.geometry.camera import resolve_intrinsics

    path = _write_jpeg_with_focal(tmp_path / "x.jpg", 800, 600, 28)
    cfg = PipelineConfig(intrinsics=(123.0, 456.0, 10.0, 20.0))
    intr = resolve_intrinsics(cfg, 800, 600, source_path=str(path), original_size=(800, 600))
    assert (intr.fx, intr.fy) == (123.0, 456.0)


def test_hfov_fallback_is_used_last():
    from roomviz.geometry.camera import resolve_intrinsics

    intr = resolve_intrinsics(PipelineConfig(hfov_deg=90.0), 100, 50)
    assert intr.fx == pytest.approx(50.0 / np.tan(np.pi / 4), rel=1e-9)


# --------------------------------------------------------------------------
# relative-depth conversion (pure numpy, no weights needed)
# --------------------------------------------------------------------------

def test_relative_depth_maps_disparity_to_a_metric_range():
    """Relative checkpoints emit disparity: larger means nearer."""
    from roomviz.perception.depth import _relative_to_metric

    disparity = np.linspace(0.0, 1.0, 200).astype(np.float32).reshape(10, 20)
    depth = _relative_to_metric(disparity, near=0.5, far=10.0)

    assert depth.shape == disparity.shape
    # Monotonically decreasing in disparity, and inside the requested range.
    flat = depth.reshape(-1)
    assert np.all(np.diff(flat) <= 1e-6)
    assert flat.min() == pytest.approx(0.5, rel=0.05)
    assert flat.max() == pytest.approx(10.0, rel=0.05)


def test_relative_depth_handles_a_constant_prediction():
    from roomviz.perception.depth import _relative_to_metric

    out = _relative_to_metric(np.full((8, 8), 3.0, np.float32), near=0.4, far=8.0)
    assert np.isfinite(out).all()
    assert out.min() > 0.0


def test_relative_depth_is_robust_to_outliers():
    """Percentile clipping must stop one spike compressing everything else."""
    from roomviz.perception.depth import _relative_to_metric

    disparity = np.linspace(0.2, 0.8, 400).astype(np.float32)
    clean = _relative_to_metric(disparity.copy(), near=0.5, far=10.0)
    spiked = disparity.copy()
    spiked[0] = 1e6
    assert np.allclose(clean[5:-5], _relative_to_metric(spiked, 0.5, 10.0)[5:-5], rtol=0.05)


# --------------------------------------------------------------------------
# sidecar lookup fallback
# --------------------------------------------------------------------------

def test_sidecar_positional_fallback_matches_sorted_order(tmp_path):
    """Unnumbered sidecars fall back to sorted position; pin that mapping down."""
    from roomviz.perception.precomputed import _find_for_frame

    directory = tmp_path / "depth"
    directory.mkdir()
    for name in ("alpha", "beta", "gamma"):
        np.save(directory / f"{name}.npy", np.ones((2, 2), np.float32))

    chosen = [_find_for_frame(directory, i, (".npy",)).stem for i in range(3)]
    assert chosen == ["alpha", "beta", "gamma"]
    with pytest.raises(FileNotFoundError):
        _find_for_frame(directory, 3, (".npy",))


def test_numbered_sidecars_beat_positional_order(tmp_path):
    from roomviz.perception.precomputed import _find_for_frame

    directory = tmp_path / "depth"
    directory.mkdir()
    for i in (0, 5, 9):
        np.save(directory / f"{i:06d}.npy", np.full((2, 2), float(i), np.float32))
    # Frame 5 must resolve to 000005, not to the second file in sorted order.
    assert _find_for_frame(directory, 5, (".npy",)).stem == "000005"
    assert _find_for_frame(directory, 9, (".npy",)).stem == "000009"


def test_labels_json_may_carry_thing_flags(tmp_path):
    """A stuff mask holding several objects must be declarable as such."""
    directory = tmp_path / "seg"
    directory.mkdir()
    ids = np.zeros((8, 8), np.int32)
    ids[:, :4] = 1
    ids[:, 4:] = 2
    np.save(directory / "000000.npy", ids)
    (directory / "labels.json").write_text(
        json.dumps({"1": "wall", "2": {"label": "clutter", "thing": False}})
    )
    cfg = PipelineConfig(seg_backend="file", seg_dir=str(directory))
    backend = build_segmentation_backend(cfg)

    from roomviz.types import Frame

    seg = backend.predict(Frame(index=0, rgb=np.zeros((8, 8, 3), np.uint8), source_index=0))
    by_id = seg.by_id()
    assert by_id[2].label == "clutter"
    assert by_id[2].is_thing is False
