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
