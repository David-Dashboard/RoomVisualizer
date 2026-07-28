"""Sidecar validation, media loading and the CLI's error paths.

The end-to-end tests only ever feed this code well-formed input, so none of the
refusals, warnings or fallbacks it contains are exercised by them.  These are
the paths a user actually hits when their export script is wrong, and the
package documents specific behaviour for each of them in the README's "Bring
your own depth" section.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from roomviz.config import PipelineConfig
from roomviz.media.loader import classify as classify_media
from roomviz.media.loader import load_frames, resize_frame
from roomviz.perception.base import build_depth_backend, build_segmentation_backend
from roomviz.perception.precomputed import (
    _parse_labels,
    _segmentation_from_ids,
    check_sidecar_numbering,
)
from roomviz.types import Frame


def frame(index: int = 0, size: tuple[int, int] = (8, 8)) -> Frame:
    return Frame(
        index=index, rgb=np.zeros((size[0], size[1], 3), np.uint8), source_index=index
    )


# --------------------------------------------------------------------------
# 1-indexed sidecar refusal
# --------------------------------------------------------------------------

def _numbered(directory, indices, ext=".npy"):
    directory.mkdir(exist_ok=True)
    for i in indices:
        np.save(directory / f"{i:06d}{ext}", np.ones((4, 4), np.float32))
    return directory


def test_one_indexed_sidecars_are_refused(tmp_path):
    """README: "A 1-indexed export is detected and refused".

    Without the refusal every frame silently pairs with the previous frame's
    depth - the file exists and has the right name, so nothing else notices.
    """
    directory = _numbered(tmp_path / "depth", range(1, 6))
    with pytest.raises(ValueError, match="numbered from 1"):
        check_sidecar_numbering(directory, (".npy",))
    with pytest.raises(ValueError, match="numbered from 1"):
        build_depth_backend(PipelineConfig(depth_backend="file", depth_dir=str(directory)))


def test_zero_indexed_sidecars_are_accepted(tmp_path):
    check_sidecar_numbering(_numbered(tmp_path / "d0", range(0, 6)), (".npy",))


@pytest.mark.parametrize(
    "indices",
    [
        [],                # nothing numbered at all
        [0],               # a single 0-indexed file
        [1],               # a lone file: too little evidence to accuse
        [2, 3, 4],         # starts at 2, not 1 - a stride, not an off-by-one
        [1, 2, 4, 9],      # gappy: max != count, so not a contiguous 1..n run
    ],
)
def test_sidecar_numbering_does_not_cry_wolf(tmp_path, indices):
    """A false refusal is worse than none: it blocks a correct export.

    Each case here fails at least one clause of the detector, and every clause
    matters - the rule is "index 0 missing AND index 1 present AND at least two
    files AND the highest index equals the count", i.e. a contiguous 1..n run.
    """
    directory = _numbered(tmp_path / f"d{len(indices)}_{sum(indices)}", indices)
    check_sidecar_numbering(directory, (".npy",))


def test_unnumbered_files_are_ignored_by_the_detector(tmp_path):
    directory = tmp_path / "mixed"
    directory.mkdir()
    for name in ("000001", "000002", "labels", "notes"):
        np.save(directory / f"{name}.npy", np.ones((2, 2), np.float32))
    # Only 000001 and 000002 parse as indices, and they are a contiguous 1..2.
    with pytest.raises(ValueError, match="numbered from 1"):
        check_sidecar_numbering(directory, (".npy",))


# --------------------------------------------------------------------------
# labels.json validation
# --------------------------------------------------------------------------

def _labels_file(tmp_path, payload, name="labels.json"):
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_labels_json_accepts_plain_names_and_objects(tmp_path):
    parsed = _parse_labels(
        _labels_file(tmp_path, {"1": "wall", "2": {"label": "books", "thing": False}})
    )
    assert parsed[1] == "wall"
    assert parsed[2] == {"label": "books", "thing": False}


def test_labels_json_accepts_a_thing_flag_of_true(tmp_path):
    parsed = _parse_labels(_labels_file(tmp_path, {"3": {"label": "sofa", "thing": True}}))
    assert parsed[3]["thing"] is True


def test_labels_json_accepts_an_object_without_a_thing_flag(tmp_path):
    parsed = _parse_labels(_labels_file(tmp_path, {"3": {"label": "sofa"}}))
    assert parsed[3] == {"label": "sofa", "thing": None}


def test_labels_json_accepts_negative_and_zero_ids(tmp_path):
    """README: "`-1` means unlabelled; `0` is a normal segment id"."""
    parsed = _parse_labels(_labels_file(tmp_path, {"0": "wall", "-1": "void"}))
    assert parsed[0] == "wall" and parsed[-1] == "void"


@pytest.mark.parametrize(
    "payload,message",
    [
        ("{not json", "not valid JSON"),
        ('["wall", "floor"]', "must be an object"),
        ('{"wall": "wall"}', "not an integer"),
        ('{"1": 7}', "must map to a string or an object"),
        ('{"1": null}', "must map to a string or an object"),
        ('{"1": {"label": 3}}', "non-string"),
        ('{"1": {}}', "non-string"),
        ('{"1": {"label": "sofa", "thing": "false"}}', "non-boolean"),
        ('{"1": {"label": "sofa", "thing": 1}}', "non-boolean"),
        ('{"1": "wall", " 1": "floor"}', "more than once"),
        ('{"12": "wall", "012": "floor"}', "more than once"),
    ],
)
def test_labels_json_refuses_malformed_input_with_a_message(tmp_path, payload, message):
    path = tmp_path / f"labels_{abs(hash(payload))}.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match=message):
        _parse_labels(path)


def test_a_json_string_false_is_not_accepted_as_a_thing_flag(tmp_path):
    """`"false"` is truthy in Python, so accepting it would invert the meaning.

    A segment declared `"thing": "false"` - stuff, splittable - would be
    treated as a thing and never split, which is exactly backwards.
    """
    with pytest.raises(ValueError, match="use true or false"):
        _parse_labels(_labels_file(tmp_path, {"2": {"label": "books", "thing": "false"}}))


def test_unlisted_ids_get_a_placeholder_label():
    segmentation = _segmentation_from_ids(np.array([[-1, 0], [5, 5]], np.int32), {})
    by_id = segmentation.by_id()
    assert set(by_id) == {0, 5}          # -1 is unlabelled and produces no segment
    assert by_id[5].label == "segment_5"


def test_an_explicit_thing_flag_beats_the_label_vocabulary():
    """The file's flag is authoritative, including when it contradicts the name."""
    ids = np.array([[1, 2]], np.int32)
    segmentation = _segmentation_from_ids(
        ids,
        {
            1: {"label": "chair", "thing": False},   # a thing class, declared stuff
            2: {"label": "books", "thing": True},    # a stuff class, declared thing
        },
    )
    by_id = segmentation.by_id()
    assert by_id[1].is_thing is False
    assert by_id[2].is_thing is True


# --------------------------------------------------------------------------
# depth sidecar semantics
# --------------------------------------------------------------------------

def _depth_backend(tmp_path, array, **cfg_kwargs):
    directory = tmp_path / "depth"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "000000.npy", array)
    cfg = PipelineConfig(depth_backend="file", depth_dir=str(directory), **cfg_kwargs)
    return build_depth_backend(cfg)


def test_depth_scale_applies_to_integers_and_not_to_floats(tmp_path, caplog):
    """README: "`--depth-scale` is ignored for float input"."""
    integer = _depth_backend(
        tmp_path / "i", np.full((4, 4), 2500, np.uint16), depth_scale=0.001
    )
    assert integer.predict(frame(size=(4, 4))).depth.mean() == pytest.approx(2.5)

    with caplog.at_level("WARNING"):
        floating = _depth_backend(
            tmp_path / "f", np.full((4, 4), 2500.0, np.float32),
            depth_scale=1.0, depth_trunc=5000.0,
        )
        out = floating.predict(frame(size=(4, 4)))
    assert out.depth.mean() == pytest.approx(2500.0)
    # ... and it says so, because silently ignoring the flag is how a
    # millimetre float export becomes a 2.5 km room.
    assert "ignored for floating-point" in caplog.text


def test_float_depth_in_metres_produces_no_scale_warning(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        backend = _depth_backend(tmp_path, np.full((4, 4), 2.5, np.float32))
        backend.predict(frame(size=(4, 4)))
    assert "depth-scale" not in caplog.text


@pytest.mark.parametrize(
    "value,expected_valid",
    [(np.nan, False), (np.inf, False), (-np.inf, False), (0.0, False), (-1.0, False),
     (0.15, True), (12.0, True)],
)
def test_invalid_depths_are_zeroed(tmp_path, value, expected_valid):
    """README: "Zero, negative, NaN and infinite depths are all treated as
    invalid" - and the depth_min / depth_trunc bounds are inclusive."""
    array = np.full((2, 2), value, np.float32)
    out = _depth_backend(tmp_path / f"v{value}", array).predict(frame(size=(2, 2)))
    assert bool(out.valid.all()) is expected_valid


def test_depth_bounds_are_inclusive_at_both_ends(tmp_path):
    array = np.array([[0.1499, 0.15], [12.0, 12.001]], np.float32)
    out = _depth_backend(tmp_path, array, depth_min=0.15, depth_trunc=12.0).predict(
        frame(size=(2, 2))
    )
    assert out.depth.tolist() == [[0.0, pytest.approx(0.15)], [12.0, 0.0]]


def test_a_sidecar_of_a_different_size_is_resized_to_the_frame(tmp_path):
    array = np.full((16, 16), 3.0, np.float32)
    out = _depth_backend(tmp_path, array).predict(frame(size=(8, 8)))
    assert out.depth.shape == (8, 8)
    assert out.depth.mean() == pytest.approx(3.0)


def test_a_transposed_sidecar_warns_that_it_is_being_stretched(tmp_path, caplog):
    """A depth map with the wrong aspect ratio is a silent geometry error."""
    with caplog.at_level("WARNING"):
        backend = _depth_backend(tmp_path, np.full((10, 40), 3.0, np.float32))
        backend.predict(
            Frame(
                index=0,
                rgb=np.zeros((40, 10, 3), np.uint8),
                source_index=0,
                original_size=(10, 40),
            )
        )
    assert "different aspect ratio" in caplog.text


def test_a_patch_snapped_resize_does_not_warn(tmp_path, caplog):
    """640x480 capture -> 644x476 working frame is a 1.5% aspect change that
    the loader itself introduced.  Warning on it fires on the documented
    quickstart, twice a frame, which is how real warnings get ignored."""
    with caplog.at_level("WARNING"):
        backend = _depth_backend(tmp_path, np.full((480, 640), 3.0, np.float32))
        backend.predict(
            Frame(
                index=0,
                rgb=np.zeros((476, 644, 3), np.uint8),
                source_index=0,
                original_size=(640, 480),
            )
        )
    assert "aspect ratio" not in caplog.text


def test_a_16_bit_png_sidecar_is_read_and_scaled(tmp_path):
    directory = tmp_path / "depth"
    directory.mkdir()
    cv2.imwrite(str(directory / "000000.png"), np.full((4, 4), 1500, np.uint16))
    cfg = PipelineConfig(depth_backend="file", depth_dir=str(directory), depth_scale=0.001)
    out = build_depth_backend(cfg).predict(frame(size=(4, 4)))
    assert out.depth.mean() == pytest.approx(1.5)


def test_a_missing_seg_directory_is_reported_as_such(tmp_path):
    cfg = PipelineConfig(seg_backend="file", seg_dir=str(tmp_path / "absent"))
    with pytest.raises(NotADirectoryError):
        build_segmentation_backend(cfg)


def test_segmentation_without_labels_json_warns_but_still_works(tmp_path, caplog):
    directory = tmp_path / "seg"
    directory.mkdir()
    np.save(directory / "000000.npy", np.zeros((4, 4), np.int32))
    with caplog.at_level("WARNING"):
        backend = build_segmentation_backend(
            PipelineConfig(seg_backend="file", seg_dir=str(directory))
        )
    assert "no labels.json" in caplog.text
    assert backend.predict(frame(size=(4, 4))).segments[0].label == "segment_0"


# --------------------------------------------------------------------------
# media loading
# --------------------------------------------------------------------------

def test_an_empty_directory_is_reported_as_having_no_images(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no images found"):
        classify_media(empty)


def test_an_unreadable_file_is_reported_as_unsupported(tmp_path):
    path = tmp_path / "notes.xyz"
    path.write_text("this is not media")
    with pytest.raises(ValueError, match="unsupported media file"):
        classify_media(path)


def test_a_single_image_yields_one_frame_with_its_original_size(tmp_path):
    path = tmp_path / "shot.png"
    cv2.imwrite(str(path), np.zeros((1200, 1600, 3), np.uint8))
    frames = load_frames(path, PipelineConfig(max_side=768))
    assert len(frames) == 1
    assert frames[0].original_size == (1600, 1200)
    assert max(frames[0].rgb.shape[:2]) <= 768
    assert frames[0].source_index == 0


def test_image_directory_stride_covers_the_whole_folder(tmp_path):
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), np.full((28, 28, 3), i * 20, np.uint8))
    frames = load_frames(tmp_path, PipelineConfig(max_frames=4))
    assert [f.source_index for f in frames] == [0, 3, 6, 9]
    assert [f.index for f in frames] == [0, 1, 2, 3]


def test_image_directory_ignores_non_image_files(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), np.zeros((16, 16, 3), np.uint8))
    (tmp_path / "readme.txt").write_text("ignore me")
    (tmp_path / "labels.json").write_text("{}")
    assert classify_media(tmp_path).frame_count == 3
    assert len(load_frames(tmp_path, PipelineConfig(max_frames=10))) == 3


@pytest.mark.parametrize(
    "height,width,cap",
    [(1080, 1920, 768), (2000, 1000, 700), (3000, 4000, 512), (1200, 1600, 1024)],
)
def test_resize_never_exceeds_the_cap_and_snaps_to_the_patch_size(height, width, cap):
    out = resize_frame(np.zeros((height, width, 3), np.uint8), cap)
    assert max(out.shape[:2]) <= cap, (height, width, cap, out.shape)
    assert out.shape[0] % 14 == 0 and out.shape[1] % 14 == 0
    assert min(out.shape[:2]) >= 14


@pytest.mark.parametrize("side,cap", [(900, 512), (1080, 768), (700, 700), (1920, 768)])
def test_resize_honours_the_cap_for_square_images_too(side, cap):
    """A square image is the case that broke the cap.

    The long side rounds *down* to a patch multiple so the cap holds; the
    short side rounds to nearest, to stay close to the true aspect ratio.  On
    a square image those are the same side, so rounding to nearest carried it
    past the cap: 900x900 at 512 gave 518x504 and 1080x1080 at the default 768
    gave 770x756 - over the documented cap, and 2.8% non-square.  It fired for
    any square input with ``cap % 14 > 7``, which both 512 and the default 768
    satisfy and 700 does not, so 700 is here as the case that always worked.
    """
    out = resize_frame(np.zeros((side, side, 3), np.uint8), cap)
    assert max(out.shape[:2]) <= cap, (side, cap, out.shape[:2])
    assert out.shape[0] == out.shape[1], f"square input came out {out.shape[:2]}"
    assert out.shape[0] % 14 == 0


def test_a_square_image_is_still_downscaled_to_roughly_the_cap():
    """Honouring the cap must not be achieved by shrinking far below it.

    Stepping back one patch is the whole correction, so the result sits within
    14 px of the cap rather than at some arbitrary smaller size.
    """
    out = resize_frame(np.zeros((900, 900, 3), np.uint8), 512)
    assert out.shape[:2] == (504, 504)
    assert 512 - 14 < max(out.shape[:2]) <= 512


def test_resize_is_a_no_op_at_exactly_the_cap():
    original = np.zeros((500, 768, 3), np.uint8)
    assert resize_frame(original, 768) is original


def test_exif_intrinsics_centre_the_principal_point(tmp_path):
    """Not just the focal length: cx/cy must land between the end pixels."""
    from PIL import ExifTags, Image

    from roomviz.geometry.camera import intrinsics_from_exif

    tag = {v: k for k, v in ExifTags.TAGS.items()}["FocalLengthIn35mmFilm"]
    path = tmp_path / "exif.jpg"
    image = Image.new("RGB", (800, 600), (128, 128, 128))
    exif = Image.Exif()
    exif[tag] = 28
    image.save(path, exif=exif)

    intr = intrinsics_from_exif(path, 800, 600)
    assert intr is not None
    assert intr.cx == pytest.approx(399.5)
    assert intr.cy == pytest.approx(299.5)
    assert intr.fx == intr.fy == pytest.approx(800 * 28 / 36.0)


def test_a_video_reports_its_frame_count_and_frame_rate(tmp_path):
    """`classify` is what decides the automatic keyframe stride."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (32, 32))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((32, 32, 3), i * 20, np.uint8))
    writer.release()

    source = classify_media(path)
    assert source.kind == "video"
    assert source.frame_count == 10
    assert source.fps == pytest.approx(15.0, rel=0.1)

    # The automatic stride spreads max_frames evenly over frame_count.
    frames = load_frames(path, PipelineConfig(max_frames=5))
    assert [f.source_index for f in frames] == [0, 2, 4, 6, 8]


def test_the_blur_filter_drops_frames_and_keeps_the_rest(tmp_path):
    """`--min-sharpness` is documented as "drop blurry frames"; with the filter
    off nothing is dropped, and with it high enough everything is."""
    rng = np.random.default_rng(0)
    path = tmp_path / "mixed.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    writer.release()

    kept = load_frames(path, PipelineConfig(max_frames=6, min_sharpness=0.0))
    assert len(kept) == 6
    with pytest.raises(ValueError, match="no usable frames"):
        load_frames(path, PipelineConfig(max_frames=6, min_sharpness=1e9))


def test_a_portrait_image_is_capped_on_its_long_side_too():
    """The tall branch of `resize_frame`, which no other test reaches.

    Every other resize case here and in test_media_and_backends is landscape,
    so the `else` arm - where the *height* is the side rounded down and the
    width is the one rounded to nearest - is only exercised from here.
    """
    out = resize_frame(np.zeros((1920, 1080, 3), np.uint8), 768)
    assert out.shape[0] <= 768
    assert out.shape[0] % 14 == 0 and out.shape[1] % 14 == 0
    assert out.shape[0] > out.shape[1], "portrait must stay portrait"
    assert abs(out.shape[1] / out.shape[0] - 1080 / 1920) < 0.03


def test_a_video_with_an_unrecognised_extension_is_still_read_as_video(tmp_path):
    """README lists the known extensions, but OpenCV gets a go at anything else.

    Nothing was covering that fallback, and it is what makes a `.dat` or
    extensionless export from a capture rig work at all.
    """
    written = tmp_path / "capture.mp4"
    writer = cv2.VideoWriter(str(written), cv2.VideoWriter_fourcc(*"mp4v"), 12, (32, 32))
    assert writer.isOpened()
    for i in range(9):
        writer.write(np.full((32, 32, 3), i * 25, np.uint8))
    writer.release()
    path = tmp_path / "capture.dat"
    path.write_bytes(written.read_bytes())
    written.unlink()

    source = classify_media(path)
    assert source.kind == "video"
    assert source.frame_count == 9
    assert source.fps == pytest.approx(12.0, rel=0.15)
    assert len(load_frames(path, PipelineConfig(max_frames=3))) == 3


def test_an_image_with_an_unrecognised_extension_is_still_read_as_an_image(tmp_path):
    path = tmp_path / "photo.bin"
    cv2.imwrite(str(tmp_path / "photo.png"), np.zeros((16, 16, 3), np.uint8))
    path.write_bytes((tmp_path / "photo.png").read_bytes())
    assert classify_media(path).kind == "image"
    assert classify_media(path).frame_count == 1


# --------------------------------------------------------------------------
# gaps found by mutation testing
# --------------------------------------------------------------------------

def test_a_zero_indexed_set_with_a_gap_is_not_mistaken_for_one_indexed(tmp_path):
    """A gap in a 0-indexed export is legitimate and must not be refused.

    The 1-indexed check keys on index 0 being *absent*.  Asking instead
    whether 0 is present *and* the set looks contiguous rejects a perfectly
    good strided export -- files 0, 1, 2, 4 have `max == len`, which is the
    same shape the 1-indexed test looks for.
    """
    for index in (0, 1, 2, 4):
        np.save(tmp_path / f"{index:06d}.npy", np.zeros((4, 4), np.float32))
    check_sidecar_numbering(tmp_path, (".npy",))  # must not raise


def test_a_one_indexed_set_is_still_refused(tmp_path):
    """The control: the check above must not have disarmed the real refusal."""
    for index in (1, 2, 3):
        np.save(tmp_path / f"{index:06d}.npy", np.zeros((4, 4), np.float32))
    with pytest.raises(ValueError, match="numbered from 1"):
        check_sidecar_numbering(tmp_path, (".npy",))


def test_a_fractional_sharpness_threshold_still_filters(tmp_path):
    """`min_sharpness` is a float, and values below 1 have to work.

    Every other fixture uses 0 (off) or a large value, so a guard reading
    `> 1` instead of `> 0` behaves identically in all of them while silently
    disabling the filter for exactly the range a user tuning it would try.
    """
    video = tmp_path / "clip.mp4"
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(12):
        blank = np.zeros((64, 64, 3), np.uint8)
        textured = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(blank if i % 2 else textured)
    writer.release()

    def kept(min_sharpness: float) -> list[int]:
        cfg = PipelineConfig(
            min_sharpness=min_sharpness, max_frames=20, max_side=64, frame_stride=1
        )
        return [f.source_index for f in load_frames(video, cfg)]

    # A fractional threshold drops the blank frames ...
    assert kept(0.5) == [0, 2, 4, 6, 8, 10]
    # ... and zero means "keep everything", which is the documented off switch.
    assert kept(0.0) == list(range(12))


def _clip(tmp_path, seconds: float, fps: int = 30, name: str = "clip.mp4"):
    path = tmp_path / name
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for _ in range(int(seconds * fps)):
        writer.write(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_sparse_keyframes_are_warned_about(tmp_path, caplog):
    """Spreading a fixed frame budget over a long clip breaks tracking.

    The failure this prevents surfaces nowhere near its cause: consecutive
    keyframes stop overlapping, every pose link degrades, and what the user
    sees is a scene with hundreds of duplicated objects and no floor.
    """
    video = _clip(tmp_path, seconds=30)
    with caplog.at_level("WARNING"):
        load_frames(video, PipelineConfig(max_frames=24, max_side=64))
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("keyframe every" in w for w in warnings), warnings
    # It must name a usable remedy, not just complain.
    assert any("--max-frames" in w for w in warnings), warnings


def test_dense_keyframes_are_not_warned_about(tmp_path, caplog):
    """The control: a warning that always fires would be ignored."""
    video = _clip(tmp_path, seconds=30, name="dense.mp4")
    with caplog.at_level("WARNING"):
        load_frames(video, PipelineConfig(max_frames=90, max_side=64))
    assert not [
        r for r in caplog.records
        if r.levelname == "WARNING" and "keyframe every" in r.getMessage()
    ]


def test_a_short_clip_is_never_warned_about(tmp_path, caplog):
    """Three seconds into 24 frames is 8 fps - comfortably trackable."""
    video = _clip(tmp_path, seconds=3, name="short.mp4")
    with caplog.at_level("WARNING"):
        load_frames(video, PipelineConfig(max_frames=24, max_side=64))
    assert not [
        r for r in caplog.records
        if r.levelname == "WARNING" and "keyframe every" in r.getMessage()
    ]


def test_the_suggested_frame_count_would_actually_clear_the_threshold(tmp_path, caplog):
    """The number in the advice has to be one that silences the warning."""
    import re

    video = _clip(tmp_path, seconds=30, name="advice.mp4")
    with caplog.at_level("WARNING"):
        load_frames(video, PipelineConfig(max_frames=24, max_side=64))
    message = next(m for m in (r.getMessage() for r in caplog.records)
                   if "--max-frames" in m)
    suggested = int(re.search(r"--max-frames (\d+)", message).group(1))

    caplog.clear()
    with caplog.at_level("WARNING"):
        load_frames(video, PipelineConfig(max_frames=suggested, max_side=64))
    assert not [
        r for r in caplog.records
        if r.levelname == "WARNING" and "keyframe every" in r.getMessage()
    ], f"taking the advice ({suggested}) still warns"


def _photo_folder(tmp_path, count: int, f35: int | None = 13, name: str = "photos"):
    """A folder of "phone photos", optionally carrying a 35mm-equivalent focal
    length in EXIF the way every camera app writes one."""
    from PIL import Image

    directory = tmp_path / name
    directory.mkdir()
    rng = np.random.default_rng(0)
    for i in range(count):
        img = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
        exif = Image.Exif()
        if f35 is not None:
            exif[41989] = f35  # FocalLengthIn35mmFilm
        img.save(directory / f"IMG_{i:04d}.jpg", exif=exif)
    return directory


def test_a_folder_of_photos_takes_its_field_of_view_from_exif(tmp_path):
    """README: EXIF is used when no field of view is given.

    This was gated on there being exactly one frame, so a *folder* of photos --
    the obvious way to shoot a room, and the case where every file carries a
    focal length -- silently fell back to the assumed default instead, and
    every reported dimension then scaled with a guess.
    """
    from roomviz.geometry.camera import resolve_intrinsics

    directory = _photo_folder(tmp_path, 12)
    cfg = PipelineConfig(max_frames=12, max_side=320)
    frames = load_frames(directory, cfg)
    intr = resolve_intrinsics(
        cfg, frames[0].width, frames[0].height,
        source_path=frames[0].source, original_size=frames[0].original_size,
    )
    assert intr.provenance == "exif"
    # 13 mm equivalent is an ultra-wide: the point is that it is nowhere near
    # the 60 degree default that would otherwise have been assumed.
    hfov = np.degrees(2 * np.arctan((intr.width / 2) / intr.fx))
    assert hfov > 100.0, hfov


def test_photos_without_exif_still_fall_back_to_the_assumed_default(tmp_path):
    """The control: reading EXIF must not invent one when there is none."""
    from roomviz.geometry.camera import resolve_intrinsics

    directory = _photo_folder(tmp_path, 6, f35=None, name="bare")
    cfg = PipelineConfig(max_frames=6, max_side=320)
    frames = load_frames(directory, cfg)
    intr = resolve_intrinsics(
        cfg, frames[0].width, frames[0].height,
        source_path=frames[0].source, original_size=frames[0].original_size,
    )
    assert intr.provenance == "assumed_default"


def test_discarding_deliberately_taken_photos_is_warned_about(tmp_path, caplog):
    directory = _photo_folder(tmp_path, 40, name="many")
    with caplog.at_level("WARNING"):
        frames = load_frames(directory, PipelineConfig(max_frames=24, max_side=320))
    assert len(frames) < 40
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("photos" in w and "--max-frames" in w for w in warnings), warnings


def test_using_every_photo_is_not_warned_about(tmp_path, caplog):
    directory = _photo_folder(tmp_path, 20, name="allofthem")
    with caplog.at_level("WARNING"):
        frames = load_frames(directory, PipelineConfig(max_frames=20, max_side=320))
    assert len(frames) == 20
    assert not [
        r for r in caplog.records
        if r.levelname == "WARNING" and "photos" in r.getMessage()
    ]
