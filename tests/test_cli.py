"""CLI-level integration tests."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from synthetic import default_room, labels_for, look_at, render

from roomviz.cli import main
from roomviz.types import CameraIntrinsics


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    """A small video with ground-truth depth and mask sidecars."""
    directory = tmp_path_factory.mktemp("capture")
    room = default_room()
    width, height, count, hfov = 320, 240, 8, 95.0
    intr = CameraIntrinsics.from_hfov(width, height, hfov)

    depth_dir = directory / "depth"
    seg_dir = directory / "segmentation"
    depth_dir.mkdir()
    seg_dir.mkdir()

    video = directory / "room.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
    )
    assert writer.isOpened()
    for i in range(count):
        s = i / (count - 1)
        pose = look_at(
            np.array([1.8 + 1.4 * s, 1.55, 0.15 + 0.15 * s]),
            np.array([2.4 + 0.25 * (s - 0.5), 0.75, 2.7]),
        )
        rgb, depth, ids = render(room, pose, intr)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_dir / f"{i:06d}.npy", depth)
        np.save(seg_dir / f"{i:06d}.npy", ids)
    writer.release()
    (seg_dir / "labels.json").write_text(
        json.dumps({str(k): v for k, v in labels_for(room).items()})
    )
    return directory, video, depth_dir, seg_dir, room, hfov


def run_cli(capture, out_dir, *extra):
    _, video, depth_dir, seg_dir, _, hfov = capture
    return main(
        [
            "reconstruct", str(video), "-o", str(out_dir),
            "--depth-backend", "file", "--depth-dir", str(depth_dir),
            "--seg-backend", "file", "--seg-dir", str(seg_dir),
            "--hfov", str(hfov), "--max-side", "320",
            *extra,
        ]
    )


def test_backends_command(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    assert "depth:" in out and "segmentation:" in out


def test_help_is_available(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "Reconstruct an interactive 3D" in capsys.readouterr().out


def test_missing_input_reports_cleanly(tmp_path, capsys):
    code = main(["reconstruct", str(tmp_path / "absent.mp4"), "-o", str(tmp_path / "o")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_reconstruct_end_to_end(capture, tmp_path, capsys):
    out_dir = tmp_path / "output"
    assert run_cli(capture, out_dir) == 0

    printed = capsys.readouterr().out
    assert "objects" in printed and "surfaces" in printed

    payload = json.loads((out_dir / "scene.json").read_text())
    room = capture[4]

    # Every object is found exactly once, with the right label.
    labels = sorted(o["label"] for o in payload["objects"])
    assert labels == sorted(b.label for b in room.boxes)

    # Dimensions are recovered in metres.  The bookcase's base is occluded by
    # the table from every viewpoint, so only its footprint is checked.
    truth = {b.label: b for b in room.boxes}
    for entry in payload["objects"]:
        box = truth[entry["label"]]
        for axis in (0, 2):
            assert abs(entry["size"][axis] - box.size[axis]) < 0.2, entry["label"]
        if entry["label"] != "bookcase":
            assert abs(entry["size"][1] - box.size[1]) < 0.2, entry["label"]

    assert payload["room"]["room_height"] == pytest.approx(room.height, abs=0.15)
    assert payload["summary"]["surfaces_by_kind"]["floor"] == 1
    assert payload["summary"]["surfaces_by_kind"]["wall"] >= 2

    for artefact in ("scene.ply", "scene.glb", "viewer.html"):
        assert (out_dir / artefact).exists()
    assert len(list((out_dir / "objects").glob("*.ply"))) == len(payload["objects"])


def test_bare_input_defaults_to_reconstruct(capture, tmp_path):
    """`roomviz room.mp4` is shorthand for `roomviz reconstruct room.mp4`."""
    _, video, depth_dir, seg_dir, _, hfov = capture
    out_dir = tmp_path / "shorthand"
    code = main(
        [
            str(video), "-o", str(out_dir),
            "--depth-backend", "file", "--depth-dir", str(depth_dir),
            "--seg-backend", "file", "--seg-dir", str(seg_dir),
            "--hfov", str(hfov), "--max-side", "320", "--no-glb", "--no-viewer",
        ]
    )
    assert code == 0
    assert (out_dir / "scene.json").exists()


def test_output_toggles_are_honoured(capture, tmp_path):
    out_dir = tmp_path / "minimal"
    assert run_cli(
        capture, out_dir, "--no-glb", "--no-ply", "--no-objects", "--no-viewer"
    ) == 0
    assert (out_dir / "scene.json").exists()
    assert not (out_dir / "scene.glb").exists()
    assert not (out_dir / "scene.ply").exists()
    assert not (out_dir / "viewer.html").exists()
    assert not (out_dir / "objects").exists()


def test_viewer_assets_are_self_contained(capture, tmp_path):
    out_dir = tmp_path / "viewer"
    assert run_cli(capture, out_dir, "--no-ply", "--no-objects") == 0

    html = (out_dir / "viewer.html").read_text()
    assert "./vendor/three.module.js" in html
    # No external origins: the viewer must work offline.
    assert "http://" not in html.replace("http://127.0.0.1", "")
    assert "https://" not in html
    assert (out_dir / "vendor" / "three.module.js").stat().st_size > 100_000
    assert (out_dir / "vendor" / "jsm" / "controls" / "OrbitControls.js").exists()
    assert (out_dir / "vendor" / "jsm" / "utils" / "BufferGeometryUtils.js").exists()
