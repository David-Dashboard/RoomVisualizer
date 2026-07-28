"""CLI-level integration tests."""

from __future__ import annotations

import json
import re

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

    assert payload["room"]["room_height"] == pytest.approx(room.height, abs=0.08)
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


# --------------------------------------------------------------------------
# error paths and the `view` server
#
# `run_cli` above only ever exercises the happy path.  Everything below is what
# a user hits when something is wrong, which is when the CLI's behaviour
# actually matters: it must give an exit code and a message, never a traceback.
# --------------------------------------------------------------------------

def test_no_arguments_prints_help_and_succeeds(capsys):
    assert main([]) == 0
    assert "Reconstruct an interactive 3D" in capsys.readouterr().out


def test_a_bad_configuration_exits_two_with_a_message(capture, tmp_path, capsys):
    """`--voxel 0` collapses the cloud to one point; validation must refuse it."""
    code = run_cli(capture, tmp_path / "out", "--voxel", "0")
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err and "voxel_size" in err


def test_an_impossible_field_of_view_exits_two(capture, tmp_path, capsys):
    """`--hfov 0` divides by tan(0) and writes a literal Infinity into JSON."""
    _, video, depth_dir, seg_dir, _, _ = capture
    code = main(
        [
            "reconstruct", str(video), "-o", str(tmp_path / "out"),
            "--depth-backend", "file", "--depth-dir", str(depth_dir),
            "--seg-backend", "file", "--seg-dir", str(seg_dir),
            "--hfov", "0",
        ]
    )
    assert code == 2
    assert "hfov" in capsys.readouterr().err


def test_a_missing_sidecar_directory_exits_cleanly(capture, tmp_path, capsys):
    _, video, _, seg_dir, _, hfov = capture
    code = main(
        [
            "reconstruct", str(video), "-o", str(tmp_path / "out"),
            "--depth-backend", "file", "--depth-dir", str(tmp_path / "absent"),
            "--seg-backend", "file", "--seg-dir", str(seg_dir),
            "--hfov", str(hfov),
        ]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_an_assumed_camera_is_printed_as_a_caveat(capture, tmp_path, capsys):
    """No --hfov and no EXIF (video carries none): the summary must say so."""
    _, video, depth_dir, seg_dir, _, _ = capture
    code = main(
        [
            "reconstruct", str(video), "-o", str(tmp_path / "assumed"),
            "--depth-backend", "file", "--depth-dir", str(depth_dir),
            "--seg-backend", "file", "--seg-dir", str(seg_dir),
            "--max-side", "320", "--no-glb", "--no-viewer", "--no-ply",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "ASSUMED" in out
    assert "caveats" in out
    assert "camera_assumed" in out


def test_the_summary_reports_a_measured_camera_and_every_dimension(
    capture, tmp_path, capsys
):
    """One run, two claims: the camera provenance and the printed summary.

    Merged into a single reconstruction on purpose - each CLI run costs about
    five seconds, and neither half needs its own.
    """
    assert run_cli(capture, tmp_path / "summary", "--no-glb", "--no-viewer") == 0
    out = capsys.readouterr().out
    # The camera was given, so it must not be reported as guessed.
    assert "from --hfov" in out
    assert "ASSUMED" not in out
    assert "95 deg horizontal FOV" in out
    # ... and the summary states its units and lists everything it found.
    assert "sizes are width x height x depth, in metres" in out
    for label in ("sofa", "table", "chair", "bookcase"):
        assert label in out
    assert "width x height, metres" in out
    assert "floor" in out
    # It says where the output went and how long it took, which is the only
    # confirmation a scripted run gets.
    assert re.search(r"Reconstructed \S*summary in \d+\.\d+s", out), out
    # Each object line carries its own point count, not a constant.
    counts = [int(m.replace(",", "")) for m in re.findall(r"\(([\d,]+) pts\)", out)]
    assert len(counts) == 4
    assert all(c > 100 for c in counts), counts
    assert len(set(counts)) > 1, counts


def _serve_once(monkeypatch):
    """Make `serve` return instead of blocking, so its setup path is testable."""
    import socketserver

    def stop(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(socketserver.TCPServer, "serve_forever", stop)


def test_view_walks_up_from_a_busy_port(tmp_path, monkeypatch, capsys):
    """README: `roomviz view` "walks up a few ports if the one you asked for is
    busy" - a stale server from an earlier run is the normal case."""
    import socket

    from roomviz.cli import serve

    _serve_once(monkeypatch)
    (tmp_path / "viewer.html").write_text("<html></html>")

    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    port = blocker.getsockname()[1]
    blocker.listen(1)
    try:
        serve(tmp_path, port=port, open_browser=False)
    finally:
        blocker.close()

    out = capsys.readouterr().out
    assert f"port {port} was busy; using {port + 1}" in out
    assert f"http://127.0.0.1:{port + 1}/viewer.html" in out


def test_view_reports_a_missing_viewer_without_refusing_to_serve(
    tmp_path, monkeypatch, capsys
):
    from roomviz.cli import serve

    _serve_once(monkeypatch)
    serve(tmp_path, port=0, open_browser=False)
    captured = capsys.readouterr()
    assert "no viewer.html" in captured.err
    assert "serving" in captured.out


def test_view_gives_up_cleanly_when_every_port_is_taken(tmp_path, monkeypatch, capsys):
    import socket

    from roomviz.cli import serve

    _serve_once(monkeypatch)
    first = socket.socket()
    first.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    first.bind(("127.0.0.1", 0))
    base = first.getsockname()[1]
    first.listen(1)
    blockers = [first]
    try:
        for offset in range(1, 20):
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", base + offset))
                sock.listen(1)
                blockers.append(sock)
            except OSError:
                sock.close()  # already taken by something else; equally blocking
        with pytest.raises(SystemExit) as excinfo:
            serve(tmp_path, port=base, open_browser=False)
    finally:
        for sock in blockers:
            sock.close()
    assert excinfo.value.code == 2
    assert "no free port" in capsys.readouterr().err
