#!/usr/bin/env python3
"""Render a synthetic room to a video plus ground-truth depth and masks.

This gives you something to run the whole pipeline on without downloading any
model weights - useful as a smoke test, and as a reference for the sidecar
layout the ``file`` backends expect.

    python examples/make_demo_room.py --out examples/demo
    roomviz reconstruct examples/demo/room.mp4 -o output \\
        --depth-backend file --depth-dir examples/demo/depth \\
        --seg-backend file  --seg-dir   examples/demo/segmentation \\
        --hfov 95
    roomviz view output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from synthetic import default_room, labels_for, look_at, render  # noqa: E402

from roomviz.types import CameraIntrinsics  # noqa: E402


def smooth_walkthrough(room, count: int) -> list[np.ndarray]:
    """A slow pan across the room, densely sampled so it looks like real video."""
    poses = []
    for i in range(count):
        s = i / max(1, count - 1)
        eye = np.array([1.8 + 1.4 * s, 1.55 + 0.05 * np.sin(s * 5.0), 0.15 + 0.15 * s])
        target = np.array([2.4 + 0.25 * (s - 0.5), 0.75, 2.7])
        poses.append(look_at(eye, target))
    return poses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="examples/demo", help="output directory")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=95.0)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args(argv)

    out = Path(args.out)
    depth_dir = out / "depth"
    seg_dir = out / "segmentation"
    for directory in (out, depth_dir, seg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    room = default_room()
    intr = CameraIntrinsics.from_hfov(args.width, args.height, args.hfov)
    poses = smooth_walkthrough(room, args.frames)

    video_path = out / "room.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open a video writer for {video_path}")

    for i, pose in enumerate(poses):
        rgb, depth, ids = render(room, pose, intr)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_dir / f"{i:06d}.npy", depth.astype(np.float32))
        np.save(seg_dir / f"{i:06d}.npy", ids.astype(np.int32))
    writer.release()

    (seg_dir / "labels.json").write_text(
        json.dumps({str(k): v for k, v in labels_for(room).items()}, indent=2)
    )
    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "room": {"width": room.width, "height": room.height, "depth": room.depth},
                "objects": [
                    {
                        "label": box.label,
                        "min": box.lo.tolist(),
                        "max": box.hi.tolist(),
                        "size": box.size.tolist(),
                    }
                    for box in room.boxes
                ],
                "intrinsics": intr.to_dict(),
                "poses": [pose.reshape(-1).tolist() for pose in poses],
            },
            indent=2,
        )
    )

    print(f"wrote {video_path} ({args.frames} frames at {args.width}x{args.height})")
    print(f"      {depth_dir}/  ground-truth depth in metres (.npy)")
    print(f"      {seg_dir}/    ground-truth segment ids + labels.json")
    print(f"      {out / 'ground_truth.json'}  room and object dimensions")
    print()
    print("now run:")
    print(f"  roomviz reconstruct {video_path} -o output \\")
    print(f"      --depth-backend file --depth-dir {depth_dir} \\")
    print(f"      --seg-backend file --seg-dir {seg_dir} --hfov {args.hfov:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
