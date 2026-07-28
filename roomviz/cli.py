"""Command-line interface."""

from __future__ import annotations

import argparse
import functools
import http.server
import logging
import socketserver
import sys
import webbrowser
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .perception.base import available_backends


def _add_reconstruct_args(parser: argparse.ArgumentParser) -> None:
    defaults = PipelineConfig()

    parser.add_argument("input", help="image, video, or directory of images")
    parser.add_argument(
        "-o", "--output", default="output", help="output directory (default: output)"
    )

    group = parser.add_argument_group("input sampling")
    group.add_argument("--max-frames", type=int, default=defaults.max_frames,
                       help="maximum keyframes taken from a video")
    group.add_argument("--frame-stride", type=int, default=defaults.frame_stride,
                       help="sample every Nth video frame (0 = automatic)")
    group.add_argument("--max-side", type=int, default=defaults.max_side,
                       help="longest image side fed to the models")
    group.add_argument("--min-sharpness", type=float, default=defaults.min_sharpness,
                       help="drop frames blurrier than this (0 = keep all)")

    group = parser.add_argument_group("camera")
    group.add_argument("--hfov", type=float, default=None,
                       help="assumed horizontal field of view in degrees")
    group.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                       help="explicit pinhole intrinsics, in input-resolution pixels")

    backends = available_backends()
    group = parser.add_argument_group("perception")
    group.add_argument("--depth-backend", default=defaults.depth_backend,
                       choices=backends["depth"])
    group.add_argument("--depth-model", default=defaults.depth_model,
                       help="depth checkpoint id or local path")
    group.add_argument("--seg-backend", default=defaults.seg_backend,
                       choices=backends["segmentation"])
    group.add_argument("--seg-model", default=defaults.seg_model,
                       help="segmentation checkpoint id or local path")
    group.add_argument("--device", default=defaults.device,
                       help="torch device: auto, cpu, cuda, mps")
    group.add_argument("--models-dir", default=None, help="weight cache directory")
    group.add_argument("--offline", action="store_true",
                       help="never hit the network; use cached weights only")
    group.add_argument("--depth-dir", default=None,
                       help="precomputed depth sidecars (for --depth-backend file)")
    group.add_argument("--seg-dir", default=None,
                       help="precomputed segment maps (for --seg-backend file)")
    group.add_argument("--depth-scale", type=float, default=defaults.depth_scale,
                       help="integer sidecar depth units to metres (default mm)")
    group.add_argument("--depth-near", type=float, default=defaults.depth_near,
                       help="nearest depth in metres assumed for a relative "
                            "(non-metric) checkpoint")
    group.add_argument("--depth-far", type=float, default=defaults.depth_far,
                       help="farthest depth in metres assumed for a relative "
                            "(non-metric) checkpoint")

    group = parser.add_argument_group("reconstruction")
    group.add_argument("--voxel", type=float, default=defaults.voxel_size,
                       help="voxel size in metres for downsampling and fusion")
    group.add_argument("--depth-trunc", type=float, default=defaults.depth_trunc,
                       help="ignore depth beyond this many metres")
    group.add_argument("--depth-min", type=float, default=defaults.depth_min)
    group.add_argument("--edge-discard", type=float, default=defaults.edge_discard,
                       help="relative depth jump treated as an occlusion edge")
    group.add_argument("--no-poses", action="store_true",
                       help="skip camera motion estimation (all frames share a pose)")
    group.add_argument("--no-align", action="store_true",
                       help="skip gravity/Manhattan alignment")
    group.add_argument("--min-object-points", type=int, default=defaults.min_object_points)
    group.add_argument("--association-iou", type=float, default=defaults.association_iou,
                       help="voxel IoU needed to merge detections across frames")
    group.add_argument("--object-split-gap", type=float, default=defaults.object_split_gap,
                       help="empty gap in metres that separates two objects "
                            "sharing one segmentation label")
    group.add_argument("--plane-threshold", type=float, default=defaults.plane_threshold,
                       help="RANSAC inlier distance in metres for plane fitting")
    group.add_argument("--plane-min-inliers", type=int, default=defaults.plane_min_inliers)
    group.add_argument("--max-walls", type=int, default=defaults.max_walls)

    group = parser.add_argument_group("output")
    group.add_argument("--no-ply", action="store_true", help="skip the PLY point cloud")
    group.add_argument("--no-glb", action="store_true", help="skip the GLB scene")
    group.add_argument("--no-objects", action="store_true",
                       help="skip per-object PLY files")
    group.add_argument("--no-viewer", action="store_true", help="skip viewer.html")
    group.add_argument("--point-budget", type=int, default=defaults.point_budget)
    group.add_argument("--open", action="store_true",
                       help="serve and open the viewer when finished")


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    defaults_hfov = PipelineConfig().hfov_deg
    return PipelineConfig(
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        min_sharpness=args.min_sharpness,
        max_side=args.max_side,
        hfov_deg=defaults_hfov if args.hfov is None else args.hfov,
        hfov_explicit=args.hfov is not None,
        intrinsics=tuple(args.intrinsics) if args.intrinsics else None,
        depth_backend=args.depth_backend,
        depth_model=args.depth_model,
        seg_backend=args.seg_backend,
        seg_model=args.seg_model,
        device=args.device,
        models_dir=args.models_dir,
        offline=args.offline,
        depth_dir=args.depth_dir,
        seg_dir=args.seg_dir,
        depth_scale=args.depth_scale,
        depth_near=args.depth_near,
        depth_far=args.depth_far,
        depth_trunc=args.depth_trunc,
        depth_min=args.depth_min,
        voxel_size=args.voxel,
        edge_discard=args.edge_discard,
        estimate_poses=not args.no_poses,
        association_iou=args.association_iou,
        object_split_gap=args.object_split_gap,
        min_object_points=args.min_object_points,
        plane_threshold=args.plane_threshold,
        plane_min_inliers=args.plane_min_inliers,
        max_walls=args.max_walls,
        align_gravity=not args.no_align,
        export_ply=not args.no_ply,
        export_glb=not args.no_glb,
        export_objects=not args.no_objects,
        export_viewer=not args.no_viewer,
        point_budget=args.point_budget,
    )


def serve(
    directory: str | Path,
    port: int = 8000,
    open_browser: bool = True,
    host: str = "127.0.0.1",
) -> None:
    """Serve a results directory over HTTP so the viewer can fetch its assets."""
    directory = Path(directory).resolve()
    if not (directory / "viewer.html").exists():
        print(f"warning: no viewer.html in {directory}", file=sys.stderr)

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    # A stale server from an earlier run is the normal case, not an error;
    # walk up a few ports rather than dumping a traceback.
    httpd = None
    for candidate in range(port, port + 20):
        try:
            httpd = Server((host, candidate), handler)
            break
        except OSError:
            continue
    if httpd is None:
        print(
            f"error: no free port in {port}-{port + 19} on {host}. "
            f"Stop the other server, or pass --port.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    with httpd:
        bound = httpd.server_address[1]
        if bound != port:
            print(f"port {port} was busy; using {bound}")
        display_host = "127.0.0.1" if host in ("127.0.0.1", "localhost") else host
        url = f"http://{display_host}:{bound}/viewer.html"
        print(f"serving {directory} at {url}  (ctrl-c to stop)")
        if host == "0.0.0.0":
            import socket

            try:
                probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                probe.connect(("10.255.255.255", 1))
                lan = probe.getsockname()[0]
                probe.close()
                print(f"  on your network:  http://{lan}:{bound}/viewer.html")
            except OSError:
                pass
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roomviz",
        description=(
            "Reconstruct an interactive 3D visualisation of a room from images "
            "or video, with objects segmented individually and walls, floor and "
            "ceiling extracted as planes."
        ),
    )
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for info, -vv for debug")
    parser.add_argument("-q", "--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command")

    reconstruct = sub.add_parser(
        "reconstruct", help="build a 3D scene from an image, video or image folder"
    )
    _add_reconstruct_args(reconstruct)

    view = sub.add_parser("view", help="serve an existing results directory")
    view.add_argument("directory", nargs="?", default="output")
    view.add_argument("-p", "--port", type=int, default=8000)
    view.add_argument("--no-browser", action="store_true")
    view.add_argument("--host", default="127.0.0.1",
                      help="address to bind; use 0.0.0.0 to reach the viewer "
                           "from a phone on the same network")

    sub.add_parser("backends", help="list the available perception backends")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `roomviz photo.jpg` is shorthand for `roomviz reconstruct photo.jpg`.
    known = {"reconstruct", "view", "backends"}
    if argv and not argv[0].startswith("-") and argv[0] not in known:
        argv.insert(0, "reconstruct")

    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING if args.quiet else (
        logging.DEBUG if args.verbose >= 2 else
        logging.INFO if args.verbose >= 1 else logging.INFO
    )
    logging.basicConfig(
        level=level, format="%(levelname).1s %(name)s: %(message)s", stream=sys.stderr
    )

    if args.command == "backends" or args.command is None and not argv:
        if args.command is None:
            parser.print_help()
            return 0
        for kind, names in available_backends().items():
            print(f"{kind}: {', '.join(names)}")
        return 0

    if args.command == "view":
        serve(args.directory, args.port, not args.no_browser, args.host)
        return 0

    if args.command != "reconstruct":
        parser.print_help()
        return 0

    from .pipeline import run  # imported late so `--help` stays fast

    cfg = _config_from_args(args)
    try:
        result = run(args.input, args.output, cfg)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        # Model-weight failures carry a multi-line explanation; print it as-is.
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        # Bad configuration, unreadable sidecars, or input that yielded no
        # usable geometry.  These already carry actionable messages; what they
        # must not do is reach the user as a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    scene = result.scene
    print()
    print(f"Reconstructed {result.output_dir} in {result.elapsed:.1f}s")
    intr = scene.intrinsics
    if intr is not None:
        note = {
            "explicit_intrinsics": "from --intrinsics",
            "hfov_flag": "from --hfov",
            "exif": "from image EXIF",
            "assumed_default": "ASSUMED -- pass --hfov or --intrinsics to measure",
        }.get(intr.provenance, intr.provenance)
        hfov = np.rad2deg(2 * np.arctan((intr.width / 2.0) / intr.fx))
        print(f"  camera   : {hfov:.0f} deg horizontal FOV, {note}")
    print("  sizes are width x height x depth, in metres")
    print(f"  points   : {scene.points.shape[0]:,}")
    print(f"  objects  : {len(scene.objects)}")
    for inst in scene.objects[:20]:
        lo, hi = inst.aabb
        size = hi - lo
        print(
            f"      [{inst.instance_id:>3}] {inst.label.split(';')[0]:<22}"
            f" {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m"
            f"  ({inst.points.shape[0]:,} pts)"
        )
    if len(scene.objects) > 20:
        print(f"      ... and {len(scene.objects) - 20} more")
    print(f"  surfaces : {len(scene.surfaces)}  (width x height, metres)")
    for surface in scene.surfaces:
        width, height = surface.extents
        print(
            f"      [{surface.surface_id:>3}] {surface.kind:<10}"
            f" {width:5.2f} x {height:5.2f} m   area {surface.area:6.2f} m2"
        )
    from .export.scene_json import build_scene_dict

    caveats = build_scene_dict(scene, cfg).get("caveats", [])
    if caveats:
        print()
        print("  caveats  :")
        for caveat in caveats:
            print(f"      - {caveat}")
    print()
    print(f"  open the viewer with:  roomviz view {result.output_dir}")

    if args.open:
        serve(result.output_dir, open_browser=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
