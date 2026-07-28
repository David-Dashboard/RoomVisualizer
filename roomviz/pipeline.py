"""End-to-end pipeline: media in, 3D scene out."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .export.gltf import safe_label, write_glb
from .export.ply import write_ply
from .export.scene_json import write_scene_json
from .export.viewer import write_viewer
from .fusion.odometry import estimate_trajectory
from .fusion.scene_fusion import fuse
from .geometry.camera import resolve_intrinsics
from .media.loader import load_frames
from .perception.base import build_depth_backend, build_segmentation_backend
from .types import Observation, Scene

log = logging.getLogger(__name__)


def object_cloud_name(inst) -> str:
    """Filename for an object's PLY.

    Uses the same sanitiser as the glTF node names so that `scene.json` can
    reference both, and a checkpoint with unusual class names cannot make the
    two disagree.
    """
    return f"{inst.instance_id:03d}_{safe_label(inst.label)}.ply"


@dataclass
class PipelineResult:
    scene: Scene
    output_dir: Path
    files: dict[str, Path]
    elapsed: float


def reconstruct(
    input_path: str | Path, cfg: PipelineConfig | None = None
) -> tuple[Scene, list[Observation]]:
    """Run perception + fusion and return the reconstructed scene."""
    cfg = cfg or PipelineConfig()
    started = time.time()

    frames = load_frames(input_path, cfg)
    intr = resolve_intrinsics(
        cfg,
        frames[0].width,
        frames[0].height,
        source_path=frames[0].source if len(frames) == 1 else None,
        original_size=frames[0].original_size,
    )
    log.info(
        "intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f", intr.fx, intr.fy, intr.cx, intr.cy
    )

    depth_backend = build_depth_backend(cfg)
    seg_backend = build_segmentation_backend(cfg)

    depths = []
    segmentations = []
    for frame in frames:
        t0 = time.time()
        depths.append(depth_backend.predict(frame))
        segmentations.append(seg_backend.predict(frame))
        log.info(
            "frame %d/%d perceived in %.2fs",
            frame.index + 1,
            len(frames),
            time.time() - t0,
        )

    poses = estimate_trajectory(frames, depths, intr, cfg)

    observations = [
        Observation(
            frame=frame,
            depth=depth,
            segmentation=segmentation,
            intrinsics=intr,
            pose=pose,
        )
        for frame, depth, segmentation, pose in zip(
            frames, depths, segmentations, poses, strict=True
        )
    ]

    scene = fuse(observations, cfg)
    scene.meta["elapsed_seconds"] = round(time.time() - started, 2)
    scene.meta["input"] = str(input_path)
    return scene, observations


def export_scene(
    scene: Scene, output_dir: str | Path, cfg: PipelineConfig | None = None
) -> dict[str, Path]:
    """Write every configured output format into ``output_dir``."""
    cfg = cfg or PipelineConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    files["scene_json"] = write_scene_json(output_dir / "scene.json", scene, cfg)

    if cfg.export_ply:
        files["cloud_ply"] = write_ply(
            output_dir / "scene.ply", scene.points, scene.colors
        )

    if cfg.export_objects:
        objects_dir = output_dir / "objects"
        # Clear first: re-running into an existing directory would otherwise
        # leave the previous scene's clouds behind, and anything globbing
        # `objects/*.ply` would pick up objects that are no longer in the scene.
        if objects_dir.exists():
            for stale in objects_dir.glob("*.ply"):
                stale.unlink()
        for inst in scene.objects:
            files[f"object_{inst.instance_id}"] = write_ply(
                objects_dir / object_cloud_name(inst),
                inst.points,
                inst.colors,
            )
        if scene.objects:
            log.info("wrote %d object clouds to %s", len(scene.objects), objects_dir)

    if cfg.export_glb:
        files["scene_glb"] = write_glb(output_dir / "scene.glb", scene)

    if cfg.export_viewer:
        files["viewer"] = write_viewer(output_dir)

    return files


def run(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: PipelineConfig | None = None,
) -> PipelineResult:
    """Reconstruct ``input_path`` and write the results to ``output_dir``."""
    cfg = cfg or PipelineConfig()
    started = time.time()
    scene, _ = reconstruct(input_path, cfg)
    files = export_scene(scene, output_dir, cfg)
    elapsed = time.time() - started
    log.info("done in %.1fs -> %s", elapsed, output_dir)
    return PipelineResult(
        scene=scene, output_dir=Path(output_dir), files=files, elapsed=elapsed
    )
