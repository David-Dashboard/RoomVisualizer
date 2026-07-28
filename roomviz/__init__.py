"""RoomVisualizer - 3D room reconstruction from images and video.

Turns an image, a video or a folder of images into an interactive 3D scene in
which every object is segmented separately and the walls, floor and ceiling are
recovered as explicit planes.

    from roomviz import PipelineConfig, run

    result = run("room.mp4", "output", PipelineConfig(max_frames=16))
    print(len(result.scene.objects), "objects")
"""

from .config import PipelineConfig
from .pipeline import PipelineResult, export_scene, reconstruct, run
from .types import (
    CameraIntrinsics,
    DepthMap,
    Frame,
    ObjectInstance,
    Observation,
    PlaneSurface,
    Scene,
    Segment,
    Segmentation,
)

__version__ = "0.1.0"

__all__ = [
    "CameraIntrinsics",
    "DepthMap",
    "Frame",
    "ObjectInstance",
    "Observation",
    "PipelineConfig",
    "PipelineResult",
    "PlaneSurface",
    "Scene",
    "Segment",
    "Segmentation",
    "__version__",
    "export_scene",
    "reconstruct",
    "run",
]
