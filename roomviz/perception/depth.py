"""Monocular depth estimation.

The default backend is Depth Anything V2.  Prefer one of the *metric* indoor
checkpoints: they emit depth directly in metres, which is what makes the
reconstruction, the camera odometry and the reported object dimensions
consistent with one another.  Relative checkpoints still work, but their output
has to be pushed through an assumed near/far range, so absolute scale is a
guess.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..types import DepthMap, Frame
from .base import register_depth, resolve_device

log = logging.getLogger(__name__)


def _relative_to_metric(
    disparity: np.ndarray, near: float = 0.4, far: float = 10.0
) -> np.ndarray:
    """Map a relative inverse-depth prediction onto a plausible metric range.

    Depth Anything's relative head predicts disparity (larger = closer).  We
    normalise it to [0, 1] and interpolate *in disparity space*, which is the
    correct domain for a projective camera and keeps near-field detail.
    """
    d = disparity.astype(np.float32)
    lo, hi = np.percentile(d, 1.0), np.percentile(d, 99.0)
    if hi - lo < 1e-6:
        return np.full_like(d, (near + far) / 2.0)
    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    inv = d * (1.0 / near - 1.0 / far) + 1.0 / far
    return (1.0 / inv).astype(np.float32)


class DepthAnythingBackend:
    """Depth Anything V2 (and any other `AutoModelForDepthEstimation`)."""

    name = "depth-anything-v2"

    def __init__(self, cfg: PipelineConfig):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "the depth-anything backend needs torch + transformers: "
                "pip install 'roomviz[models]'"
            ) from exc

        self._torch = torch
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        source = cfg.depth_model
        log.info("loading depth model %s on %s", source, self.device)

        kwargs: dict[str, Any] = {}
        if cfg.models_dir:
            kwargs["cache_dir"] = cfg.models_dir
        if cfg.offline:
            kwargs["local_files_only"] = True
        self.processor = AutoImageProcessor.from_pretrained(source, **kwargs)
        self.model = AutoModelForDepthEstimation.from_pretrained(source, **kwargs)
        self.model.to(self.device).eval()

        est_type = getattr(self.model.config, "depth_estimation_type", None)
        if est_type is not None:
            self.metric = est_type == "metric"
        else:
            # Fall back to the checkpoint *name*, and only its last component:
            # matching the whole string would call a relative model metric
            # merely because it sat in a directory called "metric-cache".
            self.metric = "metric" in str(source).rstrip("/").split("/")[-1].lower()
            log.warning(
                "%s does not declare depth_estimation_type; guessing %s from "
                "its name. Pass --depth-near/--depth-far if the scale looks wrong.",
                source, "metric" if self.metric else "relative",
            )
        if not self.metric:
            log.warning(
                "%s is a relative-depth checkpoint; absolute scale will be "
                "assumed as %.2f-%.2f m (--depth-near/--depth-far). Prefer a "
                "metric indoor checkpoint for measurable output.",
                source, cfg.depth_near, cfg.depth_far,
            )

    def predict(self, frame: Frame) -> DepthMap:
        torch = self._torch
        h, w = frame.height, frame.width
        inputs = self.processor(images=frame.rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Newer transformers exposes a post-processor that handles the resize
        # and any model-specific scaling; fall back to manual interpolation.
        post = getattr(self.processor, "post_process_depth_estimation", None)
        if post is not None:
            try:
                result = post(outputs, target_sizes=[(h, w)])[0]
                pred = result["predicted_depth"]
            except Exception:  # pragma: no cover - API drift safety net
                pred = self._manual_resize(outputs.predicted_depth, h, w)
        else:  # pragma: no cover - older transformers
            pred = self._manual_resize(outputs.predicted_depth, h, w)

        depth = pred.detach().to("cpu").float().numpy()
        depth = np.squeeze(depth)

        if not self.metric:
            depth = _relative_to_metric(
                depth, near=self.cfg.depth_near, far=self.cfg.depth_far
            )

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < self.cfg.depth_min] = 0.0
        depth[depth > self.cfg.depth_trunc] = 0.0
        return DepthMap(depth=depth.astype(np.float32), metric=self.metric)

    def _manual_resize(self, predicted_depth, h: int, w: int):
        torch = self._torch
        pred = predicted_depth
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = torch.nn.functional.interpolate(
            pred, size=(h, w), mode="bicubic", align_corners=False
        )
        return pred.squeeze()


@register_depth("depth-anything-v2")
def _build_depth_anything(cfg: PipelineConfig) -> DepthAnythingBackend:
    return DepthAnythingBackend(cfg)
