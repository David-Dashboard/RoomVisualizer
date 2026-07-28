"""Panoptic segmentation: per-pixel object instances plus room structure.

A panoptic model is the right tool here because we need both halves of the
scene in one pass: *things* (chairs, lamps, people) become individual 3D
objects, and *stuff* (wall, floor, ceiling) becomes the structural shell that
plane fitting turns into walls.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..types import Frame, Segment, Segmentation
from .base import register_segmentation, resolve_device
from .labels import classify, is_thing

log = logging.getLogger(__name__)


class Mask2FormerBackend:
    """Mask2Former / OneFormer style universal segmentation."""

    name = "mask2former"

    def __init__(self, cfg: PipelineConfig):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "the mask2former backend needs torch + transformers: "
                "pip install 'roomviz[models]'"
            ) from exc

        self._torch = torch
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        source = cfg.seg_model
        log.info("loading segmentation model %s on %s", source, self.device)

        kwargs: dict[str, Any] = {}
        if cfg.models_dir:
            kwargs["cache_dir"] = cfg.models_dir
        if cfg.offline:
            kwargs["local_files_only"] = True
        self.processor = AutoImageProcessor.from_pretrained(source, **kwargs)
        self.model = AutoModelForUniversalSegmentation.from_pretrained(source, **kwargs)
        self.model.to(self.device).eval()
        self.id2label = {
            int(k): v for k, v in getattr(self.model.config, "id2label", {}).items()
        }

    def predict(self, frame: Frame) -> Segmentation:
        torch = self._torch
        h, w = frame.height, frame.width
        inputs = self.processor(images=frame.rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        result = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(h, w)]
        )[0]

        ids = result["segmentation"].detach().to("cpu").numpy().astype(np.int32)
        segments: list[Segment] = []
        for info in result["segments_info"]:
            seg_id = int(info["id"])
            label_id = int(info["label_id"])
            label = self.id2label.get(label_id, f"class_{label_id}")
            role, kind = classify(label)
            segments.append(
                Segment(
                    segment_id=seg_id,
                    label=label,
                    role=role,
                    score=float(info.get("score", 1.0)),
                    structure_kind=kind,
                    is_thing=is_thing(label),
                )
            )

        # post_process_panoptic_segmentation marks unassigned pixels with -1.
        ids[ids < 0] = -1
        known = {s.segment_id for s in segments}
        stray = set(np.unique(ids).tolist()) - known - {-1}
        if stray:
            log.debug("dropping %d segment id(s) without metadata", len(stray))
            for sid in stray:
                ids[ids == sid] = -1

        log.debug(
            "frame %d: %d segments (%d objects, %d structural)",
            frame.index,
            len(segments),
            sum(1 for s in segments if s.role == "object"),
            sum(1 for s in segments if s.role == "structure"),
        )
        return Segmentation(ids=ids, segments=segments)


@register_segmentation("mask2former")
def _build_mask2former(cfg: PipelineConfig) -> Mask2FormerBackend:
    return Mask2FormerBackend(cfg)
