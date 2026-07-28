"""Mapping from segmentation class names to pipeline roles.

Panoptic checkpoints ship their own ``id2label`` table, so rather than pinning
a hard-coded 150-entry ADE20K list we classify by *name*.  That keeps the rest
of the pipeline working if someone swaps in a COCO, ADE or custom checkpoint.

ADE20K names are semicolon-separated synonym lists (``"floor;flooring"``), so
every synonym is considered when matching.
"""

from __future__ import annotations

import re

from ..types import ROLE_IGNORE, ROLE_OBJECT, ROLE_STRUCTURE

# Structural classes: these define the room shell and drive plane fitting.
STRUCTURE_KINDS: dict[str, tuple[str, ...]] = {
    "wall": ("wall", "walls"),
    "floor": ("floor", "flooring", "rug", "carpet", "carpeting", "ground"),
    "ceiling": ("ceiling",),
    "window": ("windowpane", "window", "screen door", "screen"),
    "door": ("door", "doorframe", "double door"),
    "column": ("column", "pillar", "pole"),
    "stairs": ("stairs", "stairway", "staircase", "step", "steps"),
}

# Classes that carry no useful indoor geometry.
IGNORE_NAMES: frozenset[str] = frozenset(
    {
        "sky",
        "water",
        "sea",
        "earth",
        "ground",
        "field",
        "sand",
        "grass",
        "mountain",
        "mount",
        "tree",
        "plant",  # outdoor greenery seen through windows; potted plants match "pot"
        "road",
        "route",
        "building",
        "edifice",
        "skyscraper",
        "house",
        "hill",
        "land",
    }
)

# Floor/ceiling coverings are structural for plane fitting, but a rug really is
# part of the floor plane, so keep it there rather than making it an object.
_FLOOR_COVERINGS = {"rug", "carpet", "carpeting"}

_SPLIT = re.compile(r"[;,/]")


def synonyms(label: str) -> list[str]:
    """Split an ADE-style synonym list into normalised names."""
    return [s.strip().lower() for s in _SPLIT.split(label or "") if s.strip()]


def structure_kind(label: str) -> str | None:
    """Return the structural sub-type for a label, or ``None`` if it is not
    part of the room shell."""
    names = set(synonyms(label))
    if not names:
        return None
    # Check the more specific kinds before the generic ones so that, e.g.,
    # "screen door" lands on `window` rather than `door`.
    for kind in ("ceiling", "wall", "window", "door", "column", "stairs", "floor"):
        if names & set(STRUCTURE_KINDS[kind]):
            return kind
    return None


def classify(label: str, is_thing: bool | None = None) -> tuple[str, str | None]:
    """Classify a segmentation label.

    Returns ``(role, structure_kind)`` where role is one of
    :data:`ROLE_OBJECT`, :data:`ROLE_STRUCTURE`, :data:`ROLE_IGNORE`.

    ``is_thing`` comes from the checkpoint's panoptic metadata when available.
    It is only advisory: ADE20K marks rugs and curtains as "stuff" but we still
    want a curtain to be an object you can toggle in the viewer.
    """
    names = set(synonyms(label))
    kind = structure_kind(label)
    if kind is not None:
        # A rug is floor, but only when it is not the sole floor evidence; we
        # keep it structural either way since it lies in the floor plane.
        if kind == "floor" and names & _FLOOR_COVERINGS:
            return ROLE_STRUCTURE, "floor"
        return ROLE_STRUCTURE, kind

    if names & IGNORE_NAMES:
        return ROLE_IGNORE, None

    return ROLE_OBJECT, None


def is_split_candidate(label: str, is_thing: bool | None) -> bool:
    """Whether a segment should be split into connected components.

    Panoptic "stuff" classes produce one mask per class per image, so three
    separate paintings arrive as a single segment.  Splitting them into
    connected components recovers per-object instances.  True "thing" classes
    are already instance-separated and must not be split, or an object
    occluded into two visible halves would become two objects.
    """
    if is_thing:
        return False
    role, _ = classify(label, is_thing)
    return role == ROLE_OBJECT
