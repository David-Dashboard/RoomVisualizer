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


def classify(label: str) -> tuple[str, str | None]:
    """Classify a segmentation label.

    Returns ``(role, structure_kind)`` where role is one of
    :data:`ROLE_OBJECT`, :data:`ROLE_STRUCTURE`, :data:`ROLE_IGNORE`.

    Thing/stuff metadata deliberately plays no part here: it decides whether a
    mask may be *split*, not what the mask is.  ADE20K marks rugs and curtains
    as stuff, and a curtain is still an object you want to toggle in the
    viewer.  (An earlier signature took an ``is_thing`` argument that the body
    never read, and whose name shadowed the module-level :func:`is_thing`
    inside this scope - a trap for anyone who tried to start using it.)
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


# Indoor "thing" classes: ones a panoptic model instance-separates, emitting a
# distinct segment id per object.  A segment carrying one of these names covers
# exactly one object, so it must never be split geometrically - an object that
# an occluder cut into two visible patches would otherwise become two objects.
#
# This is a curated list of the common indoor classes rather than the full
# ADE20K thing table.  Anything absent is reported as *unknown* rather than as
# stuff, and unknown segments fall back to geometric splitting (with rejoining
# if later views connect the pieces), which is the safer default.
THING_CLASSES: frozenset[str] = frozenset(
    {
        "sofa", "couch", "lounge", "chair", "armchair", "swivel chair", "seat",
        "bench", "stool", "table", "coffee table", "desk", "bed", "cabinet",
        "wardrobe", "closet", "press", "chest of drawers", "chest", "dresser",
        "bookcase", "shelf", "counter", "countertop", "sink", "refrigerator",
        "icebox", "oven", "stove", "microwave", "dishwasher", "washer",
        "toilet", "bathtub", "shower", "television", "tv", "crt screen",
        "screen", "monitor", "computer", "laptop", "keyboard", "mouse",
        "lamp", "chandelier", "sconce", "light", "fan", "clock", "vase",
        "pot", "flowerpot", "plaything", "toy", "painting", "picture",
        "poster", "bulletin board", "mirror", "cushion", "pillow", "blanket",
        "towel", "basket", "box", "bag", "book", "bottle", "glass", "cup",
        "plate", "food", "tray", "person", "individual", "someone", "cat",
        "dog", "bicycle", "car", "fireplace", "radiator", "piano",
        "sculpture", "statue", "ottoman", "pouf", "footstool", "barrel",
        "trash can", "ashcan", "dustbin", "fan palm", "stairs",
    }
)


def is_thing(label: str) -> bool | None:
    """Whether a class is instance-separated by a panoptic model.

    Returns ``None`` when the class is not in :data:`THING_CLASSES` - meaning
    "unknown", not "stuff".  Callers must treat unknown conservatively.
    """
    names = set(synonyms(label))
    if not names:
        return None
    if names & THING_CLASSES:
        return True
    return None


def is_split_candidate(label: str, thing: bool | None = None) -> bool:
    """Whether a segment may be split into connected components in 3D.

    Panoptic "stuff" classes produce one mask per class per image, so several
    separate objects can arrive as a single segment; splitting recovers them.
    True "thing" classes are already instance-separated and must not be split.
    """
    if thing is None:
        thing = is_thing(label)
    if thing:
        return False
    role, _ = classify(label)
    return role == ROLE_OBJECT
