"""Mapping from segmentation class names to pipeline roles.

Panoptic checkpoints ship their own ``id2label`` table, so rather than pinning
a hard-coded 150-entry ADE20K list we classify by *name*.  That keeps the rest
of the pipeline working if someone swaps in a COCO, ADE or custom checkpoint.

ADE20K names are semicolon-separated synonym lists (``"floor;flooring"``), so
every synonym is considered when matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
# distinct segment id per object.
#
# This list is a **hint of last resort**, not a decision procedure.  It is a
# curated subset of the common indoor classes, not the full ADE20K thing table,
# so membership says something only when it is positive: absence means "nobody
# typed this word", never "this class is stuff".  Nothing downstream may treat
# it as a gate - see :func:`split_policy`, which turns it into a *degree of
# scepticism* about splitting rather than a yes/no.  (It used to be a gate, and
# the result was that `painting` - which *is* in the list - made three paintings
# sharing one mask reconstruct as a single 3.6 m object, while any class nobody
# had typed was split with no restraint at all.)
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


@dataclass(frozen=True)
class SplitPolicy:
    """How willing the fusion stage should be to split one mask in 3D.

    Every object mask is splittable.  What varies with the evidence is *how
    much empty space* has to separate two blobs before they count as two
    objects, expressed as a multiple of :attr:`PipelineConfig.object_split_gap`.
    """

    gap_scale: float
    """Multiplier on the configured split gap."""

    evidence: str
    """``"thing"``, ``"stuff"`` or ``"unknown"`` - where the flag came from,
    for logging and tests."""


# The three settings below are ordered, not independent: whatever the numbers
# are, thing >= unknown >= stuff must hold, because that ordering is the whole
# point - more confidence that a mask is one object buys more reluctance to
# split it, and a class nobody has classified sits in between rather than at
# either extreme.
#
# The values are empirical, measured on the synthetic harness with the default
# 12 cm gap.  At 1.0x a pleated curtain arriving as eight panels 14 cm apart
# came back as eight objects; 2.0x (24 cm) makes it one.  2.5x (30 cm) still
# separates three paintings hung 45 cm apart, which is what a thing mask has to
# survive.  They are deliberately a soft ordering rather than a hard gate, so
# being wrong about a class degrades the split distance instead of switching
# the behaviour off.
#
# The gap is a coarse instrument on its own - an occluder's shadow is routinely
# wider than the space between two separate objects - so it is not the only
# thing holding this together: `roomviz.fusion.scene_fusion` additionally
# checks, in the image, whether something is standing in the gap.
_POLICIES = {
    "thing": SplitPolicy(gap_scale=2.5, evidence="thing"),
    "unknown": SplitPolicy(gap_scale=2.0, evidence="unknown"),
    "stuff": SplitPolicy(gap_scale=1.0, evidence="stuff"),
}


def split_policy(label: str, thing: bool | None = None) -> SplitPolicy:
    """How to split a segment of this class into 3D connected components.

    ``thing`` is the segmenter's own thing/stuff flag for the mask
    (:attr:`roomviz.types.Segment.is_thing`), which is the authoritative signal
    when the model provides one.  Only when it is ``None`` does the curated
    :data:`THING_CLASSES` word list get consulted, and then only to move the
    policy one notch, never to disable splitting.

    A "thing" mask covers one object by the segmenter's reckoning, so it takes a
    wide, unambiguous gap to overrule that.  A "stuff" mask routinely holds
    several objects, so the configured gap is taken at face value.  An unknown
    class gets the middle setting: it is neither trusted to be one object nor
    shattered at the first hole in the sampling.
    """
    if thing is None:
        thing = is_thing(label)
    if thing is True:
        return _POLICIES["thing"]
    if thing is False:
        return _POLICIES["stuff"]
    return _POLICIES["unknown"]
