"""HEIC/HEIF support, registered once per process.

HEIC is the iPhone's default photo format, so it is the format a room capture
most often arrives in - and neither of this package's image dependencies reads
it unaided.  OpenCV's wheels do not bundle ``libheif``, and Pillow has no
native HEIF codec; ``pillow-heif`` supplies both, as a plugin that has to be
registered into Pillow before ``Image.open`` will recognise the format.

The registration lives here rather than in either caller because two
unconnected places need it: decoding the pixels
(:func:`roomviz.media.loader._read_image`) and reading the focal length out of
EXIF (:func:`roomviz.geometry.camera.intrinsics_from_exif`).  A capture that
decoded but reported no EXIF would silently fall back to a guessed field of
view, which scales every dimension in the output.
"""

from __future__ import annotations

_state: bool | None = None


def enable_heif() -> bool:
    """Register the HEIF opener with Pillow.  True if HEIC can now be read.

    Idempotent and cheap to call repeatedly: the import is attempted once and
    the outcome cached, so callers may invoke it on every image.
    """
    global _state
    if _state is None:
        try:
            import pillow_heif
        except ImportError:
            _state = False
        else:
            pillow_heif.register_heif_opener()
            _state = True
    return _state


HEIF_INSTALL_HINT = (
    "HEIC/HEIF images need the 'pillow-heif' package, which is not installed.\n"
    "  * install it with:  pip install pillow-heif\n"
    "  * or convert the photos to JPEG first - but keep their EXIF, or the "
    "camera's field of view is lost and every reported dimension becomes a "
    "guess\n"
    "  * or set Settings > Camera > Formats > Most Compatible on the iPhone "
    "and reshoot"
)
