# The spatial maths: what an image's shape and extents are.
from __future__ import annotations
from .enums import CHANNELS
from typing import Optional, Tuple



def _normalise_channels(channels:object)->int:
    """Validate a requested channel count."""
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise TypeError(
            "channels must be %s, got %s"
            % (" or ".join(str(v) for v in CHANNELS), type(channels).__name__))
    if channels == 3:
        # No device image format is 3 channels wide: a sample is returned as a 1, 2 or 4
        # component value, so RGB data has to be padded. Saying that is more useful than
        # letting the caller discover it from a driver error.
        raise ValueError(
            "channels=3 is not a device image format: samples are 1, 2 or 4 components "
            "wide. Pad the data to 4 channels and pass channels=4")
    if channels not in CHANNELS:
        raise ValueError(
            "channels must be %s, got %d" % (" or ".join(str(v) for v in CHANNELS),
                                             channels))
    return channels


def _spatial_shape(shape:Tuple[int, ...], channels:Optional[int], what:str)->Tuple[int, ...]:
    """Strip the trailing channel axis when channels was given, and validate the rank.

    The channel count is never inferred: a trailing axis is a spatial extent unless the
    caller says the array carries channel data. That is what keeps (4, 4) a 4x4 grey
    image instead of a 4-pixel RGBA row, and (8, 16, 3) an RGB image instead of a volume.

    ``channels=None`` means "not given": every axis is spatial. Passing any explicit value
    -- including 1 -- makes the trailing axis the channel axis, so the two forms never
    disagree about what a shape means.
    """
    if channels is not None:
        if not shape or shape[-1] != channels:
            raise ValueError(
                "Image %s %s does not end with the %d channel(s) it was given; pass "
                "channels=%d only for data whose last axis is the channel axis"
                % (what, shape, channels, channels))
        shape = shape[:-1]

    if len(shape) not in (1, 2, 3):
        raise ValueError(
            "Image %s must have 1, 2 or 3 spatial dimensions%s, got %s"
            % (what, " plus an optional channel axis" if channels is not None else "",
               tuple(shape)))
    if any(int(v) <= 0 for v in shape):
        raise ValueError("Image %s must have positive extents, got %s"
                         % (what, tuple(shape)))
    return tuple(int(v) for v in shape)


def _extents(spatial:Tuple[int, ...])->Tuple[int, int, int]:
    """Host spatial shape -> (width, height, depth); missing axes are 1."""
    if len(spatial) == 1:
        return int(spatial[0]), 1, 1
    if len(spatial) == 2:
        return int(spatial[1]), int(spatial[0]), 1
    return int(spatial[2]), int(spatial[1]), int(spatial[0])
