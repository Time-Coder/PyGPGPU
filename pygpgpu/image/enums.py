# The filter and address enumerations, and the annotation types they name.
from __future__ import annotations
from enum import Enum
from typing import Any, Tuple, Union
import numpy as np



# What a subscript into an image may be: one number, or one per axis. The device accepts
# any of these spellings too, so the host must not narrow it further.
ImageIndex = Union[int, float, "np.integer[Any]", "np.floating[Any]", Tuple[object, ...]]

# One sampled value or one value stored: a scalar for a 1-channel image, a vector for
# 2 or 4 channels. numpy supplies both, and the dtype decides which -- so it is a numpy
# scalar or array rather than a fixed python type.
SampleValue = Union["np.generic[Any]", "np.ndarray[Any, Any]"]


class StrEnum(str, Enum):
    """A string-valued enum (the stdlib one is 3.11+, and this package supports 3.7).

    Subclassing str keeps the members comparable with plain strings, so an existing
    ``filter="linear"`` call site keeps working alongside ``filter=ImageFilter.LINEAR``.
    """

    def __str__(self)->str:
        return str(self.value)

    @classmethod
    def parse(cls, value:object, what:str)->"StrEnum":
        """Accept a member, its value, or its name (case-insensitive)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                pass
            try:
                return cls[value.strip().upper()]
            except KeyError:
                pass
        options:str = ", ".join(member.value for member in cls)
        raise ValueError("%s must be one of %s, got %r" % (what, options, value))


# Sampling rules are enums so the class stays backend-neutral while the choices stay
# discoverable. The names are the neutral ones: CUDA spells the filters POINT/LINEAR and
# OpenCL spells them NEAREST/LINEAR, and the address modes differ just as much
# (CLAMP_TO_EDGE vs CLAMP, REPEAT vs WRAP), so neither backend's vocabulary is used here.
class ImageFilter(StrEnum):
    """How a sample is computed from the texels around its coordinate."""

    POINT = "point"      # the one texel the coordinate falls in
    LINEAR = "linear"    # the neighbouring texel centres, blended by distance


class ImageAddress(StrEnum):
    """What a coordinate outside the image reads.

    Only CLAMP is portable today. WRAP and MIRROR are implemented on the python backend
    and have device objects that request them, but the GPU sampling paths read the edge
    texel instead of wrapping: with texel-space coordinates the drivers ignore
    CL_ADDRESS_REPEAT / CL_ADDRESS_MIRRORED_REPEAT and CU_TR_ADDRESS_MODE_WRAP /
    _MIRROR (measured on the NVIDIA and AMD OpenCL devices and the CUDA device available
    while developing this). Launching such an image on a GPU backend raises rather than
    returning different numbers from the same kernel on the python backend.
    """

    CLAMP = "clamp"      # the nearest edge texel
    WRAP = "wrap"        # continue from the opposite edge (python backend only)
    MIRROR = "mirror"    # reflect back inside the image (python backend only)


FILTERS:Tuple[str, ...] = tuple(member.value for member in ImageFilter)
ADDRESS_MODES:Tuple[str, ...] = tuple(member.value for member in ImageAddress)

# Channel counts an image may have. The device formats a sample can carry are 1, 2 or 4
# wide; 3-channel data is rejected explicitly (see _normalise_channels) rather than being
# guessed at, because no hardware format matches it without padding.
CHANNELS:Tuple[int, ...] = (1, 2, 4)
