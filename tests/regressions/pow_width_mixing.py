"""Bug: ``math.pow`` with a pygf-vector component and a python float is ambiguous on the device.

``math.pow(v.x, 2.2)`` -- a ``float3`` component (float32) against a python exponent (double) --
is emitted as a bare ``pow(float, double)`` call, and no C++ overload matches exactly:

    error: more than one instance of overloaded function "pow" matches the argument list

python's own ``math.pow`` is double throughout: the host promotes the float32 value and answers
a double. The device has to do the same -- widen the float32 argument to double and emit
``pow(double, double)`` -- instead of leaving two half-width candidates tied. Two doubles compile
already (``pow_double`` in the same test file), so this is purely the width-mixing case.

Found by the Light Cycle port: the gamma and pulse terms pow a ``float3`` component against a
python literal, the single most common spelling in shader code.
"""
from __future__ import annotations

import math

import numpy as np

from pygpgpu import kernel
from pygf import float3


class MixedWidthPow:
    """The gamma spelling: pow of a colour component against a python literal."""

    def __init__(self) -> None:
        self.image = np.zeros((1, 2, 3), dtype=np.uint8)

    @kernel
    def draw(self):
        v = float3(0.5, 0.5, 0.5)
        self.image[0, 0] = 255 * math.pow(v.x, 2.2)
        self.image[0, 1] = 255 * math.pow(0.5, 2.2)
