"""Bug: ``%`` on floats uses C's fmod semantics instead of python's floored remainder.

``x % 4.0`` with ``x = -1.0`` answers python's ``3.0`` on the host and ``-1.0`` on the device:
the emission is C's ``fmod`` (the result takes the dividend's sign) where python defines the
remainder by the divisor's sign. The same source therefore disagrees between the interpreter
backend and every device backend -- and GLSL's ``mod`` is floored too, so a shader ported from
Shadertoy reads the wrong cells wherever the operand can go negative.

Measured: ``-1.0 % 4.0`` -> python 3.0, device -1.0; ``-5.5 % 4.0`` -> python 2.5, device -1.5;
positive operands agree (``9.75 % 4.0`` -> 1.75 both).

The fix is the floored-mod helper the translator already owes for python semantics
(``x - y * floor(x / y)``), in the width of the operands.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def mod_four(x: np.float64, out: np.ndarray):
    out[0] = x % 4.0
