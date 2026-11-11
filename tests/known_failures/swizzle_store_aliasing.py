"""Bug: a swizzle store whose value reads the target vector aliases its own writes.

`p.xy = p.yx` is python's swap on a pygf vector, and the device's spelling of it is two
component stores (CUDA C has no swizzle lvalue):

    p.x = make_float2((p.y), (p.x)).x;
    p.y = make_float2((p.y), (p.x)).y;

The first statement overwrites `p.x`, and the second RE-READS the value expression, so it
sees the overwritten component: the swap lands `(p.y, p.y)` instead of `(p.y, p.x)`. Today
`float2(1, 2).xy = .yx` answers (2, 2) on CUDA and (2, 1) on the interpreter -- a silent
wrong answer, the kind that turns a quadrant fold into a smear.

The value has to be evaluated ONCE into a temporary before any component is stored, the
way python evaluates the right-hand side first. Found by the Light Cycle port, whose
`foldSquare` swaps the two folded coordinates exactly this way: with the bug the tunnel's
four corners collapse onto one axis, and the corner light streams flood the walls.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygf import float2


@kernel
def swap_through_swizzle(out: np.ndarray):
    v = float2(1.0, 2.0)
    v.xy = v.yx
    out[0] = v.x
    out[1] = v.y
