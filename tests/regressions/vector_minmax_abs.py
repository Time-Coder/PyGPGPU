"""Bug: the vector-wide ``abs``/``min``/``max`` builtins are refused.

GLSL spells component-wise operations on whole vectors -- ``abs(p)`` for an SDF, ``max(q, 0.0)``
for a distance clamp, ``min(v, w)`` to fold two colours. In pygpgpu every one of them is refused
at translation time:

* ``abs(float3(...))`` -> ``SyntaxError: abs() requires a numeric argument``
* ``max(float3(...), 0.0)`` -> ``SyntaxError: vector arithmetic is handled separately``
* ``min(v, float3(...))`` -> the same refusal

The scalar spellings all work (bare ``abs``/``min``/``max`` on floats -- see the min/max arity
spec for the three-argument case), the vector ARITHMETIC they would expand to works (+, *,
scalar), and the host python mutates the same objects freely. What is missing is the vector
case of these three names: expand component-wise and emit, exactly like the ``+``/``*`` paths
already do.

Found by the Light Cycle port, whose SDF helpers are one `abs(vec3)` away from the GLSL.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygf import float2, float3


@kernel
def vector_abs_and_minmax(out: np.ndarray):
    a = abs(float3(-1.0, 2.0, -3.0))
    b = max(float3(1.0, -5.0, 3.0), 0.0)
    c = min(b, float3(2.0, 2.0, 2.0))
    out[0] = a.x + a.y + a.z
    out[1] = c.x + c.y + c.z
    out[2] = abs(min(float2(-2.0, 0.5), float2(1.0, 1.0)).x)
