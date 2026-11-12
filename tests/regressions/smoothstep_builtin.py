"""Bug: ``smoothstep`` cannot be used on the device at all.

``smoothstep`` is not exported from the ``pygpgpu`` top level, and importing pygf's own fails in
translation -- the walk of its source hits a private helper the translator cannot resolve:

    SyntaxError: unknown name '_smoothstep' (only parameters, local variables and
    int/float/bool global constants are supported)

pygf is the GLSL-shaped library and smoothstep is a GLSL name, so its object is the one this
package should emit for; the fix is either a device emission in the shared tables (it is a
three-argument clamp-and-fade, ``t = clamp((x - e0) / (e1 - e0), 0, 1); t * t * (3 - 2 * t)``)
or a walk that resolves the private helper. The scalar case is the minimal repro; the vector
case is the same name once the vector builtins work (see `vector_minmax_abs.py`).

Found by the Light Cycle port, whose light-bar coverage is one smoothstep away from the GLSL.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygf import smoothstep


@kernel
def fades_one_edge(x: np.float64, out: np.ndarray):
    out[0] = smoothstep(0.2, 0.6, x)
