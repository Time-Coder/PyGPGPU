"""Bug: complex(re, im) truncates double parts to the mapped width.

Under `float=np.float64` the parts of this call are doubles (a python float argument
travels as the mapping's double), and numpy's own `complex(1.0000001234567890,
-0.12345678901234567)` is a complex128 -- but the constructor built the value at the
mapped *complex* width for every call, so both parts were cut to float32. A real part
that is a double must make the result a c128, the same widening rule every mixed
operation in the package follows: nothing is rounded down.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel(float=np.float64)
def complex_constructor_double_parts(re: float, im: float, out: np.ndarray):
    """`complex(re, im)` with double-typed parts under the double mapping."""
    for i in range(out.shape[0]):
        z = complex(re, im)
        out[i, 0] = z.real
        out[i, 1] = z.imag
