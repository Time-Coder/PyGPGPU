"""Bug: `z += c` fails to compile for a complex value (the plain spelling works)."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel(float=np.float64)
def complex_inplace_add(cre: np.float64, cim: np.float64, out: np.ndarray):
    """`z += c` over four iterations; the `z = z + c` spelling is the reference."""
    for i in range(out.shape[0]):
        c = complex(cre, cim)
        z = complex(0.0, 0.0)
        plain = complex(0.0, 0.0)
        for _ in range(4):
            z += c
            plain = plain + c
        out[i] = (z.real + plain.real) + (z.imag + plain.imag)
