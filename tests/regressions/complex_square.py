"""Bug: `z ** 2` on a complex value goes through the general exp-log power, which
loses catastrophically where numpy's own square does not (c64, |z| = 4 - 4j: numpy
answers -32 + 0j, the exp-log round trip answers -1.4e-06 - 32j)."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def complex_square_c64(a: np.ndarray, out: np.ndarray):
    """`z ** 2` at the 32-bit default width -- numpy's own answer is exact here."""
    for i in range(out.shape[0]):
        out[i] = a[i] ** 2
