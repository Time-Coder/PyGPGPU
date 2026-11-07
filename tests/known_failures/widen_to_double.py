"""Bug: a variable is refused when it widens float -> double."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def variable_widens_from_float_to_double(x: np.float64, out: np.ndarray):
    """`scale = 1.0` above the loop, a double-typed expression inside it.

    The right-hand side reads a np.float64 scalar parameter and multiplies it by a
    np.float64, so the product is a double -- the type the variable must widen to.
    (With plain `x * 2.0` the parameter narrows to float and the product stays float,
    which is why the plain spelling does not trip the refusal.)
    """
    scale = 1.0
    for i in range(out.shape[0]):
        scale = x * np.float64(2.0)
        out[i] = scale
