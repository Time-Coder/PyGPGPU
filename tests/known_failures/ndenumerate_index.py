"""Bug: an ndenumerate index tuple cannot be subscripted."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def ndenumerate_index_subscript(out: np.ndarray):
    """`for index, value in np.ndenumerate(out)` -- the index is a tuple in python."""
    for index, value in np.ndenumerate(out):
        out[index[0]] = value * 2.0
