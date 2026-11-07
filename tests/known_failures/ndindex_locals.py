"""Bug: ndindex extents held in locals are not recognized as the magic loop."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def ndindex_with_local_bounds(out: np.ndarray):
    """The extents held in locals -- the shape the original Mandelbrot example writes."""
    rows: int = out.shape[0]
    columns: int = out.shape[1]
    for i, j in np.ndindex(rows, columns):
        out[i, j] = float(i * columns + j)
