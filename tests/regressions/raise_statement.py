"""Bug: a `raise` statement inside a device method is refused.

`RaisingHelper.augmented` guards its arithmetic: on impossible input it raises, the way any
python helper would. The translator used to refuse the statement outright:
`SyntaxError: unsupported statement: Raise`. The right emission is a device-side abort of the
whole launch -- CUDA's NVRTC takes the PTX trap (`asm("trap;")`), HIP's clang takes
`__builtin_trap()` (ROCm's HIP headers declare no trap function, and inline PTX would be
rejected on AMD), the interpreter raises the python exception -- and OpenCL C and MSL, which
have no portable abort, take a comment naming the limitation so the source still compiles.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


class RaisingHelper:
    """The kernel owner: `draw` calls `augmented`, which raises on a negative count."""

    def __init__(self, count: int) -> None:
        self.count = np.int32(count)
        self.image = np.zeros((1, 2), dtype=np.float32)

    def augmented(self, x: np.float32) -> np.float32:
        if self.count < 0:
            raise ValueError("augmented() needs a non-negative count")
        return np.float32(2 * x)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.augmented(np.float32(i + j))
