"""A pygf vector as a self member: it works, and this file is the regression that pins it.

The object scan accepts a pygf vector (`float2`, `color3f`, ...) as a member of the kernel's
own class -- `self.origin.x` reads on the device and answers python's numbers. It is kept as a
*passing* case (loaded from here into `tests/test_kernel_subclass_override.py`) because the
shadertoy viewer was written under the belief that it could not be one: the viewer's uniforms
are separate float attributes where one `float2` would say the same thing more plainly. A
neighbouring repro, `vector_component_store.py`, is the case that genuinely fails.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygf import float2


class VectorMember:
    """A kernel whose receiver carries a pygf vector -- accepted, and answered correctly."""

    def __init__(self) -> None:
        self.image = np.zeros((2, 3), dtype=np.float32)
        self.origin = float2(0.5, 0.5)

    def sample(self, i: int, j: int) -> np.float32:
        return np.float32(self.origin.x + self.origin.y) * np.float32(i + j + 1)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)
