"""Bug: a class-qualified call cannot pass the receiver explicitly.

`CallsQualified.sample` reaches the base implementation with the one spelling python allows for
a one-off super-like call, `Base.sample(self, i, j)`. Today the translator refuses the
receiver: `SyntaxError: unknown name 'self' (only parameters, local variables and int/float/bool
global constants are supported)`. The call site knows the class and the instance -- it is the
receiver's own body -- so the emission is the base helper called on `self`, which the base's
own body already produces for `self.sample(i, j)`.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


class Base:
    """The kernel owner: `draw` calls `sample`, and the base answers `2 * (i + j + 1)`."""

    def __init__(self) -> None:
        self.image = np.zeros((2, 3), dtype=np.float32)

    def sample(self, i: int, j: int) -> np.float32:
        return np.float32(2 * (i + j + 1))

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)


class CallsQualified(Base):
    """The override reaches the base answer by naming the class."""

    def sample(self, i: int, j: int) -> np.float32:
        return Base.sample(self, i, j) + np.float32(1)
