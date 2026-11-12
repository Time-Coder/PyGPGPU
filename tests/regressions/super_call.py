"""Bug: a device method cannot call its base implementation through ``super()``.

`CallsSuper.sample` overrides the hook and reaches the base's answer through
`super().sample(i, j)` -- the spelling python gives for exactly this. The translator used to
refuses the name: `SyntaxError: unknown function 'super'`. The base implementation is an
ordinary device helper like any other, so the super call has to compile to the same emission
the base's own body produces when it calls `self.sample`.
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


class CallsSuper(Base):
    """The override adds one on top of whatever the base answers."""

    def sample(self, i: int, j: int) -> np.float32:
        return super().sample(i, j) + np.float32(1)
