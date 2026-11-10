"""Bug: a subclass override cannot call a module-level function either.

The same shape as `subclass_calls_base_helper`, with the callee at module scope instead of on
the base class. A *kernel body* may call such a function freely -- `examples/
digital_image_processing/image_processing.py` does exactly that with its `mean` helper -- but an
override cannot: the call is emitted as `scaled_<hash>(...)` with no definition beside it, and
the backend refuses the module with `identifier "scaled_<hash>" is undefined`.

So the gap is not "the base class is special"; it is that the helpers reachable from an
overriding method are not walked for emission, while the same helpers are when the kernel body
calls them directly.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


def scaled(x: np.float32) -> np.float32:
    """A module-level device helper, called from the subclass's override."""
    return np.float32(3 * x)


class Base:
    """The kernel owner: `draw` calls `sample`, and the subclass writes `sample`."""

    def __init__(self, height: int = 2, width: int = 3) -> None:
        self.image = np.zeros((height, width), dtype=np.float32)

    def sample(self, i: int, j: int) -> np.float32:
        """The hook a subclass overrides."""
        return np.float32(0)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)
