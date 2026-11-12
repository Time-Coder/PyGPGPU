"""The base classes whose subclass-override shapes already work.

The `@kernel` lives here and the subclasses live in the test module, on purpose: the interesting
case is a subclass in *another* module, which is the shape `examples/fractal/` uses. Two shapes
compile today and bound the gap that
`tests/test_kernel_regressions.py::test_a_subclass_override_can_call_an_inherited_helper`
describes -- an override calling a helper of its own class, and a base's own default calling a
base helper plus a subclass hook. A fix for the failing ones must leave both alone.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


class BaseWithHook:
    """`draw` calls `sample`; the base contributes nothing else."""

    def __init__(self, height: int = 2, width: int = 3) -> None:
        self.image = np.zeros((height, width), dtype=np.float32)

    def sample(self, i: int, j: int) -> np.float32:
        """The hook a subclass overrides."""
        return np.float32(0)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)


class BaseWithDefault:
    """The base's own `sample` calls a base helper and a hook the subclass writes."""

    def __init__(self, height: int = 2, width: int = 3) -> None:
        self.image = np.zeros((height, width), dtype=np.float32)

    @staticmethod
    def scale(x: np.float32) -> np.float32:
        """Owned by the base, called from the base's own default below."""
        return np.float32(2 * x)

    def count_of(self, i: int, j: int) -> np.float32:
        """The hook a subclass overrides."""
        return np.float32(0)

    def sample(self, i: int, j: int) -> np.float32:
        return self.scale(self.count_of(i, j))

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)
