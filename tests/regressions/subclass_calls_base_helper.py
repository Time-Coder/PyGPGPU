"""Bug: a subclass override cannot call a helper that the base class defines.

This is the `FractalViewer` shape. The `@kernel` belongs to the base class and its body calls
an overridable hook; a subclass in another module overrides that hook and computes through a
helper it inherits from the base. The generated source used to contain the *call*

    Base_scale_<hash>(...)

and never a definition of it, so the backend refuses the whole module:

    error: identifier "Base_scale_<hash>" is undefined

Two neighbours already work, and they bound the gap: an override that calls a helper defined in
*its own* class compiles, and the base's own default that calls a base helper and a subclass
hook compiles. So what is missing is precisely "a base-class device function reached from an
override", whichever spelling reaches it (`self.scale(x)`, `Base.scale(x)`, a staticmethod or a
plain method).
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


class Base:
    """The kernel owner: `draw` calls `sample`, and the subclass writes `sample`."""

    def __init__(self, height: int = 2, width: int = 3) -> None:
        self.image = np.zeros((height, width), dtype=np.float32)

    @staticmethod
    def scale(x: np.float32) -> np.float32:
        """The inherited helper the subclass's override reaches for."""
        return np.float32(2 * x)

    def sample(self, i: int, j: int) -> np.float32:
        """The hook a subclass overrides, standing in for `color_of_complex`."""
        return np.float32(0)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)
