"""Bug: two subclasses of one `@kernel` share the first one's compiled binary.

`@kernel` caches a compiled launch under `backend | device | argument-signature | type-map`, and
the generated C name carries the class only for the *kernel* it belongs to. The class that
supplied an overridden hook is in neither, so the second subclass to launch is served the first
one's device code and silently renders the first one's picture -- a wrong answer with no error
and no warning.

The in-memory cache is what this module pins, because it is the one that bites inside a single
process: show two viewers of the same base class in one script and the second takes the first's
colouring. A `SIG`-independent repro is simply "compile one subclass, then launch another".

The fix is an identity that includes what the hook resolved to -- the owning class, or a digest
of the device functions the kernel reached.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


class Base:
    """One kernel, two subclasses that answer differently from the same hook."""

    def __init__(self, height: int = 2, width: int = 3) -> None:
        self.image = np.zeros((height, width), dtype=np.float32)

    def sample(self, i: int, j: int) -> np.float32:
        """The hook a subclass overrides."""
        return np.float32(0)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)
