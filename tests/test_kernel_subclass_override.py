"""The subclass-override shapes that already compile, kept as the boundary of the fixes.

`tests/test_kernel_known_failures.py` describes two shapes that do not compile yet: an override
calling a helper of the base class, and one calling a module-level function. These two tests are
the other side of that line, and they pass today:

* an override calling a helper of *its own* class;
* a base class's own default calling a base helper and a subclass hook.

A fix for the failing shapes must keep both of these working -- the natural one is to extend the
walk that collects device functions, not to change how a call is resolved.
"""
from __future__ import annotations

import numpy as np

from _helpers.test_kernel_subclass_override_helpers import (BaseWithDefault,
                                                            BaseWithHook)


def _grid(factor: float) -> np.ndarray:
    """What `sample(i, j)` answers when it is `factor * (i + j + 1)`."""
    rows = np.arange(2, dtype=np.float32)[:, None]
    columns = np.arange(3, dtype=np.float32)[None, :]
    return factor * (rows + columns + 1.0)


class _OwnHelper(BaseWithHook):
    """The override reaches a staticmethod of its own class."""

    @staticmethod
    def triple(x: np.float32) -> np.float32:
        return np.float32(3 * x)

    def sample(self, i: int, j: int) -> np.float32:
        return self.triple(np.float32(i + j + 1))


class _CountOnly(BaseWithDefault):
    """The subclass supplies the hook; the base's own default does the arithmetic."""

    def count_of(self, i: int, j: int) -> np.float32:
        return np.float32(i + j + 1)


def test_an_override_can_call_a_helper_of_its_own_class(gpu_backend):
    """`self.triple(...)` from an override compiles and answers python's numbers."""
    viewer = _OwnHelper()
    viewer.draw[gpu_backend]()
    np.testing.assert_allclose(viewer.image, _grid(3.0))


def test_a_base_default_may_call_a_base_helper_and_a_subclass_hook(gpu_backend):
    """The base's own `sample` calls `self.scale` (base) and `self.count_of` (subclass)."""
    viewer = _CountOnly()
    viewer.draw[gpu_backend]()
    np.testing.assert_allclose(viewer.image, _grid(2.0))
