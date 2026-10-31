"""Same-name structures resolve as a numbered family, not as a hash.

Two modules can both define `Point`; once imported they carry one python name with
different members. Every layout still needs a C name of its own -- the name is the key
member lookups resolve through -- and the numbered spelling keeps the family readable:
`Point`, `Point_alter1`, `Point_alter2`, in the order the classes were seen.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygpgpu.kernel import structs


class PointXY:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class PointXYZ:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class PointXW:
    def __init__(self, x, w):
        self.x = x
        self.w = w


PointXY.__name__ = PointXYZ.__name__ = PointXW.__name__ = "Point"


@kernel
def use_point(p: PointXY, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p.x + p.y


@kernel
def use_other(p: PointXYZ, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p.x + p.y + p.z


@kernel
def use_w(p: PointXW, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p.x + p.w


def test_same_name_layouts_form_a_numbered_family():
    """Point, Point_alter1, Point_alter2 -- one name per layout, all distinct."""
    values = np.zeros(4, dtype=np.float32)
    use_point["cuda"](PointXY(1.0, 2.0), values)
    assert values.tolist() == [3.0, 3.0, 3.0, 3.0]
    use_other["cuda"](PointXYZ(1.0, 2.0, 3.0), values)
    assert values.tolist() == [6.0, 6.0, 6.0, 6.0]
    use_w["cuda"](PointXW(5.0, 1.0), values)
    assert values.tolist() == [6.0, 6.0, 6.0, 6.0]
    # And the first two still read their own layouts after the third compiled.
    use_point["cuda"](PointXY(1.0, 2.0), values)
    assert values.tolist() == [3.0, 3.0, 3.0, 3.0]
    use_other["cuda"](PointXYZ(1.0, 2.0, 3.0), values)
    assert values.tolist() == [6.0, 6.0, 6.0, 6.0]


def test_a_numbered_family_after_launch():
    """The hint table spells the base name first, then `_alterN` in seen order.

    The keys are the canonical dtypes (fields sorted by name, which is what
    `np.dtype(ctypes_class)` yields), so the lookup key here is built the same way. The
    table is process-global and other tests register their own structures, so this checks
    the family this module's three layouts form rather than the table's whole content.
    """
    values = np.zeros(4, dtype=np.float32)
    use_point["cuda"](PointXY(1.0, 2.0), values)
    use_other["cuda"](PointXYZ(1.0, 2.0, 3.0), values)
    use_w["cuda"](PointXW(5.0, 1.0), values)
    family:{str} = set(structs._NAME_HINTS.values())
    assert {"Point", "Point_alter1", "Point_alter2"} <= family
    assert "Point_alter3" not in family
