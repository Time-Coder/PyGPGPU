"""Same-name classes still get one C name each: `Point_<hash>`.

Two modules can both define `Point`; once imported they carry one python name with
different members. Every layout still needs a C name of its own -- the name is the key
member lookups resolve through -- so the readable half is the class name and the hash keeps
two classes of that name apart.
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


def test_same_name_layouts_stay_distinct():
    """Three layouts named `Point` keep three distinct C names, and all of them work."""
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


def test_a_structure_is_named_after_its_class_plus_a_hash():
    """`Point_<hash>`: readable, and still one name per layout.

    The keys are the canonical dtypes (fields sorted by name, which is what
    `np.dtype(ctypes_class)` yields). The table is process-global and other tests register
    their own structures, so this checks the three layouts of this module.
    """
    values = np.zeros(4, dtype=np.float32)
    use_point["cuda"](PointXY(1.0, 2.0), values)
    use_other["cuda"](PointXYZ(1.0, 2.0, 3.0), values)
    use_w["cuda"](PointXW(5.0, 1.0), values)
    families:set = {structs.struct_name(dtype)
                    for dtype in structs._NAME_BASES if structs._NAME_BASES[dtype] == "Point"}
    assert families, "the three layouts must be registered"
    # Every one of them is spelled `Point_<hash>`: the readable half is the class, and the
    # hash keeps one `Point` class from answering for another.
    assert all(name.startswith("Point_") for name in families), families
