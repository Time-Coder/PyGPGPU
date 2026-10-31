"""Two structures may carry the same python class name; each keeps its own layout.

The C name is the key every member lookup resolves through, so a name hint that two
different layouts both accept would make the second kernel read the first one's members --
silently, with plausible numbers. The registry refuses the second hint and that structure
keeps its hashed name instead.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygpgpu.kernel import structs


class PointXY:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class PointXYZ:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


# Two classes may carry one python name: this is exactly what two modules defining `Point`
# looks like once imported.
PointXY.__name__ = "Point"
PointXYZ.__name__ = "Point"


@kernel
def use_point(p: PointXY, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p.x + p.y


@kernel
def use_other(p: PointXYZ, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p.x + p.y + p.z


def test_same_name_different_layouts_keep_separate_structures():
    """Each kernel reads its own members, whichever class was seen first."""
    values = np.zeros(4, dtype=np.float32)
    use_point["cuda"](PointXY(1.0, 2.0), values)
    assert values.tolist() == [3.0, 3.0, 3.0, 3.0]
    use_other["cuda"](PointXYZ(1.0, 2.0, 3.0), values)
    assert values.tolist() == [6.0, 6.0, 6.0, 6.0]
    # And the first kernel still reads its own layout after the second compiled.
    use_point["cuda"](PointXY(1.0, 2.0), values)
    assert values.tolist() == [3.0, 3.0, 3.0, 3.0]
    # This module's two layouts resolve to different C names: one owns the readable name
    # and the other keeps its hashed one. (The process-wide registry also holds structures
    # from other tests, so only this module's two dtypes are compared.)
    xy_name:str = structs.struct_name(np.dtype(PointXY.__module__ and [
        ("x", "<f4"), ("y", "<f4")])) if False else structs.struct_name(
        np.dtype([("x", "<f4"), ("y", "<f4")]))
    del xy_name
    point_names:{str} = {name for name, dtype in structs._REGISTRY.items()
                         if dtype.names == ("x", "y")}
    other_names:{str} = {name for name, dtype in structs._REGISTRY.items()
                         if dtype.names == ("x", "y", "z")}
    # One readable name and one hashed name may both map at the same layout (a refused
    # hint falls back), but the two layouts may never share a C name: it is the key every
    # member lookup resolves through.
    assert point_names and other_names
    assert not (point_names & other_names)


def test_same_name_same_layout_shares_the_name():
    """One layout under one name in two kernels is not a collision."""
    values = np.zeros(4, dtype=np.float32)
    use_point["cuda"](PointXY(1.0, 2.0), values)
    assert values.tolist() == [3.0, 3.0, 3.0, 3.0]
    use_point["cuda"](PointXY(4.0, 5.0), values)
    assert values.tolist() == [9.0, 9.0, 9.0, 9.0]
