"""The random directions a scattered ray needs -- the built-in's spelling.

The kernel body draws with the standard library's own names (``random.random()``), and
the launch owns the seed and the work item: no ``tid`` and no ``seed`` travel through
these helpers. That is the whole reason the built-in exists -- a scattering model is
written as ordinary Python, and the thread it runs on is the device's business.

The directions are uniform on the unit sphere, by cube rejection: each component is
drawn to [-1, 1) and the triple is accepted when it falls inside the unit sphere, then
normalized. The components must be allowed to go negative -- drawing them from [0, 1)
samples only the positive octant, and no post-hoc hemisphere flip can remove that bias
(it showed up as a directional shade with no light in the scene).
"""
from pygf import vector3f, length, dot
import random


def rand_unit_vector3f() -> vector3f:
    """A direction uniform on the unit sphere, by cube rejection.

    The draws are sequential, so a rejection simply takes the next three -- the
    stream advances and the rejected triple costs nothing but the three draws.
    """
    while True:
        x = random.random() * 2.0 - 1.0
        y = random.random() * 2.0 - 1.0
        z = random.random() * 2.0 - 1.0
        result = vector3f(x, y, z)
        len_result = length(result)
        if 1e-12 < len_result <= 1:
            return result / len_result


def rand_vector3f_hemisphere(normal: vector3f) -> vector3f:
    """A direction uniform on the hemisphere around ``normal``."""
    unit_vec = rand_unit_vector3f()
    if dot(unit_vec, normal) > 0.0:
        return unit_vec
    else:
        return -unit_vec
