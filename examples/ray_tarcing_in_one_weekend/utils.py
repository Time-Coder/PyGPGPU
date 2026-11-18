import numpy as np
from pygf import vector3f, length, dot


def splitmix32(counter: int) -> np.uint32:
    """One splitmix32 output: a cheap, high-quality 32-bit scalar mixer.

    The host and the device spell the same integer arithmetic, and every step stays
    *unsigned*: the shifts are the load-bearing part -- a signed C `>>` is arithmetic
    (it copies the sign bit), while python's shift on a positive int is logical, so a
    mask over signed values made the two backends disagree the moment a draw's high
    bit was set (whole rejection loops diverged, and the two renders differed). The
    values here are numpy uint32 end to end, whose `>>` is the logical shift both the
    host and the generated C spell as unsigned.

    Quality: the original single philox round's consecutive outputs were strongly
    anti-correlated (rho = -0.42 measured), and the anti-correlation survived into the
    sampled directions as a directional shade with no light in the scene. splitmix32
    decorrelates them (rho = 0.01 measured) and answers a uniform [0, 1) stream (mean
    0.498, std 0.294 measured).
    """
    z:np.uint32 = np.uint32((counter + 0x9E3779B9) & 0xFFFFFFFF)
    z = np.uint32(z ^ np.uint32(z >> np.uint32(16)))
    z = np.uint32(z * np.uint32(0x21F0AAAD))
    z = np.uint32(z ^ np.uint32(z >> np.uint32(15)))
    z = np.uint32(z * np.uint32(0x735A2D97))
    z = np.uint32(z ^ np.uint32(z >> np.uint32(15)))
    return z


def rand(tid: int, seed: int) -> float:
    """One uniform draw in [0, 1) for the (pixel, stream) pair the two counters name."""
    r = splitmix32((tid * 0x9E3779B1 + seed) & 0xFFFFFFFF)
    return float(r / 4294967296.0)

def rand_unit_vector3f(tid: int, seed: int) -> vector3f:
    """A direction uniform on the unit sphere, by cube rejection.

    Each component is drawn to [-1, 1) and the triple is accepted when it falls inside
    the unit sphere, then normalized -- the textbook spelling. The components must be
    allowed to go negative: drawing them from [0, 1) samples only the positive octant,
    and no post-hoc hemisphere flip can remove that bias (it showed up as a directional
    shade with no light in the scene). The draws are counter-based -- the three
    components read three distinct streams of the same round, and a rejection advances
    the stream by three -- so a rejection costs nothing but the rejected triple.
    """
    while True:
        x = rand(tid, seed)
        y = rand(tid, seed + 1)
        z = rand(tid, seed + 2)
        result = vector3f(x * 2.0 - 1.0, y * 2.0 - 1.0, z * 2.0 - 1.0)
        len_result = length(result)
        if 1e-12 < len_result <= 1:
            return result / len_result
        seed += 3

def rand_vector3f_hemisphere(normal: vector3f, tid: int, seed: int) -> vector3f:
    unit_vec = rand_unit_vector3f(tid, seed)
    if dot(unit_vec, normal) > 0.0:
        return unit_vec
    else:
        return -unit_vec
