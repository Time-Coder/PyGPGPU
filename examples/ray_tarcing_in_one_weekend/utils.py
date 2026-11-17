import numpy as np
from pygf import vector3f, length, dot


def philox_round(counter: int, key: int) -> np.uint32:
    x = np.uint32(counter)
    y = np.uint32(key)

    x += y
    x ^= (x >> 16)
    x *= 0x9E3779B9

    y ^= (y << 13)
    y ^= (y >> 17)
    y += x

    return y

def rand(tid: int, seed: int) -> float:
    r = philox_round(tid, seed)
    return float(r / 4294967295.0)

def rand_unit_vector3f(tid: int, seed: int)->vector3f:
    while True:
        x = rand(tid, seed)
        seed += 1
        y = rand(tid, seed)
        seed += 1
        z = rand(tid, seed)
        seed += 1
        result = vector3f(x, y, z)
        len_result = length(result)
        if 1e-6 < len_result <= 1:
            return result / len_result

def rand_vector3f_hemisphere(normal: vector3f, tid: int, seed: int) -> vector3f:
    unit_vec = rand_unit_vector3f(tid, seed)
    if dot(unit_vec, normal) > 0.0:
        return unit_vec
    else:
        return -unit_vec
