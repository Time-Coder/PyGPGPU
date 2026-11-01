import numpy as np


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
