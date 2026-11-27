"""21_random_numbers.py -- random numbers inside a kernel body.

A kernel body draws the way it would on the host: `random.random()`,
`random.uniform(0.0, 1.0)`, `np.random.normal(0.0, 1.0)`. The launch owns the
seed and the work item, so the body never passes either -- which is what lets a
helper draw with no `tid` and no `seed`, the same contract `taichi.random`
offers.

Each work item gets its own stream, so two work items never share a sequence,
and pinning the seed replays the whole sequence.

Run:
    python examples/21_random_numbers.py
"""
import random

import numpy as np
import pygpgpu
from pygpgpu import kernel


def jitter() -> float:
    """A helper that draws: no seed, no work item -- the caller's stream is used."""
    return random.uniform(-0.1, 0.1)


@kernel
def draw_uniform(out: np.ndarray):
    """One uniform draw per element."""
    for i in range(out.shape[0]):
        out[i] = random.random()


@kernel
def draw_mixed(out: np.ndarray):
    """Several distributions, and a helper that draws, in one body."""
    for i in range(out.shape[0]):
        out[i] = (random.random()
                  + random.uniform(0.0, 1.0)
                  + np.random.normal(0.0, 1.0)
                  + jitter())


@kernel
def draw_integer(out: np.ndarray):
    """An integer draw: `random.randint` includes its high end, numpy's does not."""
    for i in range(out.shape[0]):
        out[i] = float(random.randint(1, 6))


def main():
    n = 100_000

    # A pinned seed makes the whole sequence reproducible. Left alone, every
    # process draws its own seed and consecutive runs differ.
    pygpgpu.config.random.seed = 20240101

    out = np.zeros(n, dtype=np.float32)
    draw_uniform(out)
    assert np.all(out >= 0.0) and np.all(out < 1.0)
    print(f"uniform in [0, 1)   - mean {out.mean():.4f} (expected 0.5)")

    # Every work item draws its own stream: the values are all different. The draw
    # carries 24 bits (a whole float mantissa), so a hundred thousand draws collide
    # a few hundred times -- the check is that the values are overwhelmingly
    # distinct, not that they are all different.
    unique = len(np.unique(out))
    assert unique > 0.99 * n, f"only {unique} distinct values out of {n}"
    print(f"one stream per work item - {unique} distinct values out of {n}")

    # The same seed replays the same numbers.
    again = np.zeros(n, dtype=np.float32)
    pygpgpu.config.random.seed = 20240101
    draw_uniform(again)
    assert np.array_equal(out, again)
    print("a pinned seed replays the sequence - OK")

    mixed = np.zeros(n, dtype=np.float32)
    draw_mixed(mixed)
    print(f"mixed distributions   - mean {mixed.mean():.4f}, std {mixed.std():.4f}")

    dice = np.zeros(n, dtype=np.float32)
    draw_integer(dice)
    assert np.all(dice >= 1.0) and np.all(dice <= 6.0)
    print(f"integer draw in [1, 6] - mean {dice.mean():.4f} (expected 3.5)")


if __name__ == "__main__":
    main()
