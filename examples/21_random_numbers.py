"""21_random_numbers.py -- random numbers inside a kernel body.

A kernel body draws the way it would on the host: ``random.random()``,
``random.expovariate(1.0)``, ``np.random.gamma(2.0)``. The launch owns the seed and
the work item, so the body never passes either -- which is what lets a helper draw
with no ``tid`` and no ``seed``, the same contract ``taichi.random`` offers.

The seed is the host generator's own: the first kernel that draws digests
``random.getstate()`` or ``np.random.get_state()`` once per process. Seeding the
host generator before that first draw replays the device sequence.

Run:
    python examples/21_random_numbers.py
"""
import random

import numpy as np
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
                  + np.random.poisson(4.0)
                  + jitter())


@kernel
def draw_integer(out: np.ndarray):
    """An integer draw: `random.randint` includes its high end, numpy's does not."""
    for i in range(out.shape[0]):
        out[i] = float(random.randint(1, 6))


def main():
    n = 100_000

    # Seed the host generator, not a PyGPGPU setting: the first kernel draw reads it.
    random.seed(20240101)
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

    # The launch's base seed is digested from the host generator on its first use
    # in this process; from then on successive launches walk their own streams.
    print("seed digested from the host generator on first use")

    mixed = np.zeros(n, dtype=np.float32)
    draw_mixed(mixed)
    print(f"mixed distributions   - mean {mixed.mean():.4f}, std {mixed.std():.4f}")

    dice = np.zeros(n, dtype=np.float32)
    draw_integer(dice)
    assert np.all(dice >= 1.0) and np.all(dice <= 6.0)
    print(f"integer draw in [1, 6] - mean {dice.mean():.4f} (expected 3.5)")


if __name__ == "__main__":
    main()
