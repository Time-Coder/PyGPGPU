"""The two tasks written for Numba's CPU target (`numba.njit`).

Run standalone: `python bench_numba_cpu.py --task both`. This is the CPU baseline: it shows
what the same work costs when the compiler has no device to run it on, and it is the number
a reader should compare a GPU result against before believing a speedup.

`parallel=True` spreads the outer axis over Numba's thread pool, so the comparison is against
a multi-threaded CPU build rather than a single core. `fastmath=True` is on: without it Numba
refuses the float32 arithmetic the GPU builds are allowed to contract, and a CPU build that
cannot reorder the sum is not the same program.
"""
from __future__ import annotations

import numpy as np

import common
import harness

from numba import njit, prange, set_num_threads

from common import (MANDEL_CENTER_IMAG,
                    MANDEL_CENTER_REAL, MANDEL_HEIGHT, MANDEL_MAX_ITER, MANDEL_SCALE,
                    MANDEL_WIDTH)


@njit(parallel=True, fastmath=True, cache=True)
def mandelbrot(out):
    for row in prange(out.shape[0]):
        width = float(out.shape[1])
        height = float(out.shape[0])
        for column in range(out.shape[1]):
            cr = ((float(column) + 0.5) / width - 0.5) * MANDEL_SCALE + MANDEL_CENTER_REAL
            ci = ((float(row) + 0.5) / height - 0.5) * MANDEL_SCALE * (height / width) \
                + MANDEL_CENTER_IMAG
            zr = 0.0
            zi = 0.0
            step = 0
            escaped = 0
            while step < MANDEL_MAX_ITER and escaped == 0:
                magnitude = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = magnitude
                step += 1
                if zr * zr + zi * zi > 4.0:
                    escaped = 1
            out[row, column] = float(step)


@njit(parallel=True, fastmath=True, cache=True)
def laplacian(source, out):
    for row in prange(1, out.shape[0] - 1):
        for column in range(1, out.shape[1] - 1):
            out[row, column] = (source[row - 1, column] + source[row + 1, column]
                                + source[row, column - 1] + source[row, column + 1]
                                - 4.0 * source[row, column])


def make(task: str):
    """`njit` is synchronous, so the call itself is the whole measurement."""
    if task == "mandelbrot":
        out = np.zeros((MANDEL_HEIGHT, MANDEL_WIDTH), dtype=np.float32)

        def run() -> object:
            mandelbrot(out)
            return out

        return run, None

    source = common.laplacian_input()
    out = np.zeros_like(source)

    def run() -> object:
        laplacian(source, out)
        return out

    return run, None


if __name__ == "__main__":
    arguments = harness.parse_args(__doc__ or "")
    set_num_threads(16)
    print("ARCH cpu (16 threads)")
    harness.run_tasks(arguments, make)
