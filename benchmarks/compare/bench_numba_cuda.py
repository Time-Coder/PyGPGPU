"""The two tasks written for Numba's CUDA target (`numba.cuda`).

Run standalone: `python bench_numba_cuda.py --task both`. Numba 0.61 carries the CUDA target
in the `numba` package itself; on a newer Numba the same code runs against the separate
`numba-cuda` package, whose `numba.cuda` module replaces this one with the same API.

The arrays are `cuda.to_device` copies made once in `setup`, so the timed call is the launch
and the synchronise that waits for it -- the same position the other frameworks are in.
"""
from __future__ import annotations
import math
import os
from typing import Tuple

import numpy as np

import common
import harness

# The driver on this machine is newer than the ABI Numba's own ctypes bindings assume
# (`cuCtxGetDevice` faults on the first context query), so the official NVIDIA bindings are
# asked for instead. It has to be set before `numba.cuda` is imported.
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

from numba import cuda  # noqa: E402

from common import (LAPLACIAN_HEIGHT, LAPLACIAN_WIDTH, MANDEL_CENTER_IMAG,  # noqa: E402
                    MANDEL_CENTER_REAL, MANDEL_HEIGHT, MANDEL_MAX_ITER, MANDEL_SCALE,
                    MANDEL_WIDTH)

THREADS: Tuple[int, int] = (16, 16)
F32 = np.float32


@cuda.jit
def mandelbrot(out):
    # Every local is float32, because that is what the other three frameworks compute in: a
    # float64 build of this loop is a different program, and on a consumer GPU a much slower
    # one.
    column, row = cuda.grid(2)
    if row < out.shape[0] and column < out.shape[1]:
        width = F32(out.shape[1])
        height = F32(out.shape[0])
        cr = (F32(F32(column) + F32(0.5)) / width - F32(0.5)) * F32(MANDEL_SCALE) \
            + F32(MANDEL_CENTER_REAL)
        ci = (F32(F32(row) + F32(0.5)) / height - F32(0.5)) * F32(MANDEL_SCALE) \
            * (height / width) + F32(MANDEL_CENTER_IMAG)
        zr = F32(0.0)
        zi = F32(0.0)
        step = 0
        escaped = 0
        while step < MANDEL_MAX_ITER and escaped == 0:
            magnitude = zr * zr - zi * zi + cr
            zi = F32(2.0) * zr * zi + ci
            zr = magnitude
            step += 1
            if zr * zr + zi * zi > F32(4.0):
                escaped = 1
        out[row, column] = F32(step)


@cuda.jit
def laplacian(source, out):
    column, row = cuda.grid(2)
    if 0 < row < out.shape[0] - 1 and 0 < column < out.shape[1] - 1:
        out[row, column] = (source[row - 1, column] + source[row + 1, column]
                            + source[row, column - 1] + source[row, column + 1]
                            - 4.0 * source[row, column])


def grid_for(height: int, width: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    blocks = (math.ceil(width / THREADS[0]), math.ceil(height / THREADS[1]))
    return blocks, THREADS


def make(task: str):
    """Copy the data to the device once, then time the launch plus the wait."""
    if task == "mandelbrot":
        host = np.zeros((MANDEL_HEIGHT, MANDEL_WIDTH), dtype=np.float32)
        device = cuda.to_device(host)
        blocks, threads = grid_for(MANDEL_HEIGHT, MANDEL_WIDTH)

        def run() -> object:
            mandelbrot[blocks, threads](device)
            cuda.synchronize()
            return device

        def fetch() -> np.ndarray:
            return device.copy_to_host()

        return run, fetch

    host_source = common.laplacian_input()
    device_source = cuda.to_device(host_source)
    device_out = cuda.to_device(np.zeros_like(host_source))
    blocks, threads = grid_for(LAPLACIAN_HEIGHT, LAPLACIAN_WIDTH)

    def run() -> object:
        laplacian[blocks, threads](device_source, device_out)
        cuda.synchronize()
        return device_out

    def fetch() -> np.ndarray:
        return device_out.copy_to_host()

    return run, fetch


if __name__ == "__main__":
    arguments = harness.parse_args(__doc__ or "")
    print(f"ARCH cuda ({cuda.get_current_device().name})")
    harness.run_tasks(arguments, make)
