"""The two tasks written for PyGPGPU, launched on CUDA.

Run standalone: `python bench_pygpgpu.py --task both`. The script imports PyGPGPU and
nothing else from the comparison, so its import cost is its own.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# Taken before anything is imported, so the timeline can say what the framework's own
# imports cost against the total this process spends before its first kernel exists.
T0 = time.perf_counter()

import numpy as np  # noqa: E402  (after T0, which has to precede every import)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pygpgpu import config, kernel  # noqa: E402

import common  # noqa: E402
import harness  # noqa: E402
from common import (MANDEL_CENTER_IMAG, MANDEL_CENTER_REAL, MANDEL_MAX_ITER,  # noqa: E402
                    MANDEL_SCALE)

harness.set_t0(T0)
harness.mark("pygpgpu imported")
config.print_info = False

# The backend this run launches on; `--arch` picks it (cuda, hip or opencl).
BACKEND = "cuda"


@kernel
def mandelbrot(out: np.ndarray):
    # The extents come from the array's own shape, which is the canonical spelling; a
    # module-level constant works too. The scalars are module-level names imported into this
    # module, which a kernel reads as outer values.
    for row, column in np.ndindex(out.shape[0], out.shape[1]):
        width: float = float(out.shape[1])
        height: float = float(out.shape[0])
        cr: float = ((float(column) + 0.5) / width - 0.5) * MANDEL_SCALE + MANDEL_CENTER_REAL
        ci: float = ((float(row) + 0.5) / height - 0.5) * MANDEL_SCALE * (height / width) \
            + MANDEL_CENTER_IMAG
        zr: float = 0.0
        zi: float = 0.0
        step: int = 0
        escaped: int = 0
        # The update comes first and the test after it, so `step` is the 1-based count of
        # iterations the point survived -- the same number the numpy reference records.
        while step < MANDEL_MAX_ITER and escaped == 0:
            magnitude: float = zr * zr - zi * zi + cr
            zi = 2.0 * zr * zi + ci
            zr = magnitude
            step += 1
            if zr * zr + zi * zi > 4.0:
                escaped = 1
        out[row, column] = float(step)


@kernel
def laplacian(source: np.ndarray, out: np.ndarray):
    # The border has no neighbour outside the array, so it stays at zero.
    for row, column in np.ndindex(source.shape[0], source.shape[1]):
        if 0 < row and row < source.shape[0] - 1 \
                and 0 < column and column < source.shape[1] - 1:
            out[row, column] = (
                source[row - 1, column] + source[row + 1, column]
                + source[row, column - 1] + source[row, column + 1]
                - 4.0 * source[row, column])


def make(task: str):
    """The kernel, over arrays that already live on the device, plus an untimed fetch.

    The buffers are allocated through `pygpgpu.numpy` and filled once, before any timing, so
    the steady-state number is the launch and nothing else. That is the same position the
    other frameworks are put in -- a taichi field and a `numba.cuda` device array are
    both resident by construction, and a framework that uploaded and downloaded its arrays on
    every call would be measuring the PCIe bus rather than its compiler.

    The fetch goes through `to_host`, because `np.asarray` on a `gnp` array hands back the
    host bytes it already had rather than the ones the device just wrote.
    """
    import pygpgpu.numpy as gnp

    if task == "mandelbrot":
        out = gnp.zeros((common.MANDEL_HEIGHT, common.MANDEL_WIDTH), dtype=np.float32)
        harness.mark("mandelbrot: buffers allocated")

        def run() -> object:
            mandelbrot[BACKEND](out)
            return out

        def fetch() -> np.ndarray:
            # `to_host` answers a `Transfer`; waiting on it leaves the bytes in the array's
            # own host memory, which is what the plain view then shows.
            out.to_host(wait=True)
            return np.asarray(out)

        return run, fetch

    source = gnp.asarray(common.laplacian_input())
    out = gnp.zeros(source.shape, dtype=source.dtype)
    harness.mark("laplacian: buffers allocated")

    def run() -> object:
        laplacian[BACKEND](source, out)
        return out

    def fetch() -> np.ndarray:
        out.to_host(wait=True)
        return np.asarray(out)

    return run, fetch


if __name__ == "__main__":
    arguments = harness.parse_args(__doc__ or "")
    BACKEND = arguments.arch or "cuda"
    print(f"ARCH {BACKEND}")
    harness.run_tasks(arguments, make)


