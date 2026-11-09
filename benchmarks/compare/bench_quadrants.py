"""The two tasks written for Quadrants (a Taichi derivative: `quadrants` imports as `qd`).

Run standalone: `python bench_quadrants.py --task both --arch cuda`. Quadrants keeps
Taichi's API, so this file mirrors `bench_taichi.py` call for call -- that is deliberate, so a
difference in the numbers is a difference in the compiler and runtime rather than in how the
task was written.

The data lives in `qd.field`s, and the timed call ends with `qd.sync()`: a device kernel
returns as soon as it is queued.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

import quadrants as qd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import harness  # noqa: E402

from common import (MANDEL_CENTER_IMAG, MANDEL_CENTER_REAL, MANDEL_MAX_ITER,  # noqa: E402
                    MANDEL_SCALE)

ARCHES = {"cuda": qd.cuda, "vulkan": qd.vulkan, "cpu": qd.cpu, "gpu": qd.gpu}

_MANDELBROT = None
_LAPLACIAN = None


def build(arch: str, task: str) -> None:
    """Start Quadrants and declare the kernels this run will execute."""
    global _MANDELBROT, _LAPLACIAN

    qd.init(arch=ARCHES[arch], offline_cache=True)
    print(f"ARCH {arch}")

    if task in ("mandelbrot", "both"):
        out = qd.field(dtype=qd.f32, shape=(common.MANDEL_HEIGHT, common.MANDEL_WIDTH))

        @qd.kernel
        def mandelbrot():
            for row, column in out:
                width: qd.f32 = qd.cast(common.MANDEL_WIDTH, qd.f32)
                height: qd.f32 = qd.cast(common.MANDEL_HEIGHT, qd.f32)
                cr: qd.f32 = ((qd.cast(column, qd.f32) + 0.5) / width - 0.5) * MANDEL_SCALE \
                    + MANDEL_CENTER_REAL
                ci: qd.f32 = ((qd.cast(row, qd.f32) + 0.5) / height - 0.5) * MANDEL_SCALE \
                    * (height / width) + MANDEL_CENTER_IMAG
                zr: qd.f32 = 0.0
                zi: qd.f32 = 0.0
                step: qd.i32 = 0
                escaped: qd.i32 = 0
                while step < MANDEL_MAX_ITER and escaped == 0:
                    magnitude: qd.f32 = zr * zr - zi * zi + cr
                    zi = 2.0 * zr * zi + ci
                    zr = magnitude
                    step += 1
                    if zr * zr + zi * zi > 4.0:
                        escaped = 1
                out[row, column] = qd.cast(step, qd.f32)

        _MANDELBROT = (mandelbrot, out)

    if task in ("laplacian", "both"):
        source = qd.field(dtype=qd.f32,
                          shape=(common.LAPLACIAN_HEIGHT, common.LAPLACIAN_WIDTH))
        sink = qd.field(dtype=qd.f32, shape=(common.LAPLACIAN_HEIGHT, common.LAPLACIAN_WIDTH))
        source.from_numpy(common.laplacian_input())     # uploaded once, before any timing

        @qd.kernel
        def laplacian():
            for row, column in source:
                if 0 < row and row < common.LAPLACIAN_HEIGHT - 1 \
                        and 0 < column and column < common.LAPLACIAN_WIDTH - 1:
                    sink[row, column] = (source[row - 1, column] + source[row + 1, column]
                                         + source[row, column - 1] + source[row, column + 1]
                                         - 4.0 * source[row, column])

        _LAPLACIAN = (laplacian, sink)


def make(task: str):
    """A field kernel takes no arguments; the arrays were built once in `build`."""
    if task == "mandelbrot":
        kernel, out = _MANDELBROT

        def run() -> object:
            kernel()
            qd.sync()               # the launch returns queued; the number must wait
            return out

        def fetch() -> np.ndarray:
            return out.to_numpy()

        return run, fetch

    kernel, sink = _LAPLACIAN

    def run() -> object:
        kernel()
        qd.sync()
        return sink

    def fetch() -> np.ndarray:
        return sink.to_numpy()

    return run, fetch


if __name__ == "__main__":
    arguments = harness.parse_args(__doc__ or "")
    build(arguments.arch or "cuda", arguments.task)
    harness.run_tasks(arguments, make)
