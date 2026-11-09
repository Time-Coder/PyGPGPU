"""The two tasks written for Taichi.

Run standalone: `python bench_taichi.py --task both --arch cuda`. Taichi defaults to the CPU
backend, so `--arch cuda` (or `vulkan`) picks the device; the arch actually used is printed
as `ARCH`.

The data lives in `ti.field`s, which is Taichi's own device-resident storage, and the timed
call ends with `ti.sync()`: a GPU kernel returns as soon as it is queued, so a number that
did not wait for the device would measure the queue rather than the work. Every framework
here is measured the same way.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import harness  # noqa: E402

from common import (MANDEL_CENTER_IMAG, MANDEL_CENTER_REAL, MANDEL_MAX_ITER,  # noqa: E402
                    MANDEL_SCALE)

ARCHES = {"cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu, "gpu": ti.gpu}

_MANDELBROT = None
_LAPLACIAN = None


def build(arch: str, task: str) -> None:
    """Start Taichi and declare the kernels for the tasks this run will execute."""
    global _MANDELBROT, _LAPLACIAN

    ti.init(arch=ARCHES[arch], offline_cache=True)
    print(f"ARCH {arch}")

    if task in ("mandelbrot", "both"):
        out = ti.field(dtype=ti.f32, shape=(common.MANDEL_HEIGHT, common.MANDEL_WIDTH))

        @ti.kernel
        def mandelbrot():
            for row, column in out:
                width: ti.f32 = ti.cast(common.MANDEL_WIDTH, ti.f32)
                height: ti.f32 = ti.cast(common.MANDEL_HEIGHT, ti.f32)
                cr: ti.f32 = ((ti.cast(column, ti.f32) + 0.5) / width - 0.5) * MANDEL_SCALE \
                    + MANDEL_CENTER_REAL
                ci: ti.f32 = ((ti.cast(row, ti.f32) + 0.5) / height - 0.5) * MANDEL_SCALE \
                    * (height / width) + MANDEL_CENTER_IMAG
                zr: ti.f32 = 0.0
                zi: ti.f32 = 0.0
                step: ti.i32 = 0
                escaped: ti.i32 = 0
                while step < MANDEL_MAX_ITER and escaped == 0:
                    magnitude: ti.f32 = zr * zr - zi * zi + cr
                    zi = 2.0 * zr * zi + ci
                    zr = magnitude
                    step += 1
                    if zr * zr + zi * zi > 4.0:
                        escaped = 1
                out[row, column] = ti.cast(step, ti.f32)

        _MANDELBROT = (mandelbrot, out)

    if task in ("laplacian", "both"):
        source = ti.field(dtype=ti.f32, shape=(common.LAPLACIAN_HEIGHT, common.LAPLACIAN_WIDTH))
        sink = ti.field(dtype=ti.f32, shape=(common.LAPLACIAN_HEIGHT, common.LAPLACIAN_WIDTH))
        source.from_numpy(common.laplacian_input())     # uploaded once, before any timing

        @ti.kernel
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
            ti.sync()               # the launch returns queued; the number must wait
            return out

        def fetch() -> np.ndarray:
            return out.to_numpy()

        return run, fetch

    kernel, sink = _LAPLACIAN

    def run() -> object:
        kernel()
        ti.sync()
        return sink

    def fetch() -> np.ndarray:
        return sink.to_numpy()

    return run, fetch


if __name__ == "__main__":
    arguments = harness.parse_args(__doc__ or "")
    build(arguments.arch or "cuda", arguments.task)
    harness.run_tasks(arguments, make)
