"""The two tasks written for NVIDIA Warp (`warp-lang`).

Run standalone: `python bench_warp.py --task both --arch cuda`. Warp's device is chosen at
`wp.init()` time by the `device` argument of each launch, so `--arch` here picks what the
arrays live on.

The arrays are `wp.array`s allocated once, and the timed call ends with
`wp.synchronize_device()`: a launch returns queued, and a number that did not wait for the
device would measure the queue. Every framework here is measured the same way.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import harness  # noqa: E402

from common import (MANDEL_CENTER_IMAG, MANDEL_CENTER_REAL, MANDEL_MAX_ITER,  # noqa: E402
                    MANDEL_SCALE)

DEVICE = "cuda"


@wp.kernel
def mandelbrot(out: wp.array2d(dtype=wp.float32)):
    row, column = wp.tid()
    width = wp.float32(out.shape[1])
    height = wp.float32(out.shape[0])
    cr = ((wp.float32(column) + wp.float32(0.5)) / width - wp.float32(0.5)) \
        * wp.float32(MANDEL_SCALE) + wp.float32(MANDEL_CENTER_REAL)
    ci = ((wp.float32(row) + wp.float32(0.5)) / height - wp.float32(0.5)) \
        * wp.float32(MANDEL_SCALE) * (height / width) + wp.float32(MANDEL_CENTER_IMAG)
    zr = wp.float32(0.0)
    zi = wp.float32(0.0)
    step = int(0)
    escaped = int(0)
    while step < MANDEL_MAX_ITER and escaped == 0:
        magnitude = zr * zr - zi * zi + cr
        zi = wp.float32(2.0) * zr * zi + ci
        zr = magnitude
        step += 1
        if zr * zr + zi * zi > wp.float32(4.0):
            escaped = 1
    out[row, column] = wp.float32(step)


@wp.kernel
def laplacian(source: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    row, column = wp.tid()
    if row > 0 and row < source.shape[0] - 1 and column > 0 and column < source.shape[1] - 1:
        out[row, column] = (source[row - 1, column] + source[row + 1, column]
                            + source[row, column - 1] + source[row, column + 1]
                            - wp.float32(4.0) * source[row, column])


def make(task: str):
    """The buffers live on the device from allocation, so a call is the launch and its wait."""
    if task == "mandelbrot":
        out = wp.zeros((common.MANDEL_HEIGHT, common.MANDEL_WIDTH), dtype=wp.float32,
                       device=DEVICE)

        def run() -> object:
            wp.launch(mandelbrot, dim=out.shape, inputs=[out], device=DEVICE)
            wp.synchronize_device(DEVICE)
            return out

        def fetch() -> np.ndarray:
            return out.numpy()

        return run, fetch

    source = wp.array(common.laplacian_input(), dtype=wp.float32, device=DEVICE)
    out = wp.zeros(source.shape, dtype=wp.float32, device=DEVICE)

    def run() -> object:
        wp.launch(laplacian, dim=out.shape, inputs=[source, out], device=DEVICE)
        wp.synchronize_device(DEVICE)
        return out

    def fetch() -> np.ndarray:
        return out.numpy()

    return run, fetch


if __name__ == "__main__":
    arguments = harness.parse_args(__doc__ or "")
    DEVICE = arguments.arch or "cuda"
    wp.config.quiet = True
    wp.init()
    print(f"ARCH {DEVICE} (warp {wp.__version__})")
    harness.run_tasks(arguments, make)
