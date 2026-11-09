"""The tasks every benchmark script runs, and the reference answer they are checked against.

This module is imported by each `bench_*.py`, so it holds nothing but parameters and the
checksum: importing a framework here would put its import cost inside another script's
timing, which is the one thing the comparison must not do.

Two tasks, because they fail differently. `mandelbrot` is compute-bound -- per pixel it runs
a data-dependent number of iterations of a complex recurrence -- so it measures how well a
framework compiles a real loop. `laplacian` is memory-bound -- five reads and one write per
element, no branching -- so it measures how well a framework moves bytes. A framework can be
near the top of one and far down the other.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np


# --- mandelbrot --------------------------------------------------------------

MANDEL_WIDTH: int = 1024
MANDEL_HEIGHT: int = 512
MANDEL_MAX_ITER: int = 200
# The plane is centred here and this wide, and the imaginary axis is scaled by the aspect
# ratio so the set is not stretched. Kept as two floats rather than one `complex`, because a
# kernel reads a python complex through a synthesized record while a float is a parameter.
MANDEL_CENTER_REAL: float = -0.75
MANDEL_CENTER_IMAG: float = 0.0
MANDEL_SCALE: float = 3.0


def mandelbrot_c(row: int, column: int, width: int, height: int) -> complex:
    """The point a pixel stands for: the real axis left to right, the imaginary top down."""
    real = ((column + 0.5) / width - 0.5) * MANDEL_SCALE + MANDEL_CENTER_REAL
    imag = ((row + 0.5) / height - 0.5) * MANDEL_SCALE * (height / width) + MANDEL_CENTER_IMAG
    return complex(real, imag)


def mandelbrot_reference() -> np.ndarray:
    """The escape counts, computed in numpy on the host, as float32.

    The host answer is the reference for every framework, so a mismatch means a framework
    compiled the loop differently, not that this module is a second implementation of the
    kernel under test.
    """
    height, width = MANDEL_HEIGHT, MANDEL_WIDTH
    rows = np.arange(height, dtype=np.float64)
    columns = np.arange(width, dtype=np.float64)
    real = ((columns + 0.5) / width - 0.5) * MANDEL_SCALE + MANDEL_CENTER_REAL
    imag = ((rows + 0.5) / height - 0.5) * MANDEL_SCALE * (height / width) + MANDEL_CENTER_IMAG
    plane = (real[None, :] + 1j * imag[:, None])
    z = np.zeros_like(plane)
    counts = np.full((height, width), float(MANDEL_MAX_ITER), dtype=np.float64)
    alive = np.ones((height, width), dtype=bool)
    for step in range(MANDEL_MAX_ITER):
        z[alive] = z[alive] * z[alive] + plane[alive]
        escaped = alive & (np.abs(z) > 2.0)
        counts[escaped] = float(step + 1)
        alive &= ~escaped
    return counts.astype(np.float32)


# --- laplacian ---------------------------------------------------------------

LAPLACIAN_WIDTH: int = 2048
LAPLACIAN_HEIGHT: int = 2048


def laplacian_input() -> np.ndarray:
    """A smooth, deterministic field, so every framework starts from the same bytes."""
    height, width = LAPLACIAN_HEIGHT, LAPLACIAN_WIDTH
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    return (np.sin(rows * 0.01) * np.cos(columns * 0.013)
            + 0.25 * np.sin(rows * 0.07 + columns * 0.03)).astype(np.float32)


def laplacian_reference() -> np.ndarray:
    """The 5-point stencil in numpy, with an edge that copies rather than wraps.

    The border is left at zero: a work item on the edge has no neighbour outside the array,
    and every framework is written to skip those rather than invent a rule, so the results
    stay comparable element for element.
    """
    source = laplacian_input()
    out = np.zeros_like(source)
    out[1:-1, 1:-1] = (source[:-2, 1:-1] + source[2:, 1:-1]
                       + source[1:-1, :-2] + source[1:-1, 2:]
                       - 4.0 * source[1:-1, 1:-1])
    return out


# --- shared reporting --------------------------------------------------------

def checksum(array: np.ndarray) -> float:
    """One number that catches a wrong answer without printing a whole array.

    A plain sum hides a transposed or shifted result, so the checksum also weights each
    element by its own position: two different fields collide only by construction.
    """
    flat = np.asarray(array, dtype=np.float64).ravel()
    weights = np.arange(1, flat.size + 1, dtype=np.float64)
    return float(flat.sum() + (flat * weights).sum() * 1e-9)


def tasks() -> Tuple[str, ...]:
    return ("mandelbrot", "laplacian")
