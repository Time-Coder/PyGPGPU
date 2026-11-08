"""The helpers tests, split out of `test_kernel_complex.py`.

The subject is helpers.
"""
from __future__ import annotations

from _helpers.test_kernel_complex import (
    calls_double_helper,
    calls_plain_helper,
)

import numpy as np


def test_a_helper_body_follows_the_kernel_it_is_translated_in(gpu_backend):
    """A python complex literal inside a helper takes the caller's width, not the default."""
    a = np.zeros(2, np.complex64)
    out = np.zeros(2, np.complex128)
    # Under `float=np.float64` the helper's literal is a `c128`, so the sum is exact.
    calls_double_helper[gpu_backend](a, out)
    assert out[0] == 0.2j
    # Under the default 32-bit mapping it is a `c64`, so the value is the rounded pair.
    calls_plain_helper[gpu_backend](a, out)
    assert out[0] == complex(np.complex64(0.2j))
    assert out[0] != 0.2j
