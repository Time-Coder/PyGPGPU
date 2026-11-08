"""The fft tests, split out of `test_cuda_libraries.py`.

The subject is fft.
"""
from __future__ import annotations

from _helpers.test_cuda_libraries_helpers import (
    _require_library,
)

import ctypes
import numpy as np


def test_cufft_rfft_matches_numpy(device):
    """A real-to-complex transform agrees with numpy's."""
    _require_library("cufft")
    from pygpgpu.cuda.libraries.wrappers import rfft

    rng = np.random.default_rng(23)
    signal = rng.standard_normal(16).astype(np.float32)

    spectrum = rfft(signal)
    assert spectrum.dtype == np.complex64
    assert spectrum.shape == (9,)
    np.testing.assert_allclose(spectrum, np.fft.rfft(signal), rtol=1e-4, atol=1e-5)

def test_cufft_round_trip_returns_the_signal(device):
    """The complex-to-real transform undoes the real-to-complex one."""
    _require_library("cufft")
    from pygpgpu.cuda.libraries.wrappers import irfft, rfft

    rng = np.random.default_rng(29)
    signal = rng.standard_normal(32).astype(np.float32)

    spectrum = rfft(signal)
    recovered = irfft(spectrum, 32)
    np.testing.assert_allclose(recovered, signal, rtol=1e-4, atol=1e-5)

def test_cufft_plan_and_destroy(device):
    """A plan is created and destroyed through the bound entry points."""
    _require_library("cufft")
    from pygpgpu.cuda.libraries.cufft import CUFFT, cufftHandle

    plan = cufftHandle()
    CUFFT.cufftPlan1d(ctypes.byref(plan), 16, 0x2A, 1)
    assert plan.value
    CUFFT.cufftDestroy(plan)
