"""The fft tests, split out of `test_hip_libraries.py`.

The subject is fft.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_rfft_matches_numpy(hipfft):
    """A real-to-complex transform, in numpy's own layout and scaling."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    result = hipfft.rfft(signal)
    assert result.dtype == np.complex64
    assert result.size == signal.size // 2 + 1
    np.testing.assert_allclose(result, np.fft.rfft(signal), atol=1e-4)


def test_rfft_and_irfft_round_trip(hipfft):
    """The inverse undoes the forward, which is what pins the scaling.

    hipFFT's complex-to-real transform is unnormalized exactly as cuFFT's is, so the
    wrapper divides by the signal length; without that the round trip returns the signal
    multiplied by n.
    """
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    spectrum = hipfft.rfft(signal)
    back = hipfft.irfft(spectrum, signal.size)
    np.testing.assert_allclose(back, signal, atol=1e-4)


def test_rfft_refuses_a_dtype_it_cannot_transform(hipfft):
    with pytest.raises(TypeError) as caught:
        hipfft.rfft(np.zeros(8, dtype=np.int32))
    assert "float32" in str(caught.value)
