# cuFFT: the real-to-complex transforms.
from __future__ import annotations
from ._wrappers_common import (
    _FLOAT32,
    allocate,
    free,
    synchronize,
    to_device,
    to_host,
)
from ctypes import byref
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from ..cufft import CUFFT, cufftHandle, cufftType



def rfft(x:ArrayLike)->np.ndarray:
    """The 1-D real-to-complex transform of a real array.

    The result is ``n/2 + 1`` complex numbers, in the same layout as
    ``numpy.fft.rfft``.
    """
    signal = np.ascontiguousarray(x)
    if signal.dtype != _FLOAT32:
        raise TypeError(f"rfft supports float32, got {signal.dtype}")
    if signal.ndim != 1:
        raise ValueError(f"rfft expects a 1-D array, got {signal.ndim} dimensions")

    count = signal.size
    spectrum = np.zeros(count // 2 + 1, np.complex64)

    signal_device = allocate(signal.nbytes)
    spectrum_device = allocate(spectrum.nbytes)
    plan = cufftHandle()
    CUFFT.cufftPlan1d(byref(plan), count, int(cufftType.CUFFT_R2C), 1)
    try:
        to_device(signal_device, signal)
        CUFFT.cufftExecR2C(plan, signal_device, spectrum_device)
        synchronize()
        to_host(spectrum_device, spectrum)
    finally:
        CUFFT.cufftDestroy(plan)
        free(signal_device)
        free(spectrum_device)

    return spectrum


def irfft(spectrum:ArrayLike, n:Optional[int]=None)->np.ndarray:
    """The 1-D complex-to-real transform of a spectrum.

    ``n`` is the length of the signal the spectrum came from; it defaults to
    ``2 * (len(spectrum) - 1)``, which is the length numpy's ``rfft`` implies.
    """
    values = np.ascontiguousarray(spectrum)
    if values.dtype != np.complex64:
        raise TypeError(f"irfft supports complex64, got {values.dtype}")
    if values.ndim != 1:
        raise ValueError(
            f"irfft expects a 1-D array, got {values.ndim} dimensions")

    if n is None:
        n = 2 * (values.size - 1)

    signal = np.zeros(n, np.float32)
    spectrum_device = allocate(values.nbytes)
    signal_device = allocate(signal.nbytes)
    plan = cufftHandle()
    CUFFT.cufftPlan1d(byref(plan), n, int(cufftType.CUFFT_C2R), 1)
    try:
        to_device(spectrum_device, values)
        CUFFT.cufftExecC2R(plan, spectrum_device, signal_device)
        synchronize()
        to_host(signal_device, signal)
    finally:
        CUFFT.cufftDestroy(plan)
        free(spectrum_device)
        free(signal_device)

    # cuFFT's complex-to-real transform is unnormalized; numpy's is not either, and
    # the two agree once the scale is put back.
    return signal / n
