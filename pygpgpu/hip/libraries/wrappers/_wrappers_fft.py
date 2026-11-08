# hipFFT: the real-to-complex transforms.
from __future__ import annotations
from ._wrappers_common import (
    _is_single,
    _void,
    ensure_context,
    free,
    synchronize,
    to_device,
    to_host,
)
from ctypes import byref, c_int
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from .._enums import hipfftType
from ..hipfft import HIPFFT, hipfftHandle



def rfft(x:ArrayLike)->np.ndarray:
    """The 1-D real-to-complex transform of a real array.

    The result is ``n/2 + 1`` complex numbers, in the same layout as
    ``numpy.fft.rfft``.
    """
    ensure_context()
    signal = np.ascontiguousarray(x)
    if not _is_single(signal.dtype):
        raise TypeError(f"rfft() takes a float32 array, got {signal.dtype}")
    if signal.ndim != 1:
        raise ValueError(f"rfft() takes a 1-D array, got {signal.ndim} dimensions")

    count = signal.size
    spectrum = np.zeros(count // 2 + 1, np.complex64)

    signal_buffer = to_device(signal)
    spectrum_buffer = to_device(spectrum)

    # The plan handle is a POINTER on this platform, so it is a c_void_p and the plan
    # call takes its address -- `hipfftPlan1d` is one of the two entry points that do.
    plan = hipfftHandle()
    HIPFFT.hipfftPlan1d(byref(plan), c_int(count),
                        c_int(int(hipfftType.HIPFFT_R2C)), c_int(1))
    try:
        HIPFFT.hipfftExecR2C(plan, _void(signal_buffer), _void(spectrum_buffer))
        synchronize()
        to_host(spectrum_buffer, spectrum)
    finally:
        HIPFFT.hipfftDestroy(plan)
        free(signal_buffer)
        free(spectrum_buffer)

    return spectrum


def irfft(spectrum:ArrayLike, n:Optional[int]=None)->np.ndarray:
    """The 1-D complex-to-real transform of a spectrum.

    ``n`` is the length of the signal the spectrum came from; it defaults to
    ``2 * (len(spectrum) - 1)``, which is the length numpy's ``rfft`` implies.
    """
    ensure_context()
    values = np.ascontiguousarray(spectrum)
    if values.dtype != np.complex64:
        raise TypeError(f"irfft() takes a complex64 array, got {values.dtype}")
    if values.ndim != 1:
        raise ValueError(f"irfft() takes a 1-D array, got {values.ndim} dimensions")

    if n is None:
        n = 2 * (values.size - 1)

    # The spectrum is n/2+1 complex numbers but the transform reads a full n/2 of them,
    # so the buffer is the spectrum's own memory and the plan is told the signal length.
    signal = np.zeros(n, np.float32)
    spectrum_buffer = to_device(values)
    signal_buffer = to_device(signal)

    plan = hipfftHandle()
    HIPFFT.hipfftPlan1d(byref(plan), c_int(n), c_int(int(hipfftType.HIPFFT_C2R)), c_int(1))
    try:
        HIPFFT.hipfftExecC2R(plan, _void(spectrum_buffer), _void(signal_buffer))
        synchronize()
        to_host(signal_buffer, signal)
    finally:
        HIPFFT.hipfftDestroy(plan)
        free(spectrum_buffer)
        free(signal_buffer)

    # hipFFT's complex-to-real transform is unnormalized, as cuFFT's is: putting the
    # scale back is what makes it numpy's answer.
    return signal / n
