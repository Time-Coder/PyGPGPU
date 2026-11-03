"""The registered device-side library functions, and the kernels that call them."""
from __future__ import annotations

import os

import numpy as np

from pygpgpu import device_function, kernel

# The header the registered function lives in, beside this module.
HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_library_header.h")


@device_function(
    cuda="test_device_rcp",
    opencl="native_recip",
    headers={"cuda": HEADER},
)
def test_device_rcp(x: float) -> float:
    """The host's implementation; the device gets the symbol the registry names."""
    return 1.0 / x


@kernel
def calls_registered(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = test_device_rcp(x[i])


@kernel
def calls_registered_on_cuda_only(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = test_device_rcp(x[i])
