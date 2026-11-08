"""The fixtures the HIP library tests share.

A pytest fixture has to live in a conftest.py to be found by the tests that use
it, so this is where they live.
"""
from __future__ import annotations

from _helpers.test_hip_libraries_helpers import _require_library

import pytest


@pytest.fixture(scope="module")
def hip_device():
    """The first HIP device, with its context current."""
    import pygpgpu.hip  # noqa: F401  (starts the driver's background initialization)
    from pygpgpu.hip.oop.devices import Devices

    try:
        devices = list(Devices)
    except Exception as error:
        pytest.skip(f"no HIP device is available: {error}")
    if not devices:
        pytest.skip("no HIP device is available")

    device = devices[0]
    device.default_context.make_current()
    return device


@pytest.fixture(scope="module")
def hipblas(hip_device):
    _require_library("hipblas")
    from pygpgpu.hip import libraries

    return libraries


@pytest.fixture(scope="module")
def hipfft(hip_device):
    _require_library("hipfft")
    from pygpgpu.hip import libraries

    return libraries
