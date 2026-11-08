"""The fixtures the CUDA library tests share.

A pytest fixture has to live in a conftest.py to be found by the tests that use
it, so this is where the `device` fixture lives.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def device():
    """The first CUDA device, with its context current.

    The libraries allocate their buffers through the CUDA driver, so a context has
    to be current before anything else happens.
    """
    import pygpgpu.cuda  # noqa: F401  (starts the driver's background initialization)
    from pygpgpu.cuda.oop.devices import Devices

    try:
        devices = list(Devices)
    except Exception as error:
        pytest.skip(f"no CUDA device is available: {error}")

    if not devices:
        pytest.skip("no CUDA device is available")

    the_device = devices[0]
    the_device.default_context.make_current()
    return the_device
