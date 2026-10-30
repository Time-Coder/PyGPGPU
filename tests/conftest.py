"""Shared fixtures for the @kernel feature tests.

The device backends are optional: every test that needs one is parameterized over the
backends this machine actually has, and the `python` backend always takes part so the
suite still exercises the launch semantics on a machine without a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest


DEVICE_BACKENDS: "tuple[str, ...]" = ("cuda", "hip", "opencl")


def _probe() -> "list[str]":
    """Backends with at least one device, in the documented probe order."""
    from pygpgpu.kernel.decorator import _probe_backend

    return [backend for backend in DEVICE_BACKENDS if _probe_backend(backend)]


AVAILABLE_BACKENDS = _probe()


@pytest.fixture(params=["python"] + AVAILABLE_BACKENDS)
def backend(request):
    """A backend the suite can run on: always `python`, plus the real devices present."""
    return request.param


@pytest.fixture
def gpu_backend(backend):
    """A device backend (skips when the machine has none)."""
    if backend == "python":
        pytest.skip("needs a device backend")
    return backend


@pytest.fixture
def runner(backend):
    """`runner(kernel, *args)` launches on the current backend, bracket included."""

    def launch(kernel_object, *args):
        if backend == "python":
            return kernel_object["python"](*args)
        return kernel_object[backend](*args)

    return launch


def assert_close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
