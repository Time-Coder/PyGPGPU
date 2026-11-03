"""The device-side library function registry.

A registered function names a device symbol that already exists -- a compiler intrinsic,
or a `__device__` function in a header -- and lets a kernel body call it under a name of
the user's own. What is pinned here:

* the registry recognizes a function by the *object* the callee resolves to, so a user's
  own function that happens to share a name is not mistaken for the registered one;
* the generated source carries the header the registered function asked for, so the
  symbol is actually declared;
* a backend with no implementation for the registered function is refused *by name*.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import pygpgpu
from pygpgpu import device_function
from pygpgpu.kernel.device_library import library_function_name, library_functions


def test_the_registry_is_reachable_from_the_top_level():
    assert pygpgpu.device_function is device_function
    assert "device_function" in pygpgpu.__all__


def test_a_registered_function_is_found_by_its_object():
    from device_library_kernels import test_device_rcp

    assert library_function_name(test_device_rcp) == "test_device_rcp"
    assert "test_device_rcp" in library_functions()


def test_a_function_that_is_not_registered_is_not_found():
    assert library_function_name(lambda x: x) is None
    assert library_function_name(pygpgpu.erf) is None


def test_a_registration_without_a_symbol_is_refused():
    with pytest.raises(ValueError) as caught:
        @device_function()
        def nothing(x: float) -> float:
            return x
    assert "cuda" in str(caught.value)


def test_a_user_function_sharing_a_registered_name_stays_a_helper():
    """The registry is keyed by the object, so a name collision changes nothing."""

    @device_function(cuda="test_device_rcp")
    def registered(x: float) -> float:
        return x

    def user_function(x: float) -> float:
        return x * 2.0

    assert library_function_name(registered) == "registered"
    assert library_function_name(user_function) is None


def test_the_registered_function_runs_on_the_device(gpu_backend, runner):
    """The device calls the symbol the registry names, and answers the right number."""
    from device_library_kernels import calls_registered

    values = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(calls_registered, values, out)
    np.testing.assert_allclose(out, np.array([1.0, 0.5, 0.25], dtype=np.float32))


def test_the_header_the_registration_asked_for_reaches_the_source(gpu_backend, runner):
    """On CUDA the symbol lives in a header, so the generated source has to include it."""
    from device_library_kernels import calls_registered

    if gpu_backend != "cuda":
        pytest.skip("only the CUDA registration names a header")
    values = np.array([2.0], dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    runner(calls_registered, values, out)
    np.testing.assert_allclose(out, np.array([0.5], dtype=np.float32))

    # The generated source carries the include, and the header is in the cache beside it.
    import glob
    generated = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "__cucache__", "pykernels", "*.cu"))
    assert generated, "the generated source was not found"
    text = ""
    for path in generated:
        with open(path, "r", encoding="utf-8") as file:
            text += file.read()
    assert "device_library_header.h" in text, "the header was not included"


def test_a_backend_without_the_function_is_refused_by_name(gpu_backend, runner):
    """A registration for CUDA only cannot run on OpenCL, and the refusal says so."""
    if gpu_backend != "opencl":
        pytest.skip("this check is about the OpenCL backend")

    @device_function(cuda="test_device_rcp")
    def cuda_only(x: float) -> float:
        return x

    # The kernel body has to call the registered one, so the probe is built here.
    from pygpgpu import kernel

    @kernel
    def probe(x: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = cuda_only(x[i])

    with pytest.raises(SyntaxError) as caught:
        runner(probe, np.array([1.0], dtype=np.float32), np.zeros(1, dtype=np.float32))
    message = str(caught.value)
    assert "cuda_only" in message
    assert "opencl" in message


def test_the_host_side_runs_the_body():
    """The registered function's body is the host's, so the interpreter backend runs it."""
    from device_library_kernels import test_device_rcp

    assert test_device_rcp(4.0) == 0.25
