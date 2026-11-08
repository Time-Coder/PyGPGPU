"""The surface tests, split out of `test_metal_launch_path.py`.

The subject is surface.
"""
from __future__ import annotations

import pytest


def test_the_device_class_is_reachable_under_the_shared_module_name() -> None:
    """Every backend publishes `oop.device.Device`, so Metal does too.

    OpenCL, CUDA and HIP each define the class in `oop/device.py`. Metal keeps it in
    `oop/devices.py` beside the `Devices` metaclass that consumes it, which reads well
    but breaks a caller that walks the backends generically. `oop/device.py` re-exports
    it -- re-exports, not redefines: there is one `Device` class, reachable two ways.
    """
    import importlib

    import pygpgpu.metal.oop.device as metal_device
    import pygpgpu.metal.oop.devices as metal_devices

    assert metal_device.Device is metal_devices.Device
    for backend in ("opencl", "cuda", "hip", "metal"):
        module = importlib.import_module(f"pygpgpu.{backend}.oop.device")
        assert hasattr(module, "Device"), backend


def test_the_macro_table_is_one_of_the_selectors() -> None:
    """`setPreprocessorMacros:` is the selector a mapping's `macros` field drives."""
    from pygpgpu.metal.oop.compile_options import CompileOptions

    assert "setPreprocessorMacros:" in CompileOptions.SELECTORS


def test_metal_is_a_zero_copy_backend_and_exposes_what_that_needs() -> None:
    """`StorageModeShared` already *is* host memory, and the layer is told so.

    The device-resident array layer asks `zero_copy_support(backend)` before it offers
    `to_device(zero_copy=True)`, and it refuses with the backend's own reason. Metal
    answers True because a shared buffer's bytes are the CPU's bytes; OpenCL answers
    False because `CL_MEM_USE_HOST_PTR` may keep a private copy.

    The Metal half of that path needs two things from this backend: `Buffer._contents`
    (the host address the allocation compares against the array's own pointer) and an
    instance attribute for the answer. A `__slots__` there would break the second.
    """
    from pygpgpu.numpy import zero_copy_support

    supported, reason = zero_copy_support("metal")
    assert supported, reason
    # The refusal message is empty for a backend that can; it is not empty for one
    # that cannot.
    assert reason == ""

    from pygpgpu.metal.oop.buffer import Buffer

    assert hasattr(Buffer, "_contents")
    assert not hasattr(Buffer, "__slots__")


def test_the_async_forms_reach_the_bound_launch() -> None:
    """`submit`/`async_call` on an unbracketed kernel launch with an automatic grid.

    `@kernel` offers those on every backend, so Metal answers with a launch rather
    than an AttributeError: a `MetalKernel` is bracketed (its `BoundKernel` carries
    the work size), and the async forms bracket `None` -- the same automatic
    one-threadgroup grid a synchronous call takes -- before launching. This pins the
    plumbing a machine without a Metal device can see: the refusal is gone, and the
    failure the launch meets (no device) surfaces through the future, the way a real
    failure on hardware would.
    """
    import concurrent.futures

    from pygpgpu.metal.oop.kernel import MetalKernel

    kernel = MetalKernel(None, "k")
    submit_future = kernel.submit()
    assert isinstance(submit_future, concurrent.futures.Future)
    with pytest.raises(BaseException) as submit_error:
        submit_future.result(timeout=10)
    assert not isinstance(submit_error.value, NotImplementedError)

    async_future = kernel.async_call()
    # `async_call` answers an asyncio future, whose `exception()` never raises.
    assert hasattr(async_future, "exception")
