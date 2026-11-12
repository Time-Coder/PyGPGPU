"""The kernel-side refusal of the 8/16-wide vectors (see the module there)."""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.vec_types import float4

from regressions.wide_vector_kernel_refusal import (
    builds_wide,
    reads_component,
    reads_swizzle,
    subscripts_wide,
    takes_wide,
)

WIDE_KERNELS = [takes_wide, builds_wide, reads_component, reads_swizzle, subscripts_wide]
DEVICE_BACKENDS = ["cuda", "opencl"]


@kernel
def takes_narrow(out: np.ndarray, v):
    out[0] = v.x


def _run(bad_kernel, backend):
    out = np.zeros(2)
    if bad_kernel is takes_wide:
        # A wide-vector value arrives from the OpenCL package, where the classes live.
        from pygpgpu.opencl.driver.vec_types import float8

        bad_kernel[backend](out, float8(*range(8)))
    else:
        bad_kernel[backend](out)


@pytest.mark.parametrize("bad_kernel", WIDE_KERNELS)
@pytest.mark.parametrize("backend", DEVICE_BACKENDS)
def test_a_device_kernel_refuses_the_8_and_16_wide_vectors(bad_kernel, backend):
    """Every spelling of an 8/16-wide vector in a body is refused on the devices."""
    with pytest.raises(Exception) as excinfo:
        _run(bad_kernel, backend)
    message = str(excinfo.value)
    assert "not a kernel type" in message.lower(), message
    assert "per device" in message.lower(), message


def test_the_interpreter_backend_still_runs_the_wide_value():
    """`python` executes the body as python: no translation, so no refusal."""
    from pygpgpu.opencl.driver.vec_types import float8

    out = np.zeros(2)
    takes_wide["python"](out, float8(1, 2, 3, 4, 5, 6, 7, 8))
    np.testing.assert_allclose(out, [1.0, 0.0])


def test_the_wide_classes_live_in_the_opencl_package():
    """The public vector surface stops at four components; the widths are OpenCL's."""
    import pygpgpu

    for name in ("float8", "float16", "char8", "char16", "int8", "int16", "genVec8",
                 "genVec16", "double8", "double16", "uint8", "uint16"):
        assert not hasattr(pygpgpu, name), name
        assert name not in pygpgpu.__all__, name
    # The OpenCL table still declares them: they are that backend's storage types.
    from pygpgpu.opencl.driver.clinfo import CLInfo

    for name in ("float8", "float16", "char8", "char16"):
        assert name in CLInfo.vec_types, name
    # And the registry the translator reads answers the refusal, not the class.
    from pygpgpu.kernel.translator import kernel_vector_refusal

    assert kernel_vector_refusal("float8") is not None
    assert kernel_vector_refusal("float4") is None
    assert kernel_vector_refusal("float3") is None
    assert kernel_vector_refusal("float2") is None


def test_a_wide_vector_array_argument_is_refused_with_the_same_reason():
    """An array of the wide dtype is the pointer spelling of the same refusal."""
    from pygpgpu.opencl.driver.vec_types import float8

    @kernel
    def takes_array(out, v):
        out[0] = v[0].s0

    values = np.zeros(1, dtype=float8.dtype)
    values[0] = tuple(range(8))
    with pytest.raises(Exception) as excinfo:
        takes_array["cuda"](np.zeros(1), values)
    assert "not a kernel type" in str(excinfo.value).lower()


def test_a_narrow_vector_still_travels():
    """The refusal is about the width, not about pygpgpu-owned classes."""
    out = np.zeros(2)
    takes_narrow["cuda"](out, float4(1, 2, 3, 4))
    np.testing.assert_allclose(out, [1.0, 0.0])
