"""The HIP backend carries the same scientific vocabulary as CUDA.

HIP reuses the CUDA code generation, but the two are not the same device: HIP's own
headers declare a different set of intrinsics, and the ones it lacks have to be spelled
some other way. What is pinned here:

* every function CUDA has, HIP has too -- with the same answer;
* the three round-toward-zero forms, which HIP has no intrinsic for, agree with CUDA's
  *bit for bit* (HIP spells them with a device helper, and the helper has to reproduce
  the intrinsic exactly, not approximately);
* the generated HIP source carries the helper's definition before the kernel that calls
  it, because C requires a callee before its caller.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu.kernel import device_builtins as tables


# The values the round-toward-zero forms are checked over: positive and negative, over
# three orders of magnitude, and the ones that sit exactly on a rounding boundary.
ROUNDING_VALUES = np.array(
    [3.0, -3.0, 7.0, -7.0, 625.0958, 0.1, -0.1, 1.0 / 3.0, 2.5, -2.5, 1e-3, 1e3],
    dtype=np.float32)


def _require_hip() -> None:
    """Skip when the machine has no HIP device."""
    from pygpgpu.kernel.decorator import _probe_backend

    if not _probe_backend("hip"):
        pytest.skip("this machine has no HIP device")


def _require_cuda() -> None:
    from pygpgpu.kernel.decorator import _probe_backend

    if not _probe_backend("cuda"):
        pytest.skip("this machine has no CUDA device")


# ---------------------------------------------------------------------------
# The tables themselves
# ---------------------------------------------------------------------------


def test_every_function_cuda_has_has_a_hip_spelling():
    """A name with a CUDA spelling and no HIP one would fail to compile on HIP."""
    for name, spellings in tables.CUDA_ONLY_BUILTINS.items():
        assert "cuda" in spellings, f"{name} has no CUDA spelling"
        assert "hip" in spellings, f"{name} has no HIP spelling"


def test_every_shared_function_has_a_spelling_for_all_three():
    for name, spellings in tables.SHARED_MATH_BUILTINS.items():
        for backend in ("opencl", "cuda", "hip"):
            assert backend in spellings, f"{name} has no {backend} spelling"


def test_the_round_toward_zero_forms_have_a_helper_on_hip():
    """HIP has no such intrinsic, so those three are spelled with a device helper."""
    for name in ("exact_recip_rz", "exact_divide_rz", "exact_sqrt_rz"):
        hip_spelling = tables.CUDA_ONLY_BUILTINS[name]["hip"]
        assert hip_spelling in tables.HIP_ROUND_TOWARD_ZERO_HELPERS.values(), (
            f"{name} does not use a helper on HIP")
        # CUDA has the intrinsic itself.
        assert tables.CUDA_ONLY_BUILTINS[name]["cuda"].startswith("__")


def test_every_helper_has_a_source():
    """A helper without a definition would be called but never declared."""
    for key, name in tables.HIP_ROUND_TOWARD_ZERO_HELPERS.items():
        assert key in tables.HIP_ROUND_TOWARD_ZERO_SOURCES, f"{key} has no source"
        assert name in tables.HIP_ROUND_TOWARD_ZERO_SOURCES[key], (
            f"the source of {key} does not define {name}")


# ---------------------------------------------------------------------------
# On the device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["exact_recip_rz", "exact_sqrt_rz"])
def test_hip_and_cuda_answer_the_same_for_the_round_toward_zero_forms(name):
    """The helper reproduces the intrinsic exactly.

    Not "close to": bit for bit. The rounding is a one-unit-in-the-last-place step, so an
    approximation would differ in the last bit, which is the whole point of the form.
    """
    _require_hip()
    _require_cuda()
    from sci_math_kernels import PROBES

    probe = PROBES[name]
    hip_out = np.zeros(len(ROUNDING_VALUES), dtype=np.float32)
    cuda_out = np.zeros(len(ROUNDING_VALUES), dtype=np.float32)
    probe["hip"](ROUNDING_VALUES, hip_out)
    probe["cuda"](ROUNDING_VALUES, cuda_out)
    np.testing.assert_array_equal(hip_out, cuda_out)


def test_the_division_helper_rounds_the_exact_quotient_toward_zero():
    """The quotient is checked against the exact value, not against CUDA.

    The two device backends do not round a division identically -- the hardware's own
    divide differs in the last bit -- so a helper built on each one's correctly rounded
    quotient inherits that difference. What has to hold is the rounding DIRECTION: the
    result is the exact quotient rounded toward zero, which is checked against the exact
    value computed in rational arithmetic.
    """
    _require_hip()
    from fractions import Fraction

    from sci_math_kernels import PROBES

    probe = PROBES["exact_divide_rz"]
    values = np.array([3.0, -3.0, 7.0, -7.0, 0.1, -0.1, 1.0 / 3.0, 2.5, -2.5],
                       dtype=np.float32)
    out = np.zeros(len(values), dtype=np.float32)
    probe["hip"](values, out)

    for value, result in zip(values, out):
        exact = Fraction(float(value)) / Fraction(3.0)
        # The two float32 values the exact quotient can round to: the nearest one, and its
        # neighbour on the far side. "Toward zero" means the one with the smaller
        # MAGNITUDE, which for a negative quotient is the larger number.
        nearest = np.float32(exact.numerator) / np.float32(exact.denominator)
        if exact >= Fraction(float(nearest)):
            toward_zero = nearest
        else:
            step = np.float32(np.inf) if exact < 0 else np.float32(-np.inf)
            toward_zero = np.nextafter(nearest, step)
        assert np.float32(result) == toward_zero, (
            f"the quotient of {value} by 3 came out {result}, which is not the exact "
            f"quotient {exact} rounded toward zero")


@pytest.mark.parametrize("name", ["erf", "erfc", "tgamma", "lgamma", "fmin", "fmax",
                                  "fdim", "copysign", "nextafter", "ldexp", "remainder",
                                  "rint", "fast_exp", "fast_log", "fast_sin", "fast_cos",
                                  "exact_recip", "exact_sqrt", "exact_add", "exact_fma",
                                  "exact_divide", "normal_cdf", "reciprocal_hypot",
                                  "sin_pi", "bessel_j0", "bessel_i0"])
def test_hip_and_cuda_answer_the_same(name):
    """The two device backends mean the same thing by one of these names."""
    _require_hip()
    _require_cuda()
    from sci_math_kernels import PROBES

    probe = PROBES[name]
    values = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    hip_out = np.zeros(3, dtype=np.float32)
    cuda_out = np.zeros(3, dtype=np.float32)
    probe["hip"](values, hip_out)
    probe["cuda"](values, cuda_out)
    np.testing.assert_allclose(hip_out, cuda_out, rtol=1e-6, atol=1e-7)


def test_the_helper_reaches_the_generated_hip_source():
    """The definition has to be in the source, before the kernel that calls it."""
    _require_hip()
    from sci_math_kernels import PROBES

    probe = PROBES["exact_recip_rz"]
    values = np.array([3.0], dtype=np.float32)
    probe["hip"](values, np.zeros(1, dtype=np.float32))

    import glob
    import os

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "__hipcache__", "pykernels")
    sources = glob.glob(os.path.join(folder, "*.hip"))
    assert sources, "the generated HIP source was not found"
    helper = tables.HIP_ROUND_TOWARD_ZERO_HELPERS["recip"]
    # The definition lives in the shared generated header now -- the synthesized
    # helpers have no owning module, so one header per cache carries them all and
    # every generated file includes it. The unit is the file *plus* that header.
    header_path: str = os.path.join(folder, "include", "device_helpers.h")
    assert os.path.isfile(header_path), "the shared helper header was not found"
    with open(header_path, "r", encoding="utf-8") as file:
        header_text: str = file.read()
    assert f"__device__ float {helper}" in header_text, (
        "the helper was not written into the shared header")
    text = ""
    for path in sources:
        with open(path, "r", encoding="utf-8") as file:
            text += file.read()
    # C needs the definition before the kernel that calls it: the include sits in
    # the file's header, ahead of every kernel body, and the header's own
    # forward-declaration block leads its definitions.
    include = text.index('#include "include/device_helpers.h"')
    call = text.index(f"{helper}((float)")
    kernel = text.rindex("__global__ void", 0, call)
    assert include < kernel, (
        "the helper header is included after the kernel that calls it")


def test_the_conversions_answer_the_same_on_hip():
    """A conversion is the same operation on both device backends."""
    _require_hip()
    _require_cuda()
    from sci_math_kernels import probe_float_to_int_nearest

    values = np.array([1.4, 2.6, -3.2], dtype=np.float32)
    hip_out = np.zeros(3, dtype=np.float32)
    cuda_out = np.zeros(3, dtype=np.float32)
    probe_float_to_int_nearest["hip"](values, hip_out)
    probe_float_to_int_nearest["cuda"](values, cuda_out)
    np.testing.assert_array_equal(hip_out, cuda_out)
