"""The CUDA scientific-computing vocabulary, on both sides.

Three things are pinned here:

* the functions **both** backends have run on a device and agree with the host;
* the functions **only CUDA has** are refused on OpenCL *by name* -- the compiler's own
  "undeclared identifier" would not say which of the two the user should reach for;
* the host side (the `python` backend) computes the same thing, so a kernel means the
  same on every backend.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pygpgpu
from pygpgpu.kernel import device_builtins as tables

# The functions both backends have, with the host value each one must produce. Every
# expectation takes the arguments the *call* passes, so a two-argument function's
# expectation takes two arguments too.
SHARED = [
    ("erf", math.erf),
    ("erfc", math.erfc),
    ("tgamma", math.gamma),
    ("lgamma", math.lgamma),
    ("fmin", lambda v, hi: min(v, hi)),
    ("fmax", lambda v, lo: max(v, lo)),
    ("fract", lambda v: v - math.floor(v)),
    ("sign", lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)),
    ("copysign", lambda v, sign: math.copysign(v, sign)),
    ("fdim", lambda v, other: v - other if v > other else 0.0),
    ("remainder", lambda v, other: math.remainder(v, other)),
    ("nextafter", lambda v, other: math.nextafter(v, other)),
    ("ldexp", lambda v, exponent: math.ldexp(v, exponent)),
]

# The functions only CUDA has, with the host value each one must produce. The host value
# is the accurate one; the device's fast intrinsics differ within their own error.
CUDA_ONLY = [
    ("fast_exp", math.exp),
    ("fast_log", math.log),
    ("fast_sin", math.sin),
    ("fast_cos", math.cos),
    ("exact_recip", lambda v: 1.0 / v),
    ("exact_sqrt", math.sqrt),
    ("normal_cdf", lambda v: 0.5 * math.erfc(-v / math.sqrt(2.0))),
    ("reciprocal_hypot", lambda v, other: 1.0 / math.hypot(v, other)),
    ("sin_pi", lambda v: math.sin(math.pi * v)),
    ("cos_pi", lambda v: math.cos(math.pi * v)),
]


# ---------------------------------------------------------------------------
# The shape of the API
# ---------------------------------------------------------------------------


def test_every_scientific_name_is_exported():
    """A name the translator recognizes has to be reachable as `pygpgpu.<name>`."""
    for name in (list(tables.SHARED_MATH_BUILTINS) + list(tables.SHARED_MATH_TEMPLATES)
                 + list(tables.CUDA_ONLY_BUILTINS) + list(tables.CUDA_ONLY_CONVERSIONS)):
        assert hasattr(pygpgpu, name), f"pygpgpu does not export {name}"
        assert name in pygpgpu.__all__, f"{name} is missing from pygpgpu.__all__"


def test_the_tables_and_the_module_agree():
    """Every name the translator dispatches on is a real function with a signature."""
    import inspect

    dispatched = (list(tables.SHARED_MATH_BUILTINS) + list(tables.SHARED_MATH_TEMPLATES)
                  + list(tables.CUDA_ONLY_BUILTINS) + list(tables.CUDA_ONLY_CONVERSIONS))
    for name in dispatched:
        function = getattr(pygpgpu, name)
        signature = inspect.signature(function)
        assert signature.parameters, f"{name} takes no arguments"
        for parameter in signature.parameters.values():
            assert parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD), (
                f"{name} still takes *args/**kwargs")


def test_the_host_side_refuses_a_vector_argument():
    """CUDA spells these intrinsics for scalars only, so the host must refuse one too."""
    with pytest.raises(TypeError) as caught:
        pygpgpu.fast_exp(pygpgpu.float3(1.0, 2.0, 3.0))
    assert "scalar" in str(caught.value)


# ---------------------------------------------------------------------------
# The host side
# ---------------------------------------------------------------------------


# The arguments each probe passes besides the value under test, so the expectation is
# computed exactly the way the kernel computes it.
PROBE_ARGUMENTS = {
    "fmin": (1.0,),
    "fmax": (0.0,),
    "copysign": (-1.0,),
    "fdim": (1.0,),
    "remainder": (3.0,),
    "nextafter": (2.0,),
    "ldexp": (2,),
    "reciprocal_hypot": (2.0,),
}


def _expected(name: str, host_value, values: np.ndarray) -> np.ndarray:
    """What the device must answer, computed on the host the same way."""
    rest = PROBE_ARGUMENTS.get(name, ())
    results = [host_value(value, *rest) for value in values]
    return np.array(results, dtype=np.float32)


@pytest.mark.parametrize("name,host_value", SHARED + CUDA_ONLY)
def test_the_host_computes_the_operation(name, host_value):
    """The host answers the same thing the device does."""
    function = getattr(pygpgpu, name)
    rest = PROBE_ARGUMENTS.get(name, ())
    assert function(0.5, *rest) == pytest.approx(host_value(0.5, *rest), rel=1e-5, abs=1e-6)


def test_the_conversions_round_the_way_they_say():
    assert pygpgpu.float_to_int_nearest(2.6) == 3
    assert pygpgpu.float_to_int_nearest(2.4) == 2
    assert pygpgpu.float_to_int_zero(2.6) == 2
    assert pygpgpu.float_to_int_zero(-2.6) == -2
    assert pygpgpu.float_to_int_down(2.6) == 2
    assert pygpgpu.float_to_int_down(-2.6) == -3
    assert pygpgpu.float_to_int_up(2.4) == 3
    assert pygpgpu.double_to_int_nearest(2.5) == 2      # halfway cases to even
    assert pygpgpu.int_to_float(3) == 3.0


# ---------------------------------------------------------------------------
# The device side
# ---------------------------------------------------------------------------


def _probe(name: str):
    """The kernel that calls one scientific builtin."""
    from sci_math_kernels import PROBES

    return PROBES[name]


@pytest.mark.parametrize("name,host_value", SHARED)
def test_a_shared_function_answers_the_same_on_a_device(gpu_backend, runner, name, host_value):
    """Both backends have it, so it runs everywhere and agrees with the host."""
    probe = _probe(name)
    values = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(probe, values, out)
    expected = _expected(name, host_value, values)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name,host_value", CUDA_ONLY)
def test_a_cuda_only_function_runs_on_cuda(gpu_backend, runner, name, host_value):
    """On CUDA the intrinsic exists, and it agrees with the host to its own accuracy."""
    if gpu_backend != "cuda":
        pytest.skip("this check is about the CUDA backend")
    probe = _probe(name)
    values = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(probe, values, out)
    expected = _expected(name, host_value, values)
    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("name,host_value", CUDA_ONLY)
def test_a_cuda_only_function_runs_on_hip_too(gpu_backend, runner, name, host_value):
    """HIP reuses the CUDA code generation, and it has these intrinsics as well.

    HIP's own device header declares only a part of them; the rest come from the C++
    standard library, so the generated HIP source includes `<cmath>` (see
    `hip_compiler.py`). Without that include a HIP kernel calling one of them fails to
    compile, which is why this test exists.
    """
    if gpu_backend != "hip":
        pytest.skip("this check is about the HIP backend")
    probe = _probe(name)
    values = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(probe, values, out)
    expected = _expected(name, host_value, values)
    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("name,_host_value", CUDA_ONLY)
def test_a_cuda_only_function_is_refused_on_opencl_by_name(gpu_backend, runner,
                                                           name, _host_value):
    """OpenCL has no such function, and the refusal says which one it was."""
    if gpu_backend != "opencl":
        pytest.skip("this check is about the OpenCL backend")
    probe = _probe(name)
    values = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        runner(probe, values, out)
    message = str(caught.value)
    assert name in message
    assert "OpenCL" in message


def test_the_conversions_run_on_cuda(gpu_backend, runner):
    """A conversion answers an int, and the device says so.

    The conversions are CUDA-only (CUDA spells the rounding mode into the name), so this
    is a CUDA check.
    """
    from sci_math_kernels import probe_float_to_int_nearest

    if gpu_backend != "cuda":
        pytest.skip("the conversions are CUDA-only")
    values = np.array([1.4, 2.6, -3.2], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(probe_float_to_int_nearest, values, out)
    np.testing.assert_allclose(out, np.array([1.0, 3.0, -3.0], dtype=np.float32))


def test_a_vector_argument_is_refused_by_the_translator(gpu_backend, runner):
    """The device has no vector overload for these, so the translator says so.

    The function used here is one both backends have, so the refusal is about the vector
    and not about the backend.
    """
    from sci_math_kernels import probe_vector_argument

    vectors = np.array([pygpgpu.float3(1.0, 2.0, 3.0)])
    with pytest.raises(SyntaxError) as caught:
        runner(probe_vector_argument, vectors, np.zeros(1, dtype=np.float32))
    assert "scalar" in str(caught.value)
