"""The comparison and multiply-add builtins, and where their implementation lives.

These names exist twice in the two libraries under different spellings: OpenCL calls it
`isequal`/`isless`/`mad`/`rsqrt`, GLSL and pygf call the same operations
`equal`/`lessThan`/`fma`/`inversesqrt`. Only one implementation may exist per name -- the
translator recognizes a builtin by the identity of the callee -- so the OpenCL-named export
here *is* pygf's object, and the device still emits the OpenCL spelling because that follows
the registered name.

The component-wise predicates had no test coverage at all, which is how a width bug survived:
the int vector class was derived from the operand's class *name*, and anything that was not
two components fell back to `int4`, so `isequal(float3(...), float3(...))` built an `int4`
from three values and raised.
"""
from __future__ import annotations

import numpy as np
import pytest
import pygf.funcs as pygf_funcs
from pygf import float2, float3, float4

from pygpgpu import float3 as exported_float3
from pygpgpu.vec_types import funcs


# ---------------------------------------------------------------------------
# Where the implementation lives
# ---------------------------------------------------------------------------

ALIASES = {
    "isequal": "equal",
    "isnotequal": "notEqual",
    "isgreater": "greaterThan",
    "isgreaterequal": "greaterThanEqual",
    "isless": "lessThan",
    "islessequal": "lessThanEqual",
    "mad": "fma",
    "rsqrt": "inversesqrt",
}


@pytest.mark.parametrize("ours, theirs", sorted(ALIASES.items()))
def test_the_opencl_name_is_the_pygf_object(ours, theirs):
    """One object, two spellings; the device emission follows the registered name."""
    assert getattr(funcs, ours) is getattr(pygf_funcs, theirs)


@pytest.mark.parametrize("name", [
    "dot", "cross", "normalize", "length", "distance", "clamp", "mix",
    "saturate", "step", "exp2", "any", "all",
])
def test_the_shared_names_come_from_pygf(name):
    assert getattr(funcs, name) is getattr(pygf_funcs, name)


@pytest.mark.parametrize("name", [
    "select", "isordered", "isunordered", "abs_diff", "hadd", "rhadd", "mad24", "mul24",
])
def test_the_opencl_only_names_are_still_ours(name):
    """No pygf counterpart exists, so these are defined here."""
    assert getattr(funcs, name) is not getattr(pygf_funcs, name, None)


# ---------------------------------------------------------------------------
# Every width, which is what the width bug was about
# ---------------------------------------------------------------------------


PREDICATES = {
    "isequal": lambda x, y: x == y,
    "isnotequal": lambda x, y: x != y,
    "isless": lambda x, y: x < y,
    "islessequal": lambda x, y: x <= y,
    "isgreater": lambda x, y: x > y,
    "isgreaterequal": lambda x, y: x >= y,
}


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("name", sorted(PREDICATES))
def test_a_predicate_returns_an_int_vector_of_the_same_width(width, name):
    """The regression this pins: the result width used to be read off the operand's class
    name with a two-component special case, so a three-component vector produced an `int4`
    built from three values and raised."""
    vector = {2: float2, 3: float3, 4: float4}[width]
    values = list(range(1, width + 1))
    a = vector(*values)
    b = vector(*([2] * width))
    compare = PREDICATES[name]

    result = getattr(funcs, name)(a, b)
    assert len(result) == width, f"{name} produced {type(result).__name__}"
    assert [int(component) for component in result] == [
        int(compare(x, 2)) for x in values]


def test_a_predicate_broadcasts_a_scalar_operand():
    result = funcs.isgreater(float3(1.0, 5.0, 9.0), 4.0)
    assert [int(component) for component in result] == [0, 1, 1]


@pytest.mark.parametrize("name, expected", [
    ("isless", True),
    ("islessequal", True),
    ("isgreater", False),
    ("isgreaterequal", False),
    ("isequal", False),
    ("isnotequal", True),
])
def test_a_predicate_on_two_scalars(name, expected):
    """Scalar comparisons are part of the OpenCL contract, and the device accepts them."""
    assert bool(getattr(funcs, name)(1.0, 2.0)) is expected


def test_mad_and_rsqrt_agree_with_the_pygf_operations():
    assert funcs.mad(2.0, 3.0, 1.0) == pygf_funcs.fma(2.0, 3.0, 1.0)
    assert funcs.rsqrt(4.0) == pygf_funcs.inversesqrt(4.0)
    assert funcs.mad(float3(2.0, 3.0, 4.0), float3(1.0, 2.0, 3.0),
                     float3(1.0, 1.0, 1.0)) == float3(3.0, 7.0, 13.0)


# ---------------------------------------------------------------------------
# The same three, through a launch, so the host and the device agree
# ---------------------------------------------------------------------------


def make_kernels():
    from pygpgpu import kernel

    @kernel
    def predicate_less(a: np.ndarray, b: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = isless(a[i], b[i])["x"]

    @kernel
    def predicate_equal(a: np.ndarray, b: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = isequal(a[i], b[i])["x"]

    @kernel
    def multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = mad(a[i], b[i], c[i])["x"]

    @kernel
    def reciprocal_sqrt(a: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = rsqrt(a[i])["x"]

    return predicate_less, predicate_equal, multiply_add, reciprocal_sqrt


from pygpgpu import isequal, isless, mad, rsqrt  # noqa: E402  (after the kernels that use them)

predicate_less, predicate_equal, multiply_add, reciprocal_sqrt = make_kernels()


def test_a_predicate_on_the_device_matches_the_host(runner):
    """`isless` returns 1/0 on every backend, so the device and the host agree."""
    a = np.array([float4(1.0, 0.0, 0.0, 0.0)])
    b = np.array([float4(2.0, 0.0, 0.0, 0.0)])
    out = np.zeros(1, dtype=np.float32)
    runner(predicate_less, a, b, out)
    assert float(out[0]) == float(funcs.isless(a[0], b[0])["x"])

    runner(predicate_equal, a, b, out)
    assert float(out[0]) == float(funcs.isequal(a[0], b[0])["x"])


def test_mad_and_rsqrt_on_the_device(runner):
    a = np.array([float4(2.0, 3.0, 4.0, 5.0)])
    b = np.array([float4(3.0, 3.0, 3.0, 3.0)])
    c = np.array([float4(1.0, 1.0, 1.0, 1.0)])
    out = np.zeros(1, dtype=np.float32)
    runner(multiply_add, a, b, c, out)
    assert float(out[0]) == pytest.approx(7.0)

    runner(reciprocal_sqrt, np.array([float4(4.0, 9.0, 16.0, 25.0)]), out)
    assert float(out[0]) == pytest.approx(0.5)


def test_rsqrt_answers_what_the_device_answers():
    """The host used to raise ZeroDivisionError where a GPU returns an infinity, so a
    kernel relying on it worked on a device and blew up in the interpreter. Both sides
    now say the same thing, including the sign of a zero."""
    assert funcs.rsqrt(4.0) == pytest.approx(0.5)
    assert funcs.rsqrt(0.0) == float("inf")
    assert funcs.rsqrt(-0.0) == float("-inf")
    assert funcs.rsqrt(-1.0) != funcs.rsqrt(-1.0)  # NaN


def test_rsqrt_of_zero_on_the_device_is_an_infinity(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(reciprocal_sqrt, np.array([float4(0.0, 4.0, 4.0, 4.0)]), out)
    assert float(out[0]) == float("inf")


def test_the_exported_float3_is_the_pygf_one():
    """The predicate result has to be built from these, so they must be the same types."""
    assert exported_float3 is float3
