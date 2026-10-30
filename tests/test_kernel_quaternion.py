"""Structure arithmetic and quaternion arithmetic on the device.

Two halves, both about what a structure value may do with an operator.

**A structure takes no operator at all.** `s + s`, `s * s` and `s == s` have no device
meaning: C rejects most of them outright, and `*` used to fall through to a per-component
scalar multiply that silently computed something the language never defined. Every structure
is refused now, with an explanation.

**A quaternion is the one exception, and its arithmetic is real.** `pygf` defines `q * q` as
rotation composition, `q * v` as rotating a vector and `conjugate(q)` as flipping the axis.
None is a C operator on a structure, so each becomes a generated device function whose body
mirrors `genQuat._op` -- which is what makes the device answer match the interpreter.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest
from pygf import quatf, vector3f, conjugate

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# A plain structure takes no arithmetic
# ---------------------------------------------------------------------------


class Pair(ctypes.Structure):
    _fields_ = [("a", ctypes.c_float), ("b", ctypes.c_float)]


@kernel
def struct_add(s, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (s + s).a


@kernel
def struct_multiply(s, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (s * s).a


@kernel
def struct_compare(s, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if s == s else 0.0


@kernel
def struct_subtract(s, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (s - s).a


@pytest.mark.parametrize("function, symbol", [
    (struct_add, "+"),
    (struct_multiply, "*"),
    (struct_subtract, "-"),
    (struct_compare, "=="),
], ids=["add", "multiply", "subtract", "compare"])
def test_a_structure_takes_no_operator(gpu_backend, runner, function, symbol):
    """The interpreter runs the body as written, so its own TypeError is the answer there;
    the translated backends have to refuse it with an explanation instead of emitting a
    C operator the language does not define."""
    with pytest.raises((SyntaxError, TypeError)) as caught:
        runner(function, Pair(1.0, 2.0), np.zeros(1, np.float32))
    message = str(caught.value)
    if isinstance(caught.value, SyntaxError):
        assert f"'{symbol}' operator" in message
        assert "structure" in message


def test_the_refusal_says_what_to_do_instead(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(struct_add, Pair(1.0, 2.0), np.zeros(1, np.float32))
    message = str(caught.value)
    # The two escapes the message offers: operate on members, or move it to a host method.
    assert "members" in message
    assert "host method" in message


# ---------------------------------------------------------------------------
# A quaternion's arithmetic is generated
# ---------------------------------------------------------------------------


@kernel
def rotate(q, v, out: np.ndarray):
    """`q * v` -- rotate the vector, which is what pygf means by it."""
    for i in range(out.shape[0]):
        r = q * v
        out[i, 0] = r.x
        out[i, 1] = r.y
        out[i, 2] = r.z


@kernel
def compose(q, r, out: np.ndarray):
    """`q * r` -- compose two rotations."""
    for i in range(out.shape[0]):
        p = q * r
        out[i, 0] = p.w
        out[i, 1] = p.x
        out[i, 2] = p.y
        out[i, 3] = p.z


@kernel
def scale(q, s, out: np.ndarray):
    for i in range(out.shape[0]):
        p = q * s
        out[i, 0] = p.w
        out[i, 1] = p.x


@kernel
def flip(q, out: np.ndarray):
    for i in range(out.shape[0]):
        p = conjugate(q)
        out[i, 0] = p.w
        out[i, 1] = p.x
        out[i, 2] = p.y
        out[i, 3] = p.z


def test_quaternion_rotates_a_vector(runner):
    q = quatf(0.7071, 0.7071, 0.0, 0.0)
    v = vector3f(1.0, 2.0, 3.0)
    expected = q * v
    out = np.zeros((1, 3), np.float32)
    runner(rotate, q, v, out)
    assert [float(x) for x in out[0]] == pytest.approx(
        [expected.x, expected.y, expected.z], abs=1e-5)


def test_quaternion_composition(runner):
    q = quatf(0.7071, 0.7071, 0.0, 0.0)
    r = quatf(0.5, 0.0, 0.5, 0.0)
    expected = q * r
    out = np.zeros((1, 4), np.float32)
    runner(compose, q, r, out)
    assert [float(x) for x in out[0]] == pytest.approx(
        [expected.w, expected.x, expected.y, expected.z], abs=1e-5)


def test_quaternion_scaling(runner):
    q = quatf(1.0, 2.0, 3.0, 4.0)
    expected = q * 2.0
    out = np.zeros((1, 2), np.float32)
    runner(scale, q, 2.0, out)
    assert float(out[0, 0]) == pytest.approx(expected.w, abs=1e-5)
    assert float(out[0, 1]) == pytest.approx(expected.x, abs=1e-5)


def test_quaternion_conjugate(runner):
    q = quatf(1.0, 2.0, 3.0, 4.0)
    expected = conjugate(q)
    out = np.zeros((1, 4), np.float32)
    runner(flip, q, out)
    assert [float(x) for x in out[0]] == pytest.approx(
        [expected.w, expected.x, expected.y, expected.z], abs=1e-5)


def test_the_generated_function_is_per_backend():
    """A definition carries its backend's execution-space qualifier, so the cache has to be
    keyed by the backend too -- handing a CUDA body to OpenCL (or the reverse) failed to
    compile with a `__device__` the C dialect does not have."""
    from pygpgpu.kernel import quaternion

    cu = quaternion.device_function("conjugate", "Q", np.dtype(quatf), backend="cuda")[1]
    cl = quaternion.device_function("conjugate", "Q", np.dtype(quatf), backend="opencl")[1]
    assert "__device__" in cu
    assert "__device__" not in cl


def test_a_non_quaternion_is_not_quaternion_arithmetic():
    from pygpgpu.kernel import quaternion

    assert quaternion.quat_member_names(np.dtype(Pair)) is None
    assert quaternion.quat_member_names(np.dtype(quatf)) == ("w", "x", "y", "z")
