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
from pygf import conjugate, quatd, quatf, vector3f

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


# ---------------------------------------------------------------------------
# A fixed name, so the header can declare the structure itself
# ---------------------------------------------------------------------------


def test_a_quaternion_gets_a_fixed_name_not_a_hash():
    """Other structures get a content hash; a quaternion gets a name the header can spell.

    That is what lets `quat.h` declare the type, so a generated source needs no declaration of
    its own. Two layouts exist, so two names cover every case.
    """
    from pygpgpu.kernel import structs
    from pygf import quatd

    assert structs.struct_name(np.dtype(quatf)) == "quatf"
    assert structs.struct_name(np.dtype(quatd)) == "quatd"
    # A quaternion is not a caller's structure, so it is excluded from the typedef block --
    # emitting it there as well would define the type twice.
    assert structs.render_structures([np.dtype(quatf)], "cuda") == ""
    assert structs.render_structures([np.dtype(quatd)], "opencl") == ""


def test_the_header_declares_both_structures():
    from pygpgpu.kernel import quaternion

    text = quaternion.header_source()
    assert "typedef struct { float w; float x; float y; float z; } quatf;" in text
    assert "typedef struct { double w; double x; double y; double z; } quatd;" in text
    # Normalised by the quaternion's own magnitude, not the product's: getting this wrong
    # showed up as a factor-of-three difference from the interpreter.
    assert "sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)" in text
    # OpenCL has no `sqrtf`, only `sqrt`; the header is shared by every backend.
    assert "sqrtf(" not in text


def test_the_header_is_a_real_file_in_the_package():
    """The header lives in the source tree, so it can be read and edited like any other
    source -- and `pyproject.toml` has to ship it, or an installed copy would look for a file
    that is not there."""
    import pathlib

    from pygpgpu.kernel import quaternion

    source = pathlib.Path(quaternion.HEADER_SOURCE)
    assert source.is_file(), f"missing header: {source}"
    assert source.parent.name == "include"
    assert source.name == "quat.h"
    pyproject = (pathlib.Path(quaternion.__file__).parents[2] / "pyproject.toml").read_text(
        encoding="utf-8")
    assert "kernel/include/*.h" in pyproject


def test_installing_copies_the_header_verbatim(tmp_path):
    """The copy has to match the file, since the driver never sees the source tree."""
    from pygpgpu.kernel import quaternion

    installed = quaternion.install_header(str(tmp_path))
    assert read_text(installed) == quaternion.header_source()
    assert installed == str(tmp_path / "include" / quaternion.HEADER_NAME)


def read_text(path:str)->str:
    """The text of a file, read as UTF-8."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def test_the_include_line_enables_only_the_widths_in_use(tmp_path):
    """A source says which blocks it needs, then includes the header once."""
    from pygpgpu.kernel import quaternion

    float_only = quaternion.include_line(("QUAT_NEED_FLOAT",), "cuda", str(tmp_path))
    assert float_only == ('#define QUAT_QUAL __device__\n'
                          '#define QUAT_NEED_FLOAT\n'
                          '#include "include/quat.h"')
    # The header is placed next to the sources that include it.
    assert (tmp_path / "include" / quaternion.HEADER_NAME).is_file()
    # A double block would not compile where the width is absent, so it is not asked for.
    assert "QUAT_NEED_DOUBLE" not in float_only


def test_opencl_gets_no_execution_space_qualifier(tmp_path):
    from pygpgpu.kernel import quaternion

    line = quaternion.include_line(("QUAT_NEED_FLOAT",), "opencl", str(tmp_path))
    # OpenCL C has no execution-space annotation; `__device__` there does not compile.
    assert line.splitlines()[0] == "#define QUAT_QUAL "


def test_a_cache_carries_its_own_headers(tmp_path):
    """One include folder per cache, not a shared one: a copied project still compiles."""
    from pygpgpu.kernel import quaternion

    first = tmp_path / "__cucache__"
    second = tmp_path / "__clcache__"
    quaternion.include_line(("QUAT_NEED_FLOAT",), "cuda", str(first))
    quaternion.include_line(("QUAT_NEED_FLOAT",), "opencl", str(second))
    for cache in (first, second):
        assert (cache / "include" / quaternion.HEADER_NAME).is_file()
    # Nothing is written outside the two caches.
    assert not (tmp_path / quaternion.HEADER_NAME).exists()


def test_a_non_quaternion_is_not_quaternion_arithmetic():
    from pygpgpu.kernel import quaternion

    assert quaternion.quat_member_names(np.dtype(Pair)) is None
    assert quaternion.quat_member_names(np.dtype(quatf)) == ("w", "x", "y", "z")
    assert quaternion.quat_type_name(np.dtype(Pair)) is None
    assert quaternion.needs(np.dtype(Pair)) is None


# ---------------------------------------------------------------------------
# A float and a double operand mix, and the result is the double form
# ---------------------------------------------------------------------------


@kernel
def mix_multiply(a, b, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (a * b).x


@kernel
def mix_add(a, b, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (a + b).x


@kernel
def mix_rotate(q, v, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (q * v).x


def test_a_float_and_a_double_quaternion_answer_a_double(runner):
    """pygf answers the double form; widening only ever goes one way."""
    from pygf import quatd

    float_quat = quatf(0.7071, 0.7071, 0.0, 0.0)
    double_quat = quatd(0.7071, 0.7071, 0.0, 0.0)
    for kernel_function, arithmetic in ((mix_multiply, lambda a, b: a * b),
                                        (mix_add, lambda a, b: a + b)):
        expected = arithmetic(float_quat, double_quat)
        out = np.zeros(1, np.float64)
        runner(kernel_function, float_quat, double_quat, out)
        assert float(out[0]) == pytest.approx(expected.x, abs=1e-5)


def test_a_float_quaternion_rotating_a_double_vector_answers_a_double(runner):
    from pygf import double3

    float_quat = quatf(0.7071, 0.7071, 0.0, 0.0)
    double_vector = double3(1.0, 2.0, 3.0)
    expected = float_quat * double_vector
    out = np.zeros(1, np.float64)
    runner(mix_rotate, float_quat, double_vector, out)
    assert float(out[0]) == pytest.approx(expected.x, abs=1e-6)


def test_a_double_quaternion_rotating_a_float_vector_answers_a_double(runner):
    from pygf import vector3f

    double_quat = quatd(0.7071, 0.7071, 0.0, 0.0)
    float_vector = vector3f(1.0, 2.0, 3.0)
    expected = double_quat * float_vector
    out = np.zeros(1, np.float64)
    runner(mix_rotate, double_quat, float_vector, out)
    assert float(out[0]) == pytest.approx(expected.x, abs=1e-6)


def test_the_mixed_block_needs_both_widths():
    """The header only compiles the mixed functions when both names are present, so a source
    that uses one asks for both."""
    from pygpgpu.kernel import quaternion

    assert quaternion.mixed_needs() == ("QUAT_NEED_FLOAT", "QUAT_NEED_DOUBLE")
    assert quaternion.wider(np.dtype(quatf), np.dtype(quatd)).fields["w"][0].name == "float64"

