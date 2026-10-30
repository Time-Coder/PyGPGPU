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


def test_the_header_is_a_real_file_in_the_package():
    """The fixed header lives in the source tree, so it can be read and edited like any
    other source -- and `pyproject.toml` has to ship it, or an installed copy would look for
    a file that is not there."""
    import pathlib

    from pygpgpu.kernel import quaternion

    source = pathlib.Path(quaternion.HEADER_SOURCE)
    assert source.is_file(), f"missing header: {source}"
    assert source.parent.name == "headers"
    text = source.read_text(encoding="utf-8")
    assert "PYGPGPU_QUAT_FN(_rotate)" in text
    # The package-data entry is what carries it into a wheel.
    pyproject = (pathlib.Path(quaternion.__file__).parents[2] / "pyproject.toml").read_text(
        encoding="utf-8")
    assert "kernel/headers/*.h" in pyproject


def read_text(path:str)->str:
    """The text of a file, read as UTF-8."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def test_installing_copies_the_header_verbatim(tmp_path):
    """The copy has to match the file, since the driver never sees the source tree."""
    from pygpgpu.kernel import quaternion

    installed = quaternion.install_header(str(tmp_path))
    assert read_text(installed) == quaternion.header_source()
    assert installed == str(tmp_path / "include" / quaternion.HEADER_NAME)


def test_the_header_is_fixed_and_the_shim_names_the_type(tmp_path):
    """The arithmetic is one fixed header; a per-type shim feeds it the generated C name.

    A structure's C name is a content hash, so the header cannot spell it. The shim defines
    the four macros and includes the header, and the generated source includes the shim --
    the preprocessor folds both into the text before the backend compiler runs.
    """
    from pygpgpu.kernel import quaternion

    line = quaternion.include_line("_st_abc123", np.dtype(quatf), "float3", "cuda",
                                   str(tmp_path))
    # Both files sit in the cache's include folder, so the line is relative to the generated
    # source and points into it.
    assert line == '#include "include/pygpgpu_quat__st_abc123_float3_cuda.h"'
    folder = tmp_path / "include"
    header = (folder / quaternion.HEADER_NAME).read_text(encoding="utf-8")
    assert "PYGPGPU_QUAT_FN(_rotate)" in header
    # Normalised by the quaternion's own magnitude, not the product's: getting this wrong
    # showed up as a factor-of-three difference from the interpreter.
    assert "sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)" in header
    shim = line.split('"')[1].split("/")[-1]
    body = (folder / shim).read_text(encoding="utf-8")
    assert "#define PYGPGPU_QUAT_NAME _st_abc123" in body
    assert "#define PYGPGPU_QUAT_QUAL __device__" in body
    # The shim reaches the header by its bare name, in the same folder.
    assert f'#include "{quaternion.HEADER_NAME}"' in body


def test_the_shim_has_no_device_qualifier_on_opencl(tmp_path):
    from pygpgpu.kernel import quaternion

    line = quaternion.include_line("_st_abc123", np.dtype(quatf), "float3", "opencl",
                                   str(tmp_path))
    shim = line.split('"')[1].split("/")[-1]
    body = (tmp_path / "include" / shim).read_text(encoding="utf-8")
    # OpenCL C has no execution-space annotation; `__device__` there does not compile.
    assert "#define PYGPGPU_QUAT_QUAL \n" in body


def test_a_cache_carries_its_own_headers(tmp_path):
    """One include folder per cache, not a shared one: a copied project still compiles."""
    from pygpgpu.kernel import quaternion

    first = tmp_path / "__cucache__"
    second = tmp_path / "__clcache__"
    quaternion.include_line("_st_abc123", np.dtype(quatf), "float3", "cuda", str(first))
    quaternion.include_line("_st_abc123", np.dtype(quatf), "float3", "opencl", str(second))
    for cache in (first, second):
        assert (cache / "include" / quaternion.HEADER_NAME).is_file()
    # Nothing is written outside the two caches.
    assert not (tmp_path / quaternion.HEADER_NAME).exists()


def test_a_non_quaternion_is_not_quaternion_arithmetic():
    from pygpgpu.kernel import quaternion

    assert quaternion.quat_member_names(np.dtype(Pair)) is None
    assert quaternion.quat_member_names(np.dtype(quatf)) == ("w", "x", "y", "z")
