"""Complex numbers on the device: both widths, every operation, every backend.

A complex value is the one numeric type no device language has, so the package carries it
the way numpy does -- a pair of reals -- and the arithmetic is generated rather than spelled
with a C operator. The header that declares both widths and every operation lives beside
the quaternion one, and the same contract applies: two fixed C type names, one enable macro
per width, and only the blocks a source uses are compiled.

The rule the whole feature follows is **numpy's**, because the interpreter backend runs
numpy: every operation here is compared against what numpy computes for the same two
values, so a device answer and a serial answer cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pytest
from pygf import conjugate

from pygpgpu import (clamp, exp, exp2, fmax, fmin, floor, fract, hypot, kernel, log, mad,
                     mix, rint, rsqrt, saturate, sign, sin, sqrt, step)
from pygpgpu.kernel import complex as complex_support


FLOAT = np.complex64
DOUBLE = np.complex128


def values(dtype):
    """Two complex values of the given width, with both parts non-zero."""
    return np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=dtype)


def other(dtype):
    """Two more complex values, one of them purely imaginary."""
    return np.array([5.0 + 0.0j, 0.0 + 1.0j], dtype=dtype)


# ---------------------------------------------------------------------------
# Arithmetic, at both widths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_addition_matches_numpy(runner, dtype):
    a, b = values(dtype), other(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(add, a, b, out)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_subtraction_matches_numpy(runner, dtype):
    a, b = values(dtype), other(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(subtract, a, b, out)
    np.testing.assert_allclose(out, a - b)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_multiplication_is_the_full_product(runner, dtype):
    """`z * w` is the full product, not a per-component scale."""
    a, b = values(dtype), other(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(multiply, a, b, out)
    np.testing.assert_allclose(out, a * b, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_division_matches_numpy(runner, dtype):
    a, b = values(dtype), other(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(divide, a, b, out)
    np.testing.assert_allclose(out, a / b, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_a_real_scalar_scales_and_divides(runner, dtype):
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(scale, a, np.float32(2.0), out)
    np.testing.assert_allclose(out, a * 2.0, rtol=1e-6)
    runner(scale_div, a, np.float32(2.0), out)
    np.testing.assert_allclose(out, a / 2.0, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_a_complex_scalar_argument_carries_both_parts(runner, dtype):
    """A scalar argument travels by value, and the imaginary part must survive the trip."""
    out = np.zeros(2, dtype=dtype)
    runner(scalar_scale, dtype(1.0 + 2.0j), out)
    np.testing.assert_allclose(out, np.array([3.0 + 6.0j, 3.0 + 6.0j], dtype=dtype))


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_negation_and_the_unary_plus(runner, dtype):
    a = values(dtype)
    negative = np.zeros(2, dtype=dtype)
    positive = np.zeros(2, dtype=dtype)
    runner(negate, a, negative)
    runner(identity, a, positive)
    np.testing.assert_allclose(negative, -a)
    np.testing.assert_allclose(positive, a)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_powers(runner, dtype):
    a = values(dtype)
    squared = np.zeros(2, dtype=dtype)
    runner(square_by_operator, a, squared)
    np.testing.assert_allclose(squared, a ** 2, rtol=1e-6)
    cubed = np.zeros(2, dtype=dtype)
    runner(cube, a, cubed)
    np.testing.assert_allclose(cubed, a ** 3, rtol=1e-6)


# ---------------------------------------------------------------------------
# The members and the methods a complex value has of its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_real_and_imaginary_members(runner, dtype):
    a = values(dtype)
    out = np.zeros((2, 2), dtype=np.float64 if dtype is DOUBLE else np.float32)
    runner(read_members, a, out)
    np.testing.assert_allclose(out[:, 0], a.real)
    np.testing.assert_allclose(out[:, 1], a.imag)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_conjugate_method(runner, dtype):
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(conjugate_method, a, out)
    np.testing.assert_allclose(out, np.conjugate(a))


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_free_conjugate_function(runner, dtype):
    """`conjugate(z)` is the same operation under its module-level spelling."""
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(conjugate_call, a, out)
    np.testing.assert_allclose(out, np.conjugate(a))


# ---------------------------------------------------------------------------
# The elementary functions, which numpy defines for a complex scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
@pytest.mark.parametrize("name", sorted((
    "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "square", "reciprocal", "exp2",
)))
def test_each_complex_function_matches_numpy(runner, dtype, name):
    a = values(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(COMPLEX_FUNCTIONS[name], a, out)
    expected = getattr(np, name)(a)
    np.testing.assert_allclose(out, expected,
                               rtol=1e-6 if dtype is FLOAT else 1e-12,
                               atol=1e-6 if dtype is FLOAT else 1e-12)


def test_the_magnitude_is_a_real(runner):
    """`abs(complex)` is the modulus, a real -- numpy's own rule."""
    a = values(DOUBLE)
    out = np.zeros(2, dtype=np.float64)
    runner(magnitude, a, out)
    np.testing.assert_allclose(out, np.abs(a), rtol=1e-12)


def test_the_phase_is_a_real(runner):
    a = values(DOUBLE)
    out = np.zeros(2, dtype=np.float64)
    runner(phase, a, out)
    np.testing.assert_allclose(out, np.angle(a), rtol=1e-12)


# ---------------------------------------------------------------------------
# Mixed widths widen, and nothing is rounded down
# ---------------------------------------------------------------------------


def test_a_float_and_a_double_value_answer_a_double(runner):
    """The double operand widens the whole operation, as the quaternions do."""
    single = values(FLOAT)
    double = values(DOUBLE)
    out = np.zeros(2, dtype=DOUBLE)
    runner(mixed_add, single, double, out)
    np.testing.assert_allclose(out, single.astype(DOUBLE) + double, rtol=1e-6)


def test_a_complex_value_compares_with_a_real_one(runner):
    """A real operand is widened, exactly as the arithmetic widens it."""
    a = np.array([2.0 + 0.0j, 1.0 + 1.0j], dtype=FLOAT)
    out = np.zeros(2, dtype=np.float32)
    runner(equality_with_real, a, np.float32(2.0), out)
    np.testing.assert_array_equal(out, np.array([1.0, 0.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Equality is defined, ordering is not
# ---------------------------------------------------------------------------


def test_equality_compares_both_parts(runner):
    a = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=FLOAT)
    b = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=FLOAT)
    out = np.zeros(2, dtype=np.float32)
    runner(equality, a, b, out)
    np.testing.assert_array_equal(out, np.array([1.0, 0.0], dtype=np.float32))


def test_ordering_a_complex_value_is_refused(gpu_backend, runner):
    """Python orders no complex number, so the device refuses the comparison by name."""
    a = values(FLOAT)
    with pytest.raises(SyntaxError) as caught:
        runner(less_than, a, a, np.zeros(2, dtype=np.float32))
    message = str(caught.value)
    assert "complex" in message
    assert "==" in message


# ---------------------------------------------------------------------------
# The fixed header, and the way a generated source reaches it
# ---------------------------------------------------------------------------


def test_the_two_widths_are_the_ones_numpy_has():
    assert complex_support.complex_type_name(np.dtype("complex64")) == "c64"
    assert complex_support.complex_type_name(np.dtype("complex128")) == "c128"
    assert complex_support.needs(np.dtype("complex64")) == "COMPLEX_NEED_FLOAT"
    assert complex_support.needs(np.dtype("complex128")) == "COMPLEX_NEED_DOUBLE"
    # A non-complex dtype is not one, so a caller's own two-float structure is unaffected.
    assert complex_support.complex_type_name(np.dtype("float32")) is None
    assert complex_support.needs(np.dtype([("re", "<f4"), ("im", "<f4")])) is None


def test_the_header_is_a_real_file_in_the_package():
    """The header ships with the package, or an installed copy looks for a file nothing has."""
    from pygpgpu.kernel import complex as complex_header

    assert complex_header.HEADER_SOURCE == str(
        __import__("pathlib").Path(complex_header.__file__).parent / "include" / "complex.h")

    import pathlib

    source = pathlib.Path(complex_header.HEADER_SOURCE)
    assert source.is_file(), f"missing header: {source}"
    assert source.parent.name == "include"
    assert source.name == "complex.h"
    pyproject = (pathlib.Path(complex_header.__file__).parents[2] / "pyproject.toml").read_text(
        encoding="utf-8")
    # The package data glob already covers the quaternion header; this pins that it is the
    # shared `kernel/include/*.h` one, so both headers travel together.
    assert "kernel/include/*.h" in pyproject


def test_the_header_declares_both_structures_with_numpys_layout():
    """The member names are numpy's own (`re`/`im`), which is what makes an array bind."""
    from pygpgpu.kernel import complex as complex_header

    text = complex_header.header_source()
    assert "typedef struct { float re; float im; } c64;" in text
    assert "typedef struct { double re; double im; } c128;" in text
    # OpenCL C has no `sqrtf`, only the overloaded form, so the header must not name one.
    assert "sqrtf(" not in text


def test_installing_copies_the_header_verbatim(tmp_path):
    from pygpgpu.kernel import complex as complex_header

    installed = complex_header.install_header(str(tmp_path))
    with open(installed, "r", encoding="utf-8") as handle:
        assert handle.read() == complex_header.header_source()
    assert installed == str(tmp_path / "include" / complex_header.HEADER_NAME)


def test_the_include_line_enables_only_the_widths_in_use(tmp_path):
    from pygpgpu.kernel import complex as complex_header

    float_only = complex_header.include_line(("COMPLEX_NEED_FLOAT",), "cuda", str(tmp_path))
    assert float_only == ('#define COMPLEX_QUAL __device__\n'
                          '#define COMPLEX_NEED_FLOAT\n'
                          '#include "include/complex.h"')
    assert (tmp_path / "include" / complex_header.HEADER_NAME).is_file()
    assert "COMPLEX_NEED_DOUBLE" not in float_only


def test_opencl_gets_no_execution_space_qualifier(tmp_path):
    from pygpgpu.kernel import complex as complex_header

    line = complex_header.include_line(("COMPLEX_NEED_FLOAT",), "opencl", str(tmp_path))
    assert line.splitlines()[0] == "#define COMPLEX_QUAL "


# ---------------------------------------------------------------------------
# The types the translator infers
# ---------------------------------------------------------------------------


def test_a_complex_value_is_recognized_by_its_dtype_alone():
    from pygpgpu.kernel.translator import DTYPE_MAP, infer_ctype, is_complex

    assert infer_ctype(np.complex64(1.0 + 2.0j)) == "c64"
    assert infer_ctype(np.complex128(1.0 + 2.0j)) == "c128"
    # A python complex has no width of its own, so it takes the one the python float beside
    # it takes: `float` by default, `double` under `@kernel(float=np.float64)`.
    assert infer_ctype(1.0 + 2.0j) == "c64"
    assert infer_ctype(1.0 + 2.0j, {"float": "double"}) == "c128"
    assert infer_ctype(np.zeros(2, dtype=np.complex64)) == ("*", "c64")
    assert infer_ctype(np.zeros(2, dtype=np.complex128)) == ("*", "c128")
    assert is_complex("c64") and is_complex("c128")
    assert not is_complex("float")
    assert DTYPE_MAP["complex64"] == "c64"


# ---------------------------------------------------------------------------
# The host functions take a complex scalar too
# ---------------------------------------------------------------------------


def test_the_host_functions_take_a_complex_scalar():
    """One implementation per name: the host body is pygf's, extended for complex.

    A kernel body that calls `sqrt` translates as the builtin because the object it resolves
    to *is* the builtin, and the interpreter backend runs that same body -- so the object has
    to accept a complex scalar for the two sides to agree.
    """
    import pygf

    for name, function in (("sqrt", np.sqrt), ("exp", np.exp), ("log", np.log),
                           ("sin", np.sin), ("cos", np.cos), ("tan", np.tan),
                           ("asin", np.arcsin), ("acos", np.arccos), ("atan", np.arctan),
                           ("sinh", np.sinh), ("cosh", np.cosh), ("tanh", np.tanh)):
        value = 1.0 + 2.0j
        assert getattr(pygf, name)(value) == pytest.approx(function(value))


def test_the_host_arithmetic_helpers_are_the_complex_ones():
    import pygf

    assert pygf.pow(1.0 + 2.0j, 2.0) == pytest.approx((1.0 + 2.0j) ** 2.0)
    assert pygf.conjugate(1.0 + 2.0j) == pytest.approx(1.0 - 2.0j)
    # A real pair keeps the real answer, so nothing regressed for the ordinary case.
    assert pygf.hypot(3.0, 4.0) == pytest.approx(5.0)
    with pytest.raises(TypeError):
        pygf.hypot(1.0 + 2.0j, 2.0)


def test_the_quaternions_conjugate_is_unchanged():
    import pygf
    from pygf import quatf

    flipped = pygf.conjugate(quatf(1.0, 2.0, 3.0, 4.0))
    assert (flipped.w, flipped.x, flipped.y, flipped.z) == (1.0, -2.0, -3.0, -4.0)


# ---------------------------------------------------------------------------
# The kernels the tests translate
# ---------------------------------------------------------------------------


@kernel
def add(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def subtract(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] - b[i]


@kernel
def multiply(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * b[i]


@kernel
def divide(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / b[i]


@kernel
def scale(a: np.ndarray, factor: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * factor


@kernel
def scale_div(a: np.ndarray, factor: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / factor


@kernel
def scalar_scale(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value * 3.0


@kernel
def negate(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = -a[i]


@kernel
def identity(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = +a[i]


@kernel
def square_by_operator(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] ** 2


@kernel
def cube(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] ** 3


@kernel
def read_members(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i, 0] = a[i].real
        out[i, 1] = a[i].imag


@kernel
def conjugate_method(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i].conjugate()


@kernel
def conjugate_call(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = conjugate(a[i])


@kernel
def magnitude(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = abs(a[i])


@kernel
def phase(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.angle(a[i])


@kernel
def mixed_add(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def equality(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] == b[i] else 0.0


@kernel
def equality_with_real(a: np.ndarray, value: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] == value else 0.0


@kernel
def less_than(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] < b[i] else 0.0


@kernel
def complex_sqrt(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sqrt(a[i])


@kernel
def complex_exp(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.exp(a[i])


@kernel
def complex_log(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log(a[i])


@kernel
def complex_log2(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log2(a[i])


@kernel
def complex_log10(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log10(a[i])


@kernel
def complex_sin(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sin(a[i])


@kernel
def complex_cos(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.cos(a[i])


@kernel
def complex_tan(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.tan(a[i])


@kernel
def complex_asin(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arcsin(a[i])


@kernel
def complex_acos(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arccos(a[i])


@kernel
def complex_atan(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arctan(a[i])


@kernel
def complex_sinh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sinh(a[i])


@kernel
def complex_cosh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.cosh(a[i])


@kernel
def complex_tanh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.tanh(a[i])


@kernel
def complex_square(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.square(a[i])


@kernel
def complex_reciprocal(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.reciprocal(a[i])


@kernel
def complex_exp2(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.exp2(a[i])


COMPLEX_FUNCTIONS = {
    "sqrt": complex_sqrt, "exp": complex_exp, "log": complex_log,
    "log2": complex_log2, "log10": complex_log10,
    "sin": complex_sin, "cos": complex_cos, "tan": complex_tan,
    "asin": complex_asin, "acos": complex_acos, "atan": complex_atan,
    "sinh": complex_sinh, "cosh": complex_cosh, "tanh": complex_tanh,
    "square": complex_square, "reciprocal": complex_reciprocal,
    "exp2": complex_exp2,
}

# ---------------------------------------------------------------------------
# A complex scalar in every place a value can cross the boundary
# ---------------------------------------------------------------------------
#
# Three shapes, and they are genuinely different code paths: a value **created** in the body
# (a literal above or inside the loop, or an `np.complex64(...)` cast), one **passed** as an
# argument, and one **returned** through a slot. Each used to fail on its own cause -- an
# unsupported complex literal, a C cast between two structures, and a narrowed return slot --
# which is why they are pinned separately here.


@kernel
def created_constant(out: np.ndarray):
    """A complex built above the loop: `3 + 4j` is an Add over a real and an imaginary."""
    value = 3 + 4j
    for i in range(out.shape[0]):
        out[i] = value


@kernel
def created_from_element(a: np.ndarray, out: np.ndarray):
    """`np.complex64(x)` on a device value: a cast, which is the header's conversion."""
    for i in range(out.shape[0]):
        out[i] = np.complex64(a[i])


@kernel
def created_from_real(a: np.ndarray, out: np.ndarray):
    """A real converted to a complex one: the imaginary part is zero."""
    for i in range(out.shape[0]):
        out[i] = np.complex64(a[i] * 2.0)


@kernel
def created_in_loop(a: np.ndarray, out: np.ndarray):
    """A local complex built inside the loop from the element itself."""
    for i in range(out.shape[0]):
        value = a[i] * a[i]
        out[i] = value


@kernel
def passed_scalar(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value


@kernel
def returned_element(a: np.ndarray):
    for i in range(a.shape[0]):
        return a[i]


@kernel
def returned_computed(a: np.ndarray):
    for i in range(a.shape[0]):
        return a[i] ** 2


@kernel
def returned_constant(out: np.ndarray):
    return 3 + 4j


def test_a_complex_created_above_the_loop(runner):
    """`3 + 4j` parses as `3 + (4j)`, so it goes through the arithmetic, not a constant."""
    out = np.zeros(2, np.complex128)
    runner(created_constant, out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], np.complex128))


def test_a_casted_complex_keeps_both_parts(runner):
    a = values(np.complex128)
    out = np.zeros(2, np.complex64)
    runner(created_from_element, a, out)
    np.testing.assert_allclose(out, a.astype(np.complex64), rtol=1e-6)


def test_a_real_cast_to_complex_has_a_zero_imaginary_part(runner):
    a = np.array([1.0, 3.0], np.float32)
    out = np.zeros(2, np.complex64)
    runner(created_from_real, a, out)
    np.testing.assert_allclose(out, (a * 2.0).astype(np.complex64))


def test_a_complex_local_inside_the_loop(runner):
    a = values(np.complex64)
    out = np.zeros(2, np.complex64)
    runner(created_in_loop, a, out)
    np.testing.assert_allclose(out, a * a, rtol=1e-6)


@pytest.mark.parametrize("value", [np.complex64(1 + 2j), np.complex128(1 + 2j), 1 + 2j],
                         ids=["complex64", "complex128", "python"])
def test_a_complex_scalar_argument(runner, value):
    """A python complex is a `c64` under the default mapping, so the value arrives narrowed."""
    out = np.zeros(2, np.complex128)
    runner(passed_scalar, value, out)
    np.testing.assert_allclose(out, np.array([value, value], np.complex128))


def test_a_complex_element_is_returned(runner):
    result = runner(returned_element, np.array([1 + 2j], np.complex64))
    assert result == pytest.approx(1 + 2j)


def test_a_computed_complex_is_returned(runner):
    result = runner(returned_computed, np.array([1 + 2j], np.complex64))
    assert result == pytest.approx((1 + 2j) ** 2, rel=1e-6)


def test_a_complex_constant_is_returned(runner):
    """A returned literal crosses the loop never, so the slot answers it directly."""
    result = runner(returned_constant, np.zeros(1, np.complex64))
    assert result == pytest.approx(3 + 4j)

# ---------------------------------------------------------------------------
# An outer-scope complex scalar: read, used in arithmetic, and rebound
# ---------------------------------------------------------------------------
#
# A module global or a closure cell is a kernel *parameter* here (see
# `_bind_outer_variables`): a read-only one travels by value, a `nonlocal`/`global`-declared
# one travels as a one-element buffer the host copies back after the launch. Both shapes work
# for a complex value exactly as they do for an int, which is what these pin.

SCOPE_64 = np.complex64(3 + 4j)
SCOPE_128 = np.complex128(3 + 4j)
SCOPE_PY = 3 + 4j


@kernel
def read_scope_python(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_PY


@kernel
def read_scope_complex64(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_64


@kernel
def read_scope_complex128(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_128


@kernel
def multiply_by_scope(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * SCOPE_128


@kernel
def return_scope(out: np.ndarray):
    return SCOPE_128


@kernel
def return_scope_complex64(out: np.ndarray):
    return SCOPE_64


@pytest.mark.parametrize("function, dtype", [
    (read_scope_python, np.complex128),
    (read_scope_complex64, np.complex64),
    (read_scope_complex128, np.complex128),
], ids=["python", "complex64", "complex128"])
def test_an_outer_complex_scalar_is_read(runner, function, dtype):
    out = np.zeros(2, dtype=dtype)
    runner(function, out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], dtype=dtype))


def test_an_outer_complex_scalar_takes_part_in_arithmetic(runner):
    a = np.array([1 + 1j, 2 + 2j], np.complex64)
    out = np.zeros(2, np.complex128)
    runner(multiply_by_scope, a, out)
    np.testing.assert_allclose(out, a * (3 + 4j), rtol=1e-6)


def test_an_outer_complex_scalar_is_returned(runner):
    assert runner(return_scope, np.zeros(1, np.complex64)) == pytest.approx(3 + 4j)
    assert runner(return_scope_complex64, np.zeros(1, np.complex64)) == pytest.approx(3 + 4j)


def make_scope_rebinder(dtype):
    """A closure cell a kernel may rebind, with a reader for its value.

    The rebinding parameter is called `fresh`, not `new`: CUDA and HIP compile as C++, where
    `new` is a keyword, so a parameter of that name fails to compile on those two backends
    whatever its type is (pinned separately in tests/test_kernel_argument_errors.py).
    """
    held = dtype(0 + 0j)

    @kernel
    def rebind(fresh, out: np.ndarray):
        nonlocal held
        held = fresh
        for i in range(out.shape[0]):
            out[i] = held

    return rebind, lambda: held


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128],
                         ids=["complex64", "complex128"])
def test_an_outer_complex_scalar_can_be_rebound(runner, dtype):
    function, peek = make_scope_rebinder(dtype)
    out = np.zeros(2, dtype=dtype)
    runner(function, dtype(5 + 6j), out)
    np.testing.assert_allclose(out, np.array([5 + 6j, 5 + 6j], dtype=dtype))
    # The `nonlocal` slot is copied back into the scope after the launch, so the closure
    # holds the value the kernel put there -- for a complex value just as for an int.
    assert complex(peek()) == pytest.approx(5 + 6j)

# ---------------------------------------------------------------------------
# `complex(re, im)`, python's own constructor
# -----------------------------------------------------------------------------------------------------------------------------------------


@kernel
def constructed_from_args(re: np.float32, im: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(re, im)


@kernel
def constructed_from_constants(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(3, 4)


@kernel
def constructed_from_complex(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(a[i])


@kernel
def constructed_from_real(value: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(value)


@kernel
def constructed_then_narrowed(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.complex64(complex(3, 4))


@kernel
def constructed_from_string(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex("3+4j")


def test_complex_of_two_arguments(runner):
    """`complex(re, im)` is python's own spelling, and the result is the mapped width."""
    out = np.zeros(2, dtype=np.complex128)
    runner(constructed_from_args, np.float32(3.0), np.float32(4.0), out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], dtype=np.complex128))


def test_complex_of_two_constants(runner):
    out = np.zeros(2, dtype=np.complex128)
    runner(constructed_from_constants, out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], dtype=np.complex128))


def test_complex_of_a_complex_value_is_a_copy_at_the_mapped_width(runner):
    """`complex(z)` of a complex64 is widened under the default mapping before the store."""
    out = np.zeros(2, dtype=np.complex128)
    runner(constructed_from_complex, np.array([1 + 2j, 1 + 2j], np.complex64), out)
    np.testing.assert_allclose(out, np.array([1 + 2j, 1 + 2j], dtype=np.complex128))


def test_complex_of_a_real_has_a_zero_imaginary_part(runner):
    out = np.zeros(2, dtype=np.complex128)
    runner(constructed_from_real, np.float32(5.0), out)
    np.testing.assert_allclose(out, np.array([5 + 0j, 5 + 0j], dtype=np.complex128))


def test_a_constructed_value_narrows_through_numpy_cast(runner):
    out = np.zeros(2, dtype=np.complex64)
    runner(constructed_then_narrowed, out)
    np.testing.assert_allclose(out, np.array([3 + 4j, 3 + 4j], dtype=np.complex64))


def test_complex_of_a_string_is_refused(gpu_backend, runner):
    """A string has no device value, and python would parse it on the host."""
    out = np.zeros(2, dtype=np.complex128)
    with pytest.raises(SyntaxError) as caught:
        runner(constructed_from_string, out)
    assert "complex(" in str(caught.value)

# ---------------------------------------------------------------------------
# The width of a python complex follows `float=`
# ---------------------------------------------------------------------------
#
# A python `complex` has no width of its own, so it is chosen together with the python `float`
# beside it: the 32-bit default gives `c64`, `@kernel(float=np.float64)` gives `c128`, and a
# python complex is never the one value in a body left at the other width. Three places decide
# a width for a python complex -- the scalar path (`infer_ctype`, pinned in
# tests/test_kernel_type_map.py), a body literal, and `complex(re, im)` -- and all three read
# it from `translator.python_complex_type`, so a mapping moves them together. A python complex
# used to be a `c128` everywhere, which paired a 64-bit complex with the 32-bit real a python
# float travels as.
#
# A device helper is the fourth place, by way of the first: it is translated into the kernel's
# own translation unit and monomorphized per call site, so it inherits the kernel's mapping
# (`FunctionTranslator.type_map`) and its device name carries it -- two mappings of one helper
# are two functions, and a source file holds every kernel written in it.


@kernel
def python_literal(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 3 + 4j


@kernel
def python_constructed(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(3, 4)


@kernel(float=np.float64)
def double_complex_argument(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value


def python_complex_literal_helper(value):
    """An ordinary helper whose body builds a python complex literal of its own.

    A helper is translated into the *kernel's* translation unit and monomorphized per call
    site, so it has to make the same width choice its caller does -- otherwise a helper body
    silently answered the 32-bit default while the kernel around it was mapped to a double.
    """
    return value + 0.2j


@kernel
def calls_plain_helper(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = python_complex_literal_helper(a[i])


@kernel(float=np.float64)
def calls_double_helper(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = python_complex_literal_helper(a[i])


def _generated_source(func, type_map=None):
    """The device source a one-array kernel renders, without compiling it."""
    from pygpgpu.kernel.translator import FunctionTranslator, infer_ctype

    raw = func.raw_func if hasattr(func, "raw_func") else func
    out = infer_ctype(np.zeros(2, np.complex64))
    translator = FunctionTranslator(raw, [out], "cuda", lambda name: None, {},
                                    is_kernel=True, type_map=type_map)
    translator.translate()
    return "\n\n".join(source for _, source in translator.render_kernels("k_test"))


@pytest.mark.parametrize("type_map, kind", [
    (None, "c64"),
    ({"float": "float"}, "c64"),
    ({"float": "double"}, "c128"),
], ids=["no-mapping", "float", "double"])
def test_a_python_complex_literal_is_the_mapped_width(type_map, kind):
    """`3 + 4j` is built at the width the python float beside it travels as."""
    source = _generated_source(python_literal, type_map)
    assert f"{kind}_make(" in source
    other = "c128" if kind == "c64" else "c64"
    assert f"{other}_make(" not in source


@pytest.mark.parametrize("type_map, kind", [
    (None, "c64"),
    ({"float": "float"}, "c64"),
    ({"float": "double"}, "c128"),
], ids=["no-mapping", "float", "double"])
def test_the_complex_constructor_is_the_mapped_width(type_map, kind):
    """`complex(re, im)` answers the same width a python complex travels as."""
    source = _generated_source(python_constructed, type_map)
    assert f"{kind}_make(" in source
    other = "c128" if kind == "c64" else "c64"
    assert f"{other}_make(" not in source


def test_a_python_complex_argument_is_a_c64_under_the_default_mapping(gpu_backend):
    """The device declares the 32-bit pair, so the value arrives narrowed, as a float does."""
    out = np.zeros(2, np.complex128)
    passed_scalar[gpu_backend](0.1 + 0.2j, out)
    # A `c128` parameter would have carried 0.1 + 0.2j exactly; a `c64` carries the pair
    # `np.complex64` rounds to, and the double array holds that without changing it again.
    assert out[0] == complex(np.complex64(0.1 + 0.2j))
    assert out[0] != 0.1 + 0.2j


def test_a_double_mapping_carries_a_python_complex_at_128_bits(gpu_backend):
    """`float=np.float64` is what lets a python complex hold a value float32 cannot."""
    out = np.zeros(2, np.complex128)
    double_complex_argument[gpu_backend](1e40 + 1e40j, out)
    # A `c64` parameter would have answered infinity: float32 stops at ~3.4e38.
    assert out[0] == 1e40 + 1e40j


def test_a_helper_body_follows_the_kernel_it_is_translated_in(gpu_backend):
    """A python complex literal inside a helper takes the caller's width, not the default."""
    a = np.zeros(2, np.complex64)
    out = np.zeros(2, np.complex128)
    # Under `float=np.float64` the helper's literal is a `c128`, so the sum is exact.
    calls_double_helper[gpu_backend](a, out)
    assert out[0] == 0.2j
    # Under the default 32-bit mapping it is a `c64`, so the value is the rounded pair.
    calls_plain_helper[gpu_backend](a, out)
    assert out[0] == complex(np.complex64(0.2j))
    assert out[0] != 0.2j

# ---------------------------------------------------------------------------
# The builtin spellings that had no complex path
# ---------------------------------------------------------------------------
#
# The builtin tables are keyed by *type*, and a complex value has no entry in any of them, so
# every name below used to be emitted from a path that assumed a real: `sqrtf(z)`,
# `min((int)(z), (int)(w))`, `fminf(fmaxf(z, 0.0f), 1.0f)` -- calls the compiler cannot type
# -- and a `round(z)` that returned the value unchanged. The interpreter ran pygf's helper for
# each of them and answered what numpy answers, so the device answers it too, from the complex
# header. A name the interpreter refuses (python's `round`, and the real-only `floor`,
# `rsqrt`, ...) is refused at translation time with the reason instead.


@kernel
def builtin_sign(z, out):
    for i in range(out.shape[0]):
        out[i] = sign(z)


@kernel
def numpy_sign(z, out):
    for i in range(out.shape[0]):
        out[i] = np.sign(z)


@kernel
def builtin_transcendental(z, out):
    for i in range(out.shape[0]):
        out[i] = sqrt(z) + exp(z) + log(z) + sin(z) + exp2(z)


@kernel
def numpy_transcendental(z, out):
    for i in range(out.shape[0]):
        out[i] = np.sqrt(z) + np.exp(z) + np.log(z) + np.sin(z) + np.exp2(z)


@kernel
def numpy_round(z, out):
    for i in range(out.shape[0]):
        out[i] = np.round(z)


@kernel
def numpy_rint(z, out):
    for i in range(out.shape[0]):
        out[i] = np.rint(z)


@kernel
def builtin_min_max(z, w, out):
    for i in range(out.shape[0]):
        out[i] = min(z, w) + max(z, w)


@kernel
def numpy_minimum_maximum(z, w, out):
    for i in range(out.shape[0]):
        out[i] = np.minimum(z, w) + np.maximum(z, w)


@kernel
def builtin_clamp_saturate(z, out):
    for i in range(out.shape[0]):
        out[i] = clamp(z, 0, 1) + saturate(z)


@kernel
def numpy_clip(z, out):
    for i in range(out.shape[0]):
        out[i] = np.clip(z, 0, 1) + np.clip(z, 0, 1)


@kernel
def builtin_mix_mad(z, w, v, out):
    for i in range(out.shape[0]):
        out[i] = mix(z, w, 0.5) + mad(z, w, v)


@kernel
def numpy_mix_mad(z, w, v, out):
    for i in range(out.shape[0]):
        out[i] = (z * (1 - 0.5) + w * 0.5) + (z * w + v)


@kernel
def numpy_predicates(z, flags):
    """The three predicates as a bitmask, so one run can tell them apart."""
    for i in range(flags.shape[0]):
        flags[i] = np.isnan(z) * 4 + np.isinf(z) * 2 + np.isfinite(z)


def _predicate_mask(value) -> int:
    return int(np.isnan(value)) * 4 + int(np.isinf(value)) * 2 + int(np.isfinite(value))


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_builtin_transcendentals_answer_complex(runner, dtype):
    """`sqrt(z)` and its siblings are pygf's functions, so the device calls the header."""
    z = dtype(1 + 2j)
    out = np.zeros(2, dtype=dtype)
    runner(builtin_transcendental, z, out)
    expected = (np.sqrt(z) + np.exp(z) + np.log(z) + np.sin(z) + np.exp2(z)).astype(dtype)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_sign_answers_the_unit_direction(runner, dtype):
    """numpy's complex `sign`, under both spellings -- and zero at zero, not a NaN."""
    out = np.zeros(1, dtype=dtype)
    for value in (dtype(1 + 2j), dtype(-3 + 0.5j), dtype(0)):
        runner(builtin_sign, value, out)
        np.testing.assert_allclose(out[0], np.sign(value), rtol=1e-6, atol=1e-7)
        runner(numpy_sign, value, out)
        np.testing.assert_allclose(out[0], np.sign(value), rtol=1e-6, atol=1e-7)


@kernel
def builtin_rint(z, out):
    for i in range(out.shape[0]):
        out[i] = rint(z)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_rounding_a_complex_value_rounds_each_part(runner, dtype):
    """`np.round(z)`, `np.rint(z)` and this package's `rint(z)` all round both parts to even.

    `rint` is its own host body (`sci_math`), so it answers a complex value by rounding each
    part with python's `round` -- the same rule the device's `c64_rint` implements. Python's
    *builtin* `round(z)` still refuses one, which is a different name.
    """
    out = np.zeros(1, dtype=dtype)
    for value in (dtype(1.5 + 2.5j), dtype(2.5 - 3.5j), dtype(-1.5 + 0.5j)):
        runner(numpy_round, value, out)
        np.testing.assert_allclose(out[0], np.round(value))
        runner(numpy_rint, value, out)
        np.testing.assert_allclose(out[0], np.rint(value))
        runner(builtin_rint, value, out)
        np.testing.assert_allclose(out[0], np.rint(value))


@kernel
def builtin_fmin_fmax(z, w, out):
    for i in range(out.shape[0]):
        out[i] = fmin(z, w) + fmax(z, w)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_fmin_and_fmax_ignore_a_nan_operand(runner, dtype):
    """`fmin`/`fmax` are the NaN-ignoring pair, and numpy has two families for a reason.

    `min`/`max` (and `np.minimum`/`np.maximum`) propagate a NaN, because it has a defined
    place in their order; `fmin`/`fmax` return the operand that is *not* NaN. A complex
    value is NaN when either part is, so the device needs the second function rather than
    the ordering one it used to emit.
    """
    normal = dtype(3 - 1j)
    nan_real = dtype(complex(float("nan"), 1.0))
    nan_imag = dtype(complex(1.0, float("nan")))
    out = np.zeros(1, dtype=dtype)
    for pair in ((normal, normal), (nan_real, normal), (normal, nan_real),
                 (nan_imag, normal), (normal, nan_imag)):
        out[0] = 0
        runner(builtin_fmin_fmax, pair[0], pair[1], out)
        expected = np.fmin(pair[0], pair[1]) + np.fmax(pair[0], pair[1])
        np.testing.assert_allclose(out[0], expected)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_minimum_and_maximum_are_lexicographic(runner, dtype):
    """The real part decides and the imaginary one breaks the tie, as numpy does it."""
    z = dtype(1 + 2j)
    out = np.zeros(1, dtype=dtype)
    for w in (dtype(3 - 1j), dtype(1 - 1j), dtype(3 + 0j)):
        runner(builtin_min_max, z, w, out)
        np.testing.assert_allclose(out[0], np.minimum(z, w) + np.maximum(z, w))
        runner(numpy_minimum_maximum, z, w, out)
        np.testing.assert_allclose(out[0], np.minimum(z, w) + np.maximum(z, w))


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_clamp_and_clip_are_the_lexicographic_pair(runner, dtype):
    """`clamp(z, 0, 1)`, `saturate(z)` and `np.clip(z, 0, 1)` are one operation."""
    out = np.zeros(1, dtype=dtype)
    for value in (dtype(1 + 2j), dtype(0.5 + 0.25j), dtype(4 - 3j), dtype(-1 + 0j)):
        runner(builtin_clamp_saturate, value, out)
        np.testing.assert_allclose(out[0], np.clip(value, 0, 1) * 2)
        runner(numpy_clip, value, out)
        np.testing.assert_allclose(out[0], np.clip(value, 0, 1) * 2)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_mix_and_mad_are_the_complex_arithmetic(runner, dtype):
    """`mix(z, w, 0.5)` blends and `mad(z, w, v)` is a product plus a sum."""
    z, w, v = dtype(1 + 2j), dtype(3 - 1j), dtype(0.5 + 0.5j)
    out = np.zeros(1, dtype=dtype)
    runner(builtin_mix_mad, z, w, v, out)
    runner(numpy_mix_mad, z, w, v, out)
    expected = (z * (1 - 0.5) + w * 0.5) + (z * w + v)
    np.testing.assert_allclose(out[0], expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_predicates_answer_for_either_part(runner, dtype):
    """`np.isnan`/`isinf`/`isfinite` are defined on a complex value in numpy."""
    out = np.zeros(1, dtype=np.float32)
    for value in (complex(np.nan, 1.0), complex(1.0, np.inf), dtype(1 + 2j),
                  complex(np.inf, np.nan)):
        runner(numpy_predicates, dtype(value), out)
        assert int(out[0]) == _predicate_mask(value), value


# ------------------------------------------------------- refused by name, at translation
#
# python's `round` and the real-only functions have no complex form in the interpreter either,
# so the translator refuses them and says what to write. These need no device: the refusal
# happens while the body is translated, before anything is compiled.


@kernel
def refuses_round(z, out):
    for i in range(out.shape[0]):
        out[i] = round(z)


@kernel
def refuses_rsqrt(z, out):
    for i in range(out.shape[0]):
        out[i] = rsqrt(z)


@kernel
def refuses_fract(z, out):
    for i in range(out.shape[0]):
        out[i] = fract(z)


@kernel
def refuses_step(z, w, out):
    for i in range(out.shape[0]):
        out[i] = step(z, w)


@kernel
def refuses_floor(z, out):
    for i in range(out.shape[0]):
        out[i] = floor(z)


@kernel
def refuses_hypot(z, w, out):
    for i in range(out.shape[0]):
        out[i] = hypot(z, w)


@pytest.mark.parametrize("function, expected", [
    (refuses_round, "use `np.round(z)`"),
    (refuses_rsqrt, "use `1 / np.sqrt(z)`"),
    (refuses_fract, "np.floor"),
    (refuses_step, "no ordering"),
    (refuses_floor, "no ordering"),
    (refuses_hypot, "np.abs(a - b)"),
], ids=["round", "rsqrt", "fract", "step", "floor", "hypot"])
def test_a_real_only_builtin_is_refused_for_a_complex_value(function, expected):
    """The refusal names the operation to write instead of a compiler's type error."""
    from pygpgpu.kernel.decorator import infer_ctype
    from pygpgpu.kernel.translator import FunctionTranslator

    complex_scalar = infer_ctype(1 + 2j)
    complex_array = infer_ctype(np.zeros(2, np.complex64))
    parameters: int = len(function.raw_func.__code__.co_varnames[:function.raw_func.__code__.co_argcount])
    types = [complex_scalar] * (parameters - 1) + [complex_array]
    translator = FunctionTranslator(function.raw_func, types, "cuda",
                                    lambda name: globals().get(name), {}, is_kernel=True)
    with pytest.raises(SyntaxError) as caught:
        translator.translate()
    assert expected in str(caught.value)
    assert "complex" in str(caught.value)
