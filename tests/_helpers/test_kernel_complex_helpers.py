"""The cases and the names the complex-number tests share.

A parametrized case is the operation a test runs, so it moves with the test
that uses it; what is here is what more than one test reads.
"""
from __future__ import annotations

from __future__ import annotations
import numpy as np
from pygf import conjugate
from pygpgpu import (clamp, exp, exp2, fmax, fmin, floor, fract, hypot, kernel, log, mad,
                     mix, rint, rsqrt, saturate, sign, sin, sqrt, step)


@kernel
def add(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def add_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + 2.0


@kernel
def builtin_clamp_saturate(z, out):
    for i in range(out.shape[0]):
        out[i] = clamp(z, 0, 1) + saturate(z)


@kernel
def builtin_fmin_fmax(z, w, out):
    for i in range(out.shape[0]):
        out[i] = fmin(z, w) + fmax(z, w)


@kernel
def builtin_min_max(z, w, out):
    for i in range(out.shape[0]):
        out[i] = min(z, w) + max(z, w)


@kernel
def builtin_mix_mad(z, w, v, out):
    for i in range(out.shape[0]):
        out[i] = mix(z, w, 0.5) + mad(z, w, v)


@kernel
def builtin_rint(z, out):
    for i in range(out.shape[0]):
        out[i] = rint(z)


@kernel
def builtin_sign(z, out):
    for i in range(out.shape[0]):
        out[i] = sign(z)


@kernel
def builtin_transcendental(z, out):
    for i in range(out.shape[0]):
        out[i] = sqrt(z) + exp(z) + log(z) + sin(z) + exp2(z)


@kernel(float=np.float64)
def calls_double_helper(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = python_complex_literal_helper(a[i])


@kernel
def calls_plain_helper(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = python_complex_literal_helper(a[i])


@kernel
def complex_acos(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arccos(a[i])


@kernel
def complex_asin(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arcsin(a[i])


@kernel
def complex_atan(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arctan(a[i])


@kernel
def complex_cos(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.cos(a[i])


@kernel
def complex_cosh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.cosh(a[i])


@kernel
def complex_exp(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.exp(a[i])


@kernel
def complex_exp2(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.exp2(a[i])


@kernel
def complex_log(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log(a[i])


@kernel
def complex_log10(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log10(a[i])


@kernel
def complex_log2(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log2(a[i])


@kernel
def complex_reciprocal(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.reciprocal(a[i])


@kernel
def complex_sin(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sin(a[i])


@kernel
def complex_sinh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sinh(a[i])


@kernel
def complex_sqrt(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sqrt(a[i])


@kernel
def complex_square(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.square(a[i])


@kernel
def complex_tan(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.tan(a[i])


@kernel
def complex_tanh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.tanh(a[i])


@kernel
def conjugate_call(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = conjugate(a[i])


@kernel
def conjugate_method(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i].conjugate()


@kernel
def constructed_from_args(re: np.float32, im: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(re, im)


@kernel
def constructed_from_complex(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(a[i])


@kernel
def constructed_from_constants(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(3, 4)


@kernel
def constructed_from_real(value: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(value)


@kernel
def constructed_from_string(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex("3+4j")


@kernel
def constructed_then_narrowed(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.complex64(complex(3, 4))


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
def cube(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] ** 3


@kernel
def div_by_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / 2.0


@kernel
def divide(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / b[i]


@kernel(float=np.float64)
def double_complex_argument(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value


@kernel
def equality(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] == b[i] else 0.0


@kernel
def equality_with_real(a: np.ndarray, value: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] == value else 0.0


@kernel
def identity(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = +a[i]


@kernel
def less_than(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] < b[i] else 0.0


@kernel
def magnitude(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = abs(a[i])


@kernel
def mixed_add(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def multiply(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * b[i]


@kernel
def multiply_by_scope(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * SCOPE_128


@kernel
def negate(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = -a[i]


@kernel
def numpy_clip(z, out):
    for i in range(out.shape[0]):
        out[i] = np.clip(z, 0, 1) + np.clip(z, 0, 1)


@kernel
def numpy_minimum_maximum(z, w, out):
    for i in range(out.shape[0]):
        out[i] = np.minimum(z, w) + np.maximum(z, w)


@kernel
def numpy_mix_mad(z, w, v, out):
    for i in range(out.shape[0]):
        out[i] = (z * (1 - 0.5) + w * 0.5) + (z * w + v)


@kernel
def numpy_predicates(z, flags):
    """The three predicates as a bitmask, so one run can tell them apart."""
    for i in range(flags.shape[0]):
        flags[i] = np.isnan(z) * 4 + np.isinf(z) * 2 + np.isfinite(z)


@kernel
def numpy_rint(z, out):
    for i in range(out.shape[0]):
        out[i] = np.rint(z)


@kernel
def numpy_round(z, out):
    for i in range(out.shape[0]):
        out[i] = np.round(z)


@kernel
def numpy_sign(z, out):
    for i in range(out.shape[0]):
        out[i] = np.sign(z)


@kernel
def numpy_transcendental(z, out):
    for i in range(out.shape[0]):
        out[i] = np.sqrt(z) + np.exp(z) + np.log(z) + np.sin(z) + np.exp2(z)


@kernel
def passed_scalar(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value


@kernel
def phase(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.angle(a[i])


@kernel
def pow_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] ** 2.0


@kernel
def python_constructed(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(3, 4)


@kernel
def python_literal(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 3 + 4j


@kernel
def rdiv_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 2.0 / a[i]


@kernel
def read_members(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i, 0] = a[i].real
        out[i, 1] = a[i].imag


@kernel
def read_scope_complex128(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_128


@kernel
def read_scope_complex64(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_64


@kernel
def read_scope_python(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_PY


@kernel
def refuses_floor(z, out):
    for i in range(out.shape[0]):
        out[i] = floor(z)


@kernel
def refuses_fract(z, out):
    for i in range(out.shape[0]):
        out[i] = fract(z)


@kernel
def refuses_hypot(z, w, out):
    for i in range(out.shape[0]):
        out[i] = hypot(z, w)


@kernel
def refuses_round(z, out):
    for i in range(out.shape[0]):
        out[i] = round(z)


@kernel
def refuses_rsqrt(z, out):
    for i in range(out.shape[0]):
        out[i] = rsqrt(z)


@kernel
def refuses_step(z, w, out):
    for i in range(out.shape[0]):
        out[i] = step(z, w)


@kernel
def return_scope(out: np.ndarray):
    return SCOPE_128


@kernel
def return_scope_complex64(out: np.ndarray):
    return SCOPE_64


@kernel
def returned_computed(a: np.ndarray):
    for i in range(a.shape[0]):
        return a[i] ** 2


@kernel
def returned_constant(out: np.ndarray):
    return 3 + 4j


@kernel
def returned_element(a: np.ndarray):
    for i in range(a.shape[0]):
        return a[i]


@kernel
def rpow_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 2.0 ** a[i]


@kernel
def rsub_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 2.0 - a[i]


@kernel
def scalar_scale(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value * 3.0


@kernel
def scale(a: np.ndarray, factor: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * factor


@kernel
def scale_by_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0


@kernel
def scale_div(a: np.ndarray, factor: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / factor


@kernel
def square_by_operator(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] ** 2


@kernel
def sub_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] - 2.0


@kernel
def subtract(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] - b[i]


FLOAT = np.complex64


DOUBLE = np.complex128


def values(dtype):
    """Two complex values of the given width, with both parts non-zero."""
    return np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=dtype)


def other(dtype):
    """Two more complex values, one of them purely imaginary."""
    return np.array([5.0 + 0.0j, 0.0 + 1.0j], dtype=dtype)


FLOAT_SCALAR_FORMS = [
    ("scale_by_float", lambda a: a * 2.0),
    ("add_float", lambda a: a + 2.0),
    ("sub_float", lambda a: a - 2.0),
    ("div_by_float", lambda a: a / 2.0),
    ("pow_float", lambda a: a ** 2.0),
    ("rsub_float", lambda a: 2.0 - a),
    ("rdiv_float", lambda a: 2.0 / a),
    ("rpow_float", lambda a: 2.0 ** a),
]


COMPLEX_FUNCTIONS = {
    "sqrt": complex_sqrt, "exp": complex_exp, "log": complex_log,
    "log2": complex_log2, "log10": complex_log10,
    "sin": complex_sin, "cos": complex_cos, "tan": complex_tan,
    "asin": complex_asin, "acos": complex_acos, "atan": complex_atan,
    "sinh": complex_sinh, "cosh": complex_cosh, "tanh": complex_tanh,
    "square": complex_square, "reciprocal": complex_reciprocal,
    "exp2": complex_exp2,
}


SCOPE_64 = np.complex64(3 + 4j)


SCOPE_128 = np.complex128(3 + 4j)


SCOPE_PY = 3 + 4j


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


def python_complex_literal_helper(value):
    """An ordinary helper whose body builds a python complex literal of its own.

    A helper is translated into the *kernel's* translation unit and monomorphized per call
    site, so it has to make the same width choice its caller does -- otherwise a helper body
    silently answered the 32-bit default while the kernel around it was mapped to a double.
    """
    return value + 0.2j


def _generated_source(func, type_map=None):
    """The device source a one-array kernel renders, without compiling it."""
    from pygpgpu.kernel.translator import FunctionTranslator, infer_ctype

    raw = func.raw_func if hasattr(func, "raw_func") else func
    out = infer_ctype(np.zeros(2, np.complex64))
    translator = FunctionTranslator(raw, [out], "cuda", lambda name: None, {},
                                    is_kernel=True, type_map=type_map)
    translator.translate()
    return "\n\n".join(source for _, source in translator.render_kernels("k_test"))


def _predicate_mask(value) -> int:
    return int(np.isnan(value)) * 4 + int(np.isinf(value)) * 2 + int(np.isfinite(value))
