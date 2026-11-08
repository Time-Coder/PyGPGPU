from __future__ import annotations

import numpy as np
from pygpgpu import (kernel)



@kernel
def calls_plain_helper(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = python_complex_literal_helper(a[i])


@kernel(float=np.float64)
def calls_double_helper(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = python_complex_literal_helper(a[i])


@kernel(float=np.float64)
def double_complex_argument(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value


def values(dtype):
    """Two complex values of the given width, with both parts non-zero."""
    return np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=dtype)


def other(dtype):
    """Two more complex values, one of them purely imaginary."""
    return np.array([5.0 + 0.0j, 0.0 + 1.0j], dtype=dtype)


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
