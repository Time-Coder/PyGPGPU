from __future__ import annotations

import numpy as np
from pygpgpu import (kernel)



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


@kernel
def python_literal(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 3 + 4j


@kernel
def python_constructed(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = complex(3, 4)
