"""The classes and the helpers the split test files share.

A class is the SAME class whichever group uses it -- two copies would
be two classes, and the tests compare them.
"""
from __future__ import annotations

from __future__ import annotations
from typing import Any
import numpy as np
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.translator import FunctionTranslator


class _NoDeviceProgram:
    """Just enough of a Program for a MetalKernel to be built without a device.

    A `MetalKernel` needs a program object for `context`/`device`/`file_name`/`id`, and
    every test that exercises the kernel's own logic (dispatch shape, the arg-info
    invariant, scalar packing) builds one. Kept in one place so a field the class
    grows does not have to be added in three copies.
    """

    context = None
    device = None
    file_name = "x.metal"
    id = 0


def _translate(func: Any, n_args: int) -> FunctionTranslator:
    types = [infer_ctype(np.zeros(8, np.float32)) for _ in range(n_args)]
    translator = FunctionTranslator(func, types, "metal", lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    return translator


def complex_scalar(value, out):
    for i in range(out.shape[0]):
        out[i] = value


def mixed_factor_kernel(x, out, factor):
    for i in range(out.shape[0]):
        out[i] = x[i] * factor


def two_phase(x, mid, out):
    """Two magic loops: translated into two phase kernels in one source."""
    for i in range(out.shape[0]):
        mid[i] = x[i] * 2.0
    for i in range(out.shape[0]):
        out[i] = mid[i] + 1.0


def vecadd(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]
