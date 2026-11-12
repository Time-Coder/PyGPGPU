"""The 8/16-wide vectors are OpenCL storage types, not kernel types.

A CUDA or HIP kernel has no `float8`/`float16` to translate into -- the widest
vector C for those backends declares is four components -- and a type only one
backend declares would make the same body mean different things per device. So
a device launch refuses the width at every spelling (parameter, constructor,
component, swizzle, string subscript) with a message that names the
replacements, and the classes themselves live beside the only backend that
declares them (`pygpgpu.opencl.driver.vec_types`).

The interpreter backend is NOT part of the refusal: it executes the body as
plain python -- no translation, no device types -- and any legal python goes.

The bodies here reach `float8` the way a user's module would after the move:
imported at the top of the module that defines the kernel, so the name is in
the kernel's globals.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygpgpu.opencl.driver.vec_types import float8


@kernel
def takes_wide(out: np.ndarray, v):
    out[0] = v.s0


@kernel
def builds_wide(out: np.ndarray):
    v = float8(1, 2, 3, 4, 5, 6, 7, 8)
    out[0] = v.s0


@kernel
def reads_component(out: np.ndarray):
    v = float8(1, 2, 3, 4, 5, 6, 7, 8)
    out[0] = v.s0


@kernel
def reads_swizzle(out: np.ndarray):
    v = float8(1, 2, 3, 4, 5, 6, 7, 8)
    out[0] = v.s0123[1]


@kernel
def subscripts_wide(out: np.ndarray):
    v = float8(1, 2, 3, 4, 5, 6, 7, 8)
    out[0] = v["s3"]
