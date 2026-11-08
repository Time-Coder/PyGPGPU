from __future__ import annotations

import numpy as np
from pygpgpu import (clamp, exp, exp2, fmax, fmin, floor, fract, hypot, kernel, log, mad,
                     mix, rint, rsqrt, saturate, sign, sin, sqrt, step)



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


@kernel
def builtin_rint(z, out):
    for i in range(out.shape[0]):
        out[i] = rint(z)


@kernel
def builtin_fmin_fmax(z, w, out):
    for i in range(out.shape[0]):
        out[i] = fmin(z, w) + fmax(z, w)


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
