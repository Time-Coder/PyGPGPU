"""The kernels the scientific-vocabulary tests launch.

They live in a real module because a kernel's source has to be readable (`inspect.
getsource`), which a function built with `exec` does not have.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygpgpu.kernel.builtins import (
    erf, erfc, tgamma, lgamma, fmin, fmax, fract, sign, copysign, fdim,
    remainder, nextafter, ldexp, fast_exp, fast_log, fast_sin, fast_cos,
    exact_recip, exact_sqrt, normal_cdf, reciprocal_hypot, sin_pi, cos_pi,
    float_to_int_nearest,
)


@kernel
def probe_erf(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = erf(x[i])


@kernel
def probe_erfc(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = erfc(x[i])


@kernel
def probe_tgamma(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = tgamma(x[i])


@kernel
def probe_lgamma(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = lgamma(x[i])


@kernel
def probe_fmin(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = fmin(x[i], 1.0)


@kernel
def probe_fmax(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = fmax(x[i], 0.0)


@kernel
def probe_fract(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = fract(x[i])


@kernel
def probe_sign(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = sign(x[i])


@kernel
def probe_copysign(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = copysign(x[i], -1.0)


@kernel
def probe_fdim(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = fdim(x[i], 1.0)


@kernel
def probe_remainder(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = remainder(x[i], 3.0)


@kernel
def probe_nextafter(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = nextafter(x[i], 2.0)


@kernel
def probe_ldexp(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = ldexp(x[i], 2)


@kernel
def probe_fast_exp(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = fast_exp(x[i])


@kernel
def probe_fast_log(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = fast_log(x[i])


@kernel
def probe_fast_sin(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = fast_sin(x[i])


@kernel
def probe_fast_cos(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = fast_cos(x[i])


@kernel
def probe_exact_recip(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = exact_recip(x[i])


@kernel
def probe_exact_sqrt(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = exact_sqrt(x[i])


@kernel
def probe_normal_cdf(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = normal_cdf(x[i])


@kernel
def probe_reciprocal_hypot(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = reciprocal_hypot(x[i], 2.0)


@kernel
def probe_sin_pi(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = sin_pi(x[i])


@kernel
def probe_cos_pi(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = cos_pi(x[i])


@kernel
def probe_float_to_int_nearest(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float_to_int_nearest(x[i])


@kernel
def probe_vector_argument(vectors: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = erf(vectors[i])


# The kernel each name is launched through, so a test can name the function and get the
# kernel that calls it.
PROBES = {
    "erf": probe_erf,
    "erfc": probe_erfc,
    "tgamma": probe_tgamma,
    "lgamma": probe_lgamma,
    "fmin": probe_fmin,
    "fmax": probe_fmax,
    "fract": probe_fract,
    "sign": probe_sign,
    "copysign": probe_copysign,
    "fdim": probe_fdim,
    "remainder": probe_remainder,
    "nextafter": probe_nextafter,
    "ldexp": probe_ldexp,
    "fast_exp": probe_fast_exp,
    "fast_log": probe_fast_log,
    "fast_sin": probe_fast_sin,
    "fast_cos": probe_fast_cos,
    "exact_recip": probe_exact_recip,
    "exact_sqrt": probe_exact_sqrt,
    "normal_cdf": probe_normal_cdf,
    "reciprocal_hypot": probe_reciprocal_hypot,
    "sin_pi": probe_sin_pi,
    "cos_pi": probe_cos_pi,
}
