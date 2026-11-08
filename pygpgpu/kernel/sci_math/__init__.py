"""The host side of the CUDA scientific-computing vocabulary.

Every name here is a function a kernel body may call, and each one has a device
implementation too (see ``device_builtins.py``): the table there says how the call is
emitted per backend, and the bodies below are what the ``python`` backend runs.

Two groups live here:

* the functions **both** backends have -- OpenCL C 1.2 and CUDA agree on them, so a kernel
  that uses one runs anywhere;
* the functions **only CUDA has** -- the fast approximations (``__expf`` and friends), the
  correctly-rounded forms, and the special functions CUDA adds. A kernel that uses one of
  those is a CUDA kernel, and the translator says so by name when the backend is OpenCL.

The host implementations are the mathematical definitions, so the interpreter backend and a
device agree to the last bit on the exact functions and to within the library's own
accuracy on the approximate ones (``fast_exp`` is CUDA's ``__expf``, which trades accuracy
for speed -- the host's ``math.exp`` is the accurate one, and the difference is the
device's, not the model's)."""
# This module is the facade of a split: the scientific vocabulary moved to the
# module named after its family, and every name stays importable from here -- the
# translator reads them off this module, and the tests import them by name.
from __future__ import annotations



__all__ = [
    # both backends (the ones pygf does not publish are defined here; the ones it does
    # publish -- fma, ldexp, fract, sign -- are re-exported from it, so there is exactly
    # one implementation of each name)
    "erf", "erfc", "tgamma", "lgamma", "rint", "remainder", "fmin", "fmax",
    "fdim", "copysign", "nextafter",
    "native_recip", "native_divide", "native_sqrt", "native_rsqrt", "native_exp",
    "native_log", "native_sin", "native_cos", "native_tan", "native_exp2", "native_log2",
    "native_exp10", "native_powr",
    # CUDA only
    "fast_exp", "fast_log", "fast_log2", "fast_log10", "fast_sin", "fast_cos",
    "fast_tan", "fast_exp10", "fast_pow", "fast_divide",
    "exact_recip", "exact_recip_rz", "exact_divide", "exact_divide_rz",
    "exact_sqrt", "exact_sqrt_rz", "exact_add", "exact_sub", "exact_mul", "exact_fma",
    "erf_inverse", "normal_cdf", "normal_cdf_inverse", "reciprocal_hypot",
    "sin_pi", "cos_pi",
    "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1", "bessel_i0", "bessel_i1",
    "float_to_int_nearest", "float_to_int_zero", "float_to_int_down", "float_to_int_up",
    "int_to_float", "uint_to_float",
    "double_to_int_nearest", "double_to_int_zero", "double_to_int_down", "double_to_int_up",
]


from .special import (
    erf,
    erfc,
    lgamma,
    remainder,
    rint,
    tgamma,
)
from .minmax import (
    _complex_fminmax,
    _complex_less,
    _is_complex,
    _scalar_only,
    copysign,
    fdim,
    fmax,
    fmin,
    nextafter,
)
from .native import (
    native_cos,
    native_divide,
    native_exp,
    native_exp10,
    native_exp2,
    native_log,
    native_log2,
    native_powr,
    native_recip,
    native_rsqrt,
    native_sin,
    native_sqrt,
    native_tan,
)
from .fast import (
    fast_cos,
    fast_divide,
    fast_exp,
    fast_exp10,
    fast_log,
    fast_log10,
    fast_log2,
    fast_pow,
    fast_sin,
    fast_tan,
)
from .exact import (
    exact_add,
    exact_divide,
    exact_divide_rz,
    exact_fma,
    exact_mul,
    exact_recip,
    exact_recip_rz,
    exact_sqrt,
    exact_sqrt_rz,
    exact_sub,
)
from .distribution import (
    cos_pi,
    erf_inverse,
    normal_cdf,
    normal_cdf_inverse,
    reciprocal_hypot,
    sin_pi,
)
from .bessel import (
    _bessel_i,
    _bessel_j,
    _bessel_y,
    bessel_i0,
    bessel_i1,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
)
from .conversion import (
    double_to_int_down,
    double_to_int_nearest,
    double_to_int_up,
    double_to_int_zero,
    float_to_int_down,
    float_to_int_nearest,
    float_to_int_up,
    float_to_int_zero,
    int_to_float,
    uint_to_float,
)
