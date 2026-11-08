# What the device has, keyed by backend: the shared vocabulary and the CUDA-only one.
from __future__ import annotations
from .floor import (
    HIP_ROUND_TOWARD_ZERO_HELPERS,
)
from typing import Dict, Tuple



# Math/vector builtins: (gpu name, opencl name, cuda name or None meaning special codegen is needed)
GENERIC_BUILTINS:Dict[str, Tuple[str, str]] = {
    "barrier": ("barrier", "__syncthreads()"),
    "rsqrt": ("rsqrt", "rsqrtf"),
    "exp2": ("exp2", "exp2f"),
    "clamp": ("clamp", ""),      # cuda needs special codegen
    "mix": ("mix", ""),          # cuda needs special codegen (a*(1-t)+b*t)
    "mad": ("mad", ""),          # cuda (a*b+c)
    "saturate": ("clamp", ""),   # cuda needs special codegen (clamp to 0..1)
    "step": ("step", ""),        # cuda needs special codegen (x < edge ? 0 : 1)
    "select": ("select", ""),    # cuda needs special codegen (per-component cond ? b : a)
    "any": ("any", ""),          # vector reduction, cuda needs special codegen
    "all": ("all", ""),          # vector reduction, cuda needs special codegen
    "isequal": ("isequal", ""),  # relation-function family: opencl natively returns int vectors,
    "isnotequal": ("isnotequal", ""),  # cuda generates per-component (int)(l op r)
    "isgreater": ("isgreater", ""),
    "isgreaterequal": ("isgreaterequal", ""),
    "isless": ("isless", ""),
    "islessequal": ("islessequal", ""),
    "isordered": ("isordered", ""),
    "isunordered": ("isunordered", ""),
    "abs_diff": ("abs_diff", ""),  # integer: cuda per-component abs(l-r)
    "hadd": ("hadd", ""),          # integer: cuda (a&b) + ((a^b)>>1)
    "rhadd": ("rhadd", ""),        # integer: cuda (a|b) - ((a^b)>>1)
    "mad24": ("mad24", ""),        # integer: cuda plain multiply-add
    "mul24": ("mul24", ""),        # integer: cuda plain multiply
    "dot": ("dot", ""),          # vector: opencl native, cuda per-component sum
    "cross": ("cross", ""),
    "normalize": ("normalize", ""),
    "length": ("length", ""),
    "distance": ("distance", ""),
}



# The scalar math intrinsics. OpenCL has a native vector overload for every one of
# them; CUDA has only the scalar, `f`-suffixed spelling, so the translator expands a
# vector argument per component (see FunctionTranslator._emit_generic_builtin).
MATH_BUILTINS:Dict[str, Tuple[str, str]] = {
    "sqrt": ("sqrt", "sqrtf"),
    "sin": ("sin", "sinf"),
    "cos": ("cos", "cosf"),
    "tan": ("tan", "tanf"),
    "asin": ("asin", "asinf"),
    "acos": ("acos", "acosf"),
    "atan": ("atan", "atanf"),
    "atan2": ("atan2", "atan2f"),
    "sinh": ("sinh", "sinhf"),
    "cosh": ("cosh", "coshf"),
    "tanh": ("tanh", "tanhf"),
    "exp": ("exp", "expf"),
    "log": ("log", "logf"),
    "log2": ("log2", "log2f"),
    "log10": ("log10", "log10f"),
    "pow": ("pow", "powf"),
    "floor": ("floor", "floorf"),
    "ceil": ("ceil", "ceilf"),
    "trunc": ("trunc", "truncf"),
    "fabs": ("fabs", "fabsf"),
    "fmod": ("fmod", "fmodf"),
    "hypot": ("hypot", "hypotf"),
    "cbrt": ("cbrt", "cbrtf"),
}



# The scalar math functions OpenCL C 1.2 has and CUDA spells with an `f` suffix. Both
# backends have every one of them, so they are dual-sided; the table only records the CUDA
# spelling (the OpenCL one is the name itself).
#
# The list is what the two compilers actually accept, verified by compiling each call:
# OpenCL C 1.2 has erf/erfc/tgamma/lgamma/rint/round/trunc/fmin/fmax/fma/fdim/copysign/
# nextafter/ldexp/remainder/remquo/frexp/modf/mad/fract/sign and the `native_*` family;
# CUDA has all of those except fract and sign, which it spells as an expression (there is
# no `fractf` and no `signf`), so those two carry a code template instead of a name.
# The scalar math functions all three backends have. The table records the spelling each
# backend uses, because the three are not the same: OpenCL C 1.2 has erf/erfc/tgamma/lgamma/
# rint/round/trunc/fmin/fmax/fma/fdim/copysign/nextafter/ldexp/remainder/remquo/frexp/modf/
# mad/fract/sign and the `native_*` family; CUDA has all of those except fract and sign,
# which it spells as an expression (there is no `fractf` and no `signf`), so those two carry
# a code template instead of a name; and HIP has the `f`-suffixed forms of the same set,
# with the `native_*` family under CUDA's double-underscore spellings.
SHARED_MATH_BUILTINS:Dict[str, Dict[str, str]] = {
    "erf": {"opencl": "erf", "cuda": "erff", "hip": "erff", "metal": "erf"},
    "erfc": {"opencl": "erfc", "cuda": "erfcf", "hip": "erfcf", "metal": "erfc"},
    "tgamma": {"opencl": "tgamma", "cuda": "tgammaf", "hip": "tgammaf", "metal": "tgamma"},
    "lgamma": {"opencl": "lgamma", "cuda": "lgammaf", "hip": "lgammaf", "metal": "lgamma"},
    "rint": {"opencl": "rint", "cuda": "rintf", "hip": "rintf", "metal": "rint"},
    "remainder": {"opencl": "remainder", "cuda": "remainderf", "hip": "remainderf", "metal": "remainder"},
    "fmin": {"opencl": "fmin", "cuda": "fminf", "hip": "fminf", "metal": "fmin"},
    "fmax": {"opencl": "fmax", "cuda": "fmaxf", "hip": "fmaxf", "metal": "fmax"},
    "fdim": {"opencl": "fdim", "cuda": "fdimf", "hip": "fdimf", "metal": "fdim"},
    "copysign": {"opencl": "copysign", "cuda": "copysignf", "hip": "copysignf", "metal": "copysign"},
    "nextafter": {"opencl": "nextafter", "cuda": "nextafterf", "hip": "nextafterf", "metal": "nextafter"},
    "ldexp": {"opencl": "ldexp", "cuda": "ldexpf", "hip": "ldexpf", "metal": "ldexp"},
    # `native_recip` is deliberately absent: it has no Metal spelling as a callee
    # (MSL's reciprocal is the division `1.0f / x`, a template -- see the table
    # below), and putting an expression in a table of function names would make
    # `_shared_math_call` emit `1.0f / (x)(...)`.
    "native_divide": {"opencl": "native_divide", "cuda": "__fdiv_rn", "hip": "__fdiv_rn", "metal": "fast::divide"},
    "native_sqrt": {"opencl": "native_sqrt", "cuda": "__fsqrt_rn", "hip": "__fsqrt_rn", "metal": "fast::sqrt"},
    "native_rsqrt": {"opencl": "native_rsqrt", "cuda": "__frsqrt_rn", "hip": "__frsqrt_rn", "metal": "fast::rsqrt"},
    "native_exp": {"opencl": "native_exp", "cuda": "__expf", "hip": "__expf", "metal": "fast::exp"},
    "native_log": {"opencl": "native_log", "cuda": "__logf", "hip": "__logf", "metal": "fast::log"},
    "native_sin": {"opencl": "native_sin", "cuda": "__sinf", "hip": "__sinf", "metal": "fast::sin"},
    "native_cos": {"opencl": "native_cos", "cuda": "__cosf", "hip": "__cosf", "metal": "fast::cos"},
    "native_tan": {"opencl": "native_tan", "cuda": "__tanf", "hip": "__tanf", "metal": "fast::tan"},
    "native_exp2": {"opencl": "native_exp2", "cuda": "__exp2f", "hip": "exp2f", "metal": "fast::exp2"},
    "native_log2": {"opencl": "native_log2", "cuda": "__log2f", "hip": "__log2f", "metal": "fast::log2"},
    "native_exp10": {"opencl": "native_exp10", "cuda": "__exp10f", "hip": "__exp10f", "metal": "fast::exp10"},
    "native_powr": {"opencl": "native_powr", "cuda": "__powf", "hip": "__powf", "metal": "fast::powr"},
}



# The functions the two backends spell differently, as a code template per backend.
#
# `fract` is the clearest case: OpenCL's `fract` takes a second argument (it writes the
# integer part through a pointer) and CUDA has no `fractf` at all, so both spell it as
# `x - floor(x)`. `sign` is the same story -- OpenCL has no one-argument `sign`, and CUDA
# has no `signf`, so both compare against zero.
SHARED_MATH_TEMPLATES:Dict[str, Dict[str, str]] = {
    "fract": {
        "opencl": "(({code}) - floor(({code})))",
        "cuda": "(({code}) - floorf(({code})))",
        "hip": "(({code}) - floorf(({code})))",
        "metal": "(({code}) - floor(({code})))",
    },
    "sign": {
        "opencl": "((({code}) > 0.0f) ? 1.0f : (({code}) < 0.0f) ? -1.0f : 0.0f)",
        "cuda": "((({code}) > 0.0f) ? 1.0f : (({code}) < 0.0f) ? -1.0f : 0.0f)",
        "hip": "((({code}) > 0.0f) ? 1.0f : (({code}) < 0.0f) ? -1.0f : 0.0f)",
        "metal": "((({code}) > 0.0f) ? 1.0f : (({code}) < 0.0f) ? -1.0f : 0.0f)",
    },
    # `native_recip` is a function call on all three other backends and a division
    # on Metal: MSL has no reciprocal intrinsic, so the operation is written as
    # what it is. A template is the honest place for that -- the function tables
    # hold callee names, and an expression there would be emitted as `1.0f / (x)(…)`.
    "native_recip": {
        "opencl": "native_recip({code})",
        "cuda": "__frcp_rn({code})",
        "hip": "__frcp_rn({code})",
        "metal": "(1.0f / ({code}))",
    },
}



# The device intrinsics CUDA has and OpenCL does not. They are the CUDA-only half of the
# scientific-computing vocabulary: the fast approximations (`__expf` and friends, which
# trade accuracy for speed), the correctly-rounded forms (`__frcp_rn` and friends), and the
# special functions CUDA adds (the error function's inverse, the normal distribution, the
# Bessel functions).
#
# HIP reuses the CUDA code generation and has most of these intrinsics too, so "CUDA-only"
# here means "not OpenCL": OpenCL's kernel language has no counterpart for any of them.
# The table therefore carries a spelling PER BACKEND, because the two are not the same:
#
#   * HIP has no `__exp2f` but has the ordinary `exp2f`;
#   * HIP has none of the round-toward-zero intrinsics (`__frcp_rz`, `__fdiv_rz`,
#     `__fsqrt_rz`) -- not the `_rn` ones either, only those. Those three are spelled
#     here by a device helper (see HIP_ROUND_TOWARD_ZERO_HELPERS), because a helper can
#     reproduce the exact rounding the intrinsic does, while a plain expression cannot.
#
# A kernel that uses one of them cannot run on OpenCL, so the translator says so by name
# rather than emitting a call the OpenCL compiler would reject with a bare "undeclared
# identifier".
#
# Two of the names here do NOT mean what their spelling suggests, and the device's own
# documentation is the authority:
#   * `sinpif(x)` is sin(pi*x), NOT the sinc function sin(pi*x)/(pi*x) -- CUDA has no
#     sinc, so the name here is `sin_pi` and it computes the sine of pi times x;
#   * `rhypotf(x, y)` is 1/sqrt(x*x + y*y), the *reciprocal* of the hypotenuse, so the
#     name here is `reciprocal_hypot`. The plain hypotenuse is `hypotf`, which is in the
#     shared table.
CUDA_ONLY_BUILTINS:Dict[str, Dict[str, str]] = {
    # the fast (approximate) intrinsics
    "fast_exp": {"cuda": "__expf", "hip": "__expf"},
    "fast_log": {"cuda": "__logf", "hip": "__logf"},
    "fast_log2": {"cuda": "__log2f", "hip": "__log2f"},
    "fast_log10": {"cuda": "__log10f", "hip": "__log10f"},
    "fast_sin": {"cuda": "__sinf", "hip": "__sinf"},
    "fast_cos": {"cuda": "__cosf", "hip": "__cosf"},
    "fast_tan": {"cuda": "__tanf", "hip": "__tanf"},
    "fast_exp10": {"cuda": "__exp10f", "hip": "__exp10f"},
    "fast_pow": {"cuda": "__powf", "hip": "__powf"},
    "fast_divide": {"cuda": "__fdividef", "hip": "__fdividef"},
    # the correctly-rounded forms
    "exact_recip": {"cuda": "__frcp_rn", "hip": "__frcp_rn"},
    "exact_recip_rz": {"cuda": "__frcp_rz", "hip": HIP_ROUND_TOWARD_ZERO_HELPERS["recip"]},
    "exact_divide": {"cuda": "__fdiv_rn", "hip": "__fdiv_rn"},
    "exact_divide_rz": {"cuda": "__fdiv_rz", "hip": HIP_ROUND_TOWARD_ZERO_HELPERS["divide"]},
    "exact_sqrt": {"cuda": "__fsqrt_rn", "hip": "__fsqrt_rn"},
    "exact_sqrt_rz": {"cuda": "__fsqrt_rz", "hip": HIP_ROUND_TOWARD_ZERO_HELPERS["sqrt"]},
    "exact_add": {"cuda": "__fadd_rn", "hip": "__fadd_rn"},
    "exact_sub": {"cuda": "__fsub_rn", "hip": "__fsub_rn"},
    "exact_mul": {"cuda": "__fmul_rn", "hip": "__fmul_rn"},
    "exact_fma": {"cuda": "__fmaf_rn", "hip": "__fmaf_rn"},
    # the special functions
    "erf_inverse": {"cuda": "erfinv", "hip": "erfinv"},
    "normal_cdf": {"cuda": "normcdff", "hip": "normcdff"},
    "normal_cdf_inverse": {"cuda": "normcdfinv", "hip": "normcdfinv"},
    "reciprocal_hypot": {"cuda": "rhypotf", "hip": "rhypotf"},
    "sin_pi": {"cuda": "sinpif", "hip": "sinpif"},
    "cos_pi": {"cuda": "cospif", "hip": "cospif"},
    "bessel_j0": {"cuda": "j0f", "hip": "j0f"},
    "bessel_j1": {"cuda": "j1f", "hip": "j1f"},
    "bessel_y0": {"cuda": "y0f", "hip": "y0f"},
    "bessel_y1": {"cuda": "y1f", "hip": "y1f"},
    "bessel_i0": {"cuda": "cyl_bessel_i0f", "hip": "cyl_bessel_i0f"},
    "bessel_i1": {"cuda": "cyl_bessel_i1f", "hip": "cyl_bessel_i1f"},
}



# The conversions CUDA and HIP have and OpenCL does not. They answer an int, so the
# device's own name is spelled with the rounding mode in the middle (`__float2int_rn`
# rounds to nearest, `_rz` toward zero, `_rd` down, `_ru` up). HIP spells them the same
# way -- it has every one of them.
CUDA_ONLY_CONVERSIONS:Dict[str, Dict[str, str]] = {
    "float_to_int_nearest": {"cuda": "__float2int_rn", "hip": "__float2int_rn"},
    "float_to_int_zero": {"cuda": "__float2int_rz", "hip": "__float2int_rz"},
    "float_to_int_down": {"cuda": "__float2int_rd", "hip": "__float2int_rd"},
    "float_to_int_up": {"cuda": "__float2int_ru", "hip": "__float2int_ru"},
    "int_to_float": {"cuda": "__int2float_rn", "hip": "__int2float_rn"},
    "uint_to_float": {"cuda": "__uint2float_rn", "hip": "__uint2float_rn"},
    "double_to_int_nearest": {"cuda": "__double2int_rn", "hip": "__double2int_rn"},
    "double_to_int_zero": {"cuda": "__double2int_rz", "hip": "__double2int_rz"},
    "double_to_int_down": {"cuda": "__double2int_rd", "hip": "__double2int_rd"},
    "double_to_int_up": {"cuda": "__double2int_ru", "hip": "__double2int_ru"},
}
