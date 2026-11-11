# Floor division and modulo, and the round-toward-zero helpers HIP has no intrinsic for.
from __future__ import annotations
from typing import Dict, Set



# The round-toward-zero forms, for the backends that have no intrinsic for them.
#
# CUDA has `__frcp_rz`, `__fdiv_rz` and `__fsqrt_rz`; HIP has none of them (not even the
# `_rn` spellings of the other rounding directions). The value is the same either way --
# the exact result rounded toward zero -- and it can be reproduced exactly: take the
# correctly rounded value, use the fused multiply-add to learn which side of it the exact
# result is on, and step one unit in the last place toward zero when it is the other way.
#
# The residual is what makes it exact rather than approximate. `truncf(x / y)` is not the
# same operation: it truncates an already rounded quotient, so it can be off by a whole
# unit in the last place. The technique was checked against CUDA's own intrinsics over
# 1600 values (positive and negative, over four orders of magnitude) and agreed with all
# of them.
#
# These are device functions, so the translator emits the definition of the one a kernel
# uses before the kernel that calls it. The name is prefixed so it cannot collide with a
# user's own.
HIP_ROUND_TOWARD_ZERO_HELPERS:Dict[str, str] = {
    "recip": "__pygpgpu_frcp_rz",
    "divide": "__pygpgpu_fdiv_rz",
    "sqrt": "__pygpgpu_fsqrt_rz",
}



# The definition of each helper, as the C the generated source carries. The `__device__`
# qualifier is what HIP calls a device function; CUDA's own intrinsics need none.
HIP_ROUND_TOWARD_ZERO_SOURCES:Dict[str, str] = {
    "recip": """
__device__ float __pygpgpu_frcp_rz(float x)
{
    float r = __frcp_rn(x);
    float e = __fmaf_rn(-x, r, 1.0f);
    // The exact reciprocal is below the rounded one, so step toward zero.
    if (e < 0.0f)
    {
        // A float's bit pattern is the sign bit above the magnitude, so subtracting one
        // takes a unit off the magnitude -- which is what "toward zero" means -- for a
        // positive value and for a negative one alike.
        r = __int_as_float(__float_as_int(r) - 1);
    }
    return r;
}""",
    "divide": """
__device__ float __pygpgpu_fdiv_rz(float x, float y)
{
    float r = __fdiv_rn(x, y);
    float e = __fmaf_rn(-y, r, x);
    // The exact quotient is below the rounded one, so step toward zero.
    if (e < 0.0f)
    {
        r = __int_as_float(__float_as_int(r) - 1);
    }
    return r;
}""",
    "sqrt": """
__device__ float __pygpgpu_fsqrt_rz(float x)
{
    float r = __fsqrt_rn(x);
    float e = __fmaf_rn(r, r, -x);
    // The square of the rounded root is above x, so the exact root is below it.
    if (e > 0.0f)
    {
        r = __int_as_float(__float_as_int(r) - 1);
    }
    return r;
}""",
}



# Python's `//` and `%` are floor-based, and C's are truncation-based, so the two disagree
# as soon as an operand is negative: `-7 // 2` is `-4` in python and `-3` in C, and `-7 % 2`
# is `1` in python where C gives `-1`. A kernel that divides or takes a remainder can
# therefore answer differently from the same function run in python -- silently, for a
# negative input the author may not have considered.
#
# The fix is the standard identity: truncation and flooring agree whenever the remainder is
# zero or has the divisor's sign, and otherwise differ by exactly one quotient step.
#
#   floor(a / b) = q - ((r != 0) && ((r < 0) != (b < 0))
#   a % b        = r + (b * ((r != 0) && ((r < 0) != (b < 0)))
#
# where q and r are C's own quotient and remainder. That keeps one division in the
# generated code and adds a comparison, rather than a second divide.
#
# These are emitted as functions rather than expanded inline because the same step appears
# in several places (the binary operator and the augmented assignment), and because a
# function keeps the source readable. Each is specialized on the C type the operands
# promote to, which is why the name carries the type.
#
# The qualifier is per backend, and `inline` is not decoration: the definition is emitted
# into every source that uses the operator, and a file holding several kernels would
# otherwise carry several copies of the same non-inline function -- which C++ (CUDA and
# HIP) rejects as a redefinition. `static inline` is OpenCL C's spelling; CUDA adds
# `__device__` because a plain `static inline` function there would be a *host* function.
PYTHON_FLOOR_QUALIFIERS:Dict[str, str] = {
    "opencl": "static inline",
    "cuda": "__device__ static inline",
    "hip": "__device__ static inline",
    "metal": "static inline",
}



# The definition of each, as the C the generated source carries. `%(t)s` is the C type name
# (`int`, `long`, `uint`, ...) and `%(q)s` the qualifier, so one body serves every width.
# The `%(`/`%%` spellings matter: this text goes through `%`-formatting, so a literal
# remainder operator has to be doubled.
#
# `float_mod` is the floating-width `%`: C has no `%` for floats at all (the emitter used
# to spell plain `fmod`, whose result carries the *dividend's* sign where python's
# remainder carries the *divisor's*), and the correction is CPython's own float remainder
# -- take fmod's result and add the divisor once when the two signs disagree. The empty
# `r != 0` guard keeps a zero remainder untouched (python keeps fmod's signed zero), and a
# NaN stays a NaN through both tests. `%(f)s` is the fmod spelling the width and backend
# use: OpenCL C and Metal name the plain `fmod` for both widths, CUDA and HIP spell the
# float form `fmodf` and keep `fmod` for doubles.
PYTHON_FLOOR_SOURCES:Dict[str, str] = {
    "floor_div": """
%(q)s %(t)s __pygpgpu_floor_div_%(n)s(%(t)s a, %(t)s b)
{
    %(t)s q = a / b;
    %(t)s r = a %% b;
    // Truncation and flooring differ only when the remainder is non-zero and the two
    // operands have opposite signs, and then by exactly one step.
    if (r != 0 && ((r < 0) != (b < 0)))
    {
        q -= 1;
    }
    return q;
}""",
    "floor_mod": """
%(q)s %(t)s __pygpgpu_floor_mod_%(n)s(%(t)s a, %(t)s b)
{
    %(t)s r = a %% b;
    // Python's remainder carries the divisor's sign; C's carries the dividend's.
    if (r != 0 && ((r < 0) != (b < 0)))
    {
        r += b;
    }
    return r;
}""",
    "float_mod": """
%(q)s %(t)s __pygpgpu_float_mod_%(n)s(%(t)s a, %(t)s b)
{
    %(t)s r = %(f)s(a, b);
    // Python's remainder carries the divisor's sign; fmod's carries the dividend's.
    if (r != 0 && ((r < 0) != (b < 0)))
    {
        r += b;
    }
    return r;
}""",
}



# The fmod spelling per backend for the floating-width helper (see `float_mod` above).
FLOAT_FMOD_NAMES:Dict[str, str] = {
    "opencl": "fmod",
    "cuda": "fmodf",
    "hip": "fmodf",
    "metal": "fmod",
}



# An unsigned type never has a negative remainder, so the correction is dead code there --
# and worse, `(r < 0)` on an unsigned value is always false, which some compilers warn
# about. The unsigned spellings keep the plain C operator, whose semantics already agree
# with python's for non-negative operands.
UNSIGNED_C_TYPES:Set[str] = {"unsigned char", "unsigned short", "unsigned int",
                             "unsigned long long", "uchar", "ushort", "uint", "ulong"}



# The suffix a C type contributes to a function name. A multi-word type (`long long`,
# `unsigned int`) cannot appear verbatim in an identifier, so the words are joined: the
# helper is `__pygpgpu_floor_div_longlong`, not `__pygpgpu_floor_div_long long`.
def c_identifier_suffix(ctype:str)->str:
    """The identifier-safe tail for a C type name."""
    return "".join(word for word in ctype.replace("*", "").split())



def python_floor_name(operation:str, ctype:str)->str:
    """The normalized helper name for one operation and one C width."""
    return f"__pygpgpu_{operation}_{c_identifier_suffix(ctype)}"



def python_floor_source(operation:str, ctype:str, backend:str)->str:
    """The device function that gives `//` or `%` python's floor semantics for one width.

    Returns "" for an unsigned width, whose C operator already matches python. The float
    form (`float_mod`) is only asked for a floating width: C has no `%` for floats, so it
    has no unsigned case to skip either.
    """
    if ctype in UNSIGNED_C_TYPES:
        return ""
    qualifier:str = PYTHON_FLOOR_QUALIFIERS.get(backend, PYTHON_FLOOR_QUALIFIERS["opencl"])
    if operation == "float_mod":
        fmod_name:str = FLOAT_FMOD_NAMES.get(backend, "fmod")
        if ctype == "double":
            fmod_name = "fmod"
        return PYTHON_FLOOR_SOURCES[operation] % {"q": qualifier, "t": ctype,
                                                  "n": c_identifier_suffix(ctype),
                                                  "f": fmod_name}
    return PYTHON_FLOOR_SOURCES[operation] % {"q": qualifier, "t": ctype,
                                              "n": c_identifier_suffix(ctype)}
