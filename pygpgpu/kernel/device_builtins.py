"""The device side of the ``gpu`` namespace: how each builtin is emitted per backend.

This module is translation machinery, not API. The names a user writes live in
``builtins.py`` as dual-sided functions with real signatures; here are the tables the
translator turns those names into (OpenCL C or CUDA), plus the helpers that render one
call. Nothing here inspects a Python body, and nothing here is imported by user code.

The ``python`` backend does not use these tables at all: it runs the bodies from
``builtins.py`` instead.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple



# ---------- mapping metadata used at translation time ----------

# Index builtins: (gpu name, opencl name, cuda expression template); dim argument must be an integer literal
INDEX_BUILTINS:Dict[str, Tuple[str, str]] = {
    "get_global_id": ("get_global_id", "(int)(blockIdx{c} * blockDim{c} + threadIdx{c})"),
    "get_local_id": ("get_local_id", "(int)threadIdx{c}"),
    "get_group_id": ("get_group_id", "(int)blockIdx{c}"),
    "get_global_size": ("get_global_size", "(int)blockDim{c} * gridDim{c}"),
    "get_local_size": ("get_local_size", "(int)blockDim{c}"),
    "get_num_groups": ("get_num_groups", "(int)gridDim{c}"),
}

# CUDA's index vectors: name -> the three expressions its axes stand for.
#
# They are values, not calls (`buf[blockIdx][j]`), so the translator resolves the *name* to
# this table instead of going through `_builtin_call_name`. Each axis is the same number the
# OpenCL spelling gives for that dimension, spelled in the backend's own terms, so a kernel
# ported from CUDA means exactly what the `get_*` spelling means.
INDEX_VECTOR_BUILTINS:Dict[str, Tuple[str, str]] = {
    "blockIdx": ("int3", "make_int3((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z)"),
    "threadIdx": ("int3", "make_int3((int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z)"),
    "blockDim": ("int3", "make_int3((int)blockDim.x, (int)blockDim.y, (int)blockDim.z)"),
    "gridDim": ("int3", "make_int3((int)gridDim.x, (int)gridDim.y, (int)gridDim.z)"),
}

# OpenCL has no `make_int3`; a compound literal with a struct-vs-vector spelling is why the
# expression above is CUDA-only. OpenCL builds the same value by construction instead.
INDEX_VECTOR_OPENCL:Dict[str, Tuple[str, str]] = {
    "blockIdx": ("int3", "(int3)(get_group_id(0), get_group_id(1), get_group_id(2))"),
    "threadIdx": ("int3", "(int3)(get_local_id(0), get_local_id(1), get_local_id(2))"),
    "blockDim": ("int3", "(int3)(get_local_size(0), get_local_size(1), get_local_size(2))"),
    "gridDim": ("int3", "(int3)(get_num_groups(0), get_num_groups(1), get_num_groups(2))"),
}

# The scalars behind each axis, so `threadIdx.x` is one launch term rather than
# "build the whole int3, then take its first component". Both tables above answer the same
# three numbers, so this is the same fact spelled once per backend: a reader who wants an
# axis is asking for a `get_*` call, and that is what this returns.
INDEX_VECTOR_AXES_CUDA:Dict[str, Tuple[str, str, str]] = {
    "blockIdx": ("(int)(blockIdx.x)", "(int)(blockIdx.y)", "(int)(blockIdx.z)"),
    "threadIdx": ("(int)(threadIdx.x)", "(int)(threadIdx.y)", "(int)(threadIdx.z)"),
    "blockDim": ("(int)(blockDim.x)", "(int)(blockDim.y)", "(int)(blockDim.z)"),
    "gridDim": ("(int)(gridDim.x)", "(int)(gridDim.y)", "(int)(gridDim.z)"),
}

INDEX_VECTOR_AXES_OPENCL:Dict[str, Tuple[str, str, str]] = {
    "blockIdx": ("get_group_id(0)", "get_group_id(1)", "get_group_id(2)"),
    "threadIdx": ("get_local_id(0)", "get_local_id(1)", "get_local_id(2)"),
    "blockDim": ("get_local_size(0)", "get_local_size(1)", "get_local_size(2)"),
    "gridDim": ("get_num_groups(0)", "get_num_groups(1)", "get_num_groups(2)"),
}

# Atomic ops: (gpu name, opencl name, cuda name); integer element types only.
# `atomic_cas` is the one with a different arity: (array, index, compare, value).
ATOMIC_BUILTINS:Dict[str, Tuple[str, str]] = {
    "atomic_add": ("atomic_add", "atomicAdd"),
    "atomic_sub": ("atomic_sub", "atomicSub"),
    "atomic_min": ("atomic_min", "atomicMin"),
    "atomic_max": ("atomic_max", "atomicMax"),
    "atomic_and": ("atomic_and", "atomicAnd"),
    "atomic_or": ("atomic_or", "atomicOr"),
    "atomic_xor": ("atomic_xor", "atomicXor"),
    "atomic_xchg": ("atomic_xchg", "atomicExch"),
    "atomic_cas": ("atomic_cmpxchg", "atomicCAS"),
}

# CUDA has no 64-bit atomicSub and no signed 64-bit atomicAdd, but a 64-bit element can be
# serviced through `unsigned long long` for every operation whose result is the same bit
# pattern: addition, subtraction (as adding the two's-complement negation), the bitwise
# ops, exchange and compare-and-swap. min/max are *not* in that group for a signed element,
# because unsigned ordering flips for negative values, so those keep the element's own type.
_ATOMIC_64_AS_UNSIGNED:Dict[str, str] = {
    "atomic_add": "atomicAdd",
    "atomic_sub": "atomicAdd",
    "atomic_and": "atomicAnd",
    "atomic_or": "atomicOr",
    "atomic_xor": "atomicXor",
    "atomic_xchg": "atomicExch",
    "atomic_cas": "atomicCAS",
}


def _emit_atomic_cuda_64(name:str, pointer:str, value_code:str,
                         compare_code:Optional[str], elem_type:str)->str:
    """A 64-bit atomic for CUDA, in the spelling that carries all 64 bits.

    The element type arrives as `long`/`ulong`, which is not a 64-bit type in CUDA C on
    every host, so the wide spelling is written out here instead. Without it an int64
    buffer produced `atomicAdd(long*, ...)`, which fails to compile on NVRTC (and would
    silently target 32 bits where `long` is narrow).
    """
    if name in ("atomic_min", "atomic_max"):
        wide:str = "long long" if elem_type == "long" else "unsigned long long"
        cuda_name:str = ATOMIC_BUILTINS[name][1]
        return f"{cuda_name}(({wide} *)({pointer}), ({wide})({value_code}))"

    unsigned:str = "unsigned long long"
    cuda_name = _ATOMIC_64_AS_UNSIGNED[name]
    if name == "atomic_sub":
        # a - b is a + (2^64 - b) in two's-complement arithmetic, exact for every value
        # including the most negative one, which a negation in signed arithmetic is not.
        value:str = f"0ull - ({unsigned})({value_code})"
    else:
        value = f"({unsigned})({value_code})"
    if compare_code is not None:
        argument:str = f"({unsigned})({compare_code}), {value}"
    else:
        argument = value
    return f"{cuda_name}(({unsigned} *)({pointer}), {argument})"


def emit_atomic(name:str, array_code:str, index_code:str, value_code:str,
                elem_type:str, backend:str, compare_code:Optional[str]=None,
                pointer_code:Optional[str]=None, is_local:bool=False)->str:
    """One atomic call.

    ``compare_code`` is only passed for ``atomic_cas``. ``pointer_code`` replaces the
    ``&array[index]`` address with an expression that is already a pointer to one element,
    which is how an outer scalar (`nonlocal`/`global`, a device ``T*``) is updated.
    ``is_local`` marks a `local(...)` buffer (shared memory): OpenCL needs the ``__local``
    address space in that cast, because casting ``__local`` to ``__global`` does not compile.
    """
    opencl_name, cuda_name = ATOMIC_BUILTINS[name]
    pointer:str = pointer_code if pointer_code is not None else f"&({array_code})[{index_code}]"
    arguments:str = value_code if compare_code is None else f"{compare_code}, {value_code}"
    if backend == "opencl":
        space:str = "__local" if is_local else "__global"
        return f"{opencl_name}((volatile {space} {elem_type} *)({pointer}), {arguments})"
    if backend == "metal":
        # Metal has no `atomicAdd` family. Its atomics are the C++11
        # `atomic_fetch_*_explicit` functions, taking a `device atomic_*` (or
        # `threadgroup atomic_*`) reference and a memory order. The cast selects the
        # address space, which MSL requires: a `threadgroup` buffer and a `device`
        # buffer are different atomics, exactly as OpenCL's two address spaces are.
        return _emit_atomic_metal(name, pointer, value_code, compare_code, elem_type,
                                  is_local)
    if elem_type in ("long", "ulong"):
        return _emit_atomic_cuda_64(name, pointer, value_code, compare_code, elem_type)
    return f"{cuda_name}({pointer}, {arguments})"


# MSL's atomic function for each operation. `atomic_fetch_*_explicit` returns the old
# value and takes the memory order as its last argument; every one here is a
# read-modify-write on one location, so `memory_order_relaxed` is what a work item
# needs -- it says "this update is atomic, but nothing else about it is ordered", and
# a `barrier()` is what orders the group.
_METAL_ATOMICS:Dict[str, str] = {
    "atomic_add": "atomic_fetch_add_explicit",
    "atomic_sub": "atomic_fetch_sub_explicit",
    "atomic_min": "atomic_fetch_min_explicit",
    "atomic_max": "atomic_fetch_max_explicit",
    "atomic_and": "atomic_fetch_and_explicit",
    "atomic_or": "atomic_fetch_or_explicit",
    "atomic_xor": "atomic_fetch_xor_explicit",
    "atomic_xchg": "atomic_exchange_explicit",
}

# MSL's atomic type name for an element type. `atomic_uint` is the one MSL declares
# unconditionally; the others exist for the sized integer types.
_METAL_ATOMIC_TYPES:Dict[str, str] = {
    "int": "atomic_int",
    "uint": "atomic_uint",
    "char": "atomic_char",
    "uchar": "atomic_uchar",
    "short": "atomic_short",
    "ushort": "atomic_ushort",
    "long": "atomic_long",
    "ulong": "atomic_ulong",
}


def _emit_atomic_metal(name:str, pointer:str, value_code:str,
                       compare_code:Optional[str], elem_type:str,
                       is_local:bool)->str:
    """One atomic operation, in Metal Shading Language.

    Two things are MSL's own. (1) **The address space is part of the type**: the
    atomics are overloaded on `device atomic_T&` and `threadgroup atomic_T&`, so a
    `zeros_per_group` buffer must be cast to `threadgroup` and an array argument to
    `device` -- the same distinction OpenCL draws with `__local`/`__global`, and for
    the same reason: one address space cannot be cast to the other. (2) **Compare-
    and-swap is `atomic_compare_exchange_weak_explicit`**, which takes *two*
    expected-value pointers and answers a `bool`, so its shape differs from the
    fetch family: the expected value is held in a local, tried, and the caller
    checks the answer.
    """
    atomic_type:str = _METAL_ATOMIC_TYPES.get(elem_type, f"atomic_{elem_type}")
    space:str = "threadgroup" if is_local else "device"
    target:str = f"({space} {atomic_type}*)({pointer})"
    if name == "atomic_cas":
        # `atomic_compare_exchange_weak_explicit(ptr, &expected, desired, order, fail_order)`.
        # The expected value is an input here, so it lives in a local the call can
        # take the address of; that is also why the call cannot be an expression
        # returning a value the way the fetch family can.
        return (f"({{\n"
                f"    {elem_type} _cas_expected = ({compare_code});\n"
                f"    (void)atomic_compare_exchange_weak_explicit("
                f"{target}, &_cas_expected, ({value_code}), "
                f"memory_order_relaxed, memory_order_relaxed);\n"
                f"}})")
    return (f"{_METAL_ATOMICS[name]}({target}, ({value_code}), "
            f"memory_order_relaxed)")

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

GENERIC_BUILTINS.update(MATH_BUILTINS)


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


def dim_component(dim:int)->str:
    if dim not in (0, 1, 2):
        raise ValueError(f"dimension index must be 0, 1 or 2, got {dim}")
    return (".x", ".y", ".z")[dim]


def emit_index(name:str, dim:int, backend:str)->str:
    opencl_name, cuda_template = INDEX_BUILTINS[name]
    component:str = dim_component(dim)
    if backend in ("opencl", "metal"):
        # Metal shares OpenCL's spelling: its generated prelude defines these as macros
        # over the attribute-decorated kernel parameters (see kernel/metal_codegen.py),
        # so the body keeps one spelling and the prologue is what differs.
        return f"{opencl_name}({dim})"
    return cuda_template.format(c=component)


def emit_barrier(backend:str)->str:
    """A full work-group barrier.

    Both fences are requested: a `local(...)` buffer lives in work-group shared memory, and
    OpenCL requires ``CLK_LOCAL_MEM_FENCE`` for those writes to be visible across the group,
    while ``CLK_GLOBAL_MEM_FENCE`` covers the array arguments. Asking for both is always
    correct and costs nothing measurable.
    """
    if backend == "opencl":
        return "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)"
    if backend == "metal":
        # MSL has no `__syncthreads()`: the work-group barrier there is
        # `threadgroup_barrier`, and its flags name which memory the barrier orders.
        # This spelling lives in `metal_codegen.barrier()` (next to the rest of Metal's
        # prologue) rather than inline here, so the two facts -- the call and its flags
        # -- stay in one place.
        from .metal_codegen import barrier as metal_barrier
        return metal_barrier()
    return "__syncthreads()"
