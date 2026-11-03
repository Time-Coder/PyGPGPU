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
    if elem_type in ("long", "ulong"):
        return _emit_atomic_cuda_64(name, pointer, value_code, compare_code, elem_type)
    return f"{cuda_name}({pointer}, {arguments})"

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
SHARED_MATH_BUILTINS:Dict[str, str] = {
    "erf": "erff",
    "erfc": "erfcf",
    "tgamma": "tgammaf",
    "lgamma": "lgammaf",
    "rint": "rintf",
    "remainder": "remainderf",
    "fmin": "fminf",
    "fmax": "fmaxf",
    "fdim": "fdimf",
    "copysign": "copysignf",
    "nextafter": "nextafterf",
    "ldexp": "ldexpf",
    "native_recip": "__frcp_rn",
    "native_divide": "__fdiv_rn",
    "native_sqrt": "__fsqrt_rn",
    "native_rsqrt": "__frsqrt_rn",
    "native_exp": "__expf",
    "native_log": "__logf",
    "native_sin": "__sinf",
    "native_cos": "__cosf",
    "native_tan": "__tanf",
    "native_exp2": "__exp2f",
    "native_log2": "__log2f",
    "native_exp10": "__exp10f",
    "native_powr": "__powf",
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
    },
    "sign": {
        "opencl": "((({code}) > 0.0f) ? 1.0f : (({code}) < 0.0f) ? -1.0f : 0.0f)",
        "cuda": "((({code}) > 0.0f) ? 1.0f : (({code}) < 0.0f) ? -1.0f : 0.0f)",
    },
}


# The device intrinsics CUDA has and OpenCL does not. They are the CUDA-only half of the
# scientific-computing vocabulary: the fast approximations (`__expf` and friends, which
# trade accuracy for speed), the correctly-rounded forms (`__frcp_rn` and friends), and the
# special functions CUDA adds (the error function's inverse, the normal distribution, the
# Bessel functions).
#
# HIP reuses the CUDA code generation and has these intrinsics too, so "CUDA-only" here
# means "not OpenCL": OpenCL's kernel language has no counterpart for any of them.
#
# A kernel that uses one of them cannot run on OpenCL, so the translator says so by name
# rather than emitting a call the OpenCL compiler would reject with a bare "undeclared
# identifier". The value is the CUDA spelling; the OpenCL spelling is empty because there
# is none.
#
# Two of the names here do NOT mean what their spelling suggests, and the device's own
# documentation is the authority:
#   * `sinpif(x)` is sin(pi*x), NOT the sinc function sin(pi*x)/(pi*x) -- CUDA has no
#     sinc, so the name here is `sin_pi` and it computes the sine of pi times x;
#   * `rhypotf(x, y)` is 1/sqrt(x*x + y*y), the *reciprocal* of the hypotenuse, so the
#     name here is `reciprocal_hypot`. The plain hypotenuse is `hypotf`, which is in the
#     shared table.
CUDA_ONLY_BUILTINS:Dict[str, str] = {
    # the fast (approximate) intrinsics
    "fast_exp": "__expf",
    "fast_log": "__logf",
    "fast_log2": "__log2f",
    "fast_log10": "__log10f",
    "fast_sin": "__sinf",
    "fast_cos": "__cosf",
    "fast_tan": "__tanf",
    "fast_exp10": "__exp10f",
    "fast_pow": "__powf",
    "fast_divide": "__fdividef",
    # the correctly-rounded forms
    "exact_recip": "__frcp_rn",
    "exact_recip_rz": "__frcp_rz",
    "exact_divide": "__fdiv_rn",
    "exact_divide_rz": "__fdiv_rz",
    "exact_sqrt": "__fsqrt_rn",
    "exact_sqrt_rz": "__fsqrt_rz",
    "exact_add": "__fadd_rn",
    "exact_sub": "__fsub_rn",
    "exact_mul": "__fmul_rn",
    "exact_fma": "__fmaf_rn",
    # the special functions
    "erf_inverse": "erfinv",
    "normal_cdf": "normcdff",
    "normal_cdf_inverse": "normcdfinv",
    "reciprocal_hypot": "rhypotf",
    "sin_pi": "sinpif",
    "cos_pi": "cospif",
    "bessel_j0": "j0f",
    "bessel_j1": "j1f",
    "bessel_y0": "y0f",
    "bessel_y1": "y1f",
    "bessel_i0": "cyl_bessel_i0f",
    "bessel_i1": "cyl_bessel_i1f",
}


# The CUDA-only conversions. They answer an int, so the device's own name is spelled with
# the rounding mode in the middle (`__float2int_rn` rounds to nearest, `_rz` toward zero,
# `_rd` down, `_ru` up).
CUDA_ONLY_CONVERSIONS:Dict[str, str] = {
    "float_to_int_nearest": "__float2int_rn",
    "float_to_int_zero": "__float2int_rz",
    "float_to_int_down": "__float2int_rd",
    "float_to_int_up": "__float2int_ru",
    "int_to_float": "__int2float_rn",
    "uint_to_float": "__uint2float_rn",
    "double_to_int_nearest": "__double2int_rn",
    "double_to_int_zero": "__double2int_rz",
    "double_to_int_down": "__double2int_rd",
    "double_to_int_up": "__double2int_ru",
}


def dim_component(dim:int)->str:
    if dim not in (0, 1, 2):
        raise ValueError(f"dimension index must be 0, 1 or 2, got {dim}")
    return (".x", ".y", ".z")[dim]


def emit_index(name:str, dim:int, backend:str)->str:
    opencl_name, cuda_template = INDEX_BUILTINS[name]
    component:str = dim_component(dim)
    if backend == "opencl":
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
    return "__syncthreads()"
