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
