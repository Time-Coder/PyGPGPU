# The atomic vocabulary, including the C++11 spelling MSL needs.
from __future__ import annotations
from typing import Dict, Optional, Tuple



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
    if elem_type == "float" and name == "atomic_add":
        # No hardware float atomic in this package's set: the portable CAS loop answers
        # for the carried-float shape (see float_atomic_add_source).
        return f"__pygpgpu_atomic_add_f32({pointer}, {arguments})"
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
    `local_array_per_group` buffer must be cast to `threadgroup` and an array argument to
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



# The carrier of a float carried scalar (`total = 0.0` above the loop) is a float slot,
# and no hardware in this package's set has a native float atomic add: CUDA's
# `atomicAdd(float*)` exists but OpenCL C 1.2's `atomic_add` is integer-only. The
# portable spelling is the classic bit-level CAS loop -- the float's bits ride in an
# `unsigned int` (CUDA spells the conversions `__float_as_uint`/`__uint_as_float`,
# OpenCL `as_uint`/`as_float`), and `atomicCAS`/`atomic_cmpxchg` on that unsigned slot
# retries until one work item's addition lands. The helper is emitted once per source,
# ahead of the kernels that call it (see device_helpers, the same channel the floor
# helpers travel through).
FLOAT_ATOMIC_ADD_SOURCES:Dict[str, str] = {
    "cuda": """
__device__ float __pygpgpu_atomic_add_f32(float* address, float value)
{
    unsigned int* slot = (unsigned int*)(address);
    unsigned int old_bits = *slot;
    unsigned int assumed;
    do
    {
        assumed = old_bits;
        float sum = __uint_as_float(assumed) + value;
        old_bits = atomicCAS(slot, assumed, __float_as_uint(sum));
    } while (assumed != old_bits);
    return __uint_as_float(old_bits);
}""",
    "opencl": """
float __pygpgpu_atomic_add_f32(volatile __global float* address, float value)
{
    volatile __global unsigned int* slot =
        (volatile __global unsigned int*)(address);
    unsigned int old_bits = *slot;
    unsigned int assumed;
    do
    {
        assumed = old_bits;
        float sum = as_float(assumed) + value;
        old_bits = atomic_cmpxchg(slot, assumed, as_uint(sum));
    } while (assumed != old_bits);
    return as_float(old_bits);
}""",
}



def float_atomic_add_source(backend:str)->str:
    """The float atomic-add helper for one backend, or "" where it is not spelled."""
    return FLOAT_ATOMIC_ADD_SOURCES.get(backend, "")
