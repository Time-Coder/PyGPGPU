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
                pointer_code:Optional[str]=None, is_local:bool=False,
                result_name:str="_cas_result")->str:
    """One atomic call.

    ``compare_code`` is only passed for ``atomic_cas``. ``pointer_code`` replaces the
    ``&array[index]`` address with an expression that is already a pointer to one element,
    which is how an outer scalar (`nonlocal`/`global`, a device ``T*``) is updated.
    ``is_local`` marks a `local(...)` buffer (shared memory): OpenCL needs the ``__local``
    address space in that cast, because casting ``__local`` to ``__global`` does not compile.
    ``result_name`` names the local a Metal statement block leaves its old value in; the
    caller makes it unique per call site, because two such blocks in one expression would
    otherwise read whichever ran last.
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
                                  is_local, result_name)
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
    # MSL declares only five atomic element types (metal_atomic's typedefs):
    # int, uint, ulong, float and bool. The narrow families have no atomic, so
    # an atomic on an int8/int16 array is refused upstream (`ATOMIC_ELEMENT_TYPES`);
    # the signed 64-bit spelling is built from the unsigned one -- the operations
    # are bitwise-identical for add/sub/xor/and/or, and compare-exchange through
    # the unsigned representation answers the same old-value bits.
    "int": "atomic_int",
    "uint": "atomic_uint",
    "long": "atomic_ulong",
    "ulong": "atomic_ulong",
}



def _emit_metal_cas_loop(name:str, pointer:str, value_code:str,
                         elem_type:str, space:str, op:str)->str:
    """A 64-bit read-modify-write under a spin lock on a 32-bit lock word.

    MSL's atomic builtins accept 32-bit element types only (`_valid_non_fetch_type`
    and the fetch traits list device/threadgroup int*/uint*/float*/bool* -- the raw
    `__metal_atomic_*` intrinsics reject a ulong pointer too), so the 64-bit slot
    rides under a per-slot spin lock: one extra `atomic_uint` word beside the slot.
    A thread takes the lock with a compare-exchange (acquire), reads the slot's two
    words PLAIN (the lock makes them private), computes, stores, and releases the
    lock (release). Every group gets its own lock word, allocated by the launch
    beside the counter buffers (see `metal_codegen.barrier`). Verified natively on
    this Mac at 8 groups x 1 thread and 1 group x 8 threads.

    The lock word lives at the pointer the caller names MINUS one slot: the
    generated kernel declares it as `_atomic_lock` beside the slot's own pointer
    (see `render.parameter_declarations`'s 64-bit branch), and `pointer` names the
    slot's first element, so the lock is `_atomic_lock` itself -- the extra
    parameter arrives as the `_atomic_lock` argument.
    """
    c_op:Dict[str, str] = {
        "add": "+", "sub": "-",
        "min": "min", "max": "max",
        "and": "&", "or": "|", "xor": "^",
    }
    compute:str = c_op.get(op, op)
    if op == "xchg":
        # _val is the plain element here (the value_code cast), not the union:
        # an exchange writes the operand itself.
        compute = "_val"
    elif op == "min":
        # The 64-bit min/max compares as signed when the element is `long` --
        # the element type carries the order.
        compute = "(_cur < _val ? _cur : _val)"
    elif op == "max":
        compute = "(_cur > _val ? _cur : _val)"
    else:
        compute = f"(_cur {compute} _val)"
    return (f"{{\n"
            f"    device atomic_uint* _lock = _atomic_lock;\n"
            f"    bool _done = false;\n"
            f"    while (!_done) {{\n"
            f"        bool _got = false;\n"
            f"        int _tries = 0;\n"
            f"        // An unbounded spin on a contended device atomic wedges on this\n"
            f"        // driver (see metal_codegen.barrier); a bounded retry with a\n"
            f"        // whole-RMW outer loop completes and stays correct.\n"
            f"        while (!_got && _tries < 4096) {{\n"
            f"            uint _expected = 0u;\n"
            f"            _got = atomic_compare_exchange_weak_explicit(_lock, &_expected, 1u,\n"
            f"                memory_order_relaxed, memory_order_relaxed, mem_flags::mem_device);\n"
            f"            _tries++;\n"
            f"        }}\n"
            f"        if (_got) {{\n"
            f"            device uint* _pair = (device uint*)&({pointer})[0];\n"
            f"            {elem_type} _val = ({elem_type})({value_code});\n"
            f"            long _cur = 0;\n"
            f"            _cur |= (long)_pair[0];\n"
            f"            _cur |= ((long)_pair[1]) << 32;\n"
            f"            long _operand = (long)_val;\n"
            f"            long _upd = {compute};\n"
            f"            _pair[0] = (uint)(_upd & 0xFFFFFFFFu);\n"
            f"            _pair[1] = (uint)(((unsigned long)_upd) >> 32);\n"
            f"            atomic_store_explicit(_lock, 0u, memory_order_relaxed, mem_flags::mem_device);\n"
            f"            _done = true;\n"
            f"        }}\n"
            f"    }}\n"
            f"}}")


def _emit_atomic_metal(name:str, pointer:str, value_code:str,
                       compare_code:Optional[str], elem_type:str,
                       is_local:bool, result_name:str="_cas_result")->str:
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
        # take the address of, and the call itself answers a `bool` (whether the
        # exchange happened). The value the caller reads is the expected-value
        # local AFTER the call: unchanged when the exchange succeeded, the old
        # value when it did not -- the same old-value answer the fetch family
        # gives. A GNU statement-expression `({ ... })` is not portable MSL (the
        # offline toolchain rejects a cast out of `void`), so the emission is a
        # brace STATEMENT block that leaves the old value in the `int` local
        # `_cas_result`. `atomic_cas` is a *statement* builtin on Metal: the
        # translator writes the block, then reads `_cas_result` where the body
        # used the call's value (`out[1] = ...` becomes two statements). The
        # `int` width matches every element type the atomics accept -- the block
        # casts through it, and a 64-bit element's low word is what a caller
        # reading the old value sees (the same contract the fetch family's
        # truncated answer gives a 32-bit reader). `result_name` is the caller's,
        # so two blocks in one expression do not share one slot.
        return (f"{{\n"
                f"    int {result_name};\n"
                f"    {{\n"
                f"        {elem_type} _cas_expected = ({compare_code});\n"
                f"        bool _cas_exchanged = atomic_compare_exchange_weak_explicit("
                f"{target}, &_cas_expected, ({value_code}), "
                f"memory_order_relaxed, memory_order_relaxed);\n"
                f"        {result_name} = (int)_cas_expected;\n"
                f"    }}\n"
                f"}}")
    if elem_type in ("long", "ulong"):
        # MSL's atomic fetch templates accept 32-bit element types only (the
        # `_valid_*_type` traits name device/threadgroup int*/uint*, plus float
        # where the build enables it) -- there is NO 64-bit atomic at all. The
        # 64-bit shapes all run through the bounded spin lock (see
        # `_emit_metal_cas_loop`), which is what CUDA/OpenCL implement in
        # hardware anyway.
        op_name:str = (_METAL_ATOMICS[name]
                       .replace("atomic_fetch_", "").replace("_explicit", ""))
        if op_name == "atomic_exchange":
            op_name = "xchg"  # `_cur atomic_exchange _val` is not C; the loop stores _val
        return _emit_metal_cas_loop(name, pointer, value_code, elem_type, space,
                                    op=op_name)
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
