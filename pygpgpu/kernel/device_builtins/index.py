# The work-item queries and the barrier: the two emissions with no backend table behind them.
from __future__ import annotations
from typing import Dict, Tuple



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
        from ..metal_codegen import barrier as metal_barrier
        return metal_barrier()
    return "__syncthreads()"
