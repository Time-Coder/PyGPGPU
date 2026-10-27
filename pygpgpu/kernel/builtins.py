"""Namespace of device builtin functions available inside `@kernel` kernels.

Usage::

    from pygpgpu.kernel import gpu

    @kernel
    def k(out):
        i = gpu.get_global_id(0)
        gpu.atomic_add(out, i, 1)

The translator recognizes calls on `gpu` and maps them to OpenCL/CUDA builtins per backend.
The `python` backend serves the same names from a thread-local frame instead: its
interpreter runs the kernel body directly, so `gpu.get_global_id` and friends must
report the work item the serial loop is currently emulating (see python_backend.py).

Python's builtin ``print`` works inside a kernel body too: the translator turns it into
device ``printf`` calls that print the same text (see ``FunctionTranslator._emit_print``).
The `python` backend needs no translation there -- the interpreter prints as usual.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Tuple



class PyKernelSyntaxError(SyntaxError):
    """Error type carried by the builtins module itself (a lazy import would cycle with translator)."""



# The python backend installs one frame per thread for the duration of a kernel call.
# It lives here (rather than in python_backend.py) so attribute access on `gpu` can find
# it without importing the backend, which would cycle back through the translator.
_HOST_FRAME = threading.local()


def set_host_frame(frame: Optional[Any])->None:
    """Install the active python-backend frame for this thread (None clears it)."""
    _HOST_FRAME.frame = frame


def get_host_frame()->Optional[Any]:
    """The frame installed by the python backend on this thread, or None."""
    return getattr(_HOST_FRAME, "frame", None)


class GPUNamespace:
    """Marker type: the translator recognizes `gpu.xxx(...)` calls and maps them to device builtins.

    While a python-backend call is running on the current thread, the same attributes
    resolve to the host implementations supplied by that frame instead.
    """

    def __init__(self)->None:
        self.__name__ = "pygpgpu.kernel.builtins"

    def __getattr__(self, name:str)->Callable:
        frame:Optional[Any] = get_host_frame()
        if frame is not None:
            return frame.builtin(name)

        def _not_callable(*args, **kwargs)->None:
            from .translator import PyKernelSyntaxError
            raise PyKernelSyntaxError(
                f"gpu.{name} is only available inside a @kernel function body "
                f"(it maps to a device builtin, not a host function)")
        return _not_callable


gpu = GPUNamespace()


def tex1D(texture:Any, u:Any)->Any:
    """Sample a 1D CUDA texture at a normalized coordinate, inside a @kernel body.

    The translator rewrites the call into ``tex1D<float>(texture, u)``. Sampling runs
    on the device, so this body only gives the name something to bind to.
    """
    raise PyKernelSyntaxError(
        "tex1D() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


def tex2D(texture:Any, u:Any, v:Any)->Any:
    """Sample a 2D CUDA texture at normalized coordinates, inside a @kernel body.

    The translator rewrites the call into ``tex2D<float>(texture, u, v)``. Sampling
    runs on the device, so this body only gives the name something to bind to.
    """
    raise PyKernelSyntaxError(
        "tex2D() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


def tex3D(texture:Any, u:Any, v:Any, w:Any)->Any:
    """Sample a 3D CUDA texture at normalized coordinates, inside a @kernel body.

    The translator rewrites the call into ``tex3D<float>(texture, u, v, w)``.
    Sampling runs on the device, so this body only gives the name something to bind
    to.
    """
    raise PyKernelSyntaxError(
        "tex3D() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


def surf1Dread(surface:Any, x:Any)->Any:
    """Read one element of a 1D CUDA surface, inside a @kernel body.

    The translator rewrites the call into ``surf1Dread<float>(surface, x)``.
    """
    raise PyKernelSyntaxError(
        "surf1Dread() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


def surf1Dwrite(surface:Any, value:Any, x:Any)->Any:
    """Write one element of a 1D CUDA surface, inside a @kernel body.

    The translator rewrites the call into ``surf1Dwrite<float>(value, surface, x)``.
    """
    raise PyKernelSyntaxError(
        "surf1Dwrite() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


def surf3Dread(surface:Any, x:Any, y:Any, z:Any)->Any:
    """Read one element of a 3D CUDA surface, inside a @kernel body.

    The translator rewrites the call into ``surf3Dread<float>(surface, x, y, z)``.
    """
    raise PyKernelSyntaxError(
        "surf3Dread() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


def surf3Dwrite(surface:Any, value:Any, x:Any, y:Any, z:Any)->Any:
    """Write one element of a 3D CUDA surface, inside a @kernel body.

    The translator rewrites the call into ``surf3Dwrite<float>(value, surface, x, y, z)``.
    """
    raise PyKernelSyntaxError(
        "surf3Dwrite() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


def surf2Dread(surface:Any, x:Any, y:Any)->Any:
    """Read one element of a 2D CUDA surface, inside a @kernel body.

    The translator rewrites the call into ``surf2Dread<float>(surface, x, y)``.
    """
    raise PyKernelSyntaxError(
        "surf2Dread() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


def surf2Dwrite(surface:Any, value:Any, x:Any, y:Any)->Any:
    """Write one element of a 2D CUDA surface, inside a @kernel body.

    The translator rewrites the call into ``surf2Dwrite<float>(value, surface, x, y)``.
    """
    raise PyKernelSyntaxError(
        "surf2Dwrite() is a device builtin: it is translated inside a @kernel body and "
        "cannot be called on the host")


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

# Atomic ops: (gpu name, opencl name, cuda name); signature (array, index, value), integer element types only
ATOMIC_BUILTINS:Dict[str, Tuple[str, str]] = {
    "atomic_add": ("atomic_add", "atomicAdd"),
    "atomic_sub": ("atomic_sub", "atomicSub"),
    "atomic_min": ("atomic_min", "atomicMin"),
    "atomic_max": ("atomic_max", "atomicMax"),
    "atomic_and": ("atomic_and", "atomicAnd"),
    "atomic_or": ("atomic_or", "atomicOr"),
    "atomic_xor": ("atomic_xor", "atomicXor"),
    "atomic_xchg": ("atomic_xchg", "atomicExch"),
}

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
    if backend == "opencl":
        return "barrier(CLK_GLOBAL_MEM_FENCE)"
    return "__syncthreads()"


def emit_atomic(name:str, array_code:str, index_code:str, value_code:str, elem_type:str, backend:str)->str:
    opencl_name, cuda_name = ATOMIC_BUILTINS[name]
    if backend == "opencl":
        return f"{opencl_name}((volatile __global {elem_type} *)(&({array_code})[{index_code}]), {value_code})"
    return f"{cuda_name}(&({array_code})[{index_code}], {value_code})"
