"""PyGPGPU: decorate a Python function and it runs on the GPU (OpenCL / CUDA / HIP).

The top level aggregates the commonly used public API so you do not have to
remember subpackage names::

    from pygpgpu import kernel, gpu, float4             # @kernel / device builtins / vector types
    from pygpgpu import list_devices, get_device
    from pygpgpu import climport, cuimport, hipimport    # import hooks for handwritten .cl/.cu/.hip files

Subpackage paths (pygpgpu.kernel / pygpgpu.vec_types / pygpgpu.opencl / ...)
still work. Note that top-level `pygpgpu.kernel` refers to the @kernel
decorator (same name as the subpackage; the top level exports the decorator),
so to use things from the submodules write `from pygpgpu.kernel import ...`.
"""
import importlib
from typing import Any, List

# pygf owns the single process-wide numpy.array monkey-patch. Importing it
# applies that patch (patch_nparray), which teaches numpy.array to read a
# host vector container as its components -- including the structured dtype
# path the vector-typed buffers below depend on. PyGPGPU no longer ships its
# own copy of the patch; this is a deliberate unification.
import pygf  # noqa: F401  (applies the numpy.array patch on import)


# ---------- Top-level exports: public API of kernel ----------
# kernel subpackage's __all__ copied as-is (top-level kernel is the @kernel decorator)

from .kernel import (
    DEFAULT_BACKEND_ORDER,
    PyKernel,
    PyKernelSyntaxError,
    available_devices,
    list_devices,
    get_device,
    gpu,
    kernel,
)

# ---------- Top-level exports: public API of vec_types ----------

from .vec_types import (
    char2, char4, char8, char16,
    uchar2, uchar4, uchar8, uchar16,
    short2, short4, short8, short16,
    ushort2, ushort4, ushort8, ushort16,
    int2, int4, int8, int16,
    uint2, uint4, uint8, uint16,
    float2, float4, float8, float16,
    double2, double4, double8, double16,
    genVec, genVec2, genVec3, genVec4, genVec8, genVec16,
)
from .vec_types import funcs          # dual-purpose function library: runs on the host, translates to device builtins inside @kernel

# ---------- Top-level export: the backend-neutral image type ----------
# Kernels index an Image like an array (src[y, x]); the subscript is translated into
# each backend's sampling call, and the python backend runs the host implementation
# in image.py, so one kernel yields the same numbers everywhere.
from .image import Image


# ---------- Top-level exports: import hooks for handwritten kernel files ----------

class _LazyBackendHook:
    """Lazy proxy for a backend import hook.

    Backend packages initialize their runtime at import time (nvcuda/amdhip64/
    OpenCL) and raise immediately when the runtime is missing; this defers
    loading the backend package until a hook method is actually called, so
    `from pygpgpu import climport, cuimport, hipimport` works on any machine
    and only the backend you use requires its runtime.
    """

    def __init__(self, module_name:str, attr_name:str):
        self._module_name:str = module_name
        self._attr_name:str = attr_name
        self._target:Any = None

    def _resolve(self)->Any:
        if self._target is None:
            module = importlib.import_module(self._module_name, __name__)
            self._target = getattr(module, self._attr_name)
        return self._target

    def __getattr__(self, name:str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    def __repr__(self)->str:
        return f"<lazy hook {self._module_name}.{self._attr_name}>"


climport = _LazyBackendHook(".opencl.climport", "climport")
cuimport = _LazyBackendHook(".cuda.cuimport", "cuimport")
hipimport = _LazyBackendHook(".hip.hipimport", "hipimport")


# ---------- Top-level export: the progress-logging switch ----------

# The switch itself lives in utils, below every backend, so the copy logger inside
# the numpy layer can consult the same source of truth without reaching into a
# backend driver. This name is the public entry point.
from .utils import get_print_info, set_print_info  # noqa: E402  (needs to follow the imports above)


__all__ = [
    # kernel
    "kernel", "gpu", "PyKernel", "PyKernelSyntaxError", "DEFAULT_BACKEND_ORDER",
    "list_devices", "get_device", "available_devices",
    # images
    "Image",
    # logging
    "set_print_info",
    # vec_types
    "funcs",
    "char2", "char4", "char8", "char16",
    "uchar2", "uchar4", "uchar8", "uchar16",
    "short2", "short4", "short8", "short16",
    "ushort2", "ushort4", "ushort8", "ushort16",
    "int2", "int4", "int8", "int16",
    "uint2", "uint4", "uint8", "uint16",
    "float2", "float4", "float8", "float16",
    "double2", "double4", "double8", "double16",
    "genVec", "genVec2", "genVec3", "genVec4", "genVec8", "genVec16",
    # import hooks for handwritten kernel files
    "climport", "cuimport", "hipimport",
]


def __dir__()->List[str]:
    return sorted(set(globals()) | set(__all__))
