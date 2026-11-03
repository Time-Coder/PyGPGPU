"""PyGPGPU: decorate a Python function and it runs on the GPU (OpenCL / CUDA / HIP).

The top level aggregates the commonly used public API so you do not have to
remember subpackage names::

    from pygpgpu import kernel, gpu, float4             # @kernel / device builtins / vector types
    from pygpgpu import list_devices, get_device
    from pygpgpu import kernel_import                    # import hook for handwritten .cl/.cu/.hip files

Subpackage paths (pygpgpu.kernel / pygpgpu.vec_types / pygpgpu.opencl / ...)
still work. Note that top-level `pygpgpu.kernel` refers to the @kernel
decorator (same name as the subpackage; the top level exports the decorator),
so to use things from the submodules write `from pygpgpu.kernel import ...`.
"""
from typing import List

# pygf owns the whole numpy surface for its types: importing it patches numpy.array
# (patch_nparray), the array constructors, and installs the VectorArray view whose
# elements are vectors rather than records. PyGPGPU takes no part in that -- it only
# registers the vector types it adds (8/16-wide, char/uchar/short families), so their
# arrays get the same element access. See pygf/vector_array.py.
import pygf  # noqa: F401  (applies the numpy.array patch on import)

# ---------- Top-level exports: public API of kernel ----------
# kernel subpackage's __all__ copied as-is (top-level kernel is the @kernel decorator)

from .kernel import (
    DEFAULT_BACKEND_ORDER,
    PyKernel,
    available_devices,
    get_default_device,
    list_devices,
    get_device,
    kernel,
    set_default_device,
)
# The dual-sided device builtins are plain functions, so they are exported here directly:
# `from pygpgpu import get_global_id, atomic_add`. Inside a @kernel body the translator
# recognizes each one by the object it resolves to and emits the device call.
from .kernel.builtins import *  # noqa: F403
from .kernel.builtins import __all__ as _builtin_names

# The device-side library function registry: a way to name a device function that already
# exists (a compiler intrinsic, or a `__device__` function in a header) and call it from a
# kernel body under a name of your own.
from .kernel.device_library import device_function

# ---------- Top-level exports: public API of vec_types ----------

from .vec_types import (
    char2, char4, char8, char16,
    uchar2, uchar4, uchar8, uchar16,
    short2, short4, short8, short16,
    ushort2, ushort4, ushort8, ushort16,
    int2, int3, int4, int8, int16,
    uint2, uint3, uint4, uint8, uint16,
    float2, float3, float4, float8, float16,
    double2, double3, double4, double8, double16,
    genVec, genVec2, genVec3, genVec4, genVec8, genVec16,
)
from .vec_types import funcs          # dual-purpose function library: runs on the host, translates to device builtins inside @kernel

# ---------- Top-level export: the backend-neutral image type ----------
# Kernels index an Image like an array (src[y, x]); the subscript is translated into
# each backend's sampling call, and the python backend runs the host implementation
# in image.py, so one kernel yields the same numbers everywhere.
from .image import Image, ImageAddress, ImageFilter


# ---------- Top-level export: the import hook for handwritten kernel files ----------

# The one hook: same-named `.cl`/`.cu`/`.hip` files become one module, and a single-backend
# file is a module with one backend. It is runtime-free by construction -- the file system
# until a kernel is actually called, and only that kernel's backend then -- so it imports
# directly rather than through a lazy proxy.
from .kernel_import import kernel_import  # noqa: E402  (needs the imports above)


# ---------- Top-level export: the progress-logging switch ----------

# The switch itself lives in utils, below every backend, so the copy logger inside
# the numpy layer can consult the same source of truth without reaching into a
# backend driver. This name is the public entry point.
from .utils import get_print_info, set_print_info  # noqa: E402  (needs to follow the imports above)


__all__ = [
    # kernel
    "kernel", "PyKernel", "DEFAULT_BACKEND_ORDER",
    "list_devices", "get_device", "available_devices",
    "set_default_device", "get_default_device",
    # device builtins (dual-sided functions; see pygpgpu/kernel/builtins.py)
    *_builtin_names,
    # the device-side library function registry
    "device_function",
    # images
    "Image", "ImageFilter", "ImageAddress",
    # logging
    "set_print_info",
    # vec_types
    "funcs",
    "char2", "char4", "char8", "char16",
    "uchar2", "uchar4", "uchar8", "uchar16",
    "short2", "short4", "short8", "short16",
    "ushort2", "ushort4", "ushort8", "ushort16",
    "int2", "int3", "int4", "int8", "int16",
    "uint2", "uint3", "uint4", "uint8", "uint16",
    "float2", "float3", "float4", "float8", "float16",
    "double2", "double3", "double4", "double8", "double16",
    "genVec", "genVec2", "genVec3", "genVec4", "genVec8", "genVec16",
    # the import hook for handwritten kernel files (same-named .cl/.cu/.hip sources)
    "kernel_import",
    # the host-side CUDA libraries (cuBLAS, cuSOLVER, cuFFT, cuSPARSE, the runtime);
    # the proxy imports them on first use, so `import pygpgpu` still loads no runtime
]


def __dir__()->List[str]:
    return sorted(set(globals()) | set(__all__))

