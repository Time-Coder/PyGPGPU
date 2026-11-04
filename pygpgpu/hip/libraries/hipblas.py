"""hipBLAS and hipBLASLt -- the dense linear algebra libraries, bound with ctypes.

The shape is the CUDA libraries' shape (and therefore the CUDA driver's): a metaclass
holds the DLL lazily, a ``Func`` wrapper binds the argument types and the return type on
first use, ``init()`` pre-binds the entry points, and a returned status code becomes a
``LibraryError`` carrying hipBLAS's own description of it.

Two things are **not** the same as cuBLAS's, and both are why these classes carry their
own tables:

* the **symbol names** -- hipBLAS drops the ABI version suffix CUDA 12 carries, and
  hipBLASLt spells its entry points ``hipblasLt*``;
* the **values** -- ``HIPBLAS_OP_N`` is 111 where ``CUBLAS_OP_N`` is 0, and the status
  codes are numbered differently. Passing cuBLAS's value to hipBLAS is exactly what makes
  it answer ``HIPBLAS_STATUS_INVALID_ENUM``.

The signature tables are *derived* from the CUDA ones, because the argument lists really
are the same; only the names change.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ...cuda.libraries import cublas as _cuda_cublas
from ...cuda.libraries import cublaslt as _cuda_cublaslt
from ...libraries import Library, resolve_dll
from ._enums import hipblasStatus_t
from ._symbols import hip_symbol

__all__ = ["HIPBLAS", "HIPBLASLt", "HIP_DLL_ROOTS", "hip_signatures_from"]


# Where ROCm installs its libraries. The versioned layout is the one the CUDA toolkit
# uses, so the same search finds both.
HIP_DLL_ROOTS:List[str] = [
    r"C:\Program Files\AMD\ROCm",
    "/opt/rocm",
]


def hip_signatures_from(cuda_signatures:Dict[str, Dict[str, Any]],
                        hip_library:str)->Dict[str, Dict[str, Any]]:
    """The HIP signature table, derived from the CUDA one.

    The parameters are the CUDA ones -- the API is the same -- and only the name of each
    entry point changes. The entries the HIP library does not have are left out.

    Each entry is **copied**, not shared: the binding machinery caches the bound function
    in the entry it was called through (`func_info["dll_func"]`), so a shared entry would
    make the CUDA library and the HIP one hand each other their own function objects --
    a cuBLASLt call would end up in hipBLASLt. The ctypes types themselves are shared
    safely, they are immutable.
    """
    table:Dict[str, Dict[str, Any]] = {}
    for cuda_symbol, signature in cuda_signatures.items():
        hip_name = hip_symbol(hip_library, cuda_symbol)
        if hip_name is None:
            continue
        table[hip_name] = {key: value for key, value in signature.items()
                           if key != "dll_func"}
    return table


def _no_error_check(cuda_no_error_check:List[str], signatures:Dict[str, Dict[str, Any]],
                    hip_library:str)->List[str]:
    """The entries whose return value is not an error code, under their HIP names."""
    names:List[str] = []
    for cuda_name in cuda_no_error_check:
        translated = hip_symbol(hip_library, cuda_name)
        if translated is not None and translated in signatures:
            names.append(translated)
    return names


def _status_messages(descriptions:Dict[str, str])->Dict[int, str]:
    """The error-code table for hipBLAS.

    The codes are hipBLAS's own, looked up through its status enum rather than cuBLAS's:
    the same number means something else in the two libraries.
    """
    table:Dict[int, str] = {}
    for member in hipblasStatus_t:
        table[int(member.value)] = descriptions.get(
            member.name, f"{member.name.replace('_', ' ').lower()}.")
    return table


_HIPBLAS_STATUS_MESSAGES:Dict[str, str] = {
    "HIPBLAS_STATUS_SUCCESS": "no errors.",
    "HIPBLAS_STATUS_NOT_INITIALIZED": "the library was not initialized.",
    "HIPBLAS_STATUS_ALLOC_FAILED":
        "the library could not allocate enough memory to perform the requested operation.",
    "HIPBLAS_STATUS_INVALID_VALUE":
        "one of the parameters passed to the API call is not within an acceptable range "
        "of values.",
    "HIPBLAS_STATUS_MAPPING_ERROR":
        "an internal error occurred while mapping the memory of the operands.",
    "HIPBLAS_STATUS_EXECUTION_FAILED":
        "the requested operation could not be performed on the device.",
    "HIPBLAS_STATUS_INTERNAL_ERROR": "an internal error occurred.",
    "HIPBLAS_STATUS_NOT_SUPPORTED":
        "the requested operation is not supported on this device.",
    "HIPBLAS_STATUS_ARCH_MISMATCH":
        "the architecture of the device does not support the requested operation.",
    "HIPBLAS_STATUS_HANDLE_IS_NULLPTR": "the library handle is a null pointer.",
    "HIPBLAS_STATUS_INVALID_ENUM": "an unsupported enum value was passed to the function.",
    "HIPBLAS_STATUS_UNKNOWN": "the back-end returned an unsupported status code.",
}


class _HipLibrary(Library):
    """A HIP library: the CUDA binding machinery with HIP's DLL search."""

    @classmethod
    def dll(cls)->Any:
        """The library's DLL, loaded on first use."""
        if cls._dll_instance is None:
            from ctypes import CDLL

            path:str = resolve_dll(cls._dll_name, cls._so_name, HIP_DLL_ROOTS)
            cls._dll_instance = CDLL(path)
        return cls._dll_instance


class HIPBLAS(_HipLibrary):
    """hipBLAS's entry points.

    The class is used through its metaclass, so ``HIPBLAS.hipblasSgemm`` is the bound
    function and ``HIPBLAS.hipblasCreate(ptr)`` is the call.
    """

    _library_name = "hipBLAS"
    _dll_name = "hipblas.dll"
    _so_name = "libhipblas.so"
    _signatures = hip_signatures_from(_cuda_cublas.CUBLASInfo.func_signatures, "hipblas")
    _error_codes = _status_messages(_HIPBLAS_STATUS_MESSAGES)
    _no_error_check = _no_error_check(
        _cuda_cublas.CUBLASInfo.no_need_check_error_funcs, _signatures, "hipblas")


class HIPBLASLt(_HipLibrary):
    """hipBLASLt's entry points -- the newer matmul path.

    ``INVALID_VALUE`` from this library is usually **not** a bad argument: rocBLASLt loads
    its kernels from a Tensile library file named after the device
    (``TensileLibrary_lazy_<arch>.dat``), and ROCm ships those for the discrete GPUs only.
    On a device with no such file -- many integrated ones -- every matmul fails there
    while the binding itself is fine. The message says so, because "one of the parameters
    is not within an acceptable range" points the reader at the wrong thing.
    """

    _library_name = "hipBLASLt"
    _dll_name = "hipblaslt.dll"
    _so_name = "libhipblaslt.so"
    _signatures = hip_signatures_from(_cuda_cublaslt.CUBLASLtInfo.func_signatures,
                                       "hipblaslt")
    # hipBLASLt reports hipblasStatus_t, the same codes hipBLAS does.
    _error_codes = _status_messages(_HIPBLAS_STATUS_MESSAGES)
    _no_error_check = _no_error_check(
        _cuda_cublaslt.CUBLASLtInfo.no_need_check_error_funcs, _signatures, "hipblaslt")

    @classmethod
    def _context_hint_for(cls, error_code:int, message:str)->str:
        """The message for a failure, with what the code does not say.

        This is the one code whose cause is usually the device rather than the call, so
        the explanation is attached here rather than repeated at every call site.
        """
        if error_code == hipblasStatus_t.HIPBLAS_STATUS_INVALID_VALUE:
            return (f"{message}; if the arguments look right, the cause is usually the "
                    f"device: rocBLASLt loads its kernels from a Tensile library file "
                    f"named after the GPU, and a device with no such file (many "
                    f"integrated GPUs) cannot run a matmul at all. hipblasSgemm and "
                    f"hipblasDgemm do not depend on it")
        return message
