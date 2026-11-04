"""hipSPARSE -- the sparse linear algebra library, bound with ctypes.

The signature table is written here rather than derived from cuSPARSE's, because two
entries genuinely differ in arity:

* ``hipsparseGetVersion`` takes ``(handle, int* version)`` where the cuSPARSE table lists
  one argument;
* ``hipsparseCreateDnMat`` takes an ``order`` as its seventh argument where the cuSPARSE
  table lists six.

The rest has the same shape, and the *descriptors* are opaque pointers either way, so the
`const` spellings HIP uses (`hipsparseConstSpMatDescr_t` for the read-only descriptors of
a SpMV) are the same ctypes type as the mutable ones.

The declarations these come from are ROCm's ``include/hipsparse/hipsparse.h`` and
``hipsparse-generic-*.h``.

**This ROCm install ships no rocSPARSE kernels.** `hipsparse.dll` loads and every entry
point binds, but `hipsparseCreate` answers `HIPSPARSE_STATUS_INTERNAL_ERROR` and leaves a
null handle -- because rocSPARSE resolves its kernels out of a runtime directory beside
the DLL, and ROCm 7.2's Windows install creates one for rocBLAS and rocFFT and none for
rocSPARSE (`bin/rocsparse` does not exist, while `bin/rocblas/library` does). So the
binding is verified and the *library* cannot run here; `tests/test_hip_libraries.py` skips
its hipSPARSE case with that reason rather than failing.
"""
from __future__ import annotations

from ctypes import POINTER, c_char_p, c_int, c_size_t, c_void_p, c_int64
from typing import Any, Dict

from ...libraries import Library, resolve_dll
from ._enums import hipsparseStatus_t
from .hipblas import HIP_DLL_ROOTS

__all__ = ["HIPSPARSE"]


# The handle and the descriptors. All opaque pointers.
hipsparseHandle_t = c_void_p
ptr_hipsparseHandle_t = POINTER(hipsparseHandle_t)
hipsparseSpMatDescr_t = c_void_p
ptr_hipsparseSpMatDescr_t = POINTER(hipsparseSpMatDescr_t)
hipsparseDnVecDescr_t = c_void_p
ptr_hipsparseDnVecDescr_t = POINTER(hipsparseDnVecDescr_t)
hipsparseDnMatDescr_t = c_void_p
ptr_hipsparseDnMatDescr_t = POINTER(hipsparseDnMatDescr_t)


_HIPSPARSE_STATUS_MESSAGES:Dict[str, str] = {
    "HIPSPARSE_STATUS_SUCCESS": "no errors.",
    "HIPSPARSE_STATUS_NOT_INITIALIZED": "the library was not initialized.",
    "HIPSPARSE_STATUS_ALLOC_FAILED": "the library could not allocate enough memory.",
    "HIPSPARSE_STATUS_INVALID_VALUE":
        "one of the parameters is not within an acceptable range.",
    "HIPSPARSE_STATUS_ARCH_MISMATCH":
        "the architecture of the device does not support the requested operation.",
    "HIPSPARSE_STATUS_MAPPING_ERROR":
        "an internal error occurred while mapping the memory of the operands.",
    "HIPSPARSE_STATUS_EXECUTION_FAILED":
        "the requested operation could not be performed on the device.",
    "HIPSPARSE_STATUS_INTERNAL_ERROR": "an internal error occurred.",
    "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED": "the matrix type is not supported.",
    "HIPSPARSE_STATUS_ZERO_PIVOT": "a zero pivot was encountered.",
    "HIPSPARSE_STATUS_NOT_SUPPORTED": "the requested operation is not supported.",
    "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES": "there are not enough resources.",
}


def _status_messages()->Dict[int, str]:
    return {int(member.value): _HIPSPARSE_STATUS_MESSAGES.get(
        member.name, f"{member.name.replace('_', ' ').lower()}.")
        for member in hipsparseStatus_t}


class HIPSPARSE(Library):
    """hipSPARSE's entry points.

    The class is used through its metaclass, so ``HIPSPARSE.hipsparseSpMV`` is the bound
    function and ``HIPSPARSE.hipsparseCreate(byref(handle))`` is the call.
    """

    _library_name = "hipSPARSE"
    _dll_name = "hipsparse.dll"
    _so_name = "libhipsparse.so"

    _signatures:Dict[str, Dict[str, Any]] = {
        # hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle)
        "hipsparseCreate": {
            "args": {"handle": ptr_hipsparseHandle_t},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
        "hipsparseDestroy": {
            "args": {"handle": hipsparseHandle_t},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle,
        #     hipStream_t streamId)
        "hipsparseSetStream": {
            "args": {"handle": hipsparseHandle_t, "streamId": c_void_p},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseGetVersion(hipsparseHandle_t handle, int* version)
        #
        # The handle is the FIRST argument -- cuSPARSE's table lists only the version.
        "hipsparseGetVersion": {
            "args": {"handle": hipsparseHandle_t, "version": POINTER(c_int)},
            "restype": c_int,
        },
        # const char* hipsparseGetErrorString(hipsparseStatus_t status)
        #
        # The return value is the message itself, not a status.
        "hipsparseGetErrorString": {
            "args": {"status": c_int},
            "restype": c_char_p,
        },
        # hipsparseStatus_t hipsparseCreateCsr(hipsparseSpMatDescr_t* spMatDescr,
        #     int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets,
        #     void* csrColInd, void* csrValues, hipsparseIndexType_t csrRowOffsetsType,
        #     hipsparseIndexType_t csrColIndType, hipsparseIndexBase_t idxBase,
        #     hipDataType valueType)
        "hipsparseCreateCsr": {
            "args": {"spMatDescr": ptr_hipsparseSpMatDescr_t, "rows": c_int64,
                     "cols": c_int64, "nnz": c_int64, "csrRowOffsets": c_void_p,
                     "csrColInd": c_void_p, "csrValues": c_void_p,
                     "csrRowOffsetsType": c_int, "csrColIndType": c_int,
                     "idxBase": c_int, "valueType": c_int},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseDestroySpMat(hipsparseConstSpMatDescr_t spMatDescr)
        "hipsparseDestroySpMat": {
            "args": {"spMatDescr": hipsparseSpMatDescr_t},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr,
        #     int64_t size, void* values, hipDataType valueType)
        "hipsparseCreateDnVec": {
            "args": {"dnVecDescr": ptr_hipsparseDnVecDescr_t, "size": c_int64,
                     "values": c_void_p, "valueType": c_int},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr)
        "hipsparseDestroyDnVec": {
            "args": {"dnVecDescr": hipsparseDnVecDescr_t},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t handle,
        #     hipsparseOperation_t opA, const void* alpha,
        #     hipsparseConstSpMatDescr_t matA, hipsparseConstDnVecDescr_t vecX,
        #     const void* beta, const hipsparseDnVecDescr_t vecY,
        #     hipDataType computeType, hipsparseSpMVAlg_t alg, size_t* pBufferSizeInBytes)
        "hipsparseSpMV_bufferSize": {
            "args": {"handle": hipsparseHandle_t, "opA": c_int, "alpha": c_void_p,
                     "matA": hipsparseSpMatDescr_t, "vecX": hipsparseDnVecDescr_t,
                     "beta": c_void_p, "vecY": hipsparseDnVecDescr_t,
                     "computeType": c_int, "alg": c_int,
                     "pBufferSizeInBytes": POINTER(c_size_t)},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseSpMV_preprocess(... same, with void* externalBuffer)
        "hipsparseSpMV_preprocess": {
            "args": {"handle": hipsparseHandle_t, "opA": c_int, "alpha": c_void_p,
                     "matA": hipsparseSpMatDescr_t, "vecX": hipsparseDnVecDescr_t,
                     "beta": c_void_p, "vecY": hipsparseDnVecDescr_t,
                     "computeType": c_int, "alg": c_int, "externalBuffer": c_void_p},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseSpMV(... same, with void* externalBuffer)
        "hipsparseSpMV": {
            "args": {"handle": hipsparseHandle_t, "opA": c_int, "alpha": c_void_p,
                     "matA": hipsparseSpMatDescr_t, "vecX": hipsparseDnVecDescr_t,
                     "beta": c_void_p, "vecY": hipsparseDnVecDescr_t,
                     "computeType": c_int, "alg": c_int, "externalBuffer": c_void_p},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr,
        #     int64_t rows, int64_t cols, int64_t ld, void* values,
        #     hipDataType valueType, hipsparseOrder_t order)
        #
        # The `order` is HIP's seventh argument; cuSPARSE's table here lists six.
        "hipsparseCreateDnMat": {
            "args": {"dnMatDescr": ptr_hipsparseDnMatDescr_t, "rows": c_int64,
                     "cols": c_int64, "ld": c_int64, "values": c_void_p,
                     "valueType": c_int, "order": c_int},
            "restype": c_int,
        },
        # hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr)
        "hipsparseDestroyDnMat": {
            "args": {"dnMatDescr": hipsparseDnMatDescr_t},
            "restype": c_int,
        },
    }

    _error_codes = _status_messages()
    # The version is a version and the message is a string: neither is a status.
    _no_error_check = ["hipsparseGetVersion", "hipsparseGetErrorString"]

    @classmethod
    def dll(cls)->Any:
        """The library's DLL, loaded on first use."""
        if cls._dll_instance is None:
            from ctypes import CDLL

            cls._dll_instance = CDLL(resolve_dll(cls._dll_name, cls._so_name,
                                                 HIP_DLL_ROOTS))
        return cls._dll_instance
