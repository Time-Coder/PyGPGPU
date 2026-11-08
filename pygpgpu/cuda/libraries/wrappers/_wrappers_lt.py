# cuBLASLt: the heuristics-gemm path.
from __future__ import annotations
from ._wrappers_common import (
    _FLOAT32,
    _FLOAT64,
    _as_matrix,
    _element_type,
    allocate,
    ensure_context,
    free,
    synchronize,
    to_device,
    to_host,
)
import ctypes
from ctypes import byref, c_int, c_uint64, c_void_p
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from ..cublas import (
    cublasComputeType_t,
)
from ..cublaslt import (
    CUBLASLt,
    cublasLtHandle_t,
    cublasLtMatmulAlgo_t,
    cublasLtMatmulDesc_t,
    cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreference_t,
    cublasLtMatrixLayout_t,
    cublasLtOrder_t,
    cudaDataType_t as cublasLtDataType_t,
)



def lt_gemm(a:ArrayLike, b:ArrayLike, out:Optional[ArrayLike]=None, alpha:float=1.0,
            beta:float=0.0, workspace_bytes:int=8 * 1024 * 1024)->np.ndarray:
    """A general matrix multiply through cuBLASLt, the newer cuBLAS path.

    The numbers are the same as :func:`gemm`'s -- both are a matrix multiply -- but
    the route is different: cuBLASLt picks the algorithm with a heuristic instead of
    being handed a type-specific entry point, and it reads ``alpha`` and ``beta``
    from the host rather than from the device.

    That makes this the path for a mixed-precision product or a fused epilogue; for
    a plain float32 or float64 product :func:`gemm` is the simpler call.
    """
    a_matrix = _as_matrix(a)
    b_matrix = _as_matrix(b)

    dtype = np.result_type(a_matrix.dtype, b_matrix.dtype)
    if dtype not in (_FLOAT32, _FLOAT64):
        raise TypeError(f"lt_gemm supports float32 and float64, got {dtype}")

    m, k = a_matrix.shape
    k_b, n = b_matrix.shape
    if k != k_b:
        raise ValueError(
            f"cannot multiply a {m}x{k} matrix by a {k_b}x{n} matrix")

    if out is None:
        result = np.zeros((m, n), dtype=dtype)
    else:
        result = np.asarray(out)
        if result.shape != (m, n):
            raise ValueError(
                f"the output is {result.shape}, but the product is {(m, n)}")
        if result.dtype != dtype:
            result = result.astype(dtype)

    ensure_context()
    handle = cublasLtHandle_t()
    CUBLASLt.cublasLtCreate(byref(handle))

    layouts = []
    compute = cublasLtMatmulDesc_t()
    preference = cublasLtMatmulPreference_t()
    a_device = b_device = c_device = None
    workspace_device = None
    try:
        # The layouts are ROW-major, which is what numpy holds, so the operands go
        # up unchanged. The leading dimension of a row-major matrix is its number
        # of columns.
        for rows, cols in ((m, k), (k, n), (m, n), (m, n)):
            layout = cublasLtMatrixLayout_t()
            CUBLASLt.cublasLtMatrixLayoutCreate(
                byref(layout), int(cublasLtDataType_t.CUDA_R_32F
                                   if dtype == _FLOAT32
                                   else cublasLtDataType_t.CUDA_R_64F),
                rows, cols, cols)
            order = c_int(int(cublasLtOrder_t.CUBLASLT_ORDER_ROW))
            CUBLASLt.cublasLtMatrixLayoutSetAttribute(
                layout, 1, byref(order), ctypes.sizeof(order))
            layouts.append(layout)

        CUBLASLt.cublasLtMatmulDescCreate(
            byref(compute),
            int(cublasComputeType_t.CUBLAS_COMPUTE_32F if dtype == _FLOAT32
                else cublasComputeType_t.CUBLAS_COMPUTE_64F),
            int(cublasLtDataType_t.CUDA_R_32F if dtype == _FLOAT32
                else cublasLtDataType_t.CUDA_R_64F))

        CUBLASLt.cublasLtMatmulPreferenceCreate(byref(preference))
        max_workspace = c_uint64(workspace_bytes)
        CUBLASLt.cublasLtMatmulPreferenceSetAttribute(
            preference, 1, ctypes.cast(byref(max_workspace), c_void_p),
            ctypes.sizeof(max_workspace))

        # The heuristic picks the algorithm; the first one it reports is the one
        # the library considers best.
        results = (cublasLtMatmulHeuristicResult_t * 8)()
        count = c_int(0)
        CUBLASLt.cublasLtMatmulAlgoGetHeuristic(
            handle, compute, layouts[0], layouts[1], layouts[2], layouts[3],
            preference, 8, results, byref(count))
        if count.value <= 0:
            raise RuntimeError(
                "cuBLASLt found no algorithm for this matrix multiply")

        algo = cublasLtMatmulAlgo_t()
        ctypes.memmove(ctypes.byref(algo), ctypes.byref(results[0].algo),
                       ctypes.sizeof(cublasLtMatmulAlgo_t))
        workspace_size = int(results[0].workspaceSize)

        a_device = allocate(a_matrix.nbytes)
        b_device = allocate(b_matrix.nbytes)
        c_device = allocate(result.nbytes)
        workspace_device = allocate(max(workspace_size, 1))
        to_device(a_device, a_matrix)
        to_device(b_device, b_matrix)
        to_device(c_device, result)

        # cuBLASLt reads the scalars from the host, unlike the classic API whose
        # scalars live on the device in the default pointer mode.
        element = _element_type(dtype)
        alpha_scalar = element(alpha)
        beta_scalar = element(beta)

        CUBLASLt.cublasLtMatmul(
            handle, compute,
            ctypes.cast(byref(alpha_scalar), c_void_p),
            ctypes.cast(a_device, c_void_p), layouts[0],
            ctypes.cast(b_device, c_void_p), layouts[1],
            ctypes.cast(byref(beta_scalar), c_void_p),
            ctypes.cast(c_device, c_void_p), layouts[2],
            ctypes.cast(c_device, c_void_p), layouts[3],
            byref(algo), workspace_device, workspace_size, c_void_p(0))

        synchronize()
        to_host(c_device, result)
    finally:
        for layout in layouts:
            CUBLASLt.cublasLtMatrixLayoutDestroy(layout)
        CUBLASLt.cublasLtMatmulPreferenceDestroy(preference)
        CUBLASLt.cublasLtMatmulDescDestroy(compute)
        CUBLASLt.cublasLtDestroy(handle)
        if workspace_device is not None:
            free(workspace_device)
        if c_device is not None:
            free(c_device)
        if b_device is not None:
            free(b_device)
        if a_device is not None:
            free(a_device)

    return result
