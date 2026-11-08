# hipBLASLt: the heuristics-gemm path.
from __future__ import annotations
from ._wrappers_blas import _double_pointer
from ._wrappers_common import (
    _device_pointer,
    _is_double,
    _is_single,
    _scalar_pointer,
    ensure_context,
    free,
    synchronize,
    to_device,
    to_host,
)
from ctypes import POINTER, byref, c_float, c_void_p
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from ..hipblas import HIPBLASLt



def lt_gemm(a:ArrayLike, b:ArrayLike, out:Optional[ArrayLike]=None, alpha:float=1.0, beta:float=0.0,
            transpose_a:bool=False, transpose_b:bool=False)->np.ndarray:
    """A general matrix multiply through hipBLASLt, the newer matmul path.

    The result is the same as :func:`gemm`'s; the difference is which implementation runs
    and that hipBLASLt picks its own algorithm.
    """
    ensure_context()

    a_array = np.asarray(a)
    b_array = np.asarray(b)
    if a_array.ndim != 2 or b_array.ndim != 2:
        raise ValueError("lt_gemm() takes two 2-D matrices; gemm() handles the 1-D cases")
    if a_array.dtype != b_array.dtype:
        raise TypeError(
            f"the two operands must have one dtype, got {a_array.dtype} and {b_array.dtype}")
    dtype = a_array.dtype
    if not (_is_single(dtype) or _is_double(dtype)):
        raise TypeError(f"lt_gemm() takes float32 or float64 arrays, got {dtype}")

    a_shape = (a_array.shape[1], a_array.shape[0]) if transpose_a else a_array.shape
    b_shape = (b_array.shape[1], b_array.shape[0]) if transpose_b else b_array.shape
    if a_shape[1] != b_shape[0]:
        raise ValueError(f"the shapes do not multiply: {a_shape} and {b_shape}")
    m, k = a_shape
    n = b_shape[1]

    result_shape = (m, n)
    result = np.zeros(result_shape, dtype=dtype) if out is None \
        else np.ascontiguousarray(out)
    if result.shape != result_shape:
        raise ValueError(f"out has shape {result.shape}, but the product is {result_shape}")

    a_operand = np.ascontiguousarray(a_array if not transpose_a else a_array.T)
    b_operand = np.ascontiguousarray(b_array if not transpose_b else b_array.T)
    a_buffer = to_device(a_operand)
    b_buffer = to_device(b_operand)
    c_buffer = to_device(result)

    # The same column-major trick gemm() uses: b's memory read column-major is (N, K) and
    # a's is (K, M), so the product is b^T @ a^T = (a @ b)^T, whose memory IS the row-major
    # result. The layouts describe those column-major matrices, so their leading dimension
    # is the row count.
    #
    # Plain ints, not ctypes objects: hipBLASLt declares these as c_long / c_ulonglong,
    # and ctypes only accepts a value whose type matches the declared one exactly.
    datatype = 0 if _is_single(dtype) else 1            # HIP_R_32F / HIP_R_64F
    compute_type = 2 if _is_single(dtype) else 7        # HIPBLAS_COMPUTE_32F / 64F

    handle = c_void_p()
    layout_a = c_void_p()
    layout_b = c_void_p()
    layout_c = c_void_p()
    matmul = c_void_p()
    try:
        HIPBLASLt.hipblasLtCreate(byref(handle))
        # A <- b's memory, seen column-major as (N, K)
        HIPBLASLt.hipblasLtMatrixLayoutCreate(byref(layout_a), datatype, n, k, n)
        # B <- a's memory, seen column-major as (K, M)
        HIPBLASLt.hipblasLtMatrixLayoutCreate(byref(layout_b), datatype, k, m, k)
        # C and D <- the result's memory, seen column-major as (N, M)
        HIPBLASLt.hipblasLtMatrixLayoutCreate(byref(layout_c), datatype, n, m, n)
        HIPBLASLt.hipblasLtMatmulDescCreate(byref(matmul), compute_type, datatype)

        alpha_holder, _alpha_keeper = _scalar_pointer(alpha, dtype)
        beta_holder, _beta_keeper = _scalar_pointer(beta, dtype)
        typed = POINTER(c_float) if _is_single(dtype) else _double_pointer()
        # The algorithm is left to the library (`algo=None` picks its default), and no
        # workspace is requested.
        HIPBLASLt.hipblasLtMatmul(
            handle, matmul,
            alpha_holder,
            _device_pointer(b_buffer, typed), layout_a,
            _device_pointer(a_buffer, typed), layout_b,
            beta_holder,
            _device_pointer(c_buffer, typed), layout_c,
            _device_pointer(c_buffer, typed), layout_c,
            None, c_void_p(), 0, c_void_p())
        synchronize()
        to_host(c_buffer, result)
    finally:
        for destroy, pointer in ((HIPBLASLt.hipblasLtMatmulDescDestroy, matmul),
                                 (HIPBLASLt.hipblasLtMatrixLayoutDestroy, layout_a),
                                 (HIPBLASLt.hipblasLtMatrixLayoutDestroy, layout_b),
                                 (HIPBLASLt.hipblasLtMatrixLayoutDestroy, layout_c)):
            try:
                destroy(pointer)
            except Exception:
                pass
        try:
            HIPBLASLt.hipblasLtDestroy(handle)
        except Exception:
            pass
        free(a_buffer)
        free(b_buffer)
        free(c_buffer)
    return result
