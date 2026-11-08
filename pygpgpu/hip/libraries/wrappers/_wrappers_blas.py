# hipBLAS: gemm, axpy, scal, dot.
from __future__ import annotations
from ._wrappers_common import (
    _device_pointer,
    _is_double,
    _is_single,
    _scalar_pointer,
    create_handle,
    destroy_handle,
    ensure_context,
    free,
    synchronize,
    to_device,
    to_host,
)
from ctypes import POINTER, c_float, c_int, c_void_p
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from .._enums import hipblasOperation_t
from ..hipblas import HIPBLAS



def gemm(a:ArrayLike, b:ArrayLike, out:Optional[ArrayLike]=None, alpha:float=1.0, beta:float=0.0,
         transpose_a:bool=False, transpose_b:bool=False)->np.ndarray:
    """A general matrix multiply: ``out = alpha * op(a) @ op(b) + beta * out``.

    ``a`` and ``b`` are numpy arrays, and the shapes follow numpy's own rules for ``@``:
    a 1-D array is a column when it is on the left and a row when it is on the right, and
    the product of two 1-D arrays is a scalar.

    The operands are numpy's **row-major** layout and the result is row-major too.
    hipBLAS reads column-major, so the multiply is handed to it as
    ``op(b).T @ op(a).T`` and the result comes back transposed -- the same numbers,
    without making the caller think about the layout difference.
    """
    ensure_context()

    a_array = np.asarray(a)
    b_array = np.asarray(b)
    if a_array.ndim not in (1, 2):
        raise ValueError(f"expected a 1-D or 2-D matrix, got {a_array.ndim} dimensions")
    if b_array.ndim not in (1, 2):
        raise ValueError(f"expected a 1-D or 2-D matrix, got {b_array.ndim} dimensions")
    if a_array.dtype != b_array.dtype:
        raise TypeError(
            f"the two operands must have one dtype, got {a_array.dtype} and {b_array.dtype}")
    dtype = a_array.dtype
    if not (_is_single(dtype) or _is_double(dtype)):
        raise TypeError(f"gemm() takes float32 or float64 arrays, got {dtype}")

    a_is_vector = a_array.ndim == 1
    b_is_vector = b_array.ndim == 1

    # The shapes BLAS sees: op(a) is (m, k) and op(b) is (k, n). A 1-D operand follows
    # numpy's rule for `@`: a row on the left, a column on the right.
    if a_is_vector:
        a_shape = (1, a_array.shape[0])
    elif transpose_a:
        a_shape = (a_array.shape[1], a_array.shape[0])
    else:
        a_shape = a_array.shape
    if b_is_vector:
        b_shape = (b_array.shape[0], 1)
    elif transpose_b:
        b_shape = (b_array.shape[1], b_array.shape[0])
    else:
        b_shape = b_array.shape

    if a_shape[1] != b_shape[0]:
        raise ValueError(
            f"the shapes do not multiply: {a_shape} and {b_shape}")

    m, k = a_shape
    n = b_shape[1]

    # The column-major trick, spelled out. A row-major (R, C) array has the same memory
    # as the column-major matrix of shape (C, R) that equals its transpose, so handing
    # BLAS the memory of `b` as its A and the memory of `a` as its B makes it compute
    # `b^T @ a^T`, which is `(a @ b)^T` -- and that transpose is exactly the memory of the
    # row-major result. The leading dimensions follow from the same fact: the memory of an
    # (R, C) array read as column-major has C rows, so its lda is C.
    a_operand = np.ascontiguousarray(a_array if not transpose_a else a_array.T)
    b_operand = np.ascontiguousarray(b_array if not transpose_b else b_array.T)
    a_operand_buffer = to_device(a_operand)
    b_operand_buffer = to_device(b_operand)

    result_shape = (m, n)
    if out is None:
        result = np.zeros(result_shape, dtype=dtype)
    else:
        result = np.ascontiguousarray(out)
        if result.shape != result_shape:
            raise ValueError(
                f"out has shape {result.shape}, but the product is {result_shape}")
    result_buffer = to_device(result)

    # BLAS's own dimensions: A is m x k, B is k x n, C is m x n. With b's memory as A,
    # that is (N, K); with a's memory as B, (K, M). So BLAS's m, n, k are n, m, k here.
    blas_m, blas_n, blas_k = n, m, k
    # b's memory read as a (N, K) column-major matrix has N rows, and a's as (K, M) has K.
    lda = n
    ldb = k

    handle = create_handle()
    try:
        op = c_int(hipblasOperation_t.HIPBLAS_OP_N.value)
        alpha_holder, _alpha_keeper = _scalar_pointer(alpha, dtype)
        beta_holder, _beta_keeper = _scalar_pointer(beta, dtype)
        typed = POINTER(c_float) if _is_single(dtype) else _double_pointer()
        _gemm_call(
            handle, op, op,
            blas_m, blas_n, blas_k,
            alpha_holder,
            _device_pointer(b_operand_buffer, typed), lda,
            _device_pointer(a_operand_buffer, typed), ldb,
            beta_holder,
            _device_pointer(result_buffer, typed), lda,
            dtype)
        synchronize()
        to_host(result_buffer, result)
    finally:
        destroy_handle(handle)
        free(a_operand_buffer)
        free(b_operand_buffer)
        free(result_buffer)

    if a_is_vector and b_is_vector:
        return result.reshape(-1)[:1][0]
    if a_is_vector:
        return result.reshape(n)
    if b_is_vector:
        return result.reshape(m)
    return result


def _double_pointer()->type:
    from ctypes import c_double

    return POINTER(c_double)


def _gemm_call(handle:c_void_p, op_a:object, op_b:object, m:int, n:int, k:int,
               alpha:object, A:object, lda:int, B:object, ldb:int, beta:object,
               C:object, ldc:int, dtype:np.dtype)->None:
    """One gemm, in the single or the double precision entry point."""
    if _is_single(dtype):
        HIPBLAS.hipblasSgemm(handle, op_a, op_b, c_int(m), c_int(n), c_int(k),
                             alpha, A, c_int(lda), B, c_int(ldb), beta, C, c_int(ldc))
        return
    HIPBLAS.hipblasDgemm(handle, op_a, op_b, c_int(m), c_int(n), c_int(k),
                         alpha, A, c_int(lda), B, c_int(ldb), beta, C, c_int(ldc))


def axpy(alpha:float, x:ArrayLike, y:ArrayLike)->np.ndarray:
    """``y = alpha * x + y``, elementwise, and return ``y``."""
    ensure_context()
    x_array = np.ascontiguousarray(x).reshape(-1)
    y_array = np.ascontiguousarray(y).reshape(-1)
    if x_array.shape != y_array.shape:
        raise ValueError(f"axpy() operands must have one length, got "
                         f"{x_array.shape} and {y_array.shape}")
    dtype = x_array.dtype
    if not (_is_single(dtype) or _is_double(dtype)):
        raise TypeError(f"axpy() takes float32 or float64 arrays, got {dtype}")

    x_buffer = to_device(x_array)
    y_buffer = to_device(y_array)
    handle = create_handle()
    try:
        alpha_holder, _ = _scalar_pointer(alpha, dtype)
        typed = POINTER(c_float) if _is_single(dtype) else _double_pointer()
        if _is_single(dtype):
            HIPBLAS.hipblasSaxpy(handle, c_int(x_array.size), alpha_holder,
                                  _device_pointer(x_buffer, typed), c_int(1),
                                  _device_pointer(y_buffer, typed), c_int(1))
        else:
            HIPBLAS.hipblasDaxpy(handle, c_int(x_array.size), alpha_holder,
                                  _device_pointer(x_buffer, typed), c_int(1),
                                  _device_pointer(y_buffer, typed), c_int(1))
        synchronize()
        to_host(y_buffer, y_array)
    finally:
        destroy_handle(handle)
        free(x_buffer)
        free(y_buffer)
    return y_array.reshape(np.shape(y))


def scal(alpha:float, x:ArrayLike)->np.ndarray:
    """``x = alpha * x``, elementwise, and return ``x``."""
    ensure_context()
    x_array = np.ascontiguousarray(x).reshape(-1)
    dtype = x_array.dtype
    if not (_is_single(dtype) or _is_double(dtype)):
        raise TypeError(f"scal() takes a float32 or float64 array, got {dtype}")

    x_buffer = to_device(x_array)
    handle = create_handle()
    try:
        alpha_holder, _ = _scalar_pointer(alpha, dtype)
        typed = POINTER(c_float) if _is_single(dtype) else _double_pointer()
        if _is_single(dtype):
            HIPBLAS.hipblasSscal(handle, c_int(x_array.size), alpha_holder,
                                  _device_pointer(x_buffer, typed), c_int(1))
        else:
            HIPBLAS.hipblasDscal(handle, c_int(x_array.size), alpha_holder,
                                  _device_pointer(x_buffer, typed), c_int(1))
        synchronize()
        to_host(x_buffer, x_array)
    finally:
        destroy_handle(handle)
        free(x_buffer)
    return x_array.reshape(np.shape(x))


def dot(x:ArrayLike, y:ArrayLike)->float:
    """The dot product of two 1-D arrays."""
    ensure_context()
    x_array = np.ascontiguousarray(x).reshape(-1)
    y_array = np.ascontiguousarray(y).reshape(-1)
    if x_array.shape != y_array.shape:
        raise ValueError(f"dot() operands must have one length, got "
                         f"{x_array.shape} and {y_array.shape}")
    dtype = x_array.dtype
    if not (_is_single(dtype) or _is_double(dtype)):
        raise TypeError(f"dot() takes float32 or float64 arrays, got {dtype}")

    x_buffer = to_device(x_array)
    y_buffer = to_device(y_array)
    result = np.zeros(1, dtype=dtype)
    result_buffer = to_device(result)
    handle = create_handle()
    try:
        typed = POINTER(c_float) if _is_single(dtype) else _double_pointer()
        if _is_single(dtype):
            HIPBLAS.hipblasSdot(handle, c_int(x_array.size),
                                 _device_pointer(x_buffer, typed), c_int(1),
                                 _device_pointer(y_buffer, typed), c_int(1),
                                 _device_pointer(result_buffer, typed))
        else:
            HIPBLAS.hipblasDdot(handle, c_int(x_array.size),
                                 _device_pointer(x_buffer, typed), c_int(1),
                                 _device_pointer(y_buffer, typed), c_int(1),
                                 _device_pointer(result_buffer, typed))
        synchronize()
        to_host(result_buffer, result)
    finally:
        destroy_handle(handle)
        free(x_buffer)
        free(y_buffer)
        free(result_buffer)
    return float(result[0])
