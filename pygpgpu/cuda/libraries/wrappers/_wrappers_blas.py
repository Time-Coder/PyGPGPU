# cuBLAS: gemm, axpy, scal, dot.
from __future__ import annotations
from ._wrappers_common import (
    _FLOAT32,
    _FLOAT64,
    _element_type,
    allocate,
    create_handle,
    destroy_handle,
    free,
    synchronize,
    to_device,
    to_host,
)
import ctypes
from ctypes import byref, c_float, POINTER
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from ..cublas import (
    CUBLAS,
    cublasOperation_t,
)



def gemm(a:ArrayLike, b:ArrayLike, out:Optional[ArrayLike]=None, alpha:float=1.0, beta:float=0.0,
         transpose_a:bool=False, transpose_b:bool=False)->np.ndarray:
    """A general matrix multiply: ``out = alpha * op(a) @ op(b) + beta * out``.

    ``a`` and ``b`` are numpy arrays, and the shapes follow numpy's own rules for
    ``@``: a 1-D array is a column when it is on the left and a row when it is on the
    right, and the product of two 1-D arrays is a scalar. The result is a new array
    unless ``out`` is given, in which case it is written into ``out`` and returned.

    The operands are numpy's **row-major** layout and the result is row-major too.
    cuBLAS reads column-major, so the multiply is handed to it as
    ``op(b).T @ op(a).T`` and the result comes back transposed -- the same numbers,
    without making the caller think about the layout difference.
    """
    a_array = np.asarray(a)
    b_array = np.asarray(b)

    if a_array.ndim not in (1, 2):
        raise ValueError(f"expected a 1-D or 2-D matrix, got {a_array.ndim} dimensions")
    if b_array.ndim not in (1, 2):
        raise ValueError(f"expected a 1-D or 2-D matrix, got {b_array.ndim} dimensions")

    a_is_vector = a_array.ndim == 1
    b_is_vector = b_array.ndim == 1

    # The shapes as BLAS sees them: op(a) is (m, k) and op(b) is (k, n). A 1-D
    # operand follows numpy's rule for ``@``: a row on the left and a column on the
    # right. numpy's own transpose of a 1-D array is the array, so a transpose
    # leaves a 1-D operand alone.
    if a_is_vector:
        a_matrix_shape = (1, a_array.shape[0])
    elif transpose_a:
        a_matrix_shape = (a_array.shape[1], a_array.shape[0])
    else:
        a_matrix_shape = a_array.shape

    if b_is_vector:
        b_matrix_shape = (b_array.shape[0], 1)
    elif transpose_b:
        b_matrix_shape = (b_array.shape[1], b_array.shape[0])
    else:
        b_matrix_shape = b_array.shape

    m, k = a_matrix_shape
    k_b, n = b_matrix_shape
    if k != k_b:
        raise ValueError(
            f"cannot multiply a {a_matrix_shape} matrix by a {b_matrix_shape} matrix")

    # One dtype for both operands, as HIP refuses the mixed case: `result_type`
    # would widen float32 x float64 to float64 and have cuBLAS read the float32
    # bytes as doubles -- wrong numbers with no error.
    if a_array.dtype != b_array.dtype:
        raise TypeError(
            f"the two operands must have one dtype, got {a_array.dtype} and {b_array.dtype}")
    dtype = a_array.dtype
    if dtype not in (_FLOAT32, _FLOAT64):
        raise TypeError(f"gemm supports float32 and float64, got {dtype}")

    # The operands as the 2-D matrices BLAS multiplies. A 1-D operand keeps its
    # bytes; only its interpretation changes.
    a_matrix = np.ascontiguousarray(
        a_array.T if (transpose_a and not a_is_vector) else a_array
    ).reshape(a_matrix_shape)
    b_matrix = np.ascontiguousarray(
        b_array.T if (transpose_b and not b_is_vector) else b_array
    ).reshape(b_matrix_shape)

    # numpy's own `out=` contract: the caller's array receives the product, at any
    # dtype numpy can safely cast and at any layout. `beta` also *reads* C, so the
    # buffer BLAS writes must start from the caller's values -- the temp below is
    # `out.astype(dtype)`, not zeros. The temp keeps the dtype and layout BLAS
    # writes: a differently-typed or non-contiguous `out` cannot be written through
    # directly, so the copy into it happens after the call.
    out_array:Optional[np.ndarray] = None
    if out is not None:
        out_array = np.asarray(out)
        if out_array.shape != (m, n):
            raise ValueError(
                f"the output is {out_array.shape}, but the product is {(m, n)}")
        result = out_array.astype(dtype, copy=True)
    else:
        result = np.zeros((m, n), dtype=dtype)

    # BLAS reads column-major and numpy holds row-major, so the row-major product
    # C = A @ B is the column-major product C' = B' @ A'. The bytes of a
    # C-contiguous row-major (r, c) array are exactly the column-major (c, r)
    # matrix, so no copy is needed: the operands are uploaded as they are, and the
    # leading dimension of each is its number of COLUMNS.
    #
    # In the column-major picture the operands are the transposed ones, so neither
    # operation flag is set, and the roles are swapped: B' multiplies A'.
    lda = b_matrix.shape[1]
    ldb = a_matrix.shape[1]
    ldc = result.shape[1]

    a_device = allocate(a_matrix.nbytes)
    b_device = allocate(b_matrix.nbytes)
    c_device = allocate(result.nbytes)
    handle = create_handle()
    try:
        to_device(a_device, a_matrix)
        to_device(b_device, b_matrix)
        to_device(c_device, result)

        transa = int(cublasOperation_t.CUBLAS_OP_N)
        transb = int(cublasOperation_t.CUBLAS_OP_N)

        element = _element_type(dtype)
        scalar = element(alpha)
        beta_scalar = element(beta)
        a_pointer = ctypes.cast(a_device, POINTER(element))
        b_pointer = ctypes.cast(b_device, POINTER(element))
        c_pointer = ctypes.cast(c_device, POINTER(element))

        if element is c_float:
            CUBLAS.cublasSgemm_v2(handle, transa, transb, n, m, k, byref(scalar),
                                  b_pointer, lda, a_pointer, ldb, byref(beta_scalar),
                                  c_pointer, ldc)
        else:
            CUBLAS.cublasDgemm_v2(handle, transa, transb, n, m, k, byref(scalar),
                                  b_pointer, lda, a_pointer, ldb, byref(beta_scalar),
                                  c_pointer, ldc)

        synchronize()
        to_host(c_device, result)
        if out_array is not None:
            np.copyto(out_array, result)
    finally:
        destroy_handle(handle)
        free(a_device)
        free(b_device)
        free(c_device)

    # The result takes numpy's shape for the product, not BLAS's. An `out=` call
    # answers the caller's own array, the way `np.matmul(..., out=...)` does.
    answer:np.ndarray = out_array if out_array is not None else result
    if a_is_vector and b_is_vector:
        return answer.reshape(())
    if a_is_vector:
        return answer.reshape(n)
    if b_is_vector:
        return answer.reshape(m)
    return answer


def axpy(alpha:float, x:ArrayLike, y:ArrayLike)->np.ndarray:
    """``y = alpha * x + y``, elementwise.

    ``y`` is updated in place and returned.
    """
    x_array = np.ascontiguousarray(x)
    y_array = np.ascontiguousarray(y)
    if x_array.shape != y_array.shape:
        raise ValueError(
            f"cannot combine a {x_array.shape} array with a {y_array.shape} array")

    dtype = np.result_type(x_array.dtype, y_array.dtype)
    if dtype not in (_FLOAT32, _FLOAT64):
        raise TypeError(f"axpy supports float32 and float64, got {dtype}")

    element = _element_type(dtype)
    x_device = allocate(x_array.nbytes)
    y_device = allocate(y_array.nbytes)
    handle = create_handle()
    try:
        to_device(x_device, x_array)
        to_device(y_device, y_array)
        scalar = element(alpha)
        x_pointer = ctypes.cast(x_device, POINTER(element))
        y_pointer = ctypes.cast(y_device, POINTER(element))
        count = x_array.size

        if element is c_float:
            CUBLAS.cublasSaxpy_v2(handle, count, byref(scalar), x_pointer, 1,
                                  y_pointer, 1)
        else:
            CUBLAS.cublasDaxpy_v2(handle, count, byref(scalar), x_pointer, 1,
                                  y_pointer, 1)

        synchronize()
        to_host(y_device, y_array)
    finally:
        destroy_handle(handle)
        free(x_device)
        free(y_device)

    return y_array


def scal(alpha:float, x:ArrayLike)->np.ndarray:
    """``x = alpha * x``, elementwise.

    ``x`` is updated in place and returned.
    """
    x_array = np.ascontiguousarray(x)
    dtype = x_array.dtype
    if dtype not in (_FLOAT32, _FLOAT64):
        raise TypeError(f"scal supports float32 and float64, got {dtype}")

    element = _element_type(dtype)
    x_device = allocate(x_array.nbytes)
    handle = create_handle()
    try:
        to_device(x_device, x_array)
        scalar = element(alpha)
        x_pointer = ctypes.cast(x_device, POINTER(element))

        if element is c_float:
            CUBLAS.cublasSscal_v2(handle, x_array.size, byref(scalar), x_pointer, 1)
        else:
            CUBLAS.cublasDscal_v2(handle, x_array.size, byref(scalar), x_pointer, 1)

        synchronize()
        to_host(x_device, x_array)
    finally:
        destroy_handle(handle)
        free(x_device)

    return x_array


def dot(x:ArrayLike, y:ArrayLike)->float:
    """The dot product of two 1-D arrays.

    cuBLAS writes the result to the device, so it is read back from there.
    """
    x_array = np.ascontiguousarray(x)
    y_array = np.ascontiguousarray(y)
    if x_array.shape != y_array.shape:
        raise ValueError(
            f"cannot take the dot product of a {x_array.shape} array and a "
            f"{y_array.shape} array")

    dtype = np.result_type(x_array.dtype, y_array.dtype)
    if dtype not in (_FLOAT32, _FLOAT64):
        raise TypeError(f"dot supports float32 and float64, got {dtype}")

    element = _element_type(dtype)
    x_device = allocate(x_array.nbytes)
    y_device = allocate(y_array.nbytes)
    result_device = allocate(np.dtype(dtype).itemsize)
    handle = create_handle()
    try:
        to_device(x_device, x_array)
        to_device(y_device, y_array)
        x_pointer = ctypes.cast(x_device, POINTER(element))
        y_pointer = ctypes.cast(y_device, POINTER(element))
        result_pointer = ctypes.cast(result_device, POINTER(element))

        if element is c_float:
            CUBLAS.cublasSdot_v2(handle, x_array.size, x_pointer, 1, y_pointer, 1,
                                 result_pointer)
        else:
            CUBLAS.cublasDdot_v2(handle, x_array.size, x_pointer, 1, y_pointer, 1,
                                 result_pointer)

        synchronize()

        # cuBLAS writes the result to the DEVICE, so it is read back from there.
        result = np.zeros(1, dtype=dtype)
        to_host(result_device, result)
    finally:
        destroy_handle(handle)
        free(x_device)
        free(y_device)
        free(result_device)

    return float(result[0])
