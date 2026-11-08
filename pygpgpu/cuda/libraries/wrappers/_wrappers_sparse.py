# cuSPARSE: the sparse matrix-vector product.
from __future__ import annotations
from ._wrappers_common import (
    _FLOAT32,
    allocate,
    free,
    synchronize,
    to_device,
    to_host,
)
from ctypes import byref, c_float, c_size_t, c_void_p
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from ..cusparse import (
    CUSPARSE,
    cudaDataType_t,
    cusparseIndexBase_t,
    cusparseIndexType_t,
    cusparseOperation_t,
    cusparseSpMVAlg_t,
)



def csrmv(row_offsets:ArrayLike, column_indices:ArrayLike, values:ArrayLike, x:ArrayLike,
          y:Optional[ArrayLike]=None)->np.ndarray:
    """A sparse matrix-vector multiply, for a matrix in CSR form.

    ``row_offsets`` is the row start of each row (so ``len(row_offsets)`` is the
    number of rows), ``column_indices`` and ``values`` hold the entries of each row,
    and ``x`` is the vector the matrix multiplies.

    The result is a new array unless ``y`` is given, in which case it is written
    into ``y`` and returned.
    """
    offsets = np.ascontiguousarray(row_offsets)
    indices = np.ascontiguousarray(column_indices)
    entries = np.ascontiguousarray(values)
    vector = np.ascontiguousarray(x)

    if offsets.dtype != np.int32:
        raise TypeError(f"the row offsets must be int32, got {offsets.dtype}")
    if indices.dtype != np.int32:
        raise TypeError(f"the column indices must be int32, got {indices.dtype}")
    if entries.dtype != _FLOAT32:
        raise TypeError(f"csrmv supports float32 values, got {entries.dtype}")
    if vector.dtype != _FLOAT32:
        raise TypeError(f"csrmv supports a float32 vector, got {vector.dtype}")
    if offsets.ndim != 1 or indices.ndim != 1 or entries.ndim != 1:
        raise ValueError("the matrix and the vector must be one-dimensional")

    # A CSR matrix's row offsets hold one more entry than it has rows: the last
    # one is where the final row ends.
    rows = offsets.size - 1
    if rows < 1:
        raise ValueError("the row offsets must hold at least two entries")
    if vector.size != rows:
        raise ValueError(
            f"the matrix has {rows} rows but the vector has {vector.size} entries")
    if indices.size != entries.size:
        raise ValueError(
            f"the matrix has {indices.size} column indices but {entries.size} values")

    if y is None:
        result = np.zeros(rows, np.float32)
    else:
        result = np.asarray(y)
        if result.shape != (rows,):
            raise ValueError(
                f"the output has {result.shape} entries, but the result has {rows}")

    offsets_device = allocate(offsets.nbytes)
    indices_device = allocate(indices.nbytes)
    values_device = allocate(entries.nbytes)
    x_device = allocate(vector.nbytes)
    y_device = allocate(result.nbytes)
    handle = c_void_p()
    CUSPARSE.cusparseCreate(byref(handle))
    matrix = c_void_p()
    vector_x = c_void_p()
    vector_y = c_void_p()
    try:
        to_device(offsets_device, offsets)
        to_device(indices_device, indices)
        to_device(values_device, entries)
        to_device(x_device, vector)
        to_device(y_device, result)

        CUSPARSE.cusparseCreateCsr(byref(matrix), rows, rows, entries.size,
                                   offsets_device, indices_device, values_device,
                                   int(cusparseIndexType_t.CUSPARSE_INDEX_32I),
                                   int(cusparseIndexType_t.CUSPARSE_INDEX_32I),
                                   int(cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO),
                                   int(cudaDataType_t.CUDA_R_32F))
        CUSPARSE.cusparseCreateDnVec(byref(vector_x), vector.size, x_device,
                                     int(cudaDataType_t.CUDA_R_32F))
        CUSPARSE.cusparseCreateDnVec(byref(vector_y), result.size, y_device,
                                     int(cudaDataType_t.CUDA_R_32F))

        alpha = c_float(1.0)
        beta = c_float(0.0)
        buffer_size = c_size_t(0)
        CUSPARSE.cusparseSpMV_bufferSize(
            handle, int(cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE),
            byref(alpha), matrix, vector_x, byref(beta), vector_y,
            int(cudaDataType_t.CUDA_R_32F),
            int(cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT), byref(buffer_size))

        buffer_device = allocate(max(int(buffer_size.value), 1))
        CUSPARSE.cusparseSpMV_preprocess(
            handle, int(cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE),
            byref(alpha), matrix, vector_x, byref(beta), vector_y,
            int(cudaDataType_t.CUDA_R_32F),
            int(cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT), buffer_device)
        CUSPARSE.cusparseSpMV(
            handle, int(cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE),
            byref(alpha), matrix, vector_x, byref(beta), vector_y,
            int(cudaDataType_t.CUDA_R_32F),
            int(cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT), buffer_device)

        synchronize()
        to_host(y_device, result)
    finally:
        CUSPARSE.cusparseDestroyDnVec(vector_x)
        CUSPARSE.cusparseDestroyDnVec(vector_y)
        CUSPARSE.cusparseDestroySpMat(matrix)
        CUSPARSE.cusparseDestroy(handle)
        free(buffer_device)
        free(y_device)
        free(x_device)
        free(values_device)
        free(indices_device)
        free(offsets_device)

    return result
