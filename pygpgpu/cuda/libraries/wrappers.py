"""The pythonic layer over the bound CUDA libraries.

The raw entry points are awkward to call: they take device pointers, they take the
shape of a matrix in four separate numbers, and they want the matrix in the
column-major layout BLAS uses. The wrappers here take numpy arrays and work out the
rest, because the point of the bindings is to be usable from python.

Everything here is an **array-level** operation: it moves whole arrays to the device,
runs one library call, and brings the result back. None of it is expressible inside
a ``@kernel`` body -- a kernel body is per-thread arithmetic, and a gemm is a
library call the host makes.
"""
from __future__ import annotations

import ctypes
from ctypes import byref, c_float, c_double, c_int, c_size_t, c_uint64, c_void_p, POINTER
from typing import Any, Optional

import numpy as np

from ...cuda.driver import CUDA, CUcontext, CUdeviceptr

from .cublas import (
    CUBLAS,
    cublasComputeType_t,
    cublasHandle_t,
    cublasOperation_t,
)
from .cufft import CUFFT, cufftHandle, cufftType
from .cublaslt import (
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
from .cusparse import (
    CUSPARSE,
    cudaDataType_t,
    cusparseIndexBase_t,
    cusparseIndexType_t,
    cusparseOperation_t,
    cusparseSpMVAlg_t,
)


# The element types the wrappers know how to move.
_FLOAT32 = np.dtype(np.float32)
_FLOAT64 = np.dtype(np.float64)


def _device_pointer(value:int)->c_void_p:
    """A device address as the pointer type the library entry points want."""
    return c_void_p(int(value))


def _element_type(dtype:np.dtype)->Any:
    """The ctypes scalar type of a numpy dtype, or None when it is not supported."""
    if dtype == _FLOAT32:
        return c_float
    if dtype == _FLOAT64:
        return c_double
    return None


def _data_type(dtype:np.dtype)->Optional[int]:
    """The CUDA data type of a numpy dtype, or None when it is not supported."""
    if dtype == _FLOAT32:
        return cudaDataType_t.CUDA_R_32F
    if dtype == _FLOAT64:
        return cudaDataType_t.CUDA_R_64F
    return None


def _as_matrix(array:Any)->np.ndarray:
    """A 2-D, C-contiguous view of an array, or a refusal naming what is wrong."""
    matrix = np.asarray(array)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {matrix.ndim} dimensions")
    if not matrix.flags["C_CONTIGUOUS"]:
        matrix = np.ascontiguousarray(matrix)
    return matrix


# ---------- device memory ----------

def ensure_context()->None:
    """Make a CUDA context current, if none is usable.

    A library handle belongs to the context that was current when it was created,
    and an allocation needs a context at all, so every wrapper makes sure one is
    current first. Without this the first call of a program that has not set a
    context up fails with ``CUDA_ERROR_INVALID_CONTEXT``, whose own message does
    not say what to do about it.

    The backend's default context for the first device is used. It is the context
    the rest of the CUDA backend uses, so a library buffer and a kernel buffer live
    in the same context and can be mixed.
    """
    context_id = CUcontext()
    CUDA.cuCtxGetCurrent(byref(context_id))
    if context_id.value:
        # A context is current. It may still be a DEAD one -- a context that was
        # destroyed while it was current is still reported as current, and every
        # call against it fails with INVALID_CONTEXT -- so it is probed before it
        # is trusted.
        if _context_is_usable():
            return

    from ...cuda.oop.devices import Devices

    devices = list(Devices)
    if not devices:
        raise RuntimeError(
            "no CUDA device is available: the CUDA libraries need one, and none "
            "was found on this machine")

    devices[0].default_context.make_current()


def _context_is_usable()->bool:
    """Whether the current context still works.

    A context that was destroyed while it was current is still reported as the
    current one, and every call against it fails with INVALID_CONTEXT -- so the
    cheapest possible call is used to find out.
    """
    try:
        pointer = CUdeviceptr(0)
        CUDA.cuMemAlloc_v2(byref(pointer), 1)
    except RuntimeError:
        return False

    try:
        CUDA.cuMemFree_v2(pointer)
    except RuntimeError:
        pass

    return True


def allocate(nbytes:int)->c_void_p:
    """Allocate device memory and return its address.

    The allocation goes through the CUDA driver the backend already uses, so the
    pointer is one the rest of the backend can address as well.
    """
    if nbytes <= 0:
        raise ValueError(f"cannot allocate {nbytes} bytes")

    ensure_context()
    device_pointer = CUdeviceptr(0)
    CUDA.cuMemAlloc_v2(byref(device_pointer), nbytes)
    return c_void_p(int(device_pointer.value))


def free(pointer:c_void_p)->None:
    """Release device memory allocated by :func:`allocate`."""
    CUDA.cuMemFree_v2(CUdeviceptr(int(pointer.value)))


def to_device(pointer:c_void_p, array:np.ndarray)->None:
    """Copy a host array to a device buffer."""
    CUDA.cuMemcpyHtoD_v2(CUdeviceptr(int(pointer.value)),
                         array.ctypes.data, array.nbytes)


def to_host(pointer:c_void_p, array:np.ndarray)->np.ndarray:
    """Copy a device buffer into a host array, and return the array."""
    CUDA.cuMemcpyDtoH_v2(array.ctypes.data,
                         CUdeviceptr(int(pointer.value)), array.nbytes)
    return array


def synchronize()->None:
    """Wait until the device has finished everything that was queued on it."""
    CUDA.cuCtxSynchronize()


# ---------- cuBLAS ----------

def create_handle()->cublasHandle_t:
    """Create a cuBLAS handle.

    The handle belongs to the context that is current when it is created, so a
    context is made current first if the caller has not done it.
    """
    ensure_context()
    handle = cublasHandle_t()
    CUBLAS.cublasCreate_v2(byref(handle))
    return handle


def destroy_handle(handle:cublasHandle_t)->None:
    """Release a cuBLAS handle."""
    CUBLAS.cublasDestroy_v2(handle)


def set_stream(handle:cublasHandle_t, stream:Any)->None:
    """Order a handle's work on a stream.

    The stream is the CUDA driver's stream object (a ``Stream`` of the backend's
    ``oop`` layer) or its raw id.
    """
    stream_id = getattr(stream, "id", stream)
    CUBLAS.cublasSetStream_v2(handle, _stream_pointer(stream_id))


def _stream_pointer(stream_id:Any)->c_void_p:
    """The raw pointer of a stream id, whatever shape that id has."""
    if isinstance(stream_id, c_void_p):
        return stream_id
    if isinstance(stream_id, int):
        return c_void_p(stream_id)
    value = getattr(stream_id, "value", None)
    if value is not None:
        return c_void_p(int(value))
    raise TypeError(f"cannot read a stream id from {type(stream_id).__name__}")


def gemm(a:Any, b:Any, out:Optional[Any]=None, alpha:float=1.0, beta:float=0.0,
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

    dtype = np.result_type(a_array.dtype, b_array.dtype)
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

    if out is None:
        result = np.zeros((m, n), dtype=dtype)
    else:
        result = np.asarray(out)
        if result.shape != (m, n):
            raise ValueError(
                f"the output is {result.shape}, but the product is {(m, n)}")
        if result.dtype != dtype:
            result = result.astype(dtype)

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
    finally:
        destroy_handle(handle)
        free(a_device)
        free(b_device)
        free(c_device)

    # The result takes numpy's shape for the product, not BLAS's.
    if a_is_vector and b_is_vector:
        return result.reshape(())
    if a_is_vector:
        return result.reshape(n)
    if b_is_vector:
        return result.reshape(m)
    return result


def axpy(alpha:float, x:Any, y:Any)->np.ndarray:
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


def scal(alpha:float, x:Any)->np.ndarray:
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


def dot(x:Any, y:Any)->float:
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


# ---------- cuFFT ----------

def rfft(x:Any)->np.ndarray:
    """The 1-D real-to-complex transform of a real array.

    The result is ``n/2 + 1`` complex numbers, in the same layout as
    ``numpy.fft.rfft``.
    """
    signal = np.ascontiguousarray(x)
    if signal.dtype != _FLOAT32:
        raise TypeError(f"rfft supports float32, got {signal.dtype}")
    if signal.ndim != 1:
        raise ValueError(f"rfft expects a 1-D array, got {signal.ndim} dimensions")

    count = signal.size
    spectrum = np.zeros(count // 2 + 1, np.complex64)

    signal_device = allocate(signal.nbytes)
    spectrum_device = allocate(spectrum.nbytes)
    plan = cufftHandle()
    CUFFT.cufftPlan1d(byref(plan), count, int(cufftType.CUFFT_R2C), 1)
    try:
        to_device(signal_device, signal)
        CUFFT.cufftExecR2C(plan, signal_device, spectrum_device)
        synchronize()
        to_host(spectrum_device, spectrum)
    finally:
        CUFFT.cufftDestroy(plan)
        free(signal_device)
        free(spectrum_device)

    return spectrum


def irfft(spectrum:Any, n:Optional[int]=None)->np.ndarray:
    """The 1-D complex-to-real transform of a spectrum.

    ``n`` is the length of the signal the spectrum came from; it defaults to
    ``2 * (len(spectrum) - 1)``, which is the length numpy's ``rfft`` implies.
    """
    values = np.ascontiguousarray(spectrum)
    if values.dtype != np.complex64:
        raise TypeError(f"irfft supports complex64, got {values.dtype}")
    if values.ndim != 1:
        raise ValueError(
            f"irfft expects a 1-D array, got {values.ndim} dimensions")

    if n is None:
        n = 2 * (values.size - 1)

    signal = np.zeros(n, np.float32)
    spectrum_device = allocate(values.nbytes)
    signal_device = allocate(signal.nbytes)
    plan = cufftHandle()
    CUFFT.cufftPlan1d(byref(plan), n, int(cufftType.CUFFT_C2R), 1)
    try:
        to_device(spectrum_device, values)
        CUFFT.cufftExecC2R(plan, spectrum_device, signal_device)
        synchronize()
        to_host(signal_device, signal)
    finally:
        CUFFT.cufftDestroy(plan)
        free(spectrum_device)
        free(signal_device)

    # cuFFT's complex-to-real transform is unnormalized; numpy's is not either, and
    # the two agree once the scale is put back.
    return signal / n


# ---------- cuSPARSE ----------

def csrmv(row_offsets:Any, column_indices:Any, values:Any, x:Any,
          y:Optional[Any]=None)->np.ndarray:
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


# ---------- cuBLASLt ----------

def lt_gemm(a:Any, b:Any, out:Optional[Any]=None, alpha:float=1.0,
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


__all__ = [
    "ensure_context",
    "allocate",
    "free",
    "to_device",
    "to_host",
    "synchronize",
    "create_handle",
    "destroy_handle",
    "set_stream",
    "gemm",
    "lt_gemm",
    "axpy",
    "scal",
    "dot",
    "rfft",
    "irfft",
    "csrmv",
]
