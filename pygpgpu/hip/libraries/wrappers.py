"""A numpy-friendly layer over hipBLAS and hipBLASLt.

The raw entry points take device pointers and a column-major layout; these take numpy
arrays and return the same layout the caller passed, so ``gemm(a, b)`` is ``a @ b``.

Device memory goes through the HIP runtime the backend already uses
(``hipMalloc``/``hipMemcpy``), not through a second allocator: the buffers a caller
builds with ``to_device`` are the ones the backend's own kernels read.

These are **host-side** operations, the same way the CUDA ones are: a ``gemm`` moves whole
arrays to the device and runs one library call, so it is not expressible inside a
``@kernel`` body.
"""
from __future__ import annotations

from ctypes import POINTER, byref, c_float, c_int, c_size_t, c_void_p
from typing import Any, Optional, Tuple

import numpy as np

from ...hip.driver import HIP, CUdeviceptr
from ._enums import hipblasOperation_t, hipfftType
from .hipblas import HIPBLAS, HIPBLASLt
from .hipfft import HIPFFT, hipfftHandle

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
]


# ---------------------------------------------------------------------------
# The context, which a library handle belongs to
# ---------------------------------------------------------------------------


def ensure_context()->None:
    """Make a HIP context current if none is.

    A hipBLAS handle belongs to the context that was current when it was created, so a
    call with no context is answered with an error the user cannot act on. The context
    is the backend's own (the ``oop`` layer's), so the buffers a caller builds with the
    driver are the ones the library reads.

    A context that was destroyed while current is still reported as current by the
    runtime, and every call against it then fails; so "current" alone is not enough and
    the device is asked for its name as the liveness test.
    """
    from ...hip.oop.devices import Devices

    device = Devices[0]
    context = device.default_context
    if getattr(context, "id", None) is None:
        context.make_current()
        return
    try:
        context.make_current()
        # A dead context is still "current": ask the device to prove it is alive.
        device.name
    except Exception:
        context.make_current()


# ---------------------------------------------------------------------------
# Device memory, through the HIP runtime
# ---------------------------------------------------------------------------


def allocate(nbytes:int)->CUdeviceptr:
    """Allocate device memory and return its address."""
    ensure_context()
    pointer = CUdeviceptr(0)
    HIP.hipMalloc(byref(pointer), c_size_t(nbytes))
    return pointer


def free(pointer:CUdeviceptr)->None:
    """Release device memory allocated by :func:`allocate`."""
    HIP.hipFree(pointer)


def to_device(array:Any)->CUdeviceptr:
    """Copy a host array to a fresh device buffer and return its address."""
    host = np.ascontiguousarray(array)
    pointer = allocate(host.nbytes)
    HIP.hipMemcpyHtoD(pointer, c_void_p(host.ctypes.data), c_size_t(host.nbytes))
    return pointer


def to_host(pointer:CUdeviceptr, array:np.ndarray)->np.ndarray:
    """Copy a device buffer into a host array, and return the array."""
    target = np.ascontiguousarray(array)
    HIP.hipMemcpyDtoH(c_void_p(target.ctypes.data), pointer, c_size_t(target.nbytes))
    return target


def synchronize()->None:
    """Wait until the device has finished everything that was queued on it."""
    ensure_context()
    HIP.hipDeviceSynchronize()


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


def create_handle()->c_void_p:
    """Create a hipBLAS handle, in the context that is current."""
    ensure_context()
    handle = c_void_p()
    HIPBLAS.hipblasCreate(byref(handle))
    return handle


def destroy_handle(handle:c_void_p)->None:
    """Release a hipBLAS handle."""
    HIPBLAS.hipblasDestroy(handle)


def set_stream(handle:c_void_p, stream:Any)->None:
    """Order a handle's work on a stream.

    The stream is the HIP driver's stream object or its raw id.
    """
    stream_id = getattr(stream, "id", stream)
    HIPBLAS.hipblasSetStream(handle, _stream_pointer(stream_id))


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


# ---------------------------------------------------------------------------
# The operations
# ---------------------------------------------------------------------------


def _element_type(dtype:np.dtype)->Any:
    """The ctypes scalar type of a numpy dtype, or None when it is not supported."""
    return {np.dtype(np.float32): c_float, np.dtype(np.float64): None}.get(dtype)


def _is_single(dtype:np.dtype)->bool:
    return dtype == np.dtype(np.float32)


def _is_double(dtype:np.dtype)->bool:
    return dtype == np.dtype(np.float64)


def _device_pointer(pointer:CUdeviceptr, target:Any)->Any:
    """A device address as the typed pointer an entry point wants."""
    return _cast(c_void_p(int(pointer.value)), target)


def _cast(value:Any, target:Any)->Any:
    from ctypes import cast

    return cast(value, target)


def _scalar_pointer(value:float, dtype:np.dtype)->Tuple[Any, Any]:
    """A host scalar as the pointer a level-1 call reads it through, and its keeper."""
    holder = c_float(value) if _is_single(dtype) else None
    if holder is None:
        from ctypes import c_double

        holder = c_double(value)
    return byref(holder), holder


def gemm(a:Any, b:Any, out:Optional[Any]=None, alpha:float=1.0, beta:float=0.0,
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


def _double_pointer()->Any:
    from ctypes import c_double

    return POINTER(c_double)


def _gemm_call(handle:c_void_p, op_a:Any, op_b:Any, m:int, n:int, k:int,
               alpha:Any, A:Any, lda:int, B:Any, ldb:int, beta:Any, C:Any, ldc:int,
               dtype:np.dtype)->None:
    """One gemm, in the single or the double precision entry point."""
    if _is_single(dtype):
        HIPBLAS.hipblasSgemm(handle, op_a, op_b, c_int(m), c_int(n), c_int(k),
                             alpha, A, c_int(lda), B, c_int(ldb), beta, C, c_int(ldc))
        return
    HIPBLAS.hipblasDgemm(handle, op_a, op_b, c_int(m), c_int(n), c_int(k),
                         alpha, A, c_int(lda), B, c_int(ldb), beta, C, c_int(ldc))


def axpy(alpha:float, x:Any, y:Any)->np.ndarray:
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


def scal(alpha:float, x:Any)->np.ndarray:
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


def dot(x:Any, y:Any)->float:
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


def lt_gemm(a:Any, b:Any, out:Optional[Any]=None, alpha:float=1.0, beta:float=0.0,
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


# ---------------------------------------------------------------------------
# hipFFT
# ---------------------------------------------------------------------------


def rfft(x:Any)->np.ndarray:
    """The 1-D real-to-complex transform of a real array.

    The result is ``n/2 + 1`` complex numbers, in the same layout as
    ``numpy.fft.rfft``.
    """
    ensure_context()
    signal = np.ascontiguousarray(x)
    if not _is_single(signal.dtype):
        raise TypeError(f"rfft() takes a float32 array, got {signal.dtype}")
    if signal.ndim != 1:
        raise ValueError(f"rfft() takes a 1-D array, got {signal.ndim} dimensions")

    count = signal.size
    spectrum = np.zeros(count // 2 + 1, np.complex64)

    signal_buffer = to_device(signal)
    spectrum_buffer = to_device(spectrum)

    # The plan handle is a POINTER on this platform, so it is a c_void_p and the plan
    # call takes its address -- `hipfftPlan1d` is one of the two entry points that do.
    plan = hipfftHandle()
    HIPFFT.hipfftPlan1d(byref(plan), c_int(count),
                        c_int(int(hipfftType.HIPFFT_R2C)), c_int(1))
    try:
        HIPFFT.hipfftExecR2C(plan, _void(signal_buffer), _void(spectrum_buffer))
        synchronize()
        to_host(spectrum_buffer, spectrum)
    finally:
        HIPFFT.hipfftDestroy(plan)
        free(signal_buffer)
        free(spectrum_buffer)

    return spectrum


def irfft(spectrum:Any, n:Optional[int]=None)->np.ndarray:
    """The 1-D complex-to-real transform of a spectrum.

    ``n`` is the length of the signal the spectrum came from; it defaults to
    ``2 * (len(spectrum) - 1)``, which is the length numpy's ``rfft`` implies.
    """
    ensure_context()
    values = np.ascontiguousarray(spectrum)
    if values.dtype != np.complex64:
        raise TypeError(f"irfft() takes a complex64 array, got {values.dtype}")
    if values.ndim != 1:
        raise ValueError(f"irfft() takes a 1-D array, got {values.ndim} dimensions")

    if n is None:
        n = 2 * (values.size - 1)

    # The spectrum is n/2+1 complex numbers but the transform reads a full n/2 of them,
    # so the buffer is the spectrum's own memory and the plan is told the signal length.
    signal = np.zeros(n, np.float32)
    spectrum_buffer = to_device(values)
    signal_buffer = to_device(signal)

    plan = hipfftHandle()
    HIPFFT.hipfftPlan1d(byref(plan), c_int(n), c_int(int(hipfftType.HIPFFT_C2R)), c_int(1))
    try:
        HIPFFT.hipfftExecC2R(plan, _void(spectrum_buffer), _void(signal_buffer))
        synchronize()
        to_host(signal_buffer, signal)
    finally:
        HIPFFT.hipfftDestroy(plan)
        free(spectrum_buffer)
        free(signal_buffer)

    # hipFFT's complex-to-real transform is unnormalized, as cuFFT's is: putting the
    # scale back is what makes it numpy's answer.
    return signal / n


def _void(pointer:CUdeviceptr)->c_void_p:
    """A device address as the void pointer the FFT entry points take.

    They are declared `hipfftReal*` / `hipfftComplex*`, but ctypes' `c_void_p` is the
    same pointer and accepts the address without the caller naming the element type --
    which matters because the complex buffers are not a float array ctypes could type.
    """
    return c_void_p(int(pointer.value))
