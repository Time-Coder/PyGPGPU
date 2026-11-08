"""The values tests, split out of `test_hip_libraries.py`.

The subject is values.
"""
from __future__ import annotations

from pygpgpu.hip.libraries import (
    hipblasComputeType_t,
    hipblasFillMode_t,
    hipblasOperation_t,
    hipblasStatus_t,
)


def test_the_status_codes_are_hipblas_own():
    """The numbering is not cuBLAS's, which is why the codes are declared here.

    Only SUCCESS agrees; passing cuBLAS's ALLOC_FAILED (3) to hipBLAS would be read as
    INVALID_VALUE, and its INTERNAL_ERROR (14) does not exist there at all.
    """
    assert hipblasStatus_t.HIPBLAS_STATUS_SUCCESS == 0
    assert hipblasStatus_t.HIPBLAS_STATUS_ALLOC_FAILED == 2
    assert hipblasStatus_t.HIPBLAS_STATUS_INVALID_VALUE == 3
    assert hipblasStatus_t.HIPBLAS_STATUS_INTERNAL_ERROR == 6
    assert hipblasStatus_t.HIPBLAS_STATUS_INVALID_ENUM == 10
    assert max(int(member) for member in hipblasStatus_t) == 11


def test_the_operation_codes_are_hipblas_own():
    """`HIPBLAS_OP_N` is 111, not cuBLAS's 0 -- the difference that broke a gemm."""
    assert hipblasOperation_t.HIPBLAS_OP_N == 111
    assert hipblasOperation_t.HIPBLAS_OP_T == 112
    assert hipblasOperation_t.HIPBLAS_OP_C == 113


def test_the_compute_types_are_hipblas_own():
    """A dense 0..10 numbering, not cuBLAS's 64..77."""
    assert hipblasComputeType_t.HIPBLAS_COMPUTE_32F == 2
    assert hipblasComputeType_t.HIPBLAS_COMPUTE_64F == 7


def test_hipsolvers_fill_mode_is_hipblass():
    """`uplo` is hipBLAS's enum, whose values are not cuBLAS's.

    `HIPBLAS_FILL_MODE_LOWER` is 122 and `CUBLAS_FILL_MODE_LOWER` is 0, so a
    factorization given cuBLAS's value answers `INVALID_ENUM`.
    """
    from pygpgpu.cuda.libraries.cublas import cublasFillMode_t

    assert cublasFillMode_t.CUBLAS_FILL_MODE_LOWER == 0
    assert hipblasFillMode_t.HIPBLAS_FILL_MODE_UPPER == 121
    assert hipblasFillMode_t.HIPBLAS_FILL_MODE_LOWER == 122
    assert hipblasFillMode_t.HIPBLAS_FILL_MODE_FULL == 123
