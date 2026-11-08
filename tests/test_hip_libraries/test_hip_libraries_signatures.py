"""The signatures tests, split out of `test_hip_libraries.py`.

The subject is signatures.
"""
from __future__ import annotations

import ctypes
from pygpgpu.hip.libraries import (
    HIPFFT,
    HIPSOLVER,
    HIPSPARSE,
    hipfftType,
)


def test_hipffts_handle_is_a_pointer_where_cuffts_is_an_int():
    """The one fact that makes a table copied from the CUDA side unusable.

    cuFFT's `cufftHandle` is an `int`; HIP spells its plan `struct hipfftHandle_t*` on
    this platform (the `int` spelling is inside `#ifdef __HIP_PLATFORM_NVIDIA__`). A
    table carrying the CUDA shape passes a 4-byte integer where the library reads an
    8-byte pointer, which is an access violation inside the first call rather than a
    wrong answer.
    """
    from pygpgpu.cuda.libraries import cufft

    from pygpgpu.hip.libraries.hipfft import hipfftHandle

    assert hipfftHandle is ctypes.c_void_p
    assert cufft.cufftHandle is ctypes.c_int
    assert HIPFFT._signatures["hipfftExecR2C"]["args"]["plan"] is ctypes.c_void_p
    assert HIPFFT._signatures["hipfftPlan1d"]["args"]["plan"] is not ctypes.c_void_p


def test_hipfft_declares_the_arguments_the_cuda_table_is_missing():
    """Both entries genuinely differ in arity from the cuFFT table beside them."""
    from pygpgpu.cuda.libraries import cufft

    # the direction is HIP's fourth argument; the cuFFT table lists three
    assert len(cufft.CUFFTInfo.func_signatures["cufftExecC2C"]["args"]) == 3
    assert len(HIPFFT._signatures["hipfftExecC2C"]["args"]) == 4
    assert "direction" in HIPFFT._signatures["hipfftExecC2C"]["args"]

    # `type` is HIP's third argument; the cuFFT table has `nx` then `batch`
    assert len(cufft.CUFFTInfo.func_signatures["cufftGetSize1d"]["args"]) == 4
    assert len(HIPFFT._signatures["hipfftGetSize1d"]["args"]) == 5
    assert "type" in HIPFFT._signatures["hipfftGetSize1d"]["args"]


def test_hipfft_shares_no_entry_with_the_cuda_table():
    """Each hipFFT entry is declared here, so none can carry a cuFFT function over."""
    from pygpgpu.cuda.libraries import cufft

    cuda_entries = set(id(entry) for entry in cufft.CUFFTInfo.func_signatures.values())
    for name, entry in HIPFFT._signatures.items():
        assert id(entry) not in cuda_entries, f"{name} shares its entry with cuFFT"


def test_hipsparse_declares_the_arguments_the_cuda_table_is_missing():
    """`GetVersion` takes the handle, and `CreateDnMat` takes an order."""
    from pygpgpu.cuda.libraries import cusparse

    assert len(cusparse.CUSPARSEInfo.func_signatures["cusparseGetVersion"]["args"]) == 1
    assert len(HIPSPARSE._signatures["hipsparseGetVersion"]["args"]) == 2
    assert "handle" in HIPSPARSE._signatures["hipsparseGetVersion"]["args"]

    assert len(cusparse.CUSPARSEInfo.func_signatures["cusparseCreateDnMat"]["args"]) == 6
    assert len(HIPSPARSE._signatures["hipsparseCreateDnMat"]["args"]) == 7
    assert "order" in HIPSPARSE._signatures["hipsparseCreateDnMat"]["args"]


def test_hipsolver_has_no_version_query():
    """ROCm's hipSOLVER declares none at all, where the cuSOLVER table lists one."""
    from pygpgpu.cuda.libraries import cusolver

    assert "cusolverGetVersion" in cusolver.CUSOLVERInfo.func_signatures
    assert not any("GetVersion" in name for name in HIPSOLVER._signatures)


def test_hipsolvers_getrf_takes_the_pivot_array():
    """The cuSOLVER table here has `lwork` where the API has `devIpiv`.

    So hipSOLVER's table is written from the header rather than derived: deriving would
    have carried that shape over, and the call would have passed a workspace size where
    the library writes the pivot indices.
    """
    arguments = HIPSOLVER._signatures["hipsolverDnSgetrf"]["args"]
    assert "devIpiv" in arguments
    assert "lwork" not in arguments
    assert arguments["devIpiv"]._type_ is ctypes.c_int


def test_the_hipfft_transform_kinds_carry_cuffts_values():
    """`hipfftType_t` is spelled with cuFFT's constants, unlike the status codes."""
    assert hipfftType.HIPFFT_R2C == 0x2a
    assert hipfftType.HIPFFT_C2R == 0x2c
    assert hipfftType.HIPFFT_C2C == 0x29
    assert hipfftType.HIPFFT_D2Z == 0x6a
    assert hipfftType.HIPFFT_Z2Z == 0x69
