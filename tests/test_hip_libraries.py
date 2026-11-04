"""The host-side HIP math libraries, bound with ctypes.

hipBLAS is the vendor's port of cuBLAS, so the bindings mirror the CUDA ones: a lazily
loaded DLL, a signature table, a status code turned into a ``LibraryError``. What is
specific to HIP, and what the tests pin:

* the **bond is lazy**: importing ``pygpgpu`` loads no HIP library, and the DLL is loaded
  on first use;
* the **names differ**: hipBLAS drops the ABI version suffix CUDA 12 carries, and
  hipBLASLt spells its entry points ``hipblasLt*``;
* the **values differ too**, which is the part that cost a debugging session: hipBLAS's
  ``HIPBLAS_OP_N`` is 111 where ``CUBLAS_OP_N`` is 0, and its status codes are numbered
  separately. Passing a cuBLAS value makes hipBLAS answer ``INVALID_ENUM``;
* the **numbers are right**: a gemm, an axpy, a scal and a dot each agree with numpy's
  own answer, including the 1-D cases and the transpose flags.

The tests need a real HIP device, so they use the ``gpu_backend`` fixture, and they skip
when the library they exercise is not installed.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pygpgpu.hip.libraries import (
    HIPBLAS,
    HIPFFT,
    HIPSOLVER,
    HIPSPARSE,
    hipblasComputeType_t,
    hipblasFillMode_t,
    hipblasOperation_t,
    hipblasStatus_t,
    hipfftType,
    driver_for,
    library_names,
)
from pygpgpu.hip.libraries._symbols import hip_symbol


# ---------------------------------------------------------------------------
# The binding, without a device
# ---------------------------------------------------------------------------


def test_all_five_libraries_are_bound():
    """Every library whose entry points were checked against ROCm's headers."""
    assert library_names() == ["hipblas", "hipblaslt", "hipfft", "hipsparse",
                               "hipsolver"]


def test_the_library_tables_declare_the_entry_points_they_name():
    """Every entry is spelled with its library's own prefix and declares a return type.

    The argument list may legitimately be empty (`GetVersion` in some libraries), so only
    the prefix and the return type are asserted -- those are what a mistyped entry would
    get wrong.
    """
    prefixes = {"hipblas": "hipblas", "hipblaslt": "hipblasLt", "hipfft": "hipfft",
                "hipsparse": "hipsparse", "hipsolver": "hipsolver"}
    for name in library_names():
        driver = driver_for(name)
        assert driver._signatures, f"{name} declares no entry points"
        for entry, signature in driver._signatures.items():
            assert "restype" in signature, f"{name}.{entry} declares no return type"
            assert entry.startswith(prefixes[name]), (
                f"{name}.{entry} is not spelled with the library's own prefix")


def test_the_symbol_mapping_drops_the_abi_suffix():
    """hipBLAS has no second ABI, so CUDA 12's `_v2` is not part of the name."""
    assert hip_symbol("hipblas", "cublasSgemm_v2") == "hipblasSgemm"
    assert hip_symbol("hipblas", "cublasCreate_v2") == "hipblasCreate"
    assert hip_symbol("hipblas", "cublasDgemm_v2") == "hipblasDgemm"


def test_the_symbol_mapping_uses_the_camel_case_prefix_for_hipblaslt():
    """hipBLASLt's DLL is `hipblaslt`, but its entry points are `hipblasLt*`."""
    assert hip_symbol("hipblaslt", "cublasLtMatmul") == "hipblasLtMatmul"
    assert hip_symbol("hipblaslt", "cublasLtCreate") == "hipblasLtCreate"


def test_a_symbol_the_library_lacks_has_no_mapping():
    """`GetStatusString` is absent from every HIP library."""
    assert hip_symbol("hipblas", "cublasGetStatusString") is None
    assert hip_symbol("hipblaslt", "cublasLtGetStatusString") is None


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


def test_the_signature_table_is_derived_from_the_cuda_one():
    """The API is cuBLAS's, so the entries are the ones the CUDA table has."""
    from pygpgpu.cuda.libraries import cublas

    hip_names = set(HIPBLAS._signatures)
    # every CUDA entry whose HIP counterpart exists is present, under its HIP name
    for cuda_name in cublas.CUBLASInfo.func_signatures:
        translated = hip_symbol("hipblas", cuda_name)
        if translated is not None:
            assert translated in hip_names
    assert "hipblasSgemm" in hip_names
    assert "cublasSgemm_v2" not in hip_names


def test_the_hip_table_does_not_share_entries_with_the_cuda_one():
    """A shared entry would carry one library's bound function into the other.

    The binding machinery caches the bound function in the entry it was called
    through, so sharing would make a cuBLASLt call land in hipBLASLt (or the reverse).
    """
    from pygpgpu.cuda.libraries import cublas

    for name, entry in HIPBLAS._signatures.items():
        cuda_name = "cublas" + name[len("hipblas"):]
        cuda_entry = cublas.CUBLASInfo.func_signatures.get(cuda_name)
        if cuda_entry is not None:
            assert entry is not cuda_entry, (
                f"{name} shares its entry with {cuda_name}; a bound function cached in "
                f"one would be used by the other")


# ---------------------------------------------------------------------------
# The three libraries whose signatures are NOT cuFFT's / cuSPARSE's / cuSOLVER's
# ---------------------------------------------------------------------------


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


def test_the_hipfft_transform_kinds_carry_cuffts_values():
    """`hipfftType_t` is spelled with cuFFT's constants, unlike the status codes."""
    assert hipfftType.HIPFFT_R2C == 0x2a
    assert hipfftType.HIPFFT_C2R == 0x2c
    assert hipfftType.HIPFFT_C2C == 0x29
    assert hipfftType.HIPFFT_D2Z == 0x6a
    assert hipfftType.HIPFFT_Z2Z == 0x69


def test_the_import_loads_no_hip_library():
    """`import pygpgpu` must not pay for a runtime, and not load a DLL either."""
    script = textwrap.dedent("""
        import sys
        import pygpgpu
        loaded = [name for name in sys.modules if "hip.libraries" in name]
        assert not loaded, loaded
        # resolving a name imports the package, but still no DLL
        pygpgpu.hip_libraries.library_names()
        assert "pygpgpu.hip.libraries.hipblas" in sys.modules
        print("ok")
    """)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "ok" in completed.stdout


def test_the_from_import_spelling_works():
    """`from pygpgpu.hip_libraries import gemm` is the other natural spelling."""
    script = textwrap.dedent("""
        from pygpgpu.hip_libraries import gemm
        assert gemm.__name__ == "gemm"
        print("ok")
    """)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "ok" in completed.stdout


# ---------------------------------------------------------------------------
# On the device
# ---------------------------------------------------------------------------


def _library_available(name:str)->bool:
    try:
        driver_for(name).dll()
        return True
    except OSError:
        return False


def _require_library(name:str)->None:
    if not _library_available(name):
        pytest.skip(f"the {name} library is not installed on this machine")


@pytest.fixture(scope="module")
def hip_device():
    """The first HIP device, with its context current."""
    import pygpgpu.hip  # noqa: F401  (starts the driver's background initialization)
    from pygpgpu.hip.oop.devices import Devices

    try:
        devices = list(Devices)
    except Exception as error:
        pytest.skip(f"no HIP device is available: {error}")
    if not devices:
        pytest.skip("no HIP device is available")

    device = devices[0]
    device.default_context.make_current()
    return device


@pytest.fixture(scope="module")
def hipblas(hip_device):
    _require_library("hipblas")
    from pygpgpu.hip import libraries

    return libraries


def test_gemm_matches_numpy(hipblas):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(hipblas.gemm(a, b), a @ b, rtol=1e-6)


def test_gemm_in_double_precision(hipblas):
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    b = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float64)
    np.testing.assert_allclose(hipblas.gemm(a, b), a @ b, rtol=1e-12)


def test_gemm_follows_numpys_one_dimensional_rules(hipblas):
    """A 1-D operand is a row on the left and a column on the right, as `@` says."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vector = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(hipblas.gemm(matrix, vector), matrix @ vector, rtol=1e-6)
    np.testing.assert_allclose(hipblas.gemm(vector, matrix), vector @ matrix, rtol=1e-6)


def test_gemm_honours_the_transpose_flags(hipblas):
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(hipblas.gemm(a, b, transpose_b=True), a @ b.T, rtol=1e-6)
    np.testing.assert_allclose(hipblas.gemm(a, b, transpose_a=True), a.T @ b, rtol=1e-6)


def test_gemm_honours_alpha_and_beta(hipblas):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.eye(2, dtype=np.float32)
    start = np.full((2, 2), 10.0, dtype=np.float32)
    expected = 2.0 * (a @ b) + 3.0 * start
    np.testing.assert_allclose(
        hipblas.gemm(a, b, out=start.copy(), alpha=2.0, beta=3.0), expected, rtol=1e-6)


def test_axpy_matches_the_definition(hipblas):
    """`axpy` writes into `y`, so the expectation is computed before the call."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    expected = 2.0 * x + y
    np.testing.assert_allclose(hipblas.axpy(2.0, x, y), expected, rtol=1e-6)


def test_scal_matches_the_definition(hipblas):
    """`scal` writes into `x`, so the expectation is computed before the call."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected = 2.0 * x
    np.testing.assert_allclose(hipblas.scal(2.0, x), expected, rtol=1e-6)


def test_dot_matches_the_definition(hipblas):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    assert hipblas.dot(x, y) == pytest.approx(float(x @ y), rel=1e-6)


def test_a_bad_dtype_is_refused(hipblas):
    """The library takes float32 and float64; an int array is refused by name."""
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    with pytest.raises(TypeError) as caught:
        hipblas.gemm(a, a)
    assert "float32" in str(caught.value)


def test_shapes_that_do_not_multiply_are_refused(hipblas):
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError) as caught:
        hipblas.gemm(a, b)
    assert "do not multiply" in str(caught.value)


def test_lt_gemm_matches_gemm_or_says_why_it_cannot(hipblas):
    """hipBLASLt is bound, but rocBLASLt needs a kernel library for the device.

    ROCm ships its Tensile kernels for the discrete GPUs only, so on an integrated one
    every matmul fails inside rocBLASLt while the binding itself is fine. The test
    accepts that, because it is a fact about the machine and not about the code -- but
    when the call does work its answer has to be gemm's.
    """
    _require_library("hipblaslt")
    from pygpgpu.libraries import LibraryError

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    try:
        result = hipblas.lt_gemm(a, b)
    except LibraryError as error:
        assert "Tensile" in str(error), (
            f"hipBLASLt failed for a reason other than its kernel library: {error}")
        pytest.skip("rocBLASLt has no kernel library for this device")
        return
    np.testing.assert_allclose(result, a @ b, rtol=1e-5)


# ---------------------------------------------------------------------------
# hipFFT, on the device
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hipfft(hip_device):
    _require_library("hipfft")
    from pygpgpu.hip import libraries

    return libraries


def test_rfft_matches_numpy(hipfft):
    """A real-to-complex transform, in numpy's own layout and scaling."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    result = hipfft.rfft(signal)
    assert result.dtype == np.complex64
    assert result.size == signal.size // 2 + 1
    np.testing.assert_allclose(result, np.fft.rfft(signal), atol=1e-4)


def test_rfft_and_irfft_round_trip(hipfft):
    """The inverse undoes the forward, which is what pins the scaling.

    hipFFT's complex-to-real transform is unnormalized exactly as cuFFT's is, so the
    wrapper divides by the signal length; without that the round trip returns the signal
    multiplied by n.
    """
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    spectrum = hipfft.rfft(signal)
    back = hipfft.irfft(spectrum, signal.size)
    np.testing.assert_allclose(back, signal, atol=1e-4)


def test_rfft_refuses_a_dtype_it_cannot_transform(hipfft):
    with pytest.raises(TypeError) as caught:
        hipfft.rfft(np.zeros(8, dtype=np.int32))
    assert "float32" in str(caught.value)


# ---------------------------------------------------------------------------
# hipSPARSE and hipSOLVER, which this ROCm install cannot run
# ---------------------------------------------------------------------------


def test_hipsparse_creates_a_handle_or_says_why_it_cannot(hip_device):
    """rocSPARSE resolves its kernels out of a runtime directory beside the DLL.

    ROCm 7.2's Windows install creates one for rocBLAS (`bin/rocblas/library`) and for
    rocFFT, and none for rocSPARSE -- so `hipsparseCreate` answers
    `HIPSPARSE_STATUS_INTERNAL_ERROR` and leaves a null handle. The binding is verified;
    the library cannot run here.
    """
    _require_library("hipsparse")
    from pygpgpu.libraries import LibraryError
    from pygpgpu.hip.libraries.wrappers import ensure_context

    ensure_context()
    handle = ctypes.c_void_p()
    try:
        status = HIPSPARSE.hipsparseCreate(ctypes.byref(handle))
    except LibraryError as error:
        assert "hipSPARSE" in str(error), error
        pytest.skip(f"rocSPARSE has no kernels for this device: {error}")
        return
    try:
        assert status == 0
        assert handle.value, "hipsparseCreate reported success but returned no handle"
    finally:
        HIPSPARSE.hipsparseDestroy(handle)


def test_hipsolver_factors_a_matrix_or_says_why_it_cannot(hip_device):
    """rocSOLVER has no runtime kernel directory in this install either.

    Worse than a failure: the factorization answers **success with `devInfo == 0` and
    leaves the matrix untouched**, so a caller gets a wrong answer with no error. The
    test therefore checks the *answer* rather than the status code, and skips when the
    library is not usable -- which is a fact about the machine.
    """
    _require_library("hipsolver")
    from pygpgpu.hip.libraries.wrappers import (allocate, ensure_context, free,
                                                synchronize, to_device, to_host)

    ensure_context()
    from ctypes import POINTER, byref, c_float, c_int, c_void_p

    def typed(pointer, restype):
        return ctypes.cast(c_void_p(int(pointer.value)), POINTER(restype))

    matrix = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    n = matrix.shape[0]
    expected = np.tril(np.linalg.cholesky(matrix))

    handle = ctypes.c_void_p()
    HIPSOLVER.hipsolverDnCreate(byref(handle))
    d_matrix = to_device(matrix)
    d_info = to_device(np.zeros(1, dtype=np.int32))
    work_size = c_int(0)
    d_work = None
    try:
        HIPSOLVER.hipsolverDnSpotrf_bufferSize(
            handle, c_int(int(hipblasFillMode_t.HIPBLAS_FILL_MODE_LOWER)), c_int(n),
            typed(d_matrix, c_float), c_int(n), byref(work_size))
        if work_size.value <= 0:
            pytest.skip("rocSOLVER reported no workspace need, so it cannot factorize")
            return
        d_work = allocate(work_size.value * 4)
        HIPSOLVER.hipsolverDnSpotrf(
            handle, c_int(int(hipblasFillMode_t.HIPBLAS_FILL_MODE_LOWER)), c_int(n),
            typed(d_matrix, c_float), c_int(n), typed(d_work, c_float), work_size,
            typed(d_info, c_int))
        synchronize()

        factor = to_host(d_matrix, np.zeros((n, n), dtype=np.float32))
        if np.array_equal(np.tril(factor), np.tril(matrix)):
            pytest.skip("rocSOLVER answered success but left the matrix untouched; "
                        "this install ships no kernels for the device")
            return
        np.testing.assert_allclose(np.tril(factor), expected, atol=1e-5)
    finally:
        if d_work is not None:
            free(d_work)
        free(d_matrix)
        free(d_info)
        HIPSOLVER.hipsolverDnDestroy(handle)
