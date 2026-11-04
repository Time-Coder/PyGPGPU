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

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pygpgpu.hip.libraries import (
    HIPBLAS,
    hipblasComputeType_t,
    hipblasOperation_t,
    hipblasStatus_t,
    driver_for,
    library_names,
)
from pygpgpu.hip.libraries._symbols import hip_symbol


# ---------------------------------------------------------------------------
# The binding, without a device
# ---------------------------------------------------------------------------


def test_the_two_verified_libraries_are_bound():
    """hipBLAS and hipBLASLt are the ones whose signatures were checked."""
    assert library_names() == ["hipblas", "hipblaslt"]


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
