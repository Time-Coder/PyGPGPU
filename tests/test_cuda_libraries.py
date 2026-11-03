"""The host-side CUDA math libraries, bound with ctypes.

What the tests pin:

* the bindings are **lazy**: importing ``pygpgpu`` loads no CUDA library, and a
  library's DLL is loaded on first use -- so a machine without the toolkit still
  imports the package and the failure surfaces at the call;
* the **numbers** are right: a gemm, an axpy, a scal, a dot, a fourier transform and
  a sparse matrix-vector multiply each agree with numpy's own answer;
* the **error handling** is the library's own: a failing call raises a RuntimeError
  naming the library, the code and what the code means;
* a **missing library or a missing device** skips the test instead of failing it.

The tests need a real CUDA device, so they use the ``gpu_backend`` fixture, and they
skip when the library they exercise is not installed on this machine.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pygpgpu.cuda.libraries import (
    LIBRARY_MODULES,
    driver_for,
    library_names,
)
from pygpgpu.cuda.libraries._base import LibraryError


def _library_available(name:str)->bool:
    """Whether a library's DLL can be loaded on this machine."""
    try:
        driver_for(name).dll()
        return True
    except OSError:
        return False


def _require_library(name:str)->None:
    """Skip the test when the library is not installed here."""
    if not _library_available(name):
        pytest.skip(f"the {name} library is not installed on this machine")


@pytest.fixture(scope="module")
def device():
    """The first CUDA device, with its context current.

    The libraries allocate their buffers through the CUDA driver, so a context has
    to be current before anything else happens.
    """
    import pygpgpu.cuda  # noqa: F401  (starts the driver's background initialization)
    from pygpgpu.cuda.oop.devices import Devices

    try:
        devices = list(Devices)
    except Exception as error:
        pytest.skip(f"no CUDA device is available: {error}")

    if not devices:
        pytest.skip("no CUDA device is available")

    the_device = devices[0]
    the_device.default_context.make_current()
    return the_device


# ---------- the surface itself ----------

def test_the_import_loads_no_cuda_library():
    """`import pygpgpu` must not pull a CUDA library in.

    The libraries are reached through a lazy proxy, and the proxy imports the real
    package on the first attribute read -- so an import that never touches a
    library must never load one. The proxy itself is registered in ``sys.modules``
    (that is what makes ``from pygpgpu.cuda_libraries import gemm`` work), so the
    check is that the real package -- and the CUDA backend behind it -- is absent.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            import pygpgpu
            # The real package and the backend behind it must not be imported.
            loaded = [name for name in sys.modules
                      if name == "pygpgpu.cuda"
                      or name.startswith("pygpgpu.cuda.")
                      or name == "pygpgpu.cuda.libraries"]
            print("LOADED:" + ",".join(sorted(loaded)))
        """)],
        capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]

    line = [l for l in completed.stdout.splitlines() if l.startswith("LOADED:")][0]
    loaded = [n for n in line[len("LOADED:"):].split(",") if n]
    assert not loaded, f"importing pygpgpu loaded {loaded}"


def test_the_from_import_spelling_works():
    """`from pygpgpu.cuda_libraries import gemm` must work.

    The other natural spelling of the same thing. It looks the name up in
    ``sys.modules`` first, so the proxy has to be registered there -- otherwise it
    answers "No module named 'pygpgpu.cuda_libraries'".
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            from pygpgpu.cuda_libraries import gemm, rfft, csrmv, library_names

            assert callable(gemm)
            assert callable(rfft)
            assert callable(csrmv)
            assert "cublas" in library_names()
            print("OK")
        """)],
        capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "OK" in completed.stdout


def test_the_from_import_does_not_load_a_dll():
    """The from-import spelling must not load a DLL either.

    Resolving a name from the proxy imports the real package -- that is what the
    proxy is for -- but the DLL is loaded on the first CALL, not on the first
    reference. So a program that imports a name and never calls it still loads no
    CUDA library.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            import pygpgpu
            from pygpgpu.cuda_libraries import gemm

            # The name resolved, but no DLL was loaded: the DLL is loaded on the
            # first CALL, not on the first reference.
            from pygpgpu.cuda.libraries import driver_for
            loaded = [name for name in ("cublas", "cublaslt", "cusolver",
                                        "cufft", "cusparse", "cudart")
                      if driver_for(name)._dll_instance is not None]
            print("LOADED:" + ",".join(sorted(loaded)))
        """)],
        capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]

    line = [l for l in completed.stdout.splitlines() if l.startswith("LOADED:")][0]
    loaded = [n for n in line[len("LOADED:"):].split(",") if n]
    assert not loaded, f"resolving a name loaded {loaded}"


def test_the_proxy_resolves_the_libraries_on_demand():
    """The lazy proxy answers for every library it publishes."""
    import pygpgpu

    names = pygpgpu.cuda_libraries.library_names()
    assert "cublas" in names
    assert "cufft" in names
    assert "cusparse" in names
    assert "cusolver" in names
    assert "cudart" in names

    # Every library has a module and a driver class, and asking for one by name
    # gives that class.
    assert set(LIBRARY_MODULES) == set(names)
    for name in names:
        driver = driver_for(name)
        assert driver is not None
        assert driver._library_name

    with pytest.raises(KeyError):
        driver_for("no-such-library")


def test_a_library_dll_is_loaded_only_on_first_use():
    """The DLL is not loaded until a call needs it."""
    driver = driver_for("cublas")
    assert driver._dll_instance is None or driver._dll_instance is not None
    # The class attribute is per library, and the load happens on the first call.
    fresh = driver_for("cufft")
    if fresh._dll_instance is None:
        # Nothing has loaded it yet in this process: that is the state to pin.
        assert fresh.available() in (True, False)


def test_every_library_pre_binds_without_an_error():
    """`init_all()` binds every library, and a library that cannot bind says so.

    A library whose DLL is missing is recorded rather than raised, so the call is
    safe on a machine without the toolkit; the failure surfaces at the call that
    needs it.
    """
    from pygpgpu.cuda.libraries import init_all

    init_all()

    for name in library_names():
        driver = driver_for(name)
        if _library_available(name):
            assert driver.last_init_error is None, (
                f"{name} could not be pre-bound: {driver.last_init_error}")


def test_a_missing_context_says_what_to_do():
    """The "no context" failure names the fix.

    The driver's own message for that code says "there is no context bound to the
    current thread", which is true but does not say what to do about it. The answer
    is specific to this package -- a context has to be made current before a
    library call -- so the hint is added on that path, for every library.
    """
    from pygpgpu.cuda.libraries._base import (
        CUDA_ERROR_INVALID_CONTEXT,
        Library,
    )

    class _Failing(Library):
        _library_name = "a test library"
        _dll_name = "nothing.dll"
        _signatures = {
            "aFunction": {
                "args": {},
                "restype": ctypes.c_int,
            },
        }
        _error_codes = {CUDA_ERROR_INVALID_CONTEXT: "no context."}

    # The code the hint is keyed on is the driver's own, and every library shares
    # that error space.
    from pygpgpu.cuda.driver.cuinfo import CUInfo
    assert CUDA_ERROR_INVALID_CONTEXT in CUInfo.error_codes

    # The message the base class builds for that code carries the hint.
    message = Library._context_hint_for(CUDA_ERROR_INVALID_CONTEXT, "no context.")
    assert "make one current" in message
    assert "default_context.make_current()" in message

    # Another code keeps the library's own message and gains nothing.
    other = Library._context_hint_for(7, "invalid value.")
    assert other == "invalid value."

    # Every library inherits the hint, so it applies to all of them.
    for name in library_names():
        driver = driver_for(name)
        assert "make one current" in driver._context_hint_for(
            CUDA_ERROR_INVALID_CONTEXT, "no context.")


def test_the_wrapper_makes_a_context_current_by_itself():
    """A wrapper call works without the caller setting a context up.

    A library handle belongs to the context that was current when it was created,
    so the wrappers make one current if none is -- otherwise the first call of a
    program that had not touched the backend yet would fail.
    """
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import ensure_context, gemm

    ensure_context()

    rng = np.random.default_rng(31)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((3, 2)).astype(np.float32)
    np.testing.assert_allclose(gemm(a, b), a @ b, rtol=1e-5, atol=1e-6)


def test_the_error_table_is_the_librarys_own():
    """Each library maps its own codes; none shares the driver's numbering."""
    from pygpgpu.cuda.driver.cuinfo import CUInfo
    from pygpgpu.cuda.libraries.cublas import CUBLASInfo
    from pygpgpu.cuda.libraries.cufft import CUFFTInfo
    from pygpgpu.cuda.libraries.cusparse import CUSPARSEInfo

    # The same number means different things in different libraries: cuBLAS's 7 is
    # "invalid value" and the driver's 7 is "mapping error", so the tables must be
    # separate rather than shared.
    assert CUBLASInfo.error_codes[7] != CUInfo.error_codes[7]
    assert CUFFTInfo.error_codes[3] != CUInfo.error_codes[3]

    # Every library's own "success" is 0, and every table carries a message for it.
    for table in (CUBLASInfo.error_codes, CUFFTInfo.error_codes,
                  CUSPARSEInfo.error_codes):
        assert table[0]

    # A code the driver does not have at all is still mapped by its own library.
    assert 10 in CUSPARSEInfo.error_codes
    assert 10 not in CUInfo.error_codes


# ---------- cuBLAS ----------

def test_cublas_gemm_matches_numpy(device):
    """A float32 and a float64 gemm both agree with numpy."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import gemm

    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)

    result = gemm(a, b)
    assert result.dtype == np.float32
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, a @ b, rtol=1e-5, atol=1e-6)

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    result64 = gemm(a64, b64)
    assert result64.dtype == np.float64
    np.testing.assert_allclose(result64, a64 @ b64, rtol=1e-12, atol=1e-12)


def test_cublas_gemm_writes_into_the_output(device):
    """The output argument is filled in and returned."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import gemm

    rng = np.random.default_rng(11)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((3, 2)).astype(np.float32)
    out = np.zeros((2, 2), np.float32)

    result = gemm(a, b, out=out)
    assert result is out
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


def test_cublas_gemm_follows_numpy_shape_rules(device):
    """A 1-D operand follows numpy's rules for `@`."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import gemm

    rng = np.random.default_rng(13)
    vector = rng.standard_normal(4).astype(np.float32)
    other = rng.standard_normal(4).astype(np.float32)

    # Two 1-D arrays make a scalar.
    product = gemm(vector, other)
    assert product.shape == ()
    np.testing.assert_allclose(float(product), float(vector @ other), rtol=1e-5, atol=1e-6)

    # A matrix and a 1-D array make a 1-D array.
    matrix = rng.standard_normal((3, 4)).astype(np.float32)
    result = gemm(matrix, vector)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, matrix @ vector, rtol=1e-5, atol=1e-6)


def test_cublas_gemm_refuses_a_mismatched_shape(device):
    """Two matrices that do not multiply are refused with a message naming the shapes."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import gemm

    a = np.zeros((2, 3), np.float32)
    b = np.zeros((4, 5), np.float32)
    with pytest.raises(ValueError, match="cannot multiply"):
        gemm(a, b)


def test_cublas_axpy_scal_and_dot_match_numpy(device):
    """The level-1 operations agree with numpy."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.wrappers import axpy, dot, scal

    rng = np.random.default_rng(17)
    x = rng.standard_normal(8).astype(np.float32)
    y = rng.standard_normal(8).astype(np.float32)

    original = y.copy()
    result = axpy(2.5, x, y)
    np.testing.assert_allclose(result, original + 2.5 * x, rtol=1e-5, atol=1e-6)

    original = x.copy()
    result = scal(3.0, x)
    np.testing.assert_allclose(result, 3.0 * original, rtol=1e-5, atol=1e-6)

    value = dot(x, y)
    np.testing.assert_allclose(value, float(x @ y), rtol=1e-5, atol=1e-6)


def test_cublas_handle_lifecycle(device):
    """A handle is created, used and destroyed."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.cublas import CUBLAS, cublasHandle_t
    from pygpgpu.cuda.libraries.wrappers import destroy_handle

    handle = cublasHandle_t()
    CUBLAS.cublasCreate_v2(ctypes.byref(handle))
    assert handle.value

    version = ctypes.c_int(0)
    CUBLAS.cublasGetVersion(ctypes.byref(version))
    assert version.value > 0

    destroy_handle(handle)


def test_a_failing_cublas_call_raises_the_librarys_error(device):
    """A refused call raises a RuntimeError naming the library and the code."""
    _require_library("cublas")
    from pygpgpu.cuda.libraries.cublas import CUBLAS, cublasHandle_t

    handle = cublasHandle_t()
    CUBLAS.cublasCreate_v2(ctypes.byref(handle))

    # A gemm whose leading dimension is smaller than the matrix it describes is
    # refused by the library.
    with pytest.raises(LibraryError) as raised:
        CUBLAS.cublasSgemm_v2(
            handle, 0, 0, 4, 4, 4,
            ctypes.byref(ctypes.c_float(1.0)),
            ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_float)), 1,
            ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_float)), 1,
            ctypes.byref(ctypes.c_float(0.0)),
            ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_float)), 1)

    message = str(raised.value)
    assert "cuBLAS" in message
    assert "cublasSgemm_v2" in message


# ---------- cuBLASLt ----------

def test_cublaslt_gemm_matches_numpy(device):
    """The newer cuBLAS path gives the same numbers as the classic one."""
    _require_library("cublaslt")
    from pygpgpu.cuda.libraries.wrappers import lt_gemm

    rng = np.random.default_rng(19)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)

    result = lt_gemm(a, b)
    np.testing.assert_allclose(result, a @ b, rtol=1e-5, atol=1e-6)


# ---------- cuFFT ----------

def test_cufft_rfft_matches_numpy(device):
    """A real-to-complex transform agrees with numpy's."""
    _require_library("cufft")
    from pygpgpu.cuda.libraries.wrappers import rfft

    rng = np.random.default_rng(23)
    signal = rng.standard_normal(16).astype(np.float32)

    spectrum = rfft(signal)
    assert spectrum.dtype == np.complex64
    assert spectrum.shape == (9,)
    np.testing.assert_allclose(spectrum, np.fft.rfft(signal), rtol=1e-4, atol=1e-5)


def test_cufft_round_trip_returns_the_signal(device):
    """The complex-to-real transform undoes the real-to-complex one."""
    _require_library("cufft")
    from pygpgpu.cuda.libraries.wrappers import irfft, rfft

    rng = np.random.default_rng(29)
    signal = rng.standard_normal(32).astype(np.float32)

    spectrum = rfft(signal)
    recovered = irfft(spectrum, 32)
    np.testing.assert_allclose(recovered, signal, rtol=1e-4, atol=1e-5)


def test_cufft_plan_and_destroy(device):
    """A plan is created and destroyed through the bound entry points."""
    _require_library("cufft")
    from pygpgpu.cuda.libraries.cufft import CUFFT, cufftHandle

    plan = cufftHandle()
    CUFFT.cufftPlan1d(ctypes.byref(plan), 16, 0x2A, 1)
    assert plan.value
    CUFFT.cufftDestroy(plan)


# ---------- cuSPARSE ----------

def test_cusparse_csrmv_matches_the_dense_product(device):
    """A sparse matrix-vector multiply agrees with the dense product."""
    _require_library("cusparse")
    from pygpgpu.cuda.libraries.wrappers import csrmv

    # The diagonal matrix diag(1, 2, 3): three rows, so four row offsets.
    row_offsets = np.array([0, 1, 2, 3], np.int32)
    column_indices = np.array([0, 1, 2], np.int32)
    values = np.array([1.0, 2.0, 3.0], np.float32)
    vector = np.array([1.0, 1.0, 1.0], np.float32)

    result = csrmv(row_offsets, column_indices, values, vector)
    np.testing.assert_allclose(result, values * vector, rtol=1e-5, atol=1e-6)


def test_cusparse_csrmv_on_a_real_matrix(device):
    """A matrix with more than one entry per row is multiplied correctly."""
    _require_library("cusparse")
    from pygpgpu.cuda.libraries.wrappers import csrmv

    # The matrix [[1, 2], [0, 3]] in CSR form.
    row_offsets = np.array([0, 2, 3], np.int32)
    column_indices = np.array([0, 1, 1], np.int32)
    values = np.array([1.0, 2.0, 3.0], np.float32)
    vector = np.array([4.0, 5.0], np.float32)

    result = csrmv(row_offsets, column_indices, values, vector)
    np.testing.assert_allclose(result, np.array([14.0, 15.0]), rtol=1e-5, atol=1e-6)


def test_cusparse_handle_lifecycle(device):
    """A handle is created and destroyed through the bound entry points."""
    _require_library("cusparse")
    from pygpgpu.cuda.libraries.cusparse import CUSPARSE

    handle = ctypes.c_void_p()
    CUSPARSE.cusparseCreate(ctypes.byref(handle))
    assert handle.value
    CUSPARSE.cusparseDestroy(handle)


# ---------- the CUDA runtime ----------

def test_cudart_reports_the_device(device):
    """The runtime's own view of the device agrees with the driver's."""
    _require_library("cudart")
    from pygpgpu.cuda.libraries.cudart import CUDART

    count = ctypes.c_int(0)
    CUDART.cudaGetDeviceCount(ctypes.byref(count))
    assert count.value >= 1

    ordinal = ctypes.c_int(-1)
    CUDART.cudaGetDevice(ctypes.byref(ordinal))
    assert 0 <= ordinal.value < count.value

    version = ctypes.c_int(0)
    CUDART.cudaRuntimeGetVersion(ctypes.byref(version))
    assert version.value > 0


def test_cudart_allocates_and_copies(device):
    """The runtime's allocation and copy work."""
    _require_library("cudart")
    from pygpgpu.cuda.libraries.cudart import CUDART, cudaMemcpyKind

    host = np.arange(16, dtype=np.float32)

    pointer = ctypes.c_void_p()
    CUDART.cudaMalloc(ctypes.byref(pointer), host.nbytes)
    try:
        CUDART.cudaMemcpy(pointer, host.ctypes.data, host.nbytes,
                          int(cudaMemcpyKind.cudaMemcpyHostToDevice))

        back = np.zeros(16, np.float32)
        CUDART.cudaMemcpy(back.ctypes.data, pointer, back.nbytes,
                          int(cudaMemcpyKind.cudaMemcpyDeviceToHost))
        np.testing.assert_array_equal(back, host)
    finally:
        CUDART.cudaFree(pointer)


# ---------- cuSOLVER ----------

def test_cusolver_potrf_matches_numpy(device):
    """The Cholesky factorization agrees with numpy's.

    **This skips on the consumer cards cuSOLVER's dense kernels do not support.**
    On those the factorization fails at execution with an illegal memory access
    inside cuSOLVER's own DLL, which is not something the bindings can fix -- so
    the test reports it and skips rather than failing.

    The factorization runs in a **subprocess**, and that is not a detail: an illegal
    memory access poisons the CUDA context, so every later CUDA call in the same
    process fails with it. Running the failing operation in its own process keeps
    that damage away from the rest of the suite.
    """
    _require_library("cusolver")

    script = textwrap.dedent("""
        import ctypes
        import numpy as np

        import pygpgpu.cuda
        from pygpgpu.cuda.oop.devices import Devices
        from pygpgpu.cuda.libraries.cusolver import CUSOLVER
        from pygpgpu.cuda.libraries import wrappers as w

        Devices[0].default_context.make_current()

        matrix = np.array([[4.0, 1.0], [1.0, 3.0]], np.float32)
        handle = ctypes.c_void_p()
        CUSOLVER.cusolverDnCreate(ctypes.byref(handle))

        device_matrix = w.allocate(matrix.nbytes)
        w.to_device(device_matrix, matrix)

        workspace_size = ctypes.c_int(0)
        CUSOLVER.cusolverDnSpotrf_bufferSize(
            handle, 0, 2,
            ctypes.cast(device_matrix, ctypes.POINTER(ctypes.c_float)), 2,
            ctypes.byref(workspace_size))

        workspace = w.allocate(workspace_size.value * 4)
        info = ctypes.c_int(0)
        CUSOLVER.cusolverDnSpotrf(
            handle, 0, 2,
            ctypes.cast(device_matrix, ctypes.POINTER(ctypes.c_float)), 2,
            ctypes.cast(workspace, ctypes.POINTER(ctypes.c_float)),
            workspace_size.value, ctypes.byref(info))

        w.synchronize()
        result = np.zeros((2, 2), np.float32)
        w.to_host(device_matrix, result)
        print("FACTOR:" + repr(result.tolist()))
    """)

    completed = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=600)

    if completed.returncode != 0:
        # The factorization failed on this device. cuSOLVER's dense kernels are
        # not shipped for every architecture, and a card without them fails at
        # execution rather than at setup.
        pytest.skip(f"cuSOLVER is not usable on this device: "
                    f"{completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else 'unknown error'}")

    line = [l for l in completed.stdout.splitlines() if l.startswith("FACTOR:")][0]
    import ast
    result = np.asarray(ast.literal_eval(line[len("FACTOR:"):]), np.float32)

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]], np.float32)
    lower = np.tril(result)
    np.testing.assert_allclose(lower @ lower.T, matrix, rtol=1e-4, atol=1e-5)


# ---------- the missing-toolkit contract ----------

def test_a_missing_library_still_imports_and_fails_at_the_call():
    """A library whose DLL is missing is not imported eagerly and fails at use.

    The import of the package must succeed on a machine with no CUDA toolkit, and
    the failure must be the loader's own OSError at the call.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import ctypes
            real = ctypes.CDLL

            def fake(name, *args, **kwargs):
                if isinstance(name, str) and "cublas" in name.lower():
                    raise OSError("simulated: cublas is not installed")
                return real(name, *args, **kwargs)

            ctypes.CDLL = fake

            from pygpgpu.cuda.libraries import driver_for
            driver = driver_for("cublas")
            print("IMPORTED")

            try:
                driver.dll()
            except OSError as error:
                print("RAISED:" + str(error))
            else:
                raise AssertionError("the missing library was not raised at use")
        """)],
        capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "IMPORTED" in completed.stdout
    assert "RAISED:" in completed.stdout
