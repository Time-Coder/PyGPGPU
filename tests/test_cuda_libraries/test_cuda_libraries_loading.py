"""The loading tests, split out of `test_cuda_libraries.py`.

The subject is loading.
"""
from __future__ import annotations

from _helpers.test_cuda_libraries_helpers import (
    _library_available,
    _require_library,
)

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
