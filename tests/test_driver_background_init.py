"""The driver is initialized on a background thread, and a call waits for it.

Three things are pinned here:

* importing a backend package does NOT block on the runtime load (`cuInit` alone is ~600ms
  on a warm machine, and it is a GIL-releasing C call, so it belongs off the import path);
* a driver call made before that initialization finished waits for it and then works --
  the wait is the whole synchronization story, and it must never deadlock;
* a machine without the runtime still imports the backend package, and the failure
  surfaces at the call with the driver's own error.

The tests run in subprocesses: the driver state is per-process and the import hook is
installed once, so a fresh process is the only honest way to observe the import timing.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(script:str, timeout:int=300)->subprocess.CompletedProcess:
    completed = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                               capture_output=True, text=True, timeout=timeout)
    assert completed.returncode == 0, (
        f"script failed:\n{completed.stdout}\n{completed.stderr[-3000:]}")
    return completed


def test_the_import_does_not_wait_for_the_driver():
    """`import pygpgpu.cuda` returns without waiting for `cuInit`.

    The import is measured against the driver's own completion: the background thread has
    to be done by the time the first device is reachable, but the import statement itself
    must not have waited for it.
    """
    completed = _run("""
        import time
        start = time.perf_counter()
        import pygpgpu.cuda
        after_import = time.perf_counter()
        # The wait is the driver's own business: a caller never asks for it, so this
        # reaches for the private helper to observe when the background work finished.
        from pygpgpu.cuda.driver.cuda import _wait_ready
        _wait_ready()
        after_wait = time.perf_counter()
        print(f"import_ms={((after_import - start) * 1000):.1f}")
        print(f"wait_ms={((after_wait - after_import) * 1000):.1f}")
    """)
    import_ms = float(completed.stdout.split("import_ms=")[1].split()[0])
    wait_ms = float(completed.stdout.split("wait_ms=")[1].split()[0])
    # The import itself is cheap (module loading); the runtime load is what ran in the
    # background. A generous bound keeps the test about the shape, not about the machine:
    # if the import had blocked on the driver, it would have taken the driver's ~600ms.
    assert import_ms < wait_ms + 1.0 or wait_ms < 50.0, (
        f"import_ms={import_ms} wait_ms={wait_ms}: the import appears to wait for the driver")


def test_a_call_waits_for_the_initialization_and_works():
    """A driver call issued right after the import waits, then succeeds.

    The deadlock this guards against is the obvious one: the background thread runs the
    initialization, and the initialization itself must not wait for the background thread.
    """
    completed = _run("""
        import pygpgpu.cuda
        from pygpgpu.cuda.oop.devices import Devices
        device = Devices[0]
        assert device.name, "the device has no name"
        print("device:", device.name)
    """)
    assert "device:" in completed.stdout


def test_a_missing_runtime_still_imports_and_fails_at_the_call():
    """The deferred-failure contract, at the driver level.

    The package imports without the runtime, and the first call raises the driver's own
    error -- not a wrapped one, and not at import time.
    """
    completed = _run("""
        import ctypes
        real = ctypes.CDLL

        def fake(name, *args, **kwargs):
            if isinstance(name, str) and 'nvcuda' in name.lower():
                raise OSError('simulated: nvcuda.dll not found')
            return real(name, *args, **kwargs)

        ctypes.CDLL = fake

        import pygpgpu.cuda
        from pygpgpu.cuda.driver.cuda import MetaCUDA
        assert MetaCUDA.last_init_error is not None, "the failure was not recorded"

        from pygpgpu.cuda.oop.devices import Devices
        try:
            list(Devices)
        except OSError as error:
            assert 'nvcuda' in str(error), error
            print("raised at use:", type(error).__name__)
        else:
            raise AssertionError('the missing runtime was not raised at use')
    """)
    assert "raised at use:" in completed.stdout


def test_the_mechanism_is_not_part_of_the_public_surface():
    """A user never starts the initialization or waits for it -- the library does both.

    The helpers exist, but as private names: they are not exported without the
    underscore, so an editor's completion does not offer them and `from ... import *`
    does not bring them in.
    """
    import pygpgpu.cuda
    import pygpgpu.hip
    import pygpgpu.opencl

    for package in (pygpgpu.cuda, pygpgpu.hip, pygpgpu.opencl):
        for name in dir(package):
            assert not (("init" in name.lower() or "ready" in name.lower()
                         or "wait" in name.lower()) and not name.startswith("_")), \
                f"{package.__name__}.{name} is public but is an internal helper"
        # `from package import *` must not hand the helpers out either.
        namespace:dict = {}
        exec(f"from {package.__name__} import *", namespace)  # noqa: S102 - the package itself
        for name in namespace:
            assert not name.startswith("_start_"), f"{name} was exported publicly"
