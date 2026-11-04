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
    """`import pygpgpu.cuda` does not run the runtime load on the importing thread.

    The import is cheap because the load happens *elsewhere*: module loading runs on the
    importing thread and `cuInit` runs on a named background thread, so the two overlap.
    Timing alone cannot pin this -- a machine whose module loading is slower than its
    driver load shows a large import either way, and a synchronously initialized driver
    shows a large import too -- so the assertion is on the mechanism: the thread exists,
    it is not the caller, and it is the one that owns the initialization.
    """
    completed = _run("""
        import threading
        import pygpgpu.cuda
        from pygpgpu.cuda.driver.cuda import _INIT_THREAD
        assert _INIT_THREAD is not None, "the driver initialization was not started at import"
        assert _INIT_THREAD is not threading.current_thread(), (
            "the driver initialization ran on the importing thread")
        assert _INIT_THREAD.name == "pygpgpu-cuda-init", _INIT_THREAD.name
        print("init thread:", _INIT_THREAD.name)
    """)
    assert "init thread: pygpgpu-cuda-init" in completed.stdout


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
        import platform
        real = ctypes.CDLL
        # The driver's own DLL name differs per platform, so the simulation matches
        # whichever one this machine would really load.
        target = 'nvcuda' if platform.system() == 'Windows' else 'libcuda.so'

        def fake(name, *args, **kwargs):
            if isinstance(name, str) and target in name.lower():
                raise OSError('simulated: ' + target + ' not found')
            return real(name, *args, **kwargs)

        ctypes.CDLL = fake

        import pygpgpu.cuda
        from pygpgpu.cuda.driver.cuda import MetaCUDA, _wait_ready

        # The initialization runs on a background thread started at import, so the
        # recorded failure appears a moment *after* the import returns. A test that
        # read it immediately raced the thread and saw None; waiting is what a
        # caller's first driver call does anyway, so the contract under test --
        # "the failure is recorded, not raised" -- is unchanged.
        _wait_ready()
        assert MetaCUDA.last_init_error is not None, "the failure was not recorded"

        from pygpgpu.cuda.oop.devices import Devices
        try:
            list(Devices)
        except OSError as error:
            assert target in str(error), error
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
    import pygpgpu.metal
    import pygpgpu.opencl

    for package in (pygpgpu.cuda, pygpgpu.hip, pygpgpu.metal, pygpgpu.opencl):
        for name in dir(package):
            assert not (("init" in name.lower() or "ready" in name.lower()
                         or "wait" in name.lower()) and not name.startswith("_")), \
                f"{package.__name__}.{name} is public but is an internal helper"
        # `from package import *` must not hand the helpers out either.
        namespace:dict = {}
        exec(f"from {package.__name__} import *", namespace)  # noqa: S102 - the package itself
        for name in namespace:
            assert not name.startswith("_start_"), f"{name} was exported publicly"
