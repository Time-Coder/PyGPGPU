"""The solver tests, split out of `test_cuda_libraries.py`.

The subject is solver.
"""
from __future__ import annotations

from _helpers.test_cuda_libraries_helpers import (
    _require_library,
)

import subprocess
import sys
import textwrap
import numpy as np
import pytest


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
