"""The loading tests, split out of `test_hip_libraries.py`.

The subject is loading.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


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
