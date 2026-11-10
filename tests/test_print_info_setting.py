"""`config.print_info = False` silences the backends even when it is set first.

The setting used to flip only the `print_info` flags of modules already in `sys.modules`, and
a backend module is imported lazily -- by the first call that needs it. So the natural
spelling, one line at the top of a script, did nothing at all: the class did not exist when
the call ran, and it then defaulted to verbose, printing translation, compilation, copy and
launch lines that the caller had explicitly asked to silence. The setting now records the
value as the process-wide default, and each backend takes its flag's initial value from
there.

The ordering is only observable in a *fresh* process, because the test session has already
imported the backends, so the tests that are about it run in subprocesses.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

BACKEND_MODULES = {
    "opencl": "pygpgpu.opencl.driver.cl",
    "cuda": "pygpgpu.cuda.driver.cuda",
    "hip": "pygpgpu.hip.driver.hip",
}

# What each backend's module names its flag-owning class.
FLAG_CLASSES = {
    "opencl": "CL",
    "cuda": "CUDA",
    "hip": "HIP",
}


def _run(script: str, timeout: int = 300) -> subprocess.CompletedProcess:
    completed = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                               capture_output=True, text=True, timeout=timeout)
    assert completed.returncode == 0, (
        f"script failed:\n{completed.stdout}\n{completed.stderr[-3000:]}")
    return completed


def test_the_default_survives_a_later_import_opencl():
    """The backend imported *after* the setting starts out silenced."""
    completed = _run("""
        import sys
        from pygpgpu import config
        config.print_info = False
        module_name = "pygpgpu.opencl.driver.cl"
        assert module_name not in sys.modules, "the backend was imported too early to test this"
        from pygpgpu.opencl.driver import cl
        assert cl.CL.print_info is False, "the backend imported after the setting is verbose"
        print("flag after import:", cl.CL.print_info)
    """)
    assert "flag after import: False" in completed.stdout


def test_setting_it_back_on_before_an_import_stays_verbose():
    """The value is a setting, not a one-way switch: `True` is the default and is honoured."""
    completed = _run("""
        import sys
        from pygpgpu import config
        config.print_info = False
        config.print_info = True
        module_name = "pygpgpu.opencl.driver.cl"
        assert module_name not in sys.modules
        from pygpgpu.opencl.driver import cl
        assert cl.CL.print_info is True, "a backend imported after config.print_info = True is quiet"
        assert config.print_info is True
        print("flag after import:", cl.CL.print_info)
    """)
    assert "flag after import: True" in completed.stdout


def test_a_launch_prints_nothing_once_silenced(tmp_path):
    """The end-to-end fact: a cold compile and launch print no progress lines.

    A cold cache is what makes this meaningful -- a warm one prints nothing whether the flag
    is honoured or not. The three markers bracket the launch, so only the lines the launch
    itself produced are counted. The script runs from a file rather than `-c`, because the
    cache it clears is keyed on the script's own path.
    """
    script = tmp_path / "quiet_launch.py"
    script.write_text(textwrap.dedent("""
        import shutil

        import numpy as np

        from pygpgpu import config, kernel
        from pygpgpu.kernel import source_paths

        config.print_info = False
        shutil.rmtree(source_paths.cache_root(__file__, "opencl"), ignore_errors=True)

        @kernel
        def probe(out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = i

        print("BEFORE")
        probe["opencl"](np.zeros(4, np.float32))
        print("AFTER")
    """), encoding="utf-8", newline="\n")
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                               timeout=300)
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2000:]
    between = completed.stdout.split("BEFORE")[-1].split("AFTER")[0]
    noise = [line for line in between.splitlines() if line.strip()]
    assert not noise, f"a silenced launch still printed: {noise}"


def test_a_launch_still_prints_when_it_is_on(tmp_path):
    """The other direction, so the test above cannot pass by never printing at all."""
    script = tmp_path / "verbose_launch.py"
    script.write_text(textwrap.dedent("""
        import shutil

        import numpy as np

        from pygpgpu import kernel
        from pygpgpu.kernel import source_paths

        shutil.rmtree(source_paths.cache_root(__file__, "opencl"), ignore_errors=True)

        @kernel
        def probe(out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = i

        print("BEFORE")
        probe["opencl"](np.zeros(4, np.float32))
        print("AFTER")
    """), encoding="utf-8", newline="\n")
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                               timeout=300)
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2000:]
    between = completed.stdout.split("BEFORE")[-1].split("AFTER")[0]
    noise = [line for line in between.splitlines() if line.strip()]
    assert any("building" in line for line in noise), (
        f"the verbose default printed no compile line: {noise}")


def test_the_flag_is_per_process_and_independent_of_the_setting():
    """Setting it once does not pin the flag: a direct assignment still wins for one backend.

    The class attribute stays the switch the backend reads, so silencing one backend by hand
    keeps working -- and this call is what decides the *default* for a later import, which is
    why the two are not the same thing.
    """
    completed = _run("""
        from pygpgpu import config
        from pygpgpu.opencl.driver import cl
        from pygpgpu.cuda.driver import cuda

        config.print_info = False
        assert cl.CL.print_info is False and cuda.CUDA.print_info is False
        cl.CL.print_info = True
        assert cl.CL.print_info is True, "a direct assignment must still win"
        assert cuda.CUDA.print_info is False, "one backend's flag must not drag another's"
        print("independent: ok")
    """)
    assert "independent: ok" in completed.stdout
