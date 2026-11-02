"""``KERNEL_TRANSLATING``: a compile-time switch for a body that differs on each side.

``typing.TYPE_CHECKING`` is False at runtime and True to a type checker, so one source can
say something only one reader acts on. This is the same trick for the *translator*: the
constant is False in python and True while a kernel is being translated, so

    if KERNEL_TRANSLATING:      # device branch
        ...
    if not KERNEL_TRANSLATING:  # python branch
        ...

are decided at translation time. Only one branch reaches the generated source, and no ``if``
survives into it -- it is a macro, not a runtime test. The interpreter backend runs the body
as written, which is exactly the python branch.

**The trigger is the name, not the object it resolves to.** A body that declares its own
``KERNEL_TRANSLATING`` -- a module global, anything that reads as that name -- gets the same
treatment, because a kernel author writing that name is asking for this, and requiring an
import would only make the spelling longer.
"""
from __future__ import annotations

import pathlib

import numpy as np

from pygpgpu import KERNEL_TRANSLATING, get_global_id, kernel


@kernel(domain="out")
def branches_on_the_switch(out: np.ndarray):
    """The device and the interpreter take different branches of one body."""
    i = get_global_id(0)
    if KERNEL_TRANSLATING:
        out[i] = 7.0
    else:
        out[i] = -1.0


@kernel(domain="out")
def host_only_code_is_dropped(out: np.ndarray):
    """A python-only expression under the switch must not reach the device.

    `len(str(i))` has no device meaning: if the branch were translated, the kernel would not
    compile. It compiling at all is what proves the branch was dropped.
    """
    i = get_global_id(0)
    if not KERNEL_TRANSLATING:
        out[i] = float(len(str(i)))
    else:
        out[i] = float(i)


def test_the_constant_is_false_in_python():
    """A plain `if KERNEL_TRANSLATING:` must be dead code for the interpreter."""
    assert KERNEL_TRANSLATING is False


def test_each_side_takes_its_own_branch(runner):
    """Same body, different answers -- which is the whole point of the switch."""
    out = np.zeros(4, dtype=np.float32)
    runner(branches_on_the_switch, out)
    assert not np.any(out == 0.0), out.tolist()


def test_the_python_side_keeps_the_not_form():
    """The interpreter runs the body unchanged, so `not KERNEL_TRANSLATING` is taken."""
    out = np.zeros(4, dtype=np.float32)
    branches_on_the_switch["python"][(1, 4)](out)
    np.testing.assert_array_equal(out, np.full(4, -1.0, dtype=np.float32))


def test_the_device_side_keeps_the_plain_form(gpu_backend):
    """The translator resolves the constant to True and keeps that branch."""
    out = np.zeros(4, dtype=np.float32)
    branches_on_the_switch[gpu_backend][(1, 4)](out)
    np.testing.assert_array_equal(out, np.full(4, 7.0, dtype=np.float32))


def test_host_only_code_is_dropped_from_the_device_source(gpu_backend):
    """A branch the device cannot run compiles because it was never translated."""
    out = np.zeros(4, dtype=np.float32)
    host_only_code_is_dropped[gpu_backend][(1, 4)](out)
    np.testing.assert_array_equal(out, np.arange(4, dtype=np.float32))


def test_no_if_survives_into_the_generated_source(gpu_backend):
    """A macro, not a runtime test: the branch is emitted inline."""
    source: str = _generated_source(host_only_code_is_dropped, gpu_backend)
    body: str = source.split("__global__")[-1].split("__kernel")[-1]
    assert "len(" not in body, body
    # The bounds guard the work-item head emits is the only `if` a gid-style body has.
    assert body.count("if (") <= 1, body


def _generated_source(kernel_object, backend: str) -> str:
    """The C source the kernel was compiled from."""
    cache_dir: str = kernel_object._cache_dir(backend)
    extension: str = {"cuda": "cu", "opencl": "cl", "hip": "hip"}[backend]
    files = sorted(glob_files(cache_dir, extension))
    assert files, f"no generated source under {cache_dir}"
    return pathlib.Path(files[0]).read_text(encoding="utf-8")


def glob_files(directory: str, extension: str) -> list:
    import glob

    return glob.glob(f"{directory}/*.{extension}")


def test_a_user_defined_name_triggers_it_too(gpu_backend, tmp_path):
    """The name is the trigger, not the object -- a local constant works as well."""
    module_path = tmp_path / "local_switch.py"
    module_path.write_text(
        "import numpy as np\n"
        "from pygpgpu import get_global_id, kernel\n"
        "\n"
        "KERNEL_TRANSLATING = 0\n"
        "\n"
        "@kernel(domain='out')\n"
        "def uses_its_own_constant(out: np.ndarray):\n"
        "    i = get_global_id(0)\n"
        "    if KERNEL_TRANSLATING:\n"
        "        out[i] = 99.0\n"
        "    else:\n"
        "        out[i] = -1.0\n",
        encoding="utf-8")

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("local_switch_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_switch_mod"] = module
    spec.loader.exec_module(module)
    try:
        out = np.zeros(3, dtype=np.float32)
        module.uses_its_own_constant[gpu_backend][(1, 3)](out)
        np.testing.assert_array_equal(out, np.full(3, 99.0, dtype=np.float32))
    finally:
        del sys.modules["local_switch_mod"]
