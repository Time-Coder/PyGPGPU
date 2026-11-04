"""kernel_import: one module name from same-named .cl/.cu/.hip sources.

The merged module publishes every kernel the sources declare as an object used exactly
like a `@kernel` function. What the tests pin:

* same name + same parameter list across backends = one function, several
  implementations -- a bare call runs the default backend, a bracket runs the backend it
  names, and the result agrees across all of them;
* same name + different parameter lists = overloads -- a backend that declares only the
  double form runs the double form, and a bare call lands on an implementation whose
  parameters fit the arguments;
* a name only one backend declares still exists -- a bracket naming any other backend
  raises KeyError AT the bracket, before any argument is looked at;
* structs and struct arrays cross the merged type, whichever backend runs;
* the stub is regenerated from the sources, with one __call__ per distinct parameter
  list and @overload markers when there are several.

The python backend is not part of this: `.py` sources are not merged, so these tests
parameterize over the three device backends this machine has.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

from pygpgpu import kernel_import
from pygpgpu.kernel.decorator import _probe_backend

# The sources live in examples/kernel_import; the hook resolves a name against a parent
# package's __path__, so importing through the package keeps the stub next to the sources.
_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "examples")
_KERNEL_IMPORT_DIR = os.path.join(_EXAMPLES_DIR, "kernel_import")


@pytest.fixture(scope="module")
def vecops():
    """The merged `vecops` module, imported once for the whole file.

    The example package only exists on `sys.path` because this fixture put it there, so the
    import is dynamic: a static `from kernel_import import vecops` is an unresolvable name to
    a type checker (and it collides with `pygpgpu.kernel_import`, which is a different thing).
    """
    if _EXAMPLES_DIR not in sys.path:
        sys.path.insert(0, _EXAMPLES_DIR)
    kernel_import.install()
    return importlib.import_module("kernel_import.vecops")


def test_same_sources_resolve_per_backend(vecops):
    """The merged wrapper knows which file each backend implements the kernel from."""
    sources = vecops.add_offset.sources()
    assert set(sources) == {"cuda", "hip", "opencl"}
    assert sources["opencl"].endswith(".cl")
    assert sources["cuda"].endswith(".cu")
    assert sources["hip"].endswith(".hip")


def test_bare_call_runs_the_default_backend(vecops):
    """No bracket: the documented probe order supplies the implementation."""
    n = 1024
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    out = np.empty_like(a)
    vecops.add_offset(a, b, out, np.float32(1.0), np.int32(n))
    np.testing.assert_allclose(out, a + b + 1.0)


@pytest.mark.parametrize("selector", ["cuda", "hip", "opencl"])
def test_a_backend_bracket_runs_that_implementation(vecops, selector, gpu_backend):
    """`k[selector](...)` runs the named backend's implementation of the same kernel."""
    if selector != gpu_backend:
        pytest.skip(f"the machine may not have {selector}; the gpu_backend fixture "
                    f"parameterizes over the backends present")
    n = 1024
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    out = np.zeros_like(a)
    vecops.add_offset[selector](a, b, out, np.float32(1.0), np.int32(n))
    np.testing.assert_allclose(out, a + b + 1.0)


def test_a_work_size_bracket_states_blocks_and_threads(vecops, gpu_backend):
    """`(blocks, threads)` on the merged wrapper, the same CUDA spelling everywhere."""
    n = 1024
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    out = np.zeros_like(a)
    vecops.add_offset[(8, 128)](a, b, out, np.float32(1.0), np.int32(n))
    np.testing.assert_allclose(out, a + b + 1.0)


def test_a_missing_implementation_raises_at_the_bracket(vecops, gpu_backend):
    """`negate` is declared by vecops.cl only: any other backend's bracket is KeyError."""
    out = np.zeros(8, dtype=np.float32)
    for selector in ("cuda", "hip"):
        if selector == gpu_backend:
            with pytest.raises(KeyError) as caught:
                vecops.negate[selector](out, np.int32(8))
            assert "negate" in str(caught.value)
            assert "opencl" in str(caught.value)


def test_an_opencl_only_kernel_runs_where_it_exists(vecops, gpu_backend):
    if gpu_backend != "opencl":
        pytest.skip("negate is an opencl-only declaration")
    out = np.arange(8, dtype=np.float32)
    vecops.negate(out, np.int32(8))
    np.testing.assert_allclose(out, -np.arange(8, dtype=np.float32))


def test_overloads_pick_their_backend(vecops, gpu_backend):
    """`shift` is float in .cl/.hip and double in .cu.

    The bracket names the backend; the overload is the declaration that backend made.
    A bare call walks the probe order and lands on the first implementation whose
    parameters fit the arguments -- a float array skips cuda's double-only declaration.
    """
    n = 256
    floats = np.zeros(n, dtype=np.float32)
    doubles = np.zeros(n, dtype=np.float64)

    if gpu_backend == "cuda":
        vecops.shift["cuda"](doubles, 2.0, np.int32(n))
        np.testing.assert_allclose(doubles, 2.0)
        # a float argument cannot bind cuda's double declaration, so the bare call
        # falls through to another implementing backend -- which this machine has
        # to own. A CUDA-only box (a WSL GPU passthrough with no OpenCL platform,
        # for instance) has nowhere to fall through to, and the fall-through is not
        # what this test is about.
        fallbacks = [backend for backend in vecops.shift.backends
                     if backend != "cuda" and _probe_backend(backend)]
        if not fallbacks:
            pytest.skip("the fall-through needs a second implementing backend")
        floats += 1.0
        vecops.shift(floats, np.float32(2.0), np.int32(n))
        np.testing.assert_allclose(floats, 3.0)
    elif gpu_backend == "hip":
        vecops.shift["hip"](floats, np.float32(2.0), np.int32(n))
        np.testing.assert_allclose(floats, 2.0)
    else:
        vecops.shift["opencl"](floats, np.float32(2.0), np.int32(n))
        np.testing.assert_allclose(floats, 2.0)


def test_struct_arguments_cross_the_merged_type(vecops, gpu_backend):
    """The merged `Pt` is one ctypes class whichever backend runs the kernel."""
    Pt = vecops.Pt
    points = np.array([Pt(x=3.0, y=1.0), Pt(x=0.5, y=0.5)], dtype=Pt)
    results = np.zeros(2, dtype=np.int32)
    vecops.in_circle(points, results, Pt(x=0.5, y=0.5), np.float32(1.0), np.int32(2))
    assert results.tolist() == [0, 1]


def test_the_stub_is_generated_from_the_sources(vecops):
    """The merged stub carries one __call__ per distinct parameter list."""
    stub_path = os.path.join(_KERNEL_IMPORT_DIR, "vecops.pyi")
    assert os.path.isfile(stub_path)
    with open(stub_path, encoding="utf-8") as stub_file:
        content = stub_file.read()
    # one class per kernel, plus the trailing attribute lines
    for kernel_name in ("add_offset", "shift", "in_circle", "negate"):
        assert f"class Kernel_{kernel_name}(KernelImportWrapper):" in content
        assert f"{kernel_name}: Kernel_{kernel_name}" in content
    # `shift` has two parameter lists, so its overloads appear; the single-list kernels
    # use plain @override declarations
    assert content.count("def __call__") >= 5
    assert "@overload" in content


def test_brackets_do_not_mutate_the_module_attribute(vecops, gpu_backend):
    """A bracket returns a copy: the module attribute stays bracket-free."""
    n = 64
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    out = np.zeros_like(a)
    # Take a bracketed copy, then prove the attribute itself still resolves on its own.
    bracketed = vecops.add_offset[gpu_backend]
    assert bracketed is not vecops.add_offset
    vecops.add_offset(a, b, out, np.float32(1.0), np.int32(n))
    np.testing.assert_allclose(out, a + b + 1.0)


def test_a_missing_runtime_defers_its_error_to_first_use():
    """A machine without a runtime can still import a module whose source exists.

    The driver layer records the load failure instead of failing its own import, so
    `kernel_import` parses every source it finds even on a machine with no CUDA at all;
    the error then surfaces -- the driver's own OSError, not a wrapped guess -- when a
    kernel on that backend is actually selected. Runs in a subprocess because the
    simulation replaces `ctypes.CDLL` process-wide.
    """
    script = (
        "import ctypes, sys\n"
        "real = ctypes.CDLL\n"
        "# The driver's own DLL name differs per platform (nvcuda.dll on Windows,\n"
        "# libcuda.so on Linux), so the simulation matches whichever this is.\n"
        "import platform\n"
        "target = 'nvcuda' if platform.system() == 'Windows' else 'libcuda.so'\n"
        "def fake(name, *args, **kwargs):\n"
        "    if isinstance(name, str) and target in name.lower():\n"
        "        raise OSError('simulated: ' + target + ' not found')\n"
        "    return real(name, *args, **kwargs)\n"
        "ctypes.CDLL = fake\n"
        "sys.path.insert(0, r'" + _EXAMPLES_DIR + "')\n"
        "from pygpgpu import kernel_import\n"
        "kernel_import.install()\n"
        "import kernel_import.vecops as vecops\n"
        "assert 'cuda' in vecops.add_offset.backends, 'the .cu source still declares it'\n"
        "import numpy as np\n"
        "a = np.random.rand(64).astype(np.float32)\n"
        "b = np.random.rand(64).astype(np.float32)\n"
        "out = np.empty_like(a)\n"
        "try:\n"
        "    vecops.add_offset['cuda'](a, b, out, np.float32(1.0), np.int32(64))\n"
        "except OSError as error:\n"
        "    assert target in str(error), error\n"
        "else:\n"
        "    raise AssertionError('the missing runtime was not raised at use')\n"
        "print('deferred-failure contract OK')\n"
    )
    import subprocess

    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "deferred-failure contract OK" in completed.stdout
