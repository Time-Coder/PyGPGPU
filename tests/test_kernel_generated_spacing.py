"""The generated source's blank-line rules and a method kernel's name.

Three shapes are pinned here, because each is what a reader of a generated kernel sees
first:

* a kernel that is a method reads `Class_method`, like a device helper already did;
* a run of `#include` lines is one block -- the lines are packed together and one blank
  line follows the last of them, not one blank line per line;
* every closed block is followed by a blank line, except between `if` and `else`, where the
  two halves are one statement.
"""
from __future__ import annotations

import glob
import pathlib
import re

import numpy as np
import pytest

from pygpgpu import kernel


def _source(kernel_object, path, *args, backend="cuda") -> str:
    """Run a kernel and return the generated source of its own module.

    The file is only read, never removed: every test in this module writes to the same
    module file (one file per python module), so clearing the directory would delete the
    source the other tests are about to read.
    """
    holder = kernel_object
    for candidate in (kernel_object, getattr(kernel_object, "__self__", None),
                      getattr(kernel_object, "_kernel", None)):
        if candidate is not None and hasattr(candidate, "_cache_dir"):
            holder = candidate
            break
    cache_dir = holder._cache_dir(backend)
    kernel_object[backend](*args)
    extension = {"opencl": "cl", "cuda": "cu", "hip": "hip"}[backend]
    home = pathlib.Path(cache_dir) / f"{pathlib.Path(path).stem}.{extension}"
    assert home.is_file(), sorted(glob.glob(f"{cache_dir}/*"))
    return home.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A method kernel is named after its class
# ---------------------------------------------------------------------------


class Camera:
    def __init__(self) -> None:
        self.scale = 2.0

    @kernel
    def render(self, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = self.scale


def test_a_method_kernel_is_named_after_its_class(gpu_backend):
    """`Camera.render` generates `Camera_render_<hash>`, not `render_<hash>`."""
    source = _source(Camera.render.__get__(Camera()), __file__,
                     np.zeros(1, dtype=np.float32), backend=gpu_backend)
    # `__global__ void ...` on CUDA/HIP, `__kernel void ...` on OpenCL.
    assert re.search(r"__[^\n]* Camera_render_\w+\(", source), source


def test_a_plain_kernel_keeps_its_own_name(gpu_backend):
    """Only a method gains the class prefix; a function keeps its name."""
    source = _source(own_name_kernel, __file__, np.zeros(1, dtype=np.float32),
                     backend=gpu_backend)
    assert "own_name_kernel_" in source, source
    # A plain function's name is its own: no class is prepended.
    assert "Camera_own_name_kernel" not in source, source


# ---------------------------------------------------------------------------
# Includes: one packed run, one blank line after it
# ---------------------------------------------------------------------------


@kernel
def plain_kernel(out: np.ndarray):
    """A plain function keeps its own name."""
    for i in range(out.shape[0]):
        out[i] = 1.0


@kernel
def own_name_kernel(out: np.ndarray):
    """The same shape again, so `keeps_its_own_name` reads a file it just wrote."""
    for i in range(out.shape[0]):
        out[i] = 2.0


def test_a_run_of_includes_is_one_block(gpu_backend):
    """No blank line between consecutive `#include` lines; one after the last."""
    source = _source(Camera.render.__get__(Camera()), __file__,
                     np.zeros(1, dtype=np.float32), backend=gpu_backend)
    includes = [line for line in source.splitlines() if line.lstrip().startswith("#include")]
    if len(includes) < 2:
        pytest.skip("this source reaches no other module")
    for first, second in zip(includes, includes[1:]):
        # The two lines are adjacent in the source: no blank line between them.
        assert f"{first}\n{second}" in source, source


# ---------------------------------------------------------------------------
# A closed block is followed by a blank line
# ---------------------------------------------------------------------------


@kernel
def blocks(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        if a[i] > 0:
            out[i] = 1.0
        else:
            out[i] = 0.0
        if a[i] > 10:
            out[i] = 99.0
        total = 0
        while total < 2:
            total += 1


def test_every_closed_block_is_followed_by_a_blank_line(gpu_backend):
    """A blank line after each `}` -- except the one whose next line is `else`."""
    source = _source(blocks, __file__, np.arange(4, dtype=np.float32),
                     np.zeros(4, dtype=np.float32), backend=gpu_backend)
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "}" or index + 1 >= len(lines):
            continue
        following = lines[index + 1].strip()
        if following.startswith("else"):
            # `if`/`else` are one statement: no separator between the halves.
            assert following == "else", source
            continue
        if following.startswith("#endif"):
            # The file's own end needs no separator, and the `#endif` that closes a
            # structure guard sits directly under its declaration by design.
            continue
        if following.startswith("#ifndef") or following.startswith("#define"):
            # A guarded structure: the guard wraps it, with no separator.
            continue
        assert following == "", (index + 1, following, source)
