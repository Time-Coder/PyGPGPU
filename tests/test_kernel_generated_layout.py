"""The generated source's semantic shape: structure pointers and the receiver struct.

Two rules are pinned here, because each is visible to whoever reads a generated kernel and
each was wrong before:

* a structure parameter is a pointer, and one the body never writes is `const`;
* a `@kernel`-decorated instance method sends its receiver as ONE structure, so a scalar
  member is a field of that pointer and an ndarray member stays its own argument.

The code *layout* rules (blank lines, braced guards, the element-loop declaration) are
checked where they are cheapest to see: the ray-tracing example's generated source, a real
kernel of six device functions.
"""
from __future__ import annotations

import ctypes
import glob
import pathlib
import re
import shutil

import numpy as np
import pytest
from pygf import float3, point3f

from pygpgpu import kernel


def _cache_dir_of(kernel_object, backend) -> str:
    """The generated-source directory a kernel writes into.

    A bound method (camera.render) does not expose it; the callable it wraps does.
    """
    for candidate in (kernel_object, getattr(kernel_object, "_kernel", None),
                      getattr(kernel_object, "__func__", None)):
        if candidate is not None and hasattr(candidate, "_cache_dir"):
            return candidate._cache_dir(backend)
    raise AssertionError(f"no cache_dir on {kernel_object!r}")


def _generated_source(kernel_object, *args, backend="cuda") -> str:
    """Run a kernel and return the C source it was compiled from."""
    # A bound kernel method's own PyKernel lives on the instance (camera.render);
    # the descriptor's __get__ result carries it as __self__.
    cache_dir = _cache_dir_of(kernel_object, backend)
    shutil.rmtree(cache_dir, ignore_errors=True)
    kernel_object[backend](*args)
    extension = {"cuda": "cu", "opencl": "cl", "hip": "hip"}[backend]
    files = sorted(glob.glob(f"{cache_dir}/*.{extension}"))
    assert files, f"no generated source under {cache_dir}"
    newest = max(files, key=lambda path: pathlib.Path(path).stat().st_mtime_ns)
    return pathlib.Path(newest).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A structure parameter is a pointer, and a read-only one is const
# ---------------------------------------------------------------------------


class _Region(ctypes.Structure):
    _fields_ = [("left", ctypes.c_int), ("right", ctypes.c_int)]


@kernel
def reads_a_region(region, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = region["left"] + region["right"]


class _WritableRegion(ctypes.Structure):
    _fields_ = [("left", ctypes.c_int), ("right", ctypes.c_int)]


@kernel
def writes_a_region(region, out: np.ndarray):
    for i in range(out.shape[0]):
        region.left = i
        out[i] = region.left


def test_a_read_only_structure_is_a_const_pointer(gpu_backend):
    out = np.zeros(1, dtype=np.float32)
    source = _generated_source(reads_a_region, _Region(2, 3), out, backend=gpu_backend)
    # The structure's C name is generated (`_st_<hash>`), so the predicate matches the
    # pointer form rather than the caller's class name.
    assert re.search(r"const \w+\* region", source), source
    # The member read goes through the pointer, never through a value copy.
    assert "(region)->left" in source, source


def test_a_written_structure_is_a_plain_pointer(gpu_backend):
    out = np.zeros(1, dtype=np.float32)
    source = _generated_source(writes_a_region, _WritableRegion(2, 3), out,
                               backend=gpu_backend)
    assert not re.search(r"const \w+\* region", source), source
    assert re.search(r"\w+\* region", source), source


def test_a_written_member_is_visible_to_the_body(runner):
    """A store through the pointer is read back by the body that made it."""
    out = np.zeros(1, dtype=np.float32)
    runner(writes_a_region, _WritableRegion(0, 0), out)
    assert float(out[0]) == 0.0


# ---------------------------------------------------------------------------
# The receiver is one structure; array members stay their own arguments
# ---------------------------------------------------------------------------


class _Camera:
    def __init__(self, scale: float, height: int, width: int) -> None:
        self.scale = scale
        self.image_height = height
        self.image_width = width
        self.image = np.zeros((height, width), dtype=np.float32)

    @kernel
    def render(self, out: np.ndarray):
        for i, j in np.ndindex(self.image_height, self.image_width):
            out[i, j] = self.image[i, j] + self.scale


def test_the_receiver_is_one_structure_argument(gpu_backend):
    camera = _Camera(2.0, 2, 3)
    out = np.zeros((2, 3), dtype=np.float32)
    source = _generated_source(camera.render, out, backend=gpu_backend)
    # One struct pointer named for the receiver, and the ndarray member as its own
    # parameter -- never one `_self_` scalar per attribute.
    assert re.search(r"const \w+\* self", source), source
    assert "_self_scale" not in source, source
    assert "_self_image" in source, source
    # A scalar member reads through the receiver pointer.
    assert "(self)->scale" in source, source


def test_a_receiver_member_can_be_the_loop_extent(runner):
    """`np.ndindex(self.image_height, ...)`: the extent is that member's value."""
    camera = _Camera(2.0, 2, 3)
    out = np.zeros((2, 3), dtype=np.float32)
    runner(camera.render, out)
    np.testing.assert_allclose(out, np.full((2, 3), 2.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# A method's device function is named after the class it is defined on
# ---------------------------------------------------------------------------


class _Ray(ctypes.Structure):
    _fields_ = [("origin", point3f), ("direction", float3)]

    def at(self, t: float) -> point3f:
        return self.origin + t * self.direction


@kernel
def calls_a_method(ray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = ray.at(2.0)["x"]


def test_a_method_is_named_after_its_own_class(gpu_backend):
    out = np.zeros(1, dtype=np.float32)
    source = _generated_source(calls_a_method,
                               _Ray(point3f(1, 2, 3), float3(0, 0, -1)), out,
                               backend=gpu_backend)
    assert "Ray_at" in source, source


def test_the_named_method_answers_its_result(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_method, _Ray(point3f(1.0, 2.0, 3.0), float3(0.0, 0.0, -1.0)), out)
    assert float(out[0]) == pytest.approx(1.0)
