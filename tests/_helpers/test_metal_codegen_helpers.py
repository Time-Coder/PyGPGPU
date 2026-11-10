"""The classes and the helpers the split test files share.

A class is the SAME class whichever group uses it -- two copies would
be two classes, and the tests compare them.
"""
from __future__ import annotations

from __future__ import annotations
from typing import Any
import numpy as np
import pytest
from pygpgpu import (barrier, blockDim, blockIdx, get_global_id,
                     get_local_id, threadIdx, local_array_per_group)
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.translator import FunctionTranslator


def _compile_with_stub(function) -> None:
    """Compile one kernel on the Metal path with only the `Program` stubbed."""
    import os
    import tempfile

    import pygpgpu.metal.oop.program as program_module
    from pygpgpu.kernel.decorator import PyKernel
    from pygpgpu.metal.oop import program_parser  # noqa: F401  (import surface only)
    from pygpgpu.kernel import source_paths

    directory = tempfile.mkdtemp()

    class StubProgram:
        def __init__(self, file_name, **_options):
            pass

        @property
        def kernel_infos(self):
            return {}

        @property
        def kernels(self):
            return {}

    saved_cache = PyKernel._cache_dir
    saved_root = source_paths.cache_root
    saved_program = program_module.Program
    PyKernel._cache_dir = lambda self, backend: os.path.join(directory, backend)
    source_paths.cache_root = lambda source_file, backend: os.path.join(directory, backend)
    program_module.Program = StubProgram
    try:
        args = [np.zeros(8, np.float32)] * 2
        __import__("pygpgpu").kernel(function)._compile(
            args, [infer_ctype(a) for a in args], [], [], None, "sig", "metal", object())
    finally:
        PyKernel._cache_dir = saved_cache
        source_paths.cache_root = saved_root
        program_module.Program = saved_program


def _compiled_source(function, args, tmp_path: Any, monkeypatch, **decorator_args) -> str:
    """Compile one kernel on the Metal path and return the source it wrote.

    Only the `Program` constructor is stubbed: it needs Metal.framework. Everything
    else is the real translation and caching the backend uses.
    """
    import pygpgpu.metal.oop.program as program_module
    from pygpgpu.kernel.decorator import PyKernel
    from pygpgpu.kernel import source_paths

    monkeypatch.setattr(PyKernel, "_cache_dir",
                        lambda self, backend: str(tmp_path / backend))
    monkeypatch.setattr(source_paths, "cache_root",
                        lambda source_file, backend: str(tmp_path / backend))

    captured: dict = {}

    class StubProgram:
        def __init__(self, file_name, **_options):
            captured["source"] = open(file_name, encoding="utf-8").read()

        @property
        def kernel_infos(self):
            return {}

        @property
        def kernels(self):
            return {}

    kernel_object = __import__("pygpgpu").kernel(function, **decorator_args)
    original = program_module.Program
    program_module.Program = StubProgram
    try:
        try:
            kernel_object._compile(args, [infer_ctype(a) for a in args], [], [], None,
                                   "sig", "metal", object())
        except KeyError:
            pass
    finally:
        program_module.Program = original
    return captured["source"]


def _image_helper_resolver():
    return lambda name: {"image_passing_helper": image_passing_helper}.get(name)


def _no_slice_marker_leak(source: str) -> None:
    """Neither internal type marker may reach the generated source.

    The translator carries a parameter's type as a tuple for two kinds of value --
    `('shared_slice', element, shape)` and `('image', channels, type)` -- and both used
    to fall through `_device_type`, which printed the tuple itself into the signature:

        float f(('shared_slice', 'float', (64,)) buf, int i)
        float f(('image', 4, 'float') img, int y, int x)

    No compiler accepts either. The markers are internal; the parameters they describe
    are pointers and image types.
    """
    assert "('shared_slice'" not in source, source[:400]
    assert "('image'" not in source, source[:400]


def _syntax_diagnostics(source: str) -> tuple:
    """Compile the emitted MSL as C++14; skip the assertion when no compiler exists."""
    from metal_syntax_check import check, cpp_compiler

    if cpp_compiler() is None:
        pytest.skip("no C++ compiler to check the emitted source with")
    return check(source)


def body_with_unknown_call(x, out):
    """A body calling a name that is no device function (the refusal under test)."""
    for i in range(out.shape[0]):
        # Deliberately undefined: the translator must refuse an unknown device helper and
        # report where the name sits. Both linters are told, because neither understands
        # that the call is the test.
        out[i] = unknown_device_function(x[i])  # noqa: F821  # ty: ignore[unresolved-reference]


def cuda_index_body(a, out):
    """A CUDA reader's index spelling, on a body that reads the work item itself."""
    i = blockIdx.x * blockDim.x + threadIdx.x
    out[i] = a[i]


def generate(func, n_args: int, backend: str = "metal") -> str:
    """The rendered source for `func`, without compiling it."""
    types = [infer_ctype(np.zeros(8, np.float32)) for _ in range(n_args)]
    translator = FunctionTranslator(func, types, backend, lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    return "\n\n".join(source for _, source in translator.render_kernels("k_test"))


def gid_style_body(out):
    """A gid-style body: it names its own domain and reads the work item itself."""
    i = get_global_id(0)
    out[i] = i * 2.0


def histogram(pixels, bins):
    """A data-driven index -- the racy shape that must become an atomic."""
    for i in range(pixels.shape[0]):
        bins[pixels[i]] += 1


def image_passing_helper(img, y, x):
    """Receives an image but never samples it -- the shape that leaked."""
    return 1.0


def image_uses_helper(img, out):
    for i in range(out.shape[0]):
        out[i] = image_passing_helper(img, i, 0)


def index_reading_helper(a):
    """A device helper that reads the work item itself."""
    return a[get_global_id(0)]


def kernel_using_shared_a(x, out):
    for i in range(out.shape[0]):
        out[i] = shared_across_kernels(x[i])


def kernel_using_shared_b(x, out):
    for i in range(out.shape[0]):
        out[i] = shared_across_kernels(x[i]) + 1.0


def plain_helper(v):
    """A helper that reads nothing but its argument."""
    return v * 3.0


def printer(x, out):
    for i in range(out.shape[0]):
        print(x[i])
        out[i] = x[i]


def read_image(src, out):
    """Read one image row: the shape an image parameter takes."""
    for i in range(out.shape[0]):
        out[i] = src[0, i]


def reads_shared(buf, i):
    """A device helper that reads one element of a group's shared slice."""
    return buf[i]


def returning(x, out):
    """A kernel whose work items return a value: the first one to reach it wins."""
    for i in range(out.shape[0]):
        out[i] = x[i]
        return x[i]


def returns_member(parts, i):
    """A helper returning a scalar read from a structure."""
    return parts[i].pos


def returns_record(parts, i):
    """A helper returning the whole structure, by value."""
    return parts[i]


def returns_vector(v, i):
    """A helper returning a vector element."""
    return v[i]


def shared_across_kernels(v):
    """One helper reached by two different kernels."""
    return v * 3.0


def shared_buffer(x, out):
    """A `local_array_per_group` buffer: the shape a work-group scratch array takes."""
    buf = local_array_per_group(256, "float32")
    gid = get_global_id(0)
    out[gid] = buf[blockIdx][gid] + x[gid]


def shared_scratch(x, out):
    """A `local_array_per_group` buffer, written, barred, then read."""
    scratch = local_array_per_group(256, np.float32)
    lane = get_local_id(0)
    scratch[blockIdx][lane] = x[lane]
    barrier()
    out[lane] = scratch[blockIdx][lane]


def structured_add(parts, out):
    """A ctypes-structure argument: the array's dtype names the record."""
    for i in range(out.shape[0]):
        out[i] = parts[i].pos + parts[i].vel


def structured_sum(parts, out):
    """A ctypes-structure argument, read through its members."""
    for i in range(out.shape[0]):
        out[i] = parts[i].pos + parts[i].vel


def uses_image_helper(img, out):
    for i in range(out.shape[0]):
        out[i] = image_passing_helper(img, i, 0)


def uses_index_helper(a, out):
    for i in range(out.shape[0]):
        out[i] = index_reading_helper(a)


def uses_plain_helper(a, out):
    for i in range(out.shape[0]):
        out[i] = plain_helper(a[i])


def uses_reads_shared(a, out):
    scratch = local_array_per_group(64, np.float32)
    lane = get_local_id(0)
    scratch[blockIdx][lane] = a[lane]
    barrier()
    out[lane] = reads_shared(scratch[blockIdx], lane)


def uses_record_member(parts, out):
    for i in range(out.shape[0]):
        out[i] = returns_member(parts, i)


def uses_record_value(parts, out):
    for i in range(out.shape[0]):
        out[i] = returns_record(parts, i).pos


def uses_vector_return(v, out):
    for i in range(out.shape[0]):
        out[i] = returns_vector(v, i) * 2.0


def vecadd(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


def vector_scale(src, out):
    """A vector kernel: one float4 per element."""
    for i in range(out.shape[0]):
        out[i] = src[i] * 2.0


def vector_scaled(src, out):
    """A vector kernel: one float4 per element, scaled by a scalar."""
    for i in range(out.shape[0]):
        out[i] = src[i] * 2.0


def write_image(dst, src):
    """Write one image row: the other direction, which needs a surface."""
    for i in range(src.shape[0]):
        dst[0, i] = src[i]
