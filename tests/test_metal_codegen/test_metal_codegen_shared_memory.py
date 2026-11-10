"""The shared_memory tests, split out of `test_metal_codegen.py`.

The subject is shared_memory.
"""
from __future__ import annotations


from typing import Any
import numpy as np
from pygpgpu.kernel.decorator import infer_ctype


def test_shared_memory_is_a_threadgroup_parameter(tmp_path: Any, monkeypatch) -> None:
    """`local_array_per_group` becomes a `threadgroup` buffer parameter, not `__local`.

    The kernel and the compile flow are the real ones (`PyKernel._compile`, whose
    resolver recognizes the builtins); only the Metal `Program` is stubbed, because
    compiling needs Metal.framework. What is asserted is the *source text* the
    decorator writes.
    """
    import sys

    sys.path.insert(0, "tests")
    try:
        import test_kernel_local_memory as shared_kernels
    finally:
        sys.path.pop(0)

    from pygpgpu.kernel.decorator import PyKernel

    monkeypatch.setattr(PyKernel, "_cache_dir",
                        lambda self, backend: str(tmp_path / backend))
    from pygpgpu.kernel import source_paths

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

    import pygpgpu.metal.oop.program as program_module

    original = program_module.Program
    program_module.Program = StubProgram
    try:
        kernel_object = shared_kernels.shared_histogram
        args = [np.zeros(256, np.uint32) for _ in range(2)]
        kernel_object._compile(args, [infer_ctype(a) for a in args], [], [], None,
                               "sig", "metal", object())
    except KeyError:
        pass                      # the stub reports no kernels to look up
    finally:
        program_module.Program = original

    source: str = captured["source"]
    assert "threadgroup uint*" in source
    assert "__local" not in source
    # The barrier and the atomics are Metal's own spellings too.
    assert "threadgroup_barrier" in source
    assert "atomic_fetch_add_explicit" in source
    assert "atomicAdd" not in source
    # Both address spaces appear: the threadgroup scratch buffer and the device
    # histogram are different atomics.
    assert "threadgroup atomic_uint" in source
    assert "device atomic_uint" in source


def test_a_block_local_array_is_a_threadgroup_parameter_too(tmp_path: Any,
                                                            monkeypatch) -> None:
    """`local_array`/`shared_array` declare the same buffer, indexed directly.

    The device declaration is the same one the per-group pair builds -- the difference is
    the host model, not the emitted C -- so what this pins is that the selection of the
    declaration does not go through `buf[blockIdx]` (there is no group subscript to fold)
    and that the element access is still emitted against the declared array.
    """
    import sys

    sys.path.insert(0, "tests")
    try:
        import test_kernel_block_memory as block_kernels
    finally:
        sys.path.pop(0)

    from pygpgpu.kernel.decorator import PyKernel

    monkeypatch.setattr(PyKernel, "_cache_dir",
                        lambda self, backend: str(tmp_path / backend))
    from pygpgpu.kernel import source_paths

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

    import pygpgpu.metal.oop.program as program_module

    original = program_module.Program
    program_module.Program = StubProgram
    try:
        kernel_object = block_kernels.block_scratch
        args = [np.zeros(8, np.int32) for _ in range(2)]
        kernel_object._compile(args, [infer_ctype(a) for a in args], [], [], None,
                               "sig", "metal", object())
    except KeyError:
        pass                      # the stub reports no kernels to look up
    finally:
        program_module.Program = original

    source: str = captured["source"]
    assert "threadgroup int*" in source
    assert "__local" not in source
    assert "threadgroup_barrier" in source
    # The element access names the declared array directly, with no group subscript left
    # to fold: `_local0[lane]`, never `_local0[blockIdx][lane]` or an offset from one.
    assert "_local0[lane]" in source
    assert "blockIdx" not in source
