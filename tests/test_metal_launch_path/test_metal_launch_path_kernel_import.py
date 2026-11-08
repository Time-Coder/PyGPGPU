"""The kernel_import tests, split out of `test_metal_launch_path.py`.

The subject is kernel_import.
"""
from __future__ import annotations

from _helpers.test_metal_launch_path_helpers import (
    mixed_factor_kernel,
    two_phase,
)

from typing import Any
import numpy as np
import pytest
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.metal_codegen import PROLOGUE_PARAMETERS
from pygpgpu.metal.oop.program_parser import ProgramParser


def test_a_metal_source_takes_part_in_the_same_named_import(tmp_path: Any) -> None:
    """A `.metal` file joins the other backends' sources under one kernel name.

    This is what `kernel_import` exists for: one handwritten source per backend, merged
    into a module whose bracket selects the implementation. Adding Metal to it is what
    makes a Mac able to use the same call site the other backends use -- the wrapper
    lists every implementation, and a backend with none reports the ones there are.
    """
    import sys

    from pygpgpu.kernel_import import kernel_import

    source_dir = tmp_path / "pkg"
    source_dir.mkdir()
    (source_dir / "vecops.cl").write_text(
        "__kernel void add(__global const float* a, __global float* out,\n"
        "                  const int n) {\n"
        "    int i = get_global_id(0);\n"
        "    if (i < n) { out[i] = a[i] * 2.0f; }\n}\n", encoding="utf-8")
    (source_dir / "vecops.metal").write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void add(device const float* a, device float* out, const int n,\n"
        "                uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid < (uint)n) { out[gid] = a[gid] * 2.0f; }\n}\n", encoding="utf-8")

    sys.path.insert(0, str(source_dir))
    try:
        kernel_import.install()
        module = __import__("vecops")
    finally:
        sys.path.remove(str(source_dir))

    declared = getattr(module.add, "_declarations", {})
    assert "metal" in declared, declared
    assert "opencl" in declared, declared

    # A backend with no source says which ones exist, rather than answering nothing.
    if "cuda" not in declared:
        with pytest.raises(KeyError) as missing:
            module.add["cuda"]
        message = str(missing.value)
        assert "no cuda implementation" in message
        assert "metal" in message


def test_a_metal_only_module_imports_without_a_device(tmp_path: Any) -> None:
    """A `.metal` source parses on a machine with no Metal, so the module still imports.

    `kernel_import` reads signatures without a device -- that is what lets a Windows
    machine inspect a Mac-targeted source. Only the compile step needs the framework.
    """
    import sys

    from pygpgpu.kernel_import import kernel_import

    source_dir = tmp_path / "pkg"
    source_dir.mkdir()
    (source_dir / "solo.metal").write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void scale(device const float* a, device float* out,\n"
        "                  uint gid [[thread_position_in_grid]]) {\n"
        "    out[gid] = a[gid] * 3.0f;\n}\n", encoding="utf-8")

    sys.path.insert(0, str(source_dir))
    try:
        kernel_import.install()
        module = __import__("solo")
    finally:
        sys.path.remove(str(source_dir))

    assert list(getattr(module.scale, "_declarations", {})) == ["metal"]


def test_a_two_phase_body_generates_two_kernels_the_scanner_reads(tmp_path: Any,
                                                                  monkeypatch) -> None:
    """Every phase kernel is declared in one MSL file, and the scanner finds both.

    A body with two magic loops is two passes (see the phase rule). Each pass becomes
    its own kernel, `<name>_p0` and `<name>_p1`, and the launch has to find both by
    name -- so the declaration scanner must read a file that holds several entry
    points, not just the last one.
    """
    import pygpgpu.metal.oop.program as program_module
    from pygpgpu.kernel.decorator import PyKernel

    monkeypatch.setattr(PyKernel, "_cache_dir",
                        lambda self, backend: str(tmp_path / backend))
    from pygpgpu.kernel import source_paths

    monkeypatch.setattr(source_paths, "cache_root",
                        lambda source_file, backend: str(tmp_path / backend))

    captured: dict = {}

    class StubProgram:
        def __init__(self, file_name, **_options):
            captured["path"] = file_name

        @property
        def kernel_infos(self):
            return {}

        @property
        def kernels(self):
            return {}

    kernel_object = __import__("pygpgpu").kernel(two_phase)
    original = program_module.Program
    program_module.Program = StubProgram
    try:
        args = [np.zeros(8, np.float32) for _ in range(3)]
        try:
            kernel_object._compile(args, [infer_ctype(a) for a in args], [], [], None,
                                   "sig", "metal", object())
        except KeyError:
            pass
    finally:
        program_module.Program = original

    parser = ProgramParser()
    parser.parse(captured["path"], options=None)
    names = list(parser.kernel_infos)
    assert len(names) == 2, names
    assert names[0].endswith("_p0") and names[1].endswith("_p1")
    # Both declare the same host parameters (the launch aligns by position) and the
    # same six attribute parameters.
    for name in names:
        info = parser.kernel_infos[name]
        assert info.args
        assert len(info.dispatch_parameters) == len(PROLOGUE_PARAMETERS)


def test_two_scalar_types_specialize_two_kernels(tmp_path: Any, monkeypatch) -> None:
    """A scalar's type is part of the specialisation, so the C names differ.

    `@kernel` caches one launch per backend + device + argument-type signature. The
    scalar's *declared* type is part of that signature: `int factor` and `float factor`
    are two kernels, with two C names and two parameter types in one file. Sharing a
    name would mean one of them silently ran the other's code.
    """
    from pygpgpu.kernel.decorator import PyKernel
    from pygpgpu.kernel import source_paths
    from pygpgpu.metal.oop.program_parser import ProgramParser

    monkeypatch.setattr(PyKernel, "_cache_dir",
                        lambda self, backend: str(tmp_path / backend))
    monkeypatch.setattr(source_paths, "cache_root",
                        lambda source_file, backend: str(tmp_path / backend))

    import pygpgpu.metal.oop.program as program_module

    written: dict = {}

    class StubProgram:
        def __init__(self, file_name, **_options):
            written["path"] = file_name

        @property
        def kernel_infos(self):
            return {}

        @property
        def kernels(self):
            return {}

    array = np.zeros(8, np.float32)
    pointer = infer_ctype(array)
    kernel_object = __import__("pygpgpu").kernel(mixed_factor_kernel)

    original = program_module.Program
    program_module.Program = StubProgram
    try:
        for scalar_type in ("int", "float"):
            try:
                kernel_object._compile([array, array, 2], [pointer, pointer, scalar_type],
                                       [], [], None, f"sig-{scalar_type}", "metal", object())
            except KeyError:
                pass
    finally:
        program_module.Program = original

    parser = ProgramParser()
    parser.parse(written["path"], options=None)
    declarations = list(parser.kernel_infos)
    assert len(declarations) == 2, declarations
    assert declarations[0] != declarations[1]
    # And the parameter really differs, not just the name.
    types = sorted(info.args["factor"].type_str
                   for info in parser.kernel_infos.values())
    assert types == ["float", "int"], types
