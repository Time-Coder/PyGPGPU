"""The imports tests, split out of `test_metal_launch_path.py`.

The subject is imports.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from pygpgpu.metal.oop.program_parser import ProgramParser


def test_a_handwritten_metal_file_imports_as_a_module(tmp_path: Any) -> None:
    """`kernel_import` accepts a `.metal` source, and its stub omits the attributes.

    Three things the shared import hook requires of every backend's parser: it takes
    a `options` keyword, it exposes `_struct_types` and `_struct_infos`, and its
    `kernel_infos` lists only host-bindable parameters. The last one matters here:
    a `[[thread_position_in_grid]]` argument is filled in by the dispatch, so a stub
    that listed it would demand a `gid` argument the caller cannot supply.
    """
    import sys

    source_dir = tmp_path / "pkg"
    source_dir.mkdir()
    (source_dir / "vecprobe.metal").write_text(
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void scale(device const float* a, device float* out,"
        " uint gid [[thread_position_in_grid]]) {\n"
        "    out[gid] = a[gid] * 2.0f;\n"
        "}\n",
        encoding="utf-8")

    sys.path.insert(0, str(source_dir))
    try:
        from pygpgpu.kernel_import import kernel_import

        kernel_import.install()
        module = __import__("vecprobe")
    finally:
        sys.path.remove(str(source_dir))

    kernel = module.scale
    assert "[metal]" in repr(kernel) or "metal" in repr(kernel), repr(kernel)
    # The parser recorded the attribute separately, so the call signature is bound-only.
    stub = (source_dir / "vecprobe.pyi").read_text(encoding="utf-8")
    # The annotation names the dtype the table resolved, which is why the stub used to
    # read `NDArray[float]` and now reads `NDArray[np.float32]`.
    assert ("def __call__(self, a: NDArray[np.float32], out: NDArray[np.float32])"
            in stub)
    assert "gid" not in stub


def test_a_metal_module_type_checks_its_arguments() -> None:
    """A `.metal` module checks dtypes now, rather than skipping the check.

    `_scalar_tables("metal")` used to be empty, and an empty table makes
    `_declaration_accepts` answer `None` ("cannot tell") for every parameter -- so a
    wrong dtype reached the launch and failed on the device instead of at the call.
    The table is now derived from OpenCL's (a vector is a layout of its components,
    and does not depend on which device reads it), filtered to MSL's own names and
    to the widths MSL has: 2, 3 and 4, never 8 or 16.
    """
    from pygpgpu.kernel_import import KernelImportWrapper, _scalar_tables

    tables = _scalar_tables("metal")
    vectors = tables["vec"]
    assert "float4" in vectors and "float3" in vectors and "float2" in vectors
    # MSL declares no 8- or 16-component vector, unlike OpenCL.
    assert "float8" not in vectors and "float16" not in vectors
    assert "int8" not in vectors

    signature = (("a", "float*"), ("out", "float*"))
    wrapper = KernelImportWrapper("scale", {"metal": object()}, {"metal": signature})
    # The right dtype binds; a wider one is refused at the call.
    assert wrapper._declaration_accepts(
        "metal", signature, (np.zeros(8, np.float32), np.zeros(8, np.float32)), {})
    assert not wrapper._declaration_accepts(
        "metal", signature, (np.zeros(8, np.float64), np.zeros(8, np.float32)), {})
    assert not wrapper._declaration_accepts(
        "metal", signature, (np.zeros(8, np.int32), np.zeros(8, np.float32)), {})


def test_a_vector_parameter_needs_an_array_of_that_vector() -> None:
    """A `float4*` parameter takes `float4` elements, not four floats."""
    from pygpgpu import float4
    from pygpgpu.kernel_import import KernelImportWrapper

    signature = (("v", "float4*"), ("out", "float4*"))
    wrapper = KernelImportWrapper("v", {"metal": object()}, {"metal": signature})
    as_vectors = np.zeros(4, dtype=float4)
    as_floats = np.zeros(4, np.float32)
    assert wrapper._declaration_accepts(
        "metal", signature, (as_vectors, as_vectors), {})
    assert not wrapper._declaration_accepts(
        "metal", signature, (as_floats, as_floats), {})


def test_a_handwritten_metal_source_needs_no_attributes(tmp_path: Any) -> None:
    """A plain `.metal` kernel binds every parameter: nothing is dispatch-supplied."""
    source_file = tmp_path / "plain.metal"
    source_file.write_text(
        '#include <metal_stdlib>\n'
        'using namespace metal;\n\n'
        'kernel void scale(device float* a, device float* out, int n) {\n'
        '    uint gid = get_global_id(0);\n'
        '    if (gid < (uint)n) { out[gid] = a[gid] * 2.0f; }\n'
        '}\n',
        encoding="utf-8")
    parser = ProgramParser()
    parser.parse(str(source_file))
    kernel = parser.kernel_infos["scale"]
    assert list(kernel.args) == ["a", "out", "n"]
    assert not any(getattr(arg, "dispatch_supplied", False)
                   for arg in kernel.args.values())
    assert kernel.args["a"].is_ptr
    assert not kernel.args["n"].is_ptr


@pytest.mark.parametrize("bad_source", [
    "kernel void broken(device float* a",  # no parameter list terminator
])
def test_an_unreadable_source_raises_rather_than_guessing(tmp_path: Any,
                                                          bad_source: str) -> None:
    """A source the scanner cannot read is refused with its own message."""
    source_file = tmp_path / "bad.metal"
    source_file.write_text(bad_source, encoding="utf-8")
    parser = ProgramParser()
    with pytest.raises(ValueError):
        parser.parse(str(source_file))
