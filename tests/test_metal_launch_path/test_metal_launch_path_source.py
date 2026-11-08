"""The source tests, split out of `test_metal_launch_path.py`.

The subject is source.
"""
from __future__ import annotations

from _helpers.test_metal_launch_path_helpers import (
    _NoDeviceProgram,
    _translate,
    vecadd,
)

from typing import Any
import pytest
from pygpgpu.kernel.metal_codegen import PROLOGUE_PARAMETERS, prelude
from pygpgpu.kernel.source_paths import CACHE_FOLDERS, EXTENSIONS
from pygpgpu.metal.oop.program_parser import ProgramParser


def test_metal_has_its_own_cache_folder_and_extension() -> None:
    """The cache layout names Metal like every other backend, so nothing is shared."""
    assert EXTENSIONS["metal"] == "metal"
    assert CACHE_FOLDERS["metal"] == "__metalcache__"
    # No other backend may claim them.
    assert list(EXTENSIONS.values()).count("metal") == 1
    assert list(CACHE_FOLDERS.values()).count("__metalcache__") == 1


def test_generated_source_carries_the_prelude_and_the_kernel() -> None:
    """The written file is MSL: the prelude first, then the kernel definition."""
    translator = _translate(vecadd, 3)
    rendered = "\n\n".join(source for _, source in translator.render_kernels("k"))
    full = prelude() + "\n" + rendered
    assert full.startswith("#include <metal_stdlib>")
    assert "#define get_global_id(dim)" in full
    assert "kernel void k(" in full
    # The macros must be defined before the body that calls them.
    assert full.index("#define get_global_id(dim)") < full.index("kernel void k(")


def test_parser_reads_back_the_generated_signature(tmp_path: Any) -> None:
    """What the parser reads is exactly what the translator declared.

    This is the contract a launch depends on: the parser's argument order *is* the
    binding order, so a mismatch would bind arrays to the wrong parameters.
    """
    translator = _translate(vecadd, 3)
    body = "\n\n".join(source for _, source in translator.render_kernels("k_metal"))
    source_file = tmp_path / "generated.metal"
    source_file.write_text(prelude() + "\n" + body, encoding="utf-8")

    parser = ProgramParser()
    parser.parse(str(source_file))
    assert list(parser.kernel_infos) == ["k_metal"]
    info = parser.kernel_infos["k_metal"]

    # Every host-bound parameter is present, in the order they were declared.
    host_bound = [name for name, arg in info.args.items()
                  if not getattr(arg, "dispatch_supplied", False)]
    assert host_bound[:3] == ["a", "b", "out"]
    assert info.args["a"].is_ptr
    assert info.args["a"].base_type_str == "float"

    # The dispatch parameters are NOT part of `args`: that map is the host-visible
    # signature (`kernel_import` and the stub generator read it), and a
    # dispatch-supplied value is never bound by the host. They are recorded beside it.
    assert info.dispatch_parameters == [name for _t, name, _a in PROLOGUE_PARAMETERS]
    assert not any(getattr(arg, "dispatch_supplied", False)
                   for arg in info.args.values())
    # The host-bound tail is the extent the body indexes, and it stays bindable.
    assert "_out_shape0" in info.args


def test_parser_numbers_every_dispatch_parameter(tmp_path: Any) -> None:
    """The parser sees all six attributes, so the launch skips all six slots."""
    translator = _translate(vecadd, 3)
    body = "\n\n".join(source for _, source in translator.render_kernels("k_metal"))
    source_file = tmp_path / "generated.metal"
    source_file.write_text(prelude() + "\n" + body, encoding="utf-8")

    info = ProgramParser()
    info.parse(str(source_file))
    kernel = info.kernel_infos["k_metal"]
    attributes = [name for _t, name, _a in PROLOGUE_PARAMETERS]
    assert kernel.dispatch_parameters == attributes
    # And they are absent from the host's call signature.
    assert "a" in kernel.args and "out" in kernel.args
    # `_out_shape0` is a host-bound extent (the translator appends it after the
    # arrays), so it stays in the signature; only the attribute parameters are out.
    assert list(kernel.args) == ["a", "b", "out", "_out_shape0"]


@pytest.mark.parametrize("body,host,dispatch", [
    # A template argument's comma must not split the parameter list.
    ("kernel void t(array<float, 4> v, device float* out) {}", ["v", "out"], []),
    # `[[buffer(0)]]` is a host-bound slot, not a dispatch-supplied value.
    ("kernel void r(device float* a [[buffer(0)]], device float* out [[buffer(1)]]) {}",
     ["a", "out"], []),
    # A multi-line list, with the attribute parameters last.
    ("kernel void m(\n    device const float* a,\n    device float* out,\n"
     "    uint gid [[thread_position_in_grid]],\n"
     "    uint tptg [[threads_per_threadgroup]]\n) {}",
     ["a", "out"], ["gid", "tptg"]),
])
def test_the_scanner_handles_msl_declaration_shapes(tmp_path: Any,
                                                    body: str, host: list,
                                                    dispatch: list) -> None:
    """Templates, buffer attributes and multi-line lists all parse correctly.

    The parameter splitter tracks `<>` as well as `()`, because `array<float, 4>`
    carries a comma of its own; and only a *dispatch* attribute removes a parameter
    from the host signature -- `[[buffer(0)]]` names the host's own slot and must
    stay bindable.
    """
    source_file = tmp_path / "shapes.metal"
    source_file.write_text("#include <metal_stdlib>\nusing namespace metal;\n"
                           + body + "\n", encoding="utf-8")
    parser = ProgramParser()
    parser.parse(str(source_file), options=None)
    kernel = next(iter(parser.kernel_infos.values()))
    assert list(kernel.args) == host
    assert kernel.dispatch_parameters == dispatch


def test_a_dispatch_parameter_cannot_appear_in_the_bindable_arguments() -> None:
    """The attribute parameters must be absent from `args`, or every index shifts.

    A host-bound parameter's position in `args` is its buffer index only because the
    device signature declares the attribute parameters after them. A parser change
    that let one slip into `args` would bind buffers one slot off -- silently, on the
    device -- so the launch refuses the state instead.
    """
    from pygpgpu.metal.oop.kernel import MetalKernel
    from pygpgpu.metal.oop.program_info import ArgInfo

    kernel = MetalKernel(_NoDeviceProgram(), "bad")
    kernel.args = {"a": ArgInfo("a", "float*", (8,)),
                   "_mtl_thread_position": ArgInfo("_mtl_thread_position", "uint3", ())}
    kernel.dispatch_parameters = ["_mtl_thread_position"]

    with pytest.raises(RuntimeError) as refused:
        kernel._bind_args((object(), object()), {})
    message = str(refused.value)
    assert "_mtl_thread_position" in message
    assert "must not be bound" in message
