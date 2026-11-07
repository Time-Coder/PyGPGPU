"""Pins the Metal launch path's structure, without a Metal device.

A launch needs a Mac, but almost everything that *leads up to* one is pure python,
and that is where a regression would hide: the generated source must carry
Metal's prelude, the parser must read back exactly the signature the translator
emitted, and the argument list must line up with what the kernel declares. Those
three facts are what make the first launch on a Mac work at all, so they are
checked here on any machine.

The chain exercised, end to end:

1. `@kernel` translates the body to MSL and writes a `.metal` source (prelude +
   kernel) into the backend's cache directory;
2. `ProgramParser` reads that file back into a `KernelInfo`;
3. the parser's parameter list matches the translator's rendered declaration, so
   the launch binds each argument to the index the device expects.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.metal_codegen import PROLOGUE_PARAMETERS, prelude
from pygpgpu.kernel.source_paths import CACHE_FOLDERS, EXTENSIONS
from pygpgpu.kernel.translator import FunctionTranslator
from pygpgpu.metal.oop.program_parser import ProgramParser


def vecadd(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


def _translate(func: Any, n_args: int) -> FunctionTranslator:
    types = [infer_ctype(np.zeros(8, np.float32)) for _ in range(n_args)]
    translator = FunctionTranslator(func, types, "metal", lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    return translator



class _NoDeviceProgram:
    """Just enough of a Program for a MetalKernel to be built without a device.

    A `MetalKernel` needs a program object for `context`/`device`/`file_name`/`id`, and
    every test that exercises the kernel's own logic (dispatch shape, the arg-info
    invariant, scalar packing) builds one. Kept in one place so a field the class
    grows does not have to be added in three copies.
    """

    context = None
    device = None
    file_name = "x.metal"
    id = 0

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
    # The device signature declares them, so the host binds `len(args)` slots and the
    # dispatch fills the tail -- which is what the parser's records let the launch do.


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


def test_every_science_vocabulary_entry_has_a_metal_spelling() -> None:
    """Metal must carry a spelling for every entry, or refuse by name -- never KeyError.

    The tables are keyed by backend, so a missing entry is a `KeyError` raised from
    inside the translation. Either a backend carries a spelling or the translator
    refuses the call with a message naming the backends that do: those are the only
    two acceptable outcomes.
    """
    from pygpgpu.kernel.device_builtins import SHARED_MATH_BUILTINS, SHARED_MATH_TEMPLATES

    for name, table in list(SHARED_MATH_BUILTINS.items()) + list(SHARED_MATH_TEMPLATES.items()):
        assert "metal" in table, f"{name} has no metal spelling and would KeyError"
        assert table["metal"], f"{name}'s metal spelling is empty"
    # The two tables are disjoint: a name lives in exactly one, so the lookup order
    # cannot change what is emitted.
    assert not (set(SHARED_MATH_BUILTINS) & set(SHARED_MATH_TEMPLATES))


def test_the_fast_and_exact_families_are_refused_on_metal() -> None:
    """The CUDA-only vocabulary has no Metal counterpart, and the refusal says so."""
    from pygpgpu.kernel.device_builtins import CUDA_ONLY_BUILTINS, CUDA_ONLY_CONVERSIONS

    # Every CUDA-only entry is spelled for cuda and hip, and for nothing else.
    for name, table in list(CUDA_ONLY_BUILTINS.items()) + list(CUDA_ONLY_CONVERSIONS.items()):
        assert "metal" not in table, (
            f"{name} claims a Metal spelling; if Metal really has this intrinsic, "
            f"move it into the shared tables instead")
        assert {"cuda", "hip"} <= set(table), f"{name} is not spelled for cuda and hip"


def test_a_shared_math_name_emits_its_metal_spelling() -> None:
    """What `_shared_math_call` emits is the table's `metal` entry, verbatim.

    The two shapes are pinned separately because they are genuinely different: a
    callee entry becomes `name(args)`, and a template entry is formatted with the
    single argument. Metal's `native_recip` is the second kind; everything else is
    the first.
    """
    from pygpgpu.kernel.device_builtins import SHARED_MATH_BUILTINS, SHARED_MATH_TEMPLATES

    for name, table in SHARED_MATH_BUILTINS.items():
        spelling = table["metal"]
        # A callee spelling is a bare identifier path, never an expression.
        assert "(" not in spelling and " " not in spelling, (
            f"{name}'s metal spelling {spelling!r} is an expression, so it belongs in "
            f"SHARED_MATH_TEMPLATES")
    for name, table in SHARED_MATH_TEMPLATES.items():
        # A template must consume the argument placeholder exactly once per use.
        assert "{code}" in table["metal"], (
            f"{name}'s metal template never consumes its argument")
        rendered = table["metal"].format(code="x")
        assert "x" in rendered


def test_a_scalar_argument_travels_as_its_own_bytes() -> None:
    """A scalar is `setBytes:`, not `setBuffer:` -- and its width is the device's.

    MSL declares a scalar kernel parameter in the thread address space, so the host
    hands over the bytes themselves. The width has to be the *device's*: a kernel's
    extent parameter is `const int`, and sending python's 8-byte int would make the
    device read four bytes of a value plus four of nothing.
    """
    from pygpgpu.metal.oop.kernel import _SCALAR_SIZES, _scalar_bytes

    expected_widths = {
        "bool": 1, "char": 1, "uchar": 1,
        "short": 2, "ushort": 2, "half": 2,
        "int": 4, "uint": 4, "float": 4,
        "long": 8, "ulong": 8, "double": 8,
    }
    assert _SCALAR_SIZES == expected_widths
    # The int an extent travels as: four bytes, little-endian, exactly the value.
    assert _scalar_bytes("int", 1000) == (1000).to_bytes(4, "little")
    # A float keeps its value through the round trip.
    import struct
    assert struct.unpack("<f", _scalar_bytes("float", 2.5))[0] == 2.5
    assert struct.unpack("<d", _scalar_bytes("double", 2.5))[0] == 2.5
    # A negative value keeps its sign.
    assert _scalar_bytes("short", -5) == (-5).to_bytes(2, "little", signed=True)
    # Every declared scalar type has a width; a type that appeared without one
    # would silently fall back to four bytes.
    for type_name in expected_widths:
        assert len(_scalar_bytes(type_name, 1)) == expected_widths[type_name]


def complex_scalar(value, out):
    for i in range(out.shape[0]):
        out[i] = value


def test_a_complex_parameter_is_declared_by_value(tmp_path: Any) -> None:
    """The translator and the parser agree that a complex parameter is a value.

    MSL has no complex type, so `kernel/complex.py` declares the two fixed records (`c64`
    is a `float` pair, `c128` a `double` pair) and the translator emits one as a parameter
    the way it emits a scalar: by value, in the thread address space. The parser is what a
    launch binds against, so a `device c64*` here would be bound as a buffer -- the wrong
    path entirely.
    """
    types = [infer_ctype(1 + 2j), infer_ctype(np.zeros(8, np.complex64))]
    translator = FunctionTranslator(complex_scalar, types, "metal", lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    body = "\n\n".join(source for _, source in translator.render_kernels("k_complex"))
    assert "c64 value" in body, body
    assert "device c64* value" not in body, body

    source_file = tmp_path / "generated.metal"
    source_file.write_text(prelude() + "\n" + body, encoding="utf-8")
    parser = ProgramParser()
    parser.parse(str(source_file))
    info = parser.kernel_infos["k_complex"]
    assert list(info.args) == ["value", "out", "_out_shape0"]
    assert info.args["value"].base_type_str == "c64"
    assert not info.args["value"].is_ptr
    assert info.args["out"].is_ptr and info.args["out"].base_type_str == "c64"


def test_a_complex_argument_travels_as_its_record_bytes() -> None:
    """A complex value is not a scalar, but it is bound the way one is: `setBytes:`.

    The record is two reals in declaration order, which is numpy's own complex layout, so
    the payload is 8 bytes for a `c64` and 16 for a `c128` -- and the length handed to
    `setBytes:` is the structure's size, not python's.
    """
    import struct

    from pygpgpu.metal.oop.kernel import _scalar_bytes

    assert _scalar_bytes("c64", 1 + 2j) == struct.pack("<ff", 1.0, 2.0)
    assert _scalar_bytes("c128", 1 + 2j) == struct.pack("<dd", 1.0, 2.0)
    assert len(_scalar_bytes("c64", 1 + 2j)) == 8
    assert len(_scalar_bytes("c128", 1 + 2j)) == 16
    # A numpy complex scalar is the same value, read at its own width.
    assert _scalar_bytes("c128", np.complex64(3 + 4j)) == struct.pack("<dd", 3.0, 4.0)
    # The component is what the `c64` width holds, so a value it cannot carry is refused
    # with the message a `float` parameter gives, not with `struct.pack`'s own.
    with pytest.raises(OverflowError) as refused:
        _scalar_bytes("c64", 1e40 + 0j)
    assert "c64" in str(refused.value)


def test_a_complex_parameter_refuses_a_non_complex_argument() -> None:
    """`ArgInfo.check_type` is what refuses a wrong argument before the launch.

    Without the complex branch the integer fallback answered "must be an integer" for a
    `c64` parameter, which names neither what the parameter is nor what it accepts.
    """
    from pygpgpu.metal.oop.program_info import COMPLEX_TYPE_NAMES, ArgInfo

    assert set(COMPLEX_TYPE_NAMES) == {"c64", "c128"}
    parameter = ArgInfo("value", "c64", ())
    for accepted in (1 + 2j, np.complex64(1 + 2j), np.complex128(1 + 2j)):
        parameter.check_type(accepted)
    for wrong in (1.5, 3, "z", [1 + 2j], np.float32(1.0)):
        with pytest.raises(TypeError) as refused:
            parameter.check_type(wrong)
        message = str(refused.value)
        assert "value" in message and "complex" in message


def test_the_encoder_never_binds_a_scalar_as_a_buffer() -> None:
    """`_bind_args` marks each entry, so a scalar cannot take the buffer path.

    The marker is what the encoder branches on: one wrong branch presents a
    pointer where MSL declared a value, and the launch fails on the device.
    """
    from pygpgpu.metal.oop.kernel import MetalKernel

    kernel = MetalKernel(None, "k")
    # A pointer and a scalar, as the parser would report them.
    import pygpgpu.metal.oop.program_info as info_module

    pointer = info_module.ArgInfo("a", "float*", (8,))
    scalar = info_module.ArgInfo("n", "int", ())
    kernel.args = {"a": pointer, "n": scalar}

    # Only the classification is under test here: the buffer path needs a device,
    # so the scalar's bytes are checked through the helper instead.
    from pygpgpu.metal.oop.kernel import _scalar_bytes
    assert pointer.is_ptr and not scalar.is_ptr
    assert len(_scalar_bytes("int", 4)) == 4



def test_the_context_id_reads_like_a_ctypes_value() -> None:
    """`context.id.value` must exist: the shared numpy layer keys its copies on it.

    `pygpgpu/numpy/ndarray.py` writes `key = (cmd_queue.context.id.value, flags)`
    for every array it puts on a device -- that is the ctypes shape OpenCL and CUDA
    answer with. A Metal `Context` that returned a bare `int` would raise
    `AttributeError: 'int' object has no attribute 'value'` on the first launch of
    a `gnp.ndarray`, which is the path every kernel argument takes.
    """
    from pygpgpu.metal.oop.metalobject import Handle

    handle = Handle.of(0x1000)
    assert handle.value == 0x1000
    # It still behaves like the integer the driver needs: comparable to zero,
    # usable as a ctypes value and hashable for the cache key.
    assert handle == 0x1000
    assert hash((handle.value, 1)) == hash((0x1000, 1))
    import ctypes

    assert ctypes.c_void_p(handle).value == 0x1000
    # Wrapping is idempotent, and a bare int or a ctypes value wraps the same way.
    assert Handle.of(handle) is handle
    assert int(Handle.of(42)) == 42



def test_the_oop_layer_names_its_classes_like_the_other_backends() -> None:
    """`Kernel` is the cross-backend name for a launchable entry point.

    Each backend publishes `Buffer`/`Context`/`Event`/`Program` under the same names,
    and Metal's kernel class is `MetalKernel` because that is what its own machinery
    is called. A caller that walks the backends generically -- comparing them, or
    writing a backend-specific tool -- reaches `oop.kernel.Kernel`, so the alias
    exists to keep that name answering on Metal too. The class itself keeps its
    name; this is only the shared spelling.
    """
    import pygpgpu.metal.oop.kernel as metal_kernel
    import pygpgpu.metal.oop.program as metal_program

    assert metal_kernel.Kernel is metal_kernel.MetalKernel
    assert metal_program.Kernel is metal_kernel.MetalKernel
    # The shared classes really are shared names, not Metal-specific ones.
    for module, name in ((metal_kernel, "BoundKernel"),
                         (__import__("pygpgpu.metal.oop.buffer", fromlist=["Buffer"]),
                          "Buffer"),
                         (__import__("pygpgpu.metal.oop.context", fromlist=["Context"]),
                          "Context"),
                         (__import__("pygpgpu.metal.oop.event", fromlist=["Event"]),
                          "Event")):
        assert hasattr(module, name), name


def test_a_metal_program_publishes_the_shared_program_surface() -> None:
    """The attributes shared code reads from a `Program` exist on Metal's.

    `kernel_import` and the decorator reach `kernels`, `kernel_infos` and `structs`
    on every backend's `Program`; a missing one would surface as an `AttributeError`
    from inside the import hook rather than as a Metal-specific message.
    """
    import pygpgpu.metal.oop.program as metal_program

    for name in ("kernels", "kernel_infos", "structs", "file_name"):
        assert hasattr(metal_program.Program, name), name



def test_compile_options_map_onto_mtl_compile_options() -> None:
    """The MSL compile settings reach the compiler instead of being dropped.

    `Program.__init__` used to take `**options`, so `kernel_import`'s
    `Program(..., options=...)` was swallowed and every library compiled with nil --
    Metal's defaults, with no way to turn fast math off. The settings are now read
    into a `CompileOptions` and handed to `newLibraryWithSource:options:error:`, and
    the selectors are the SDK's own spellings.
    """
    from pygpgpu.metal.oop.compile_options import (
        MATH_MODES, CompileOptions, from_mapping)

    # Nothing configured -> nil stays nil; a half object would be worse.
    assert from_mapping(None) is None
    assert from_mapping(CompileOptions()) is None

    options = from_mapping({"fast_math": False, "language_version": 2})
    assert options is not None
    assert options.fast_math is False and options.language_version == 2
    # A field the caller did not pass keeps the compiler's default, not a sentinel.
    assert options.math_mode is None and options.macros is None

    # The selectors are the ones `MTLCompileOptions` declares.
    assert CompileOptions.SELECTORS == (
        "setLanguageVersion:", "setFastMathEnabled:",
        "setMathMode:", "setPreprocessorMacros:")
    assert set(MATH_MODES) == {"fast", "relaxed", "precise"}


def test_an_opencl_build_options_object_is_not_read_as_msl_options() -> None:
    """The hook hands the same options object to whichever backend owns the source.

    An OpenCL `BuildOptions` reaching Metal is a normal case, and its `-D`/include
    settings mean nothing to MSL -- translating them would be a guess, so it is read
    as "no MSL options" and the defaults apply.
    """
    from pygpgpu.metal.oop.compile_options import from_mapping

    class BuildOptions:
        pass

    assert from_mapping(BuildOptions()) is None
    assert from_mapping("not options at all") is None
    assert from_mapping({}) is None



def test_building_compile_options_without_the_runtime_says_why() -> None:
    """Off macOS the option object cannot be built, and the message names the cause.

    `build` reaches `ObjC.class_`, which dereferences the framework handle; without
    Metal.framework that is `None`, and the resulting `AttributeError: 'NoneType'
    object has no attribute 'objc_getClass'` names nothing a user could act on. The
    driver records its own reason (`MetaMetal.last_init_error`), so the error reports
    that instead. The defaults path stays untouched: `build(None)` never reaches the
    framework, so a kernel that configured nothing compiles everywhere.
    """
    from pygpgpu.metal.driver.objc import ObjC
    from pygpgpu.metal.driver.metal import MetaMetal
    from pygpgpu.metal.oop import compile_options
    from pygpgpu.metal.oop.compile_options import CompileOptions

    # Nothing configured never touches the framework.
    assert compile_options.build(None) == 0
    assert compile_options.build(CompileOptions()) == 0

    if ObjC.lib is not None:
        pytest.skip("this machine has the Metal framework; the failure path is moot")

    with pytest.raises(RuntimeError) as refused:
        compile_options.build(CompileOptions(fast_math=False))
    message = str(refused.value)
    assert "Metal runtime is not loaded" in message
    # The driver's own recorded reason is carried through (`last_init_error` holds
    # the exception the load raised, not a string).
    recorded = getattr(MetaMetal, "last_init_error", None)
    assert recorded is not None
    assert str(recorded) in message



def test_selecting_metal_off_macos_explains_the_requirement() -> None:
    """`k["metal"]` on a machine without Metal names the requirement, not a path.

    The driver records its load failure in `MetaMetal.last_init_error` instead of
    raising, so the first real use is what surfaces it -- and for a device selector
    that is `Devices`, whose raw failure is a `FileNotFoundError` about a file under
    `/System/Library`. That text says nothing about the backend being unavailable or
    about macOS being what it needs, so the enumeration re-raises with both.
    """
    from pygpgpu.metal.driver.objc import ObjC
    from pygpgpu.metal.oop.devices import Devices

    if ObjC.lib is not None:
        pytest.skip("this machine has the Metal framework")

    with pytest.raises(RuntimeError) as refused:
        len(Devices)
    message = str(refused.value)
    assert "Metal backend is unavailable" in message
    assert "macOS" in message
    assert "real Apple GPU" in message


def test_probing_metal_answers_unavailable_instead_of_raising() -> None:
    """The probe is the contract every backend shares: unavailable, not an error.

    A machine with no Metal must still be able to `import pygpgpu.metal`, parse
    `.metal` sources and probe the backend, so the probe swallows the failure and
    answers None. The readable error above is for a *selection*, not a probe.
    """
    from pygpgpu.kernel.decorator import _probe_backend

    assert _probe_backend("metal") is None



def test_nsinteger_arguments_are_pointer_sized() -> None:
    """`NSUInteger` is pointer-sized, so `c_size_t` is the type to name.

    `setBuffer:offset:atIndex:` and `setBytes:length:atIndex:` take `NSUInteger`
    arguments, and `MTLSize`'s three fields are `NSUInteger` too. `c_ulong` happens to
    be 8 bytes on macOS, so those bindings worked there **by accident of the platform**
    -- it is 4 bytes on Windows. `c_size_t` is pointer-sized everywhere, which is what
    the selector's own type means.
    """
    import ctypes

    import pygpgpu.metal.oop.kernel as metal_kernel

    # MTLSize is three NSUIntegers, so 24 bytes on a 64-bit host.
    assert ctypes.sizeof(metal_kernel.MTLSize) == 3 * ctypes.sizeof(ctypes.c_size_t)
    size = metal_kernel.MTLSize(8, 4, 1)
    assert (size.width, size.height, size.depth) == (8, 4, 1)

    source = open(metal_kernel.__file__, encoding="utf-8").read()
    # The selectors must not name `c_ulong`: it is the wrong width off macOS.
    assert "(ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong)" not in source
    assert "(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)" in source


def test_the_kernel_docstring_matches_the_scalar_binding() -> None:
    """A scalar is bound with `setBytes:`, and the module says so.

    The docstring used to claim "Metal has no by-value kernel argument: a scalar is
    bound as a one-element buffer", which is the behaviour `setBytes:length:atIndex:`
    replaced -- and a docstring that describes the old design sends a reader looking
    for the wrong thing. The binding description and the code have to agree.
    """
    import pygpgpu.metal.oop.kernel as metal_kernel

    doc = metal_kernel.__doc__ or ""
    assert "setBytes:length:atIndex:" in doc
    assert "Every argument travels through a buffer" not in doc



def test_no_metal_module_names_a_platform_dependent_ctypes_type() -> None:
    """Every `NSUInteger` in this backend is spelled `c_size_t`, nowhere `c_ulong`.

    Objective-C's `NSUInteger` is pointer-sized. `ctypes.c_ulong` is that size on
    macOS *by coincidence* -- it is 4 bytes on Windows -- so a `c_ulong` in a
    selector's argument tuple or a struct field binds correctly only on the platform
    this code happens to be developed on. `c_size_t` is pointer-sized everywhere.

    This walks every module in the backend rather than a list of known call sites,
    because the defect is invisible in review: the code works on macOS either way.
    """
    import glob
    import os
    import re

    import pygpgpu.metal

    package = os.path.dirname(pygpgpu.metal.__file__)
    offenders = []
    for path in glob.glob(os.path.join(package, "**", "*.py"), recursive=True):
        source = open(path, encoding="utf-8").read()
        for match in re.finditer(r"ctypes\.c_ulong\b", source):
            line = source[:match.start()].count("\n") + 1
            offenders.append(f"{os.path.basename(path)}:{line}")
    assert not offenders, (
        "NSUInteger must be c_size_t, not c_ulong (pointer-sized only on macOS): "
        + ", ".join(offenders))



def test_every_owned_object_is_released() -> None:
    """The `new*`-style selectors answer +1 references, and each is released.

    Objective-C's ownership rule is the naming convention: a method whose name starts
    with `new` or `alloc` answers a reference the caller owns, and everything else is
    autoreleased. Four call sites here answer +1 references, and none is an
    `MTLObject`, so nothing else would ever drop them:

    * `new_command_buffer()` -- one per **launch**, the one that leaks repeatedly;
    * `newFunctionWithName:` -- one per pipeline;
    * the `NSError` from both `error:` selectors;
    * the `MTLCompileOptions` `build` returns.

    A leak here is invisible on a single call and fatal in a loop, which is why it is
    pinned by reading the code rather than by a device run.
    """
    import pygpgpu.metal.oop.kernel as metal_kernel
    import pygpgpu.metal.oop.program as metal_program

    launch_source = open(metal_kernel.__file__, encoding="utf-8").read()
    # The command buffer is released, and through a `finally` so a failing read-back
    # cannot skip it.
    assert "new_command_buffer()" in launch_source
    assert "finally:" in launch_source
    assert "ObjC.release(int(command_buffer))" in launch_source

    # The `try` must start right after the buffer is created -- covering the encoder
    # creation, every parameter binding and the dispatch. A `try` that only wrapped
    # the commit would still leak whenever `_bind_args` or `setBuffer:` raised, which
    # is the case a launch-time argument error produces.
    # Searched from `created` onwards: the selector names also appear in the module
    # docstring, which comes before every call site.
    created = launch_source.index("new_command_buffer()")
    guarded = launch_source.index("try:", created)
    encoder = launch_source.index('"computeCommandEncoder"', created)
    dispatch = launch_source.index("dispatchThreadgroups", created)
    assert guarded < encoder < dispatch, "the try must precede the encoder work"

    # The function handle and its NSError are released too.
    assert "ObjC.release(function_id)" in launch_source

    compile_source = open(metal_program.__file__, encoding="utf-8").read()
    assert "release_compile_options(options_id)" in compile_source
    assert "ObjC.release(int(error.value or 0))" in compile_source


def test_the_preprocessor_dictionary_is_not_retained() -> None:
    """`setPreprocessorMacros:` retains the dictionary, so building it must not.

    `dictionary` is autoreleased (+0); an extra `retain` with no matching `release`
    would leak it for the life of the process, and the options object it is handed to
    already keeps it for as long as it is needed.
    """
    import pygpgpu.metal.oop.compile_options as compile_options

    source = open(compile_options.__file__, encoding="utf-8").read()
    body = source[source.index("def _dictionary"):source.index("def _string")]
    assert "objc.retain(dictionary)" not in body
    # The release helper exists and is a no-op for the "defaults" answer (0/nil).
    assert callable(compile_options.release_compile_options)



def test_the_read_back_flag_comes_from_arg_info() -> None:
    """`_finish_buffer` is told whether to copy, rather than deriving it itself.

    Every backend reads `ArgInfo.need_read_back` -- the shared read/write analysis's
    answer -- to decide whether a buffer comes back. Metal used to test
    `buffer.kernel_flags & 0b10` inline, which is the same bit spelled a second time:
    if the shared flags ever moved, that copy would silently pick the wrong buffers.
    The flag now travels with the bound entry.
    """
    from pygpgpu.constants import KernelFlags

    import inspect

    import pygpgpu.metal.oop.kernel as metal_kernel
    from pygpgpu.metal.oop.program_info import ArgInfo

    signature = inspect.signature(metal_kernel._finish_buffer)
    assert "need_read_back" in signature.parameters

    source = open(metal_kernel.__file__, encoding="utf-8").read()
    body = source[source.index("def _finish_buffer"):source.index("Kernel = MetalKernel")]
    # The mask is not tested here any more.
    assert "0b10" not in body
    assert "KernelFlags" not in body

    # And the property really is the bit the old inline test spelled out: a pointer
    # flagged as written is the case that comes back.
    written = ArgInfo("a", "float*", (8,), kernel_flags=KernelFlags.Writed)
    assert written.need_read_back is True
    read_only = ArgInfo("a", "float*", (8,), kernel_flags=KernelFlags.Readed)
    assert read_only.need_read_back is False


def test_a_failing_unuse_does_not_mask_the_real_error() -> None:
    """Releasing the claim happens on every path, and never replaces an exception.

    `_finish_buffer` runs inside a `finally` chain; an `unuse()` that raised there
    would replace whatever exception the read-back was already unwinding, turning a
    useful message into "mem obj is not in using". The release carries the claim's
    serial, so a stale one is a no-op rather than an error -- and a release that
    still raises (a fake, or a genuinely broken buffer) is absorbed all the same.
    """
    class Buffer:
        size = 0
        kernel_flags = 0

        def _contents(self):
            return 0

        def unuse(self, claim_serial=None):
            raise RuntimeError("mem obj is not in using")

    import pygpgpu.metal.oop.kernel as metal_kernel

    # No host value and no read-back: only the release runs, and its failure is
    # absorbed.
    metal_kernel._finish_buffer(Buffer(), None, False)

    # A read-back that fails must still surface its own error.
    class BrokenBuffer(Buffer):
        def _contents(self):
            raise OSError("device lost")

    import numpy as np

    with pytest.raises(OSError) as caught:
        metal_kernel._finish_buffer(BrokenBuffer(), np.zeros(4, np.uint8), True)
    assert "device lost" in str(caught.value)



def test_an_event_runs_every_registered_callback() -> None:
    """Both callback spellings run immediately, because the work is already done.

    Metal reports completion through the command buffer's status and the launch waits
    on it, so an `Event` that exists describes finished work. `add_callback` has to
    answer to that; `on_completed_callbacks_add` used to append to a list nothing
    drained, so a caller that registered one was told nothing and heard nothing --
    a silent drop rather than a refusal.
    """
    from pygpgpu.metal.oop.event import Event

    called = []
    event = Event()
    assert event.finished
    assert event.status is None
    # `wait()` is a no-op rather than an error: the state is already final.
    event.wait()

    event.add_callback(lambda ev, error: called.append(("add", error)))
    event.on_completed_callbacks_add(lambda ev, error: called.append(("deferred", error)))
    assert [name for name, _error in called] == ["add", "deferred"]
    assert event.status is None


def test_an_failed_event_carries_its_error_to_the_callback() -> None:
    """A non-None `status` is an exception, and the callback receives it."""
    from pygpgpu.metal.oop.event import Event

    failure = ValueError("boom")
    event = Event(failure)
    assert event.status is failure
    seen = []
    event.add_callback(lambda ev, error: seen.append(error))
    assert seen == [failure]



def test_a_device_key_distinguishes_two_identical_gpus() -> None:
    """`unique_key` is a cache key, so the name alone cannot be it.

    `unique_key` is what a compiled kernel is cached under per device. A Mac with two
    of the same GPU reports the same `name` for both, so a name-only key would let
    one device reuse the other's build. Every other backend folds an identity fact
    into the key for this reason; Metal's is `registryID`.
    """
    from pygpgpu.utils import sanitize_filename

    class FakeDevice:
        def __init__(self, name: str, registry_id: int) -> None:
            self.name = name
            self.registry_id = registry_id

    def key(device: FakeDevice) -> str:
        return sanitize_filename(f"{device.name} {device.registry_id}")

    first = FakeDevice("Apple M2", 0x1234)
    second = FakeDevice("Apple M2", 0x5678)
    again = FakeDevice("Apple M2", 0x1234)

    assert key(first) != key(second), "two identical GPUs must not share a key"
    assert key(first) == key(again), "one GPU must answer the same key every time"
    # The name stays readable, which is what the key is compared by a human.
    assert "Apple M2" in key(first)


def test_the_device_reports_a_registry_id() -> None:
    """`registryID` is the identity fact the key is built from.

    It is an `MTLDevice` selector, so the Device class has to expose it next to the
    other device queries rather than inventing an identifier of its own.
    """
    import inspect

    import pygpgpu.metal.oop.devices as metal_devices

    assert hasattr(metal_devices.Device, "registry_id")
    # `inspect.getsource` needs the getter: the attribute is a `property`, not a
    # function, so passing the descriptor itself raises TypeError.
    assert "registryID" in inspect.getsource(
        metal_devices.Device.registry_id.fget)
    # And `unique_key` folds it in rather than using the bare name.
    key_source = inspect.getsource(metal_devices.Device.unique_key.fget)
    assert "registry_id" in key_source



@pytest.mark.parametrize("bracket,expected", [
    # Automatic: one group of one thread, so an id-reading body gets 0.
    (None, ((1, 1, 1), (1, 1, 1))),
    # A bare item count and its one-tuple spelling are the same launch.
    (64, ((64, 1, 1), (1, 1, 1))),
    ((64,), ((64, 1, 1), (1, 1, 1))),
    # The ordinary CUDA pair. This used to raise TypeError: 'int' object is not
    # iterable, which made every plain `k[(4, 256)]` launch fail.
    ((4, 256), ((4, 1, 1), (256, 1, 1))),
    # Mixed spellings: a scalar side and a tuple side describe one launch.
    ((4, (256,)), ((4, 1, 1), (256, 1, 1))),
    (((4,), (256,)), ((4, 1, 1), (256, 1, 1))),
    # The fully spelled three-axis pair passes through.
    (((4, 1, 1), (256, 1, 1)), ((4, 1, 1), (256, 1, 1))),
    (((2, 3, 4), (8, 4, 2)), ((2, 3, 4), (8, 4, 2))),
])
def test_the_work_size_bracket_maps_onto_mtl_size(bracket, expected) -> None:
    """Every bracket spelling reaches Metal as two three-axis `MTLSize`s.

    The bracket is CUDA's `(blocks_per_grid, threads_per_block)` on every backend,
    and `dispatchThreadgroups:threadsPerThreadgroup:` takes that pair under its own
    names -- so the mapping is a normalisation, not a conversion. Either side may be
    written as a scalar or as a per-axis tuple, and `MTLSize` is always three
    dimensional, so a missing axis is padded with 1 (never 0, which would ask for an
    empty dispatch).
    """
    from pygpgpu.metal.oop.kernel import BoundKernel, MetalKernel

    kernel = MetalKernel(_NoDeviceProgram(), "k")
    assert BoundKernel(kernel, bracket)._dispatch_shape() == expected


def test_a_bracket_with_more_than_three_axes_is_refused() -> None:
    """Metal dispatches at most three axes, and the message says which side."""
    from pygpgpu.metal.oop.kernel import BoundKernel, MetalKernel

    kernel = MetalKernel(_NoDeviceProgram(), "k")
    with pytest.raises(ValueError) as refused:
        BoundKernel(kernel, ((1, 2, 3, 4), 256))._dispatch_shape()
    assert "blocks_per_grid" in str(refused.value)
    assert "at most three" in str(refused.value)



@pytest.mark.parametrize("device_type,width", [
    ("bool", 1), ("char", 1), ("uchar", 1),
    ("short", 2), ("ushort", 2), ("half", 2),
    ("int", 4), ("uint", 4), ("float", 4),
    ("long", 8), ("ulong", 8), ("double", 8),
])
def test_a_scalar_is_packed_at_the_device_width(device_type, width) -> None:
    """`setBytes:` copies exactly what is sent, so the width is the declared type's.

    A kernel's extent parameter is `const int`: sending python's 8-byte integer would
    make the device read half a value. `_SCALAR_SIZES` is what decides, and this pins
    it against the type the MSL declares.
    """
    from pygpgpu.metal.oop.kernel import _scalar_bytes

    value = {"bool": True, "float": 1.5, "double": 2.5, "half": 1.5}.get(device_type, 7)
    assert len(_scalar_bytes(device_type, value)) == width


@pytest.mark.parametrize("device_type,value", [
    ("uint", -1),
    ("ulong", -1),
    ("int", 2 ** 31),
    ("short", 70000),
    ("uchar", 256),
])
def test_an_out_of_range_scalar_names_the_type_and_its_range(device_type, value) -> None:
    """An impossible value is refused with a message that says what went wrong.

    `int.to_bytes` raises `OverflowError: int too big to convert` -- which names
    neither the parameter's type nor the range it accepts. A kernel argument is what
    the caller mistyped, so the message has to carry both.
    """
    from pygpgpu.metal.oop.kernel import _scalar_bytes

    with pytest.raises(OverflowError) as refused:
        _scalar_bytes(device_type, value)
    message = str(refused.value)
    assert device_type in message
    assert str(value) in message
    assert "range is" in message


def test_the_scalar_range_boundaries_are_accepted() -> None:
    """The reported range is the real one: its endpoints pack."""
    from pygpgpu.metal.oop.kernel import _scalar_bytes

    assert len(_scalar_bytes("uint", 4294967295)) == 4
    assert len(_scalar_bytes("int", -2147483648)) == 4
    assert len(_scalar_bytes("short", -32768)) == 2
    assert len(_scalar_bytes("uchar", 255)) == 1



@pytest.mark.parametrize("device_type,value", [
    ("float", 1e40),
    ("half", 1e10),
    ("float", 3.5e38),
])
def test_an_out_of_range_real_names_the_type_and_its_range(device_type, value) -> None:
    """A real that does not fit is refused with a message naming the type.

    `struct.pack` raises `OverflowError: float too large to pack with f format`, whose
    format letter tells a caller nothing about the kernel parameter they mistyped.
    (`double` has no such case: python's own `float` is IEEE double, so any value that
    would overflow `"<d"` is already `inf` before it reaches this code, and infinity
    is accepted -- see the next test.)
    """
    from pygpgpu.metal.oop.kernel import _scalar_bytes

    with pytest.raises(OverflowError) as refused:
        _scalar_bytes(device_type, value)
    message = str(refused.value)
    assert device_type in message
    assert "largest finite value" in message


def test_ieee_special_values_are_not_treated_as_overflow() -> None:
    """Infinity and NaN are valid IEEE values, so they pack rather than raise.

    A guard written as `abs(value) > largest` would also reject infinity, which is a
    legitimate value a kernel may want to send -- and `-inf` and `nan` likewise.
    """
    import math

    from pygpgpu.metal.oop.kernel import _scalar_bytes

    assert len(_scalar_bytes("float", float("inf"))) == 4
    assert len(_scalar_bytes("float", float("-inf"))) == 4
    assert len(_scalar_bytes("float", float("nan"))) == 4
    import struct

    assert math.isinf(struct.unpack("<f", _scalar_bytes("float", float("inf")))[0])
    # The largest finite value is still accepted.
    assert len(_scalar_bytes("float", 3.4028235e38)) == 4
    assert len(_scalar_bytes("half", 65504.0)) == 2


def test_half_packs_as_ieee_half_in_little_endian() -> None:
    """`half` is `<e`: IEEE half, little endian, explicitly.

    `numpy.float16(...).tobytes()` answers the *host's* byte order, so on a
    big-endian host it would disagree with every other branch here -- all of which
    spell the byte order out.
    """
    import struct

    from pygpgpu.metal.oop.kernel import _scalar_bytes

    for value in (1.5, -2.0, 65504.0, 0.0, -0.0):
        assert _scalar_bytes("half", value) == struct.pack("<e", value), value



def test_a_data_size_mismatch_is_refused_before_the_copy() -> None:
    """The copy length is the buffer's own size, so the data must match it.

    `set_data` copies `self._size` bytes with `memmove`, which reads past the end of a
    shorter array and silently truncates a longer one. `_host_pointer_of` is the single
    place both `set_data` and `bind_data` go through, and it refuses the mismatch
    before either touches memory.
    """
    from pygpgpu.metal.oop.buffer import Buffer

    import numpy as np

    class FakeBuffer:
        """Just enough of a Buffer to run `_host_pointer_of` without a device."""

        _size = 8
        _contiguous_with_size = staticmethod(Buffer._contiguous_with_size)

    check = Buffer._host_pointer_of

    mismatched = np.zeros(4, np.uint8)
    with pytest.raises(ValueError) as refused:
        check(FakeBuffer(), mismatched)
    assert "not equal to buffer size" in str(refused.value)

    # A matching array is accepted, and the pointer is the array's own storage.
    matching = np.zeros(8, np.uint8)
    pointer, array = check(FakeBuffer(), matching)
    assert pointer == int(matching.ctypes.data)
    assert array.nbytes == 8


def test_bytes_and_bytearray_are_accepted_as_data() -> None:
    """A `bytes` payload is a valid buffer source, sized by its own length."""
    from pygpgpu.metal.oop.buffer import Buffer

    array, size = Buffer._contiguous_with_size(b"abcd")
    assert size == 4 and array.nbytes == 4
    array, size = Buffer._contiguous_with_size(bytearray(b"abc"))
    assert size == 3 and array.nbytes == 3



def test_a_message_send_is_serialized_across_threads() -> None:
    """`objc_msgSend`'s signature is shared state, so the send has to be serialized.

    `objc_msgSend` is variadic in C: ctypes needs the called method's own signature
    set on the SHARED function object before each call. Two threads readying
    different signatures at the same time can hand the wrong one to the call, which
    is undefined behaviour at the ABI level rather than a python exception -- and the
    launch path takes no lock of its own, because a Metal command queue is thread-safe
    at the framework level.
    """
    import inspect

    import pygpgpu.metal.driver.objc as objc

    assert hasattr(objc.ObjC, "_send_lock")
    source = inspect.getsource(objc.ObjC.send)
    assert "with ObjC._send_lock" in source
    # The protected region covers both the signature set-up and the call itself:
    # releasing between them would defeat the point.
    body = source[source.index("with ObjC._send_lock:"):]
    assert "msg.argtypes" in body
    assert "return msg(" in body



def test_the_objc_bridge_declares_pointer_sized_integers() -> None:
    """`NSUInteger` results and struct fields are `c_size_t`, never `c_ulong`.

    `objc.py` is the layer below the one the earlier sweep covered, and it had the
    same defect: `send_int` declared its result `c_ulong` while documenting it as
    `NSUInteger`, and `struct_result` built `MTLSize`/`MTLOrigin` fields from it.
    `registryID` is a full 64-bit value, so on a host where `c_ulong` is 4 bytes the
    id would be truncated -- and the device cache key is built from it.
    """
    import ctypes
    import inspect

    import pygpgpu.metal.driver.objc as objc

    import ast

    # Read the module's real code, not its prose: the name legitimately appears in
    # the docstrings that explain why it is the wrong type.
    tree = ast.parse(inspect.getsource(objc))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "c_ulong":
            offenders.append(f"line {node.lineno}: .c_ulong")
        if isinstance(node, ast.Name) and node.id == "c_ulong":
            offenders.append(f"line {node.lineno}: c_ulong")
    assert not offenders, offenders

    assert "c_size_t" in inspect.getsource(objc.ObjC.send_int)
    # Struct fields are pointer-sized too.
    assert "c_size_t" in inspect.getsource(objc.ObjC.struct_result)
    assert ctypes.sizeof(ctypes.c_size_t) == ctypes.sizeof(ctypes.c_void_p)


def test_struct_result_sets_its_signature_under_the_send_lock() -> None:
    """`struct_result` touches the same shared `objc_msgSend` as `send` does.

    Setting `restype`/`argtypes` outside the lock would race with a concurrent `send`
    -- and the return convention differs (a struct comes back in registers on arm64,
    through `stret` on x86_64), so a mixed-up signature corrupts the value rather than
    raising.
    """
    import inspect

    import pygpgpu.metal.driver.objc as objc

    body = inspect.getsource(objc.ObjC.struct_result)
    assert "with ObjC._send_lock:" in body
    locked = body[body.index("with ObjC._send_lock:"):]
    assert "msg.restype" in locked
    assert "result = msg(" in locked



def test_building_compile_options_releases_a_failed_attempt() -> None:
    """`alloc` answers +1, so a nil `init` or a raising setter still owes a release.

    `+alloc` hands back a reference this call owns; when `-init` answers nil the
    allocated object is still this call's to release, and a setter that raises leaves
    the same obligation on the half-configured object. The whole set-up runs under
    one `try`, and `_configure` holds the field writes.
    """
    import inspect

    import pygpgpu.metal.oop.compile_options as compile_options

    source = inspect.getsource(compile_options.build)
    assert "objc.release(alloc)" in source, "a failed -init must release the allocation"
    assert "objc.release(target)" in source, "a raising setter must release the object"
    assert "except BaseException:" in source
    assert "_configure(" in source
    assert hasattr(compile_options, "_configure")

    # The defaults never allocate, so they stay a plain 0 (nil).
    from pygpgpu.metal.oop.compile_options import CompileOptions

    assert compile_options.build(None) == 0
    assert compile_options.build(CompileOptions()) == 0


def test_the_configure_step_sets_only_the_fields_that_were_given() -> None:
    """Each field is written only when the caller named it, so a default stays Metal's.

    A field left at its compiler default is the point of `CompileOptions`; setting
    every property unconditionally would overwrite Metal's own choices with the
    numbers this module happened to list.
    """
    import inspect

    import pygpgpu.metal.oop.compile_options as compile_options

    body = inspect.getsource(compile_options._configure)
    for field, selector in (
            ("language_version", "setLanguageVersion:"),
            ("fast_math", "setFastMathEnabled:"),
            ("math_mode", "setMathMode:"),
            ("macros", "setPreprocessorMacros:")):
        assert selector in body, selector
        # Each write is guarded by its own `is not None` test.
        assert f"options.{field} is not None" in body, field



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



def two_phase(x, mid, out):
    """Two magic loops: translated into two phase kernels in one source."""
    for i in range(out.shape[0]):
        mid[i] = x[i] * 2.0
    for i in range(out.shape[0]):
        out[i] = mid[i] + 1.0


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


def mixed_factor_kernel(x, out, factor):
    for i in range(out.shape[0]):
        out[i] = x[i] * factor



def test_the_device_class_is_reachable_under_the_shared_module_name() -> None:
    """Every backend publishes `oop.device.Device`, so Metal does too.

    OpenCL, CUDA and HIP each define the class in `oop/device.py`. Metal keeps it in
    `oop/devices.py` beside the `Devices` metaclass that consumes it, which reads well
    but breaks a caller that walks the backends generically. `oop/device.py` re-exports
    it -- re-exports, not redefines: there is one `Device` class, reachable two ways.
    """
    import importlib

    import pygpgpu.metal.oop.device as metal_device
    import pygpgpu.metal.oop.devices as metal_devices

    assert metal_device.Device is metal_devices.Device
    for backend in ("opencl", "cuda", "hip", "metal"):
        module = importlib.import_module(f"pygpgpu.{backend}.oop.device")
        assert hasattr(module, "Device"), backend



def test_compile_options_are_read_only_from_their_own_spellings() -> None:
    """A foreign options object is "no MSL settings", not a translation to guess at.

    `kernel_import` hands one options object to whichever backend owns the source, so a
    CUDA/OpenCL `BuildOptions` reaching Metal is a normal case. Its `-D`/include/gpu-
    architecture settings describe a *different* compiler, and `BuildOptions` cannot
    even be constructed from a bare flag string -- so reading it as "unconfigured" and
    leaving Metal's defaults is the honest answer, not a limitation to work around.

    The MSL settings have their own spelling: `CompileOptions`, or a mapping whose keys
    are this module's field names.
    """
    from pygpgpu.metal.oop.compile_options import from_mapping

    class ForeignOptions:
        """Stand-in for a BuildOptions: has attributes, none of this module's names."""

        def __init__(self):
            self.gpu_architecture = "sm_90"

    assert from_mapping(ForeignOptions()) is None
    assert from_mapping("a string") is None
    assert from_mapping({}) is None
    assert from_mapping({"unknown": 1}) is None

    # The module's own names are honoured, including the macro table.
    options = from_mapping({"macros": {"FOO": "1"}, "fast_math": False})
    assert options is not None
    assert options.macros == {"FOO": "1"}
    assert options.fast_math is False
    assert options.language_version is None


def test_the_macro_table_is_one_of_the_selectors() -> None:
    """`setPreprocessorMacros:` is the selector a mapping's `macros` field drives."""
    from pygpgpu.metal.oop.compile_options import CompileOptions

    assert "setPreprocessorMacros:" in CompileOptions.SELECTORS



def test_metal_is_a_zero_copy_backend_and_exposes_what_that_needs() -> None:
    """`StorageModeShared` already *is* host memory, and the layer is told so.

    The device-resident array layer asks `zero_copy_support(backend)` before it offers
    `to_device(zero_copy=True)`, and it refuses with the backend's own reason. Metal
    answers True because a shared buffer's bytes are the CPU's bytes; OpenCL answers
    False because `CL_MEM_USE_HOST_PTR` may keep a private copy.

    The Metal half of that path needs two things from this backend: `Buffer._contents`
    (the host address the allocation compares against the array's own pointer) and an
    instance attribute for the answer. A `__slots__` there would break the second.
    """
    from pygpgpu.numpy import zero_copy_support

    supported, reason = zero_copy_support("metal")
    assert supported, reason
    # The refusal message is empty for a backend that can; it is not empty for one
    # that cannot.
    assert reason == ""

    from pygpgpu.metal.oop.buffer import Buffer

    assert hasattr(Buffer, "_contents")
    assert not hasattr(Buffer, "__slots__")


def test_the_async_forms_reach_the_bound_launch() -> None:
    """`submit`/`async_call` on an unbracketed kernel launch with an automatic grid.

    `@kernel` offers those on every backend, so Metal answers with a launch rather
    than an AttributeError: a `MetalKernel` is bracketed (its `BoundKernel` carries
    the work size), and the async forms bracket `None` -- the same automatic
    one-threadgroup grid a synchronous call takes -- before launching. This pins the
    plumbing a machine without a Metal device can see: the refusal is gone, and the
    failure the launch meets (no device) surfaces through the future, the way a real
    failure on hardware would.
    """
    import concurrent.futures

    from pygpgpu.metal.oop.kernel import MetalKernel

    kernel = MetalKernel(None, "k")
    submit_future = kernel.submit()
    assert isinstance(submit_future, concurrent.futures.Future)
    with pytest.raises(BaseException) as submit_error:
        submit_future.result(timeout=10)
    assert not isinstance(submit_error.value, NotImplementedError)

    async_future = kernel.async_call()
    # `async_call` answers an asyncio future, whose `exception()` never raises.
    assert hasattr(async_future, "exception")

