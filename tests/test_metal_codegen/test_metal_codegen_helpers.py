"""The helpers tests, split out of `test_metal_codegen.py`.

The subject is helpers.
"""
from __future__ import annotations

from _helpers.test_metal_codegen_helpers import (
    _compiled_source,
    _image_helper_resolver,
    _no_slice_marker_leak,
    _syntax_diagnostics,
    cuda_index_body,
    gid_style_body,
    histogram,
    image_uses_helper,
    kernel_using_shared_a,
    kernel_using_shared_b,
    returning,
    shared_scratch,
    uses_index_helper,
    uses_plain_helper,
    uses_reads_shared,
    uses_record_member,
    uses_record_value,
    uses_vector_return,
    vecadd,
    vector_scale,
)

from typing import Any
import numpy as np
import pytest
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.metal_codegen import (
    PROLOGUE_PARAMETERS,
    prelude,
)
from pygpgpu.kernel.translator import FunctionTranslator


def test_a_helper_taking_shared_memory_declares_a_threadgroup_pointer(tmp_path: Any,
                                                                      monkeypatch) -> None:
    """A shared slice reaching a helper is a `threadgroup` pointer, not a printed tuple.

    The translator tracks the parameter as `('shared_slice', <element>, <shape>)`, an
    internal marker. `render_device` used to fall through to `_device_type(tuple)`,
    which printed the tuple into the C source -- `(('shared_slice', 'float', (64,))
    buf, int i)` -- a signature no compiler accepts. The marker is now understood and
    rendered as the pointer it means, in the space the storage lives in.
    """
    from metal_address_space import check_address_spaces
    from metal_attributes import check_attributes

    source = _compiled_source(uses_reads_shared,
                              [np.zeros(64, np.float32)] * 2, tmp_path, monkeypatch)
    # The helper's signature names a threadgroup pointer, and the internal marker never
    # reaches the source.
    signature = next(line for line in source.splitlines()
                     if "reads_shared" in line and "(" in line
                     and "{" not in line and "=" not in line)
    assert "threadgroup float*" in signature, signature
    # The internal marker is a tuple literal; its textual form is what used to leak.
    assert "('shared_slice'" not in source, "the internal type marker leaked"

    assert not check_attributes(source), check_attributes(source)
    assert not check_address_spaces(source), check_address_spaces(source)
    ok, diagnostics = _syntax_diagnostics(source)
    assert ok, diagnostics


def test_a_gid_style_body_compiles_and_declares_no_extent(tmp_path: Any,
                                                         monkeypatch) -> None:
    """`@kernel(domain=...)` emits no extent parameter, and the source is valid MSL.

    A magic loop knows its array's extent, so the translator passes it as a `const int`.
    A gid-style body reads the work item directly, so the domain is the launch's -- there
    is no extent to send, and the signature is just the buffers and the six attribute
    parameters. All three checks have to agree on that shape.
    """
    import pygpgpu.metal.oop.program as program_module
    from pygpgpu.kernel.decorator import PyKernel
    from pygpgpu.kernel import source_paths
    from metal_address_space import check_address_spaces
    from metal_attributes import check_attributes

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

    kernel_object = __import__("pygpgpu").kernel(gid_style_body, domain="out")
    original = program_module.Program
    program_module.Program = StubProgram
    try:
        args = [np.zeros(8, np.float32)]
        try:
            kernel_object._compile(args, [infer_ctype(a) for a in args], [], [], None,
                                   "sig", "metal", object())
        except KeyError:
            pass
    finally:
        program_module.Program = original

    source = captured["source"]
    # No extent parameter: the domain is the launch's, not the body's.
    assert "_out_shape0" not in source
    assert not check_attributes(source), check_attributes(source)
    assert not check_address_spaces(source), check_address_spaces(source)
    ok, diagnostics = _syntax_diagnostics(source)
    assert ok, diagnostics


@pytest.mark.parametrize("case,arg_count,expectations", [
    ("scalar", 3, ()),
    ("returns", 2, ("_ret0", "device float* x")),
    ("vector", 2, ()),
    ("histogram", 2, ("atomic_fetch_add_explicit", "(device atomic_uint*)")),
    ("shared", 2, ("threadgroup float*", "threadgroup_barrier")),
], ids=["scalar", "returns", "vector", "histogram", "shared"])
def test_every_shape_produces_valid_msl(case, arg_count, expectations, tmp_path: Any,
                                        monkeypatch) -> None:
    """One kernel of each shape reaches a Metal compiler as valid MSL.

    The shapes differ in what they declare and what they call: a return adds slot
    buffers and a claim word, a vector needs MSL's own width, a racy update becomes
    the C++11 atomic family, and shared memory becomes a `threadgroup` parameter.
    Each is checked three ways -- C++ syntax, attribute validity, address-space
    presence -- because each check catches a different class of mistake.
    """
    from metal_address_space import check_address_spaces
    from metal_attributes import check_attributes

    from pygpgpu import float4

    # Each shape's own element type and count: a vector kernel takes `float4`s, a
    # histogram takes the index buffer and its bins, and `vecadd` takes three arrays.
    if case == "vector":
        args = [np.zeros(4, dtype=float4)] * arg_count
    elif case == "histogram":
        args = [np.zeros(256, np.uint32), np.zeros(300, np.uint32)]
    elif case == "scalar":
        args = [np.zeros(8, np.float32)] * arg_count
    else:
        args = [np.zeros(256, np.float32)] * arg_count
    functions = {"scalar": vecadd, "returns": returning, "vector": vector_scale,
                 "histogram": histogram, "shared": shared_scratch}
    source = _compiled_source(functions[case], list(args), tmp_path, monkeypatch)
    assert source, case

    for expected in expectations:
        assert expected in source, (case, expected)

    assert not check_attributes(source), check_attributes(source)
    assert not check_address_spaces(source), check_address_spaces(source)


def test_every_index_builtin_has_a_prelude_macro() -> None:
    """The two sets are the same size, name for name.

    MSL has no `get_global_id()`: the prelude defines each one as a macro over an
    attribute-decorated parameter. A builtin without a macro would compile to an
    undeclared call, and a macro without a builtin would be dead text -- so the two
    lists have to match in both directions.
    """
    import re

    from pygpgpu.kernel.device_builtins import INDEX_BUILTINS

    macros = re.findall(r"#define (\w+)\(", prelude())
    assert sorted(macros) == sorted(INDEX_BUILTINS)
    # And each expands onto a parameter the prologue declares.
    attributes = {name for _t, name, _a in PROLOGUE_PARAMETERS}
    for macro_body in re.findall(r"#define \w+\(dim\) .*", prelude()):
        for parameter in attributes:
            if parameter in macro_body:
                break
        else:
            raise AssertionError(f"a macro names no prologue parameter: {macro_body}")


@pytest.mark.parametrize("case,arg_count,expectations", [
    ("cuda_names", 2, ("get_group_id(0)", "get_local_id(0)", "get_local_size(0)")),
], ids=["cuda_names"])
def test_cuda_index_names_resolve_through_the_macros(case, arg_count, expectations,
                                                    tmp_path: Any, monkeypatch) -> None:
    """`blockIdx.x`/`threadIdx.x` work on Metal, spelled as the OpenCL calls it maps to.

    The index vectors are values on every backend, so `blockIdx.x * blockDim.x +
    threadIdx.x` is the same source a CUDA reader writes. On Metal those names resolve
    to the OpenCL per-axis calls, which the prelude's macros then turn into the
    attribute parameters -- so a ported kernel needs no rewrite.
    """
    source = _compiled_source(cuda_index_body, [np.zeros(8, np.float32)] * arg_count,
                              tmp_path, monkeypatch, domain="out")
    for expected in expectations:
        assert expected in source, (case, expected)
    ok, diagnostics = _syntax_diagnostics(source)
    assert ok, diagnostics


def test_a_helper_that_reads_a_work_item_receives_the_prologue(tmp_path: Any,
                                                               monkeypatch) -> None:
    """On Metal the work-item numbers are parameters, so a helper has to receive them.

    OpenCL and CUDA provide `get_global_id`/`threadIdx` as device intrinsics -- symbols
    every function can name. Metal has no such intrinsic: the prelude defines those
    names as macros over attribute-decorated **kernel parameters**, which exist only in
    the entry point's scope. A helper that calls one used to expand to a reference to a
    parameter it never had, and the Metal compiler answered with a name the user never
    wrote (`'_mtl_thread_position' was not declared in this scope`).

    The fix passes the six parameters down: the helper declares them, and its call site
    forwards its own.
    """
    from metal_syntax_check import check

    source = _compiled_source(uses_index_helper, [np.zeros(8, np.float32)] * 2,
                              tmp_path, monkeypatch)
    # The helper's own signature carries the prologue...
    signature = next(line for line in source.splitlines()
                     if "index_reading_helper" in line and "(" in line
                     and "{" not in line and "=" not in line)
    assert "_mtl_thread_position" in signature, signature
    # ...and the call passes it, or the two would disagree.
    call = next(line for line in source.splitlines()
                if "out[i] = index_reading_helper" in line)
    assert "_mtl_thread_position" in call, call

    ok, diagnostics = check(source)
    assert ok, diagnostics


def test_a_helper_without_an_index_call_keeps_its_plain_signature(tmp_path: Any,
                                                                  monkeypatch) -> None:
    """Only the helper that needs the parameters receives them.

    A helper whose body reads no work item stays `f(device float* x)` with a plain
    call, on every backend -- passing parameters nothing reads would be noise in the
    signature and work at the call site for no benefit.
    """
    source = _compiled_source(uses_plain_helper, [np.zeros(8, np.float32)] * 2,
                              tmp_path, monkeypatch)
    assert "index_reading_helper(" not in source
    signature = next(line for line in source.splitlines()
                     if "plain_helper" in line and "(" in line and "{" not in line)
    assert "_mtl_thread" not in signature, signature


@pytest.mark.parametrize("backend,expected", [
    ("opencl", "image2d_t"),
    ("cuda", "cudaTextureObject_t"),
])
def test_an_image_parameter_reaching_a_helper_uses_the_image_type(backend, expected) -> None:
    """A helper that takes an image declares the backend's own image type.

    The image path is OpenCL's and CUDA's; Metal refuses images outright, so this is
    checked on the two that support them. What matters is that the parameter is an
    image type rather than the translator's internal marker.
    """
    translator = FunctionTranslator(image_uses_helper, [("image", 4, "float"),
                                                       infer_ctype(np.zeros(8, np.float32))],
                                    backend, _image_helper_resolver(), {}, is_kernel=True)
    translator.translate()
    code = next(entry["code"] for entry in translator.device_cache.values()
                if "code" in entry)
    assert expected in code.splitlines()[0], code.splitlines()[0]
    _no_slice_marker_leak(code)


@pytest.mark.parametrize("case,args,expected_return", [
    ("member", "record", "float "),
    ("record", "record", "_st_"),
    ("vector", "vector", "float4 "),
], ids=["struct member", "struct value", "vector"])
def test_a_helper_returns_the_type_its_body_produced(case, args, expected_return,
                                                     tmp_path: Any, monkeypatch) -> None:
    """Struct members, whole structures and vectors all come back as their own type.

    A helper's return type is not always the C scalar: reading a structure member
    yields that member's type, returning the structure yields the structure (which MSL
    passes by value), and a vector stays a vector. Getting this wrong would either
    fail to compile or silently narrow the value.
    """
    from pygpgpu import float4

    record = np.zeros(4, dtype=[("pos", np.float32), ("vel", np.float32)])
    if args == "record":
        call_args = [record, np.zeros(4, np.float32)]
    else:
        call_args = [np.zeros(4, dtype=float4)] * 2

    functions = {"member": uses_record_member, "record": uses_record_value,
                 "vector": uses_vector_return}
    source = _compiled_source(functions[case], call_args, tmp_path, monkeypatch)
    helper_line = next(line for line in source.splitlines()
                       if line.startswith(expected_return) and "_" in line
                       and "(" in line and "kernel" not in line)
    assert helper_line.startswith(expected_return), helper_line
    ok, diagnostics = _syntax_diagnostics(source)
    assert ok, diagnostics


def test_two_kernels_in_one_file_share_one_helper_definition(tmp_path: Any,
                                                            monkeypatch) -> None:
    """The helper is emitted once, and both kernels resolve to it.

    Two kernels in one module write to one generated file, and each contributes its
    own definitions. A helper both of them reach must appear **once** -- a second copy
    would be a duplicate definition, which C++ rejects -- while both kernels still call
    it by the same name. The dependency order is what makes that work: the helper
    precedes the kernels that use it.
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

    args = [np.zeros(8, np.float32)] * 2
    types = [infer_ctype(a) for a in args]
    first = __import__("pygpgpu").kernel(kernel_using_shared_a)
    second = __import__("pygpgpu").kernel(kernel_using_shared_b)

    original = program_module.Program
    program_module.Program = StubProgram
    try:
        # Four compiles, so each kernel is written and then written again: the second
        # pass is what would duplicate a definition that did not dedupe.
        for kernel_object, signature in ((first, "sigA"), (second, "sigB"),
                                         (first, "sigA"), (second, "sigB")):
            try:
                kernel_object._compile(args, types, [], [], None, signature,
                                       "metal", object())
            except KeyError:
                pass
    finally:
        program_module.Program = original

    source = captured["source"]
    assert source.count("float shared_across_kernels_") == 1, "the helper is duplicated"
    assert source.count("kernel void") == 2, "both kernels must be in the file"
    # The helper precedes both kernels: C needs the callee before its caller.
    helper_at = source.index("float shared_across_kernels_")
    kernel_positions = [index for index in range(len(source))
                        if source.startswith("kernel void", index)]
    assert len(kernel_positions) == 2
    assert all(position > helper_at for position in kernel_positions)

    ok, diagnostics = _syntax_diagnostics(source)
    assert ok, diagnostics
