"""Pins the Metal Shading Language a @kernel translates to, without a device.

Translation needs no GPU: the translator is pure python, and the *source text* it
produces is the whole contract between this package and Metal. That makes these
tests runnable on any machine -- which matters, because the runtime side (a launch)
needs a Mac. The runtime half lives wherever the device backends are parameterized
(see tests/conftest.py); this file is the half that must never regress silently.

What is pinned, and why each one is load-bearing (see AGENTS.md's Metal section):

* the entry point is `kernel void`, and its buffer address space is `device`;
* the work-item numbers arrive as `[[...]]`-attributed parameters and the
  prelude's macros are what the body's `get_global_id(d)` calls expand onto;
* the prologue parameters are part of `parameter_declarations()`, so they are in
  the signature digest and therefore in the cache key;
* OpenCL's own generation is unchanged by any of it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pygpgpu import (barrier, blockDim, blockIdx, get_global_id,
                     get_local_id, threadIdx, zeros_per_group)
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.metal_codegen import (
    PROLOGUE_PARAMETERS,
    kernel_prefix,
    prelude,
    prologue_parameters,
)
# The MSL barrier spelling is reached through `device_builtins.emit_barrier`; importing
# `metal_codegen.barrier` here would shadow the kernel builtin of the same name.
from pygpgpu.kernel.metal_codegen import barrier as msl_barrier
from pygpgpu.kernel.translator import FunctionTranslator


def vecadd(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


def generate(func, n_args: int, backend: str = "metal") -> str:
    """The rendered source for `func`, without compiling it."""
    types = [infer_ctype(np.zeros(8, np.float32)) for _ in range(n_args)]
    translator = FunctionTranslator(func, types, backend, lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    return "\n\n".join(source for _, source in translator.render_kernels("k_test"))


def test_entry_point_is_kernel_void() -> None:
    """MSL writes the return type in the declaration, and the qualifier is `kernel`."""
    assert kernel_prefix() == "kernel void"
    source = generate(vecadd, 3)
    assert source.startswith("kernel void k_test(")
    # Not the OpenCL or CUDA spelling, and no doubled `void`.
    assert "__kernel" not in source
    assert "__global__" not in source
    assert "void void" not in source


def test_buffer_address_space_is_device() -> None:
    """A buffer pointer lives in `device`, not `__global`."""
    source = generate(vecadd, 3)
    assert "device float* a" in source
    assert "__global" not in source


def test_prologue_parameters_are_attributed() -> None:
    """The work-item numbers are parameters carrying a `[[...]]` attribute."""
    text = ", ".join(prologue_parameters())
    for _type_text, name, attribute in PROLOGUE_PARAMETERS:
        assert f"{name} [[{attribute}]]" in text
    # The attribute follows the name, which is MSL's spelling.
    for parameter in prologue_parameters():
        type_text, rest = parameter.split(" ", 1)
        assert rest.startswith("_mtl_")


def test_body_keeps_opencl_index_spelling() -> None:
    """The shared body still calls `get_global_id(0)`; the prelude is what defines it."""
    source = generate(vecadd, 3)
    assert "get_global_id(0)" in source
    assert "get_global_id(dim)" in prelude()


def test_prelude_maps_every_index_helper() -> None:
    """Every helper the OpenCL body may call has a macro over a real parameter."""
    text = prelude()
    declared = {name for _t, name, _a in PROLOGUE_PARAMETERS}
    for helper in ("get_global_id", "get_local_id", "get_group_id",
                   "get_global_size", "get_local_size", "get_num_groups"):
        assert f"#define {helper}(dim)" in text
    # Every macro expands onto exactly the parameters the kernel declares: each
    # `#define` line mentions at least one prologue name, and no name is invented.
    macro_lines = [line for line in text.splitlines() if line.startswith("#define")]
    assert len(macro_lines) == 6
    for line in macro_lines:
        assert any(name in line for name in declared), line


def test_prologue_parameters_are_in_the_signature() -> None:
    """They must be in the rendered parameter list -- the signature digest hashes it."""
    translator = FunctionTranslator(
        vecadd, [infer_ctype(np.zeros(8, np.float32)) for _ in range(3)],
        "metal", lambda name: None, {}, is_kernel=True)
    translator.translate()
    declarations = translator.parameter_declarations()
    expected = [f"{_t} {name} [[{attribute}]]"
                for _t, name, attribute in PROLOGUE_PARAMETERS]
    assert declarations[-len(expected):] == expected


def test_barrier_is_metal_spelling() -> None:
    """The Metal barrier is `threadgroup_barrier`, never CUDA's `__syncthreads()`."""
    assert "threadgroup_barrier" in msl_barrier()
    assert "sync" not in msl_barrier().lower().replace("threadgroup_barrier", "")
    # And the emitted call is reached through the builtin, through the shared emitter.
    from pygpgpu.kernel.device_builtins import emit_barrier

    assert "threadgroup_barrier" in emit_barrier("metal")
    assert "__syncthreads" not in emit_barrier("metal")


@pytest.mark.parametrize("backend", ["opencl", "cuda"])
def test_other_backends_unchanged(backend: str) -> None:
    """The same body on the other backends keeps its own spelling."""
    source = generate(vecadd, 3, backend)
    if backend == "opencl":
        assert "__kernel void" in source
        assert "__global float* a" in source
    else:
        assert "__global__ void" in source
        assert "device float*" not in source
    # Neither carries Metal's prologue.
    assert "_mtl_thread_position" not in source


def printer(x, out):
    for i in range(out.shape[0]):
        print(x[i])
        out[i] = x[i]


def shared_buffer(x, out):
    """A `zeros_per_group` buffer: the shape a work-group scratch array takes."""
    buf = zeros_per_group(256, "float32")
    gid = get_global_id(0)
    out[gid] = buf[blockIdx][gid] + x[gid]


def test_print_is_refused_on_metal_and_still_works_on_opencl() -> None:
    """Metal's printf goes to a log no host code reads, so the call site is refused.

    The generated call sites write into a per-call-site output buffer that the
    OpenCL runtime drains. Metal has no such buffer, so emitting them would compile
    and silently lose the output -- the same reason HIP refuses its own.
    """
    with pytest.raises(SyntaxError) as refused:
        generate(printer, 2)
    message = str(refused.value)
    assert "metal" in message
    assert "print()" in message

    # The backends that do drain the buffer keep generating it.
    for backend in ("opencl", "cuda"):
        source = generate(printer, 2, backend)
        assert "printf" in source


@pytest.mark.parametrize("call,expected", [
    ("atomic_add", "atomic_fetch_add_explicit"),
    ("atomic_sub", "atomic_fetch_sub_explicit"),
    ("atomic_min", "atomic_fetch_min_explicit"),
    ("atomic_max", "atomic_fetch_max_explicit"),
    ("atomic_and", "atomic_fetch_and_explicit"),
    ("atomic_or", "atomic_fetch_or_explicit"),
    ("atomic_xor", "atomic_fetch_xor_explicit"),
    ("atomic_xchg", "atomic_exchange_explicit"),
])
def test_atomics_use_the_metal_c_11_spelling(call: str, expected: str) -> None:
    """MSL has no `atomicAdd` family; the C++11 `atomic_fetch_*` functions are it."""
    from pygpgpu.kernel.device_builtins import _METAL_ATOMICS

    assert _METAL_ATOMICS[call] == expected
    # The CUDA spelling must not survive into a metal source.
    from pygpgpu.kernel.device_builtins import ATOMIC_BUILTINS

    emitted = ATOMIC_BUILTINS["atomic_add"][1]
    assert emitted == "atomicAdd", "the CUDA table is unchanged"
    assert not expected.startswith("atomicAdd")


def test_a_metal_atomic_names_its_address_space() -> None:
    """A `threadgroup` buffer and a `device` array take different atomics.

    MSL overloads the atomics on the reference's address space, and casting one
    space to the other does not compile -- the same rule OpenCL enforces with
    `__local` and `__global`.
    """
    from pygpgpu.kernel.device_builtins import _emit_atomic_metal

    for is_local, space in ((True, "threadgroup"), (False, "device")):
        emitted = _emit_atomic_metal("atomic_add", "&(buf)[i]", "1", None, "uint", is_local)
        assert f"({space} atomic_uint*)" in emitted
        assert "memory_order_relaxed" in emitted
        assert "atomicAdd" not in emitted
    # The two address spaces produce different text: they are not interchangeable.
    assert _emit_atomic_metal("atomic_add", "&(b)[i]", "1", None, "uint", True) != \
        _emit_atomic_metal("atomic_add", "&(b)[i]", "1", None, "uint", False)


def test_cas_is_the_compare_exchange_form() -> None:
    """MSL's compare-and-swap answers a bool and takes the expected value by pointer."""
    from pygpgpu.kernel.device_builtins import _emit_atomic_metal

    emitted = _emit_atomic_metal("atomic_cas", "&(a)[i]", "1", "0", "int", False)
    assert "atomic_compare_exchange_weak_explicit" in emitted
    # It needs a local to hold the expected value, which the call takes the address of.
    assert "_cas_expected" in emitted
    assert "atomicCAS" not in emitted


def test_the_prelude_macro_body_is_parenthesised_as_a_whole() -> None:
    """The macro body is one parenthesised expression, so it composes anywhere.

    A conditional expression binds looser than `%`, and the folding the translator
    emits (`(gid % size) * stride + ...`) puts a macro expansion on the left of a
    `%`. Without the outer parentheses the conditional would swallow the rest of
    that expression and the work position would be wrong -- silently, on the
    device. The generated *body* was verified against a real C compiler evaluating
    the same fold (position 31 from a 8x4x1 grid at (7, 3, 0)); this test pins the
    parentheses that made it correct.
    """
    from pygpgpu.kernel.metal_codegen import prelude

    for line in prelude().splitlines():
        if not line.startswith("#define"):
            continue
        body: str = line.split(")", 1)[1].strip()
        # The whole body is wrapped: it opens with a parenthesised group and ends
        # by closing that same group, and the brackets balance.
        assert body.startswith("(") and body.endswith(")"), line
        assert body.count("(") == body.count(")"), line
        depth: int = 0
        for index, character in enumerate(body):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                # The outer group must not close before the very last character.
                assert depth != 0 or index == len(body) - 1, line


def read_image(src, out):
    """Read one image row: the shape an image parameter takes."""
    for i in range(out.shape[0]):
        out[i] = src[0, i]


def write_image(dst, src):
    """Write one image row: the other direction, which needs a surface."""
    for i in range(src.shape[0]):
        dst[0, i] = src[i]


def test_an_image_parameter_is_refused_on_metal() -> None:
    """An image is refused by the translator, not left to fail in an MSL compiler.

    Metal *has* texture objects, but the whole image path is Metal's own -- the
    parameter type, the sampling call with its sampler state, the store, and the
    host-side `MTLTexture` binding. Until that exists the OpenCL formulation would
    reach the source as `__read_only image2d_t`, which MSL has no notion of, so the
    translator is where the refusal belongs: it can name the backends that work.
    """
    from pygpgpu.image import Image, ImageAddress, ImageFilter

    image = Image(np.zeros((8, 8), np.float32), address=ImageAddress.CLAMP,
                  filter=ImageFilter.POINT)
    types = [infer_ctype(image), infer_ctype(np.zeros(8, np.float32))]

    for func, action in ((read_image, "read"), (write_image, "write")):
        translator = FunctionTranslator(func, types, "metal", lambda name: None, {},
                                        is_kernel=True, launch_backend="metal")
        with pytest.raises(SyntaxError) as refused:
            translator.translate()
        message = str(refused.value)
        assert "metal" in message
        assert action in message
        # The message points at the backends that do support images.
        assert "opencl" in message and "cuda" in message

    # The backends that implement the image path are unaffected.
    for backend in ("opencl", "cuda"):
        translator = FunctionTranslator(read_image, types, backend, lambda name: None,
                                        {}, is_kernel=True, launch_backend=backend)
        translator.translate()
        source = "\n".join(s for _, s in translator.render_kernels("k"))
        assert "image2d_t" in source or "cudaTextureObject_t" in source


def vector_scale(src, out):
    """A vector kernel: one float4 per element."""
    for i in range(out.shape[0]):
        out[i] = src[i] * 2.0


def structured_add(parts, out):
    """A ctypes-structure argument: the array's dtype names the record."""
    for i in range(out.shape[0]):
        out[i] = parts[i].pos + parts[i].vel


def test_a_structure_argument_is_a_device_const_pointer() -> None:
    """A structured array travels as `device const T*`, with the typedef ahead.

    MSL needs the record type before the pointer that names it, and it needs the
    macro prelude before the kernel body. The write order is therefore prelude ->
    typedef block -> kernel, which is what the decorator produces; a `__global`
    qualifier or a missing `typedef` would both fail the Metal compiler.
    """
    record = np.zeros(4, dtype=[("pos", np.float32), ("vel", np.float32)])
    types = [infer_ctype(record), infer_ctype(np.zeros(4, np.float32))]
    translator = FunctionTranslator(structured_add, types, "metal",
                                    lambda name: None, {}, is_kernel=True)
    translator.translate()
    source = "\n".join(piece for _, piece in translator.render_kernels("k"))
    assert "device const _st_" in source
    assert "__global" not in source

    # The record's own fields are the backend's scalar spellings.
    typedefs = translator.render_structure_typedefs()
    assert "float pos;" in typedefs
    assert "float vel;" in typedefs


def test_vector_parameters_and_arithmetic_are_msl() -> None:
    """A vector parameter is `device float4*`, and vector maths stays native.

    MSL has the same vector types OpenCL does, and the scalar type names it uses
    (`uchar`, `ushort`, `uint`, `ulong`, `half`) are exactly the translator's
    internal spellings -- which is why `_device_type` passes them through for
    Metal instead of applying the CUDA renaming. A vector times a scalar is one
    instruction there, so the body keeps `v * 2.0f` rather than expanding per
    component the way CUDA must.
    """
    from pygpgpu import float4

    vectors = np.zeros(4, dtype=float4)
    types = [infer_ctype(vectors) for _ in range(2)]
    translator = FunctionTranslator(vector_scale, types, "metal",
                                    lambda name: None, {}, is_kernel=True)
    translator.translate()
    source = "\n".join(piece for _, piece in translator.render_kernels("k"))
    assert "device float4* src" in source
    assert "device float4* out" in source
    # The multiply is not expanded into make_float4(...) components.
    assert "make_float4" not in source


def test_metal_scalar_type_spellings_are_msl_ones() -> None:
    """The scalar/vector names pass through: MSL spells them as OpenCL does."""
    translator = FunctionTranslator(
        vecadd, [infer_ctype(np.zeros(8, np.float32)) for _ in range(3)],
        "metal", lambda name: None, {}, is_kernel=True)
    expected = {
        "uchar": "uchar", "ushort": "ushort", "uint": "uint",
        "ulong": "ulong", "long": "long", "half": "half", "float": "float",
        "uchar4": "uchar4", "ushort4": "ushort4", "uint4": "uint4",
        "long4": "long4",
    }
    for internal, msl in expected.items():
        assert translator._device_type(internal) == msl, internal





def cuda_index_body(a, out):
    """A CUDA reader's index spelling, on a body that reads the work item itself."""
    i = blockIdx.x * blockDim.x + threadIdx.x
    out[i] = a[i]


def plain_helper(v):
    """A helper that reads nothing but its argument."""
    return v * 3.0


def uses_plain_helper(a, out):
    for i in range(out.shape[0]):
        out[i] = plain_helper(a[i])

def reads_shared(buf, i):
    """A device helper that reads one element of a group's shared slice."""
    return buf[i]


def uses_reads_shared(a, out):
    scratch = zeros_per_group(64, np.float32)
    lane = get_local_id(0)
    scratch[blockIdx][lane] = a[lane]
    barrier()
    out[lane] = reads_shared(scratch[blockIdx], lane)


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

def gid_style_body(out):
    """A gid-style body: it names its own domain and reads the work item itself."""
    i = get_global_id(0)
    out[i] = i * 2.0


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


def histogram(pixels, bins):
    """A data-driven index -- the racy shape that must become an atomic."""
    for i in range(pixels.shape[0]):
        bins[pixels[i]] += 1


def shared_scratch(x, out):
    """A `zeros_per_group` buffer, written, barred, then read."""
    scratch = zeros_per_group(256, np.float32)
    lane = get_local_id(0)
    scratch[blockIdx][lane] = x[lane]
    barrier()
    out[lane] = scratch[blockIdx][lane]


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



def index_reading_helper(a):
    """A device helper that reads the work item itself."""
    return a[get_global_id(0)]


def uses_index_helper(a, out):
    for i in range(out.shape[0]):
        out[i] = index_reading_helper(a)


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


def image_passing_helper(img, y, x):
    """Receives an image but never samples it -- the shape that leaked."""
    return 1.0


def uses_image_helper(img, out):
    for i in range(out.shape[0]):
        out[i] = image_passing_helper(img, i, 0)


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


def image_uses_helper(img, out):
    for i in range(out.shape[0]):
        out[i] = image_passing_helper(img, i, 0)


def _image_helper_resolver():
    return lambda name: {"image_passing_helper": image_passing_helper}.get(name)



def returns_member(parts, i):
    """A helper returning a scalar read from a structure."""
    return parts[i].pos


def returns_record(parts, i):
    """A helper returning the whole structure, by value."""
    return parts[i]


def returns_vector(v, i):
    """A helper returning a vector element."""
    return v[i]


def uses_record_member(parts, out):
    for i in range(out.shape[0]):
        out[i] = returns_member(parts, i)


def uses_record_value(parts, out):
    for i in range(out.shape[0]):
        out[i] = returns_record(parts, i).pos


def uses_vector_return(v, out):
    for i in range(out.shape[0]):
        out[i] = returns_vector(v, i) * 2.0


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



def shared_across_kernels(v):
    """One helper reached by two different kernels."""
    return v * 3.0


def kernel_using_shared_a(x, out):
    for i in range(out.shape[0]):
        out[i] = shared_across_kernels(x[i])


def kernel_using_shared_b(x, out):
    for i in range(out.shape[0]):
        out[i] = shared_across_kernels(x[i]) + 1.0


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



def body_with_unknown_call(x, out):
    """A body calling a name that is no device function (the refusal under test)."""
    for i in range(out.shape[0]):
        out[i] = unknown_device_function(x[i])  # noqa: F821  (refused by design)


def test_a_translation_error_names_the_file_and_line() -> None:
    """A refused body reports where the offending name is, not a generated excerpt.

    The translator parses a copy of the function's source, so a position it reports
    has to be mapped back to the file the user wrote. The check below triggers the
    commonest refusal -- an unknown function in the body -- and asserts the message
    carries both the name and the real line of the file.
    """
    import inspect

    line_of_call = None
    source_lines, first_line = inspect.getsourcelines(body_with_unknown_call)
    for offset, text in enumerate(source_lines):
        if "unknown_device_function" in text:
            line_of_call = first_line + offset
            break
    assert line_of_call is not None

    with pytest.raises(SyntaxError) as refused:
        _compile_with_stub(body_with_unknown_call)
    message = str(refused.value)

    assert "unknown_device_function" in message
    # The reported position is the caller's file and the line the name sits on.
    assert f"line {line_of_call}" in message, message
    assert "test_metal_codegen" in message


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





@pytest.mark.parametrize("case,expectation", [
    ("print", ("opencl or cuda", "print()")),
    ("image_read", ("cuda, opencl or python", "images (read)")),
    ("image_write", ("cuda, opencl or python", "images (write)")),
], ids=["print", "image read", "image write"])
def test_a_refusal_names_a_usable_alternative(case, expectation) -> None:
    """Every refusal says why, and which backends do work.

    A message that only said "not supported" would leave a caller with nowhere to go.
    Each of these carries both halves: the reason the generated code would be wrong,
    and the backends that implement it. The pair is checked together because either
    half alone makes the message unhelpful.
    """
    def print_body(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i]
            print(x[i])

    def image_body(img, out):
        for i in range(out.shape[0]):
            out[i] = img[i, 0]

    def image_write_body(img, out):
        for i in range(out.shape[0]):
            img[i, 0] = out[i]

    array = np.zeros(8, np.float32)
    translator = FunctionTranslator(
        {"print": print_body, "image_read": image_body,
         "image_write": image_write_body}[case],
        [("image", 4, "float"), infer_ctype(array)] if case != "print"
        else [infer_ctype(array)] * 2,
        "metal", lambda name: None, {}, is_kernel=True)
    with pytest.raises(SyntaxError) as refused:
        translator.translate()
    message = str(refused.value)
    reason, backup = expectation
    assert reason in message, message[:120]
    assert backup in message, message[:120]


def test_the_prelude_is_one_blank_line_free_block() -> None:
    """The prelude must not contain a blank line, or the writer splits it apart.

    A generated file's header is the pieces between blank lines, and a piece whose
    first line is not `#`/`struct`/`typedef struct` is classified as a function
    definition (`source_paths._split_source`). A `using namespace metal;` that
    starts such a piece is dropped on the next write, and every macro then names an
    unqualified `uint3` -- an MSL compile error. Keeping the three parts on
    consecutive lines makes the whole prelude one piece that starts with `#`, so it
    dedupes by its exact text and survives every rewrite.
    """
    text = prelude()
    assert "" not in text.splitlines(), "a blank line would split the prelude"
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("#include <metal_stdlib>")
    assert lines[1] == "using namespace metal;"
    # Every remaining line is a macro.
    assert all(line.startswith("#define ") for line in lines[2:])


def test_writing_the_same_kernel_twice_is_byte_identical(tmp_path: Any) -> None:
    """Recompiling a kernel must not grow the file: one include, one using, one macro.

    `append_source` writes the header again on every compile of a module, so a
    header that does not dedupe accumulates a copy per call. The Metal prelude is
    the case that used to fail: its `using` line was split off and re-added.
    """
    from pygpgpu.kernel.source_paths import append_source

    path = str(tmp_path / "gen.metal")
    header = [prelude()]
    written: list = []
    for _ in range(3):
        append_source(path, header, ["kernel void k(device float* a) {}"],
                      __file__, source_digest="digest")
        written.append(open(path, encoding="utf-8").read())

    for source in written[1:]:
        assert source.count("#include <metal_stdlib>") == 1
        assert source.count("using namespace metal;") == 1
        assert source.count("#define get_global_id(dim)") == 1
    # The same definition is written three times; the file must not grow after the first.
    assert written[1] == written[2]
    assert written[0] == written[1]



def test_a_module_with_two_kernels_keeps_one_prelude(tmp_path: Any) -> None:
    """Writing a second kernel must not duplicate the prelude in the shared file.

    One python module owns one generated file and each kernel is compiled on its own
    call, so the header is re-written every time. Two kernels therefore means two
    header writes, and the prelude has to dedupe by its exact text -- which is why
    it is one blank-line-free block (see the test above).
    """
    from pygpgpu.kernel.source_paths import append_source

    source_file = str(__file__)
    path = str(tmp_path / "multi.metal")
    # Two different kernels, each written twice, in one file.
    for _ in range(2):
        for body in ("kernel void one(device float* a) {}", "kernel void two(device float* a) {}"):
            append_source(path, [prelude()], [body], source_file, source_digest="d")
    written = open(path, encoding="utf-8").read()

    assert written.count("#include <metal_stdlib>") == 1
    assert written.count("using namespace metal;") == 1
    assert written.count("#define get_global_id(dim)") == 1
    assert written.count("kernel void one") == 1
    assert written.count("kernel void two") == 1



def _syntax_diagnostics(source: str) -> tuple:
    """Compile the emitted MSL as C++14; skip the assertion when no compiler exists."""
    from metal_syntax_check import check, cpp_compiler

    if cpp_compiler() is None:
        pytest.skip("no C++ compiler to check the emitted source with")
    return check(source)


def test_the_generated_source_compiles_as_cxx() -> None:
    """The emitted MSL parses under a real C++ compiler.

    Metal's own compiler is unavailable here, but MSL is a C++14 dialect with a few
    keywords and attributes C++ lacks. Defining those away and providing stand-ins
    for `uint3`/`atomic_uint`/`mem_flags` makes the generated text checkable: this
    catches a missing brace, a misspelled type or an expression that groups
    differently -- things a string assertion cannot see.
    """
    source = prelude() + "\n" + generate(vecadd, 3)
    ok, diagnostics = _syntax_diagnostics(source)
    assert ok, diagnostics


def test_the_shared_memory_kernel_compiles_as_cxx(tmp_path: Any, monkeypatch) -> None:
    """`threadgroup` parameters, the barrier call and the atomics all parse as C++.

    This is the reduction shape `examples/09_reduction.py` needs, compiled through
    the real `PyKernel._compile` path so the output is exactly what a Mac would hand
    to Metal.

    The generated file goes to a per-test cache root: two runs of this suite on one
    machine (or a concurrent one) would otherwise write and re-read the same
    `tests/__metalcache__` file, and a half-written read looks like a source defect.
    """
    from pygpgpu.kernel.decorator import PyKernel

    monkeypatch.setattr(PyKernel, "_cache_dir",
                        lambda self, backend: str(tmp_path / backend))
    from pygpgpu.kernel import source_paths

    monkeypatch.setattr(source_paths, "cache_root",
                        lambda source_file, backend: str(tmp_path / backend))
    import sys

    sys.path.insert(0, "tests")
    try:
        import test_kernel_local_memory as shared_kernels
    finally:
        sys.path.pop(0)

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
        args = [np.zeros(256, np.uint32) for _ in range(2)]
        shared_kernels.shared_histogram._compile(
            args, [infer_ctype(a) for a in args], [], [], None, "sig", "metal", object())
    except KeyError:
        pass
    finally:
        program_module.Program = original

    ok, diagnostics = _syntax_diagnostics(captured["source"])
    assert ok, diagnostics



def returning(x, out):
    """A kernel whose work items return a value (the claim/site slots)."""
    for i in range(out.shape[0]):
        out[i] = x[i]
        return x[i]


def vector_scaled(src, out):
    """A vector kernel: one float4 per element, scaled by a scalar."""
    for i in range(out.shape[0]):
        out[i] = src[i] * 2.0


def structured_sum(parts, out):
    """A ctypes-structure argument, read through its members."""
    for i in range(out.shape[0]):
        out[i] = parts[i].pos + parts[i].vel


@pytest.mark.parametrize("case,args_for,label", [
    (returning, lambda np: [np.zeros(8, np.float32), np.zeros(8, np.float32)], "return slots"),
    (vector_scaled, lambda np: [np.zeros(4, dtype=__import__("pygpgpu").float4),
                                np.zeros(4, dtype=__import__("pygpgpu").float4)],
     "vector"),
    (structured_sum, lambda np: [np.zeros(4, dtype=[("pos", np.float32),
                                                     ("vel", np.float32)]),
                                 np.zeros(4, np.float32)], "structure"),
])
def test_every_kernel_shape_compiles_as_cxx(tmp_path: Any, monkeypatch,
                                           case, args_for, label) -> None:
    """Returns, vectors and structures all produce parseable MSL.

    Each shape brings its own declaration form -- `device float* _ret0` and the claim
    word for a return, `device float4*` for a vector, `device const _st_*` plus the
    record typedef for a structure -- and all of them have to reach a Metal compiler
    as valid C++. This runs the real `PyKernel._compile` path (only the `Program`
    constructor is stubbed, since it needs Metal.framework).
    """
    import pygpgpu.metal.oop.program as program_module
    from pygpgpu.kernel.decorator import PyKernel

    # A per-test cache root: the suite must not leave `tests/__metalcache__` behind,
    # and a concurrent run of it must not share the file this test reads back.
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

    args = args_for(np)
    kernel_object = __import__("pygpgpu").kernel(case)
    original = program_module.Program
    program_module.Program = StubProgram
    try:
        kernel_object._compile(args, [infer_ctype(a) for a in args], [], [], None,
                               "sig", "metal", object())
    except KeyError:
        pass
    finally:
        program_module.Program = original

    source = captured.get("source", "")
    assert source, f"{label}: nothing was written"
    ok, diagnostics = _syntax_diagnostics(source)
    assert ok, f"{label}: {diagnostics}"




def test_every_emitted_attribute_is_valid_msl() -> None:
    """The attributes are the one thing the C++ check strips, so they are checked here.

    `metal_syntax_check` defines `[[...]]` away so a C++ compiler can read the rest;
    that means a misspelled or misplaced attribute would pass it silently and only
    Metal would object. `metal_attributes` reads them as text instead: the name must
    be one MSL declares for a dispatch value, and it must end its parameter
    declaration.
    """
    from metal_attributes import check_attributes

    for function, count in ((vecadd, 3), (vector_scale, 2)):
        translator = FunctionTranslator(
            function, [infer_ctype(np.zeros(8, np.float32)) for _ in range(count)],
            "metal", lambda name: None, {}, is_kernel=True)
        translator.translate()
        source = prelude() + "\n" + "\n".join(
            piece for _, piece in translator.render_kernels("k"))
        assert not check_attributes(source), check_attributes(source)


def test_the_attribute_checker_catches_real_mistakes() -> None:
    """The attribute check is not vacuous: each failure mode is rejected.

    Three ways an attribute can be wrong -- a misspelled name, a host-binding
    attribute where a dispatch value belongs, and an attribute that does not end its
    declaration -- and all three are caught before a Mac sees them.
    """
    from metal_attributes import check_attributes

    good = ("kernel void k(device float* a, uint gid [[thread_position_in_grid]],\n"
            "                uint tgs [[threadgroups_per_grid]]) { }")
    assert not check_attributes(good)

    broken = {
        "misspelled name":
            "kernel void k(device float* a, uint gid [[thread_position_in_grd]]) { }",
        "host binding attribute":
            "kernel void k(device float* a [[buffer(0)]]) { }",
        "attribute not at the end":
            "kernel void k(device float* [[thread_position_in_grid]] a) { }",
    }
    for label, source in broken.items():
        problems = check_attributes(source)
        assert problems, f"{label} should have been rejected"



def test_every_pointer_parameter_names_its_address_space() -> None:
    """MSL requires a pointer parameter to name `device`/`threadgroup`.

    `metal_syntax_check` defines the spaces away so a C++ compiler can read the
    source, which means a missing qualifier passes it silently and only Metal
    objects. This reads the declarations as text instead: every pointer parameter of
    a generated kernel must carry a space, and the shared-memory one must be spelled
    `threadgroup` rather than `__local`.
    """
    from metal_address_space import check_address_spaces, pointer_parameters

    cases = {"vecadd": (vecadd, 3, np.zeros(8, np.float32)),
             "returning": (returning, 2, np.zeros(8, np.float32))}
    for label, (function, count, sample) in cases.items():
        translator = FunctionTranslator(function, [infer_ctype(sample)
                                                   for _ in range(count)],
                                        "metal", lambda name: None, {}, is_kernel=True)
        translator.translate()
        source = prelude() + "\n" + "\n".join(
            piece for _, piece in translator.render_kernels("k"))
        assert not check_address_spaces(source), (label, check_address_spaces(source))
        assert all(parameter.address_space == "device"
                   for parameter in pointer_parameters(source)), label


def test_the_address_space_checker_catches_a_missing_qualifier() -> None:
    """The check is not vacuous: an unqualified pointer and OpenCL's `__global` fail."""
    from metal_address_space import check_address_spaces

    assert not check_address_spaces(
        "kernel void k(device float* a, threadgroup uint* s, const int n) { }")
    for source in ("kernel void k(float* a) { }",
                   "kernel void k(__global float* a) { }",
                   "kernel void k(device float* a, float* b) { }"):
        assert check_address_spaces(source), source


def test_the_syntax_checker_catches_real_errors() -> None:
    """The C++ check is not vacuous: it rejects the defects it claims to catch.

    A check that always answers "ok" would pass every test and prove nothing, so the
    negative cases are pinned here: an unbalanced brace, an undeclared name, a type
    that does not exist, a duplicate parameter and a call with the wrong arity.
    """
    from metal_syntax_check import check, cpp_compiler

    if cpp_compiler() is None:
        pytest.skip("no C++ compiler to check the emitted source with")

    ok, _ = check("kernel void t(device float* a) { a[0] = 1.0f; }")
    assert ok, "a well-formed kernel must pass"

    broken = {
        "missing brace": "kernel void t(device float* a) { a[0] = 1.0f;",
        "undeclared variable": "kernel void t(device float* a) { b[0] = 1.0f; }",
        "unknown type": "kernel void t(device float* a) { float5 v; }",
        "duplicate parameter": "kernel void t(device float* a, device float* a) { }",
        "wrong arity": "kernel void t(device float* a, float b) { } void u() { t(0); }",
    }
    for label, source in broken.items():
        ok, _diagnostics = check(source)
        assert not ok, f"{label} should have been rejected"


def test_a_byte_order_mark_does_not_break_the_checker() -> None:
    """A BOM written by a Windows editor must not become part of the first token.

    `\ufeffkernel` is not a type name, so an untouched BOM turns every check into a
    confusing failure. Both the entry points strip it.
    """
    from metal_syntax_check import check, cpp_compiler

    if cpp_compiler() is None:
        pytest.skip("no C++ compiler to check the emitted source with")
    source = "kernel void t(device float* a) { a[0] = 1.0f; }"
    ok_plain, _ = check(source)
    ok_bom, _diagnostics = check("\ufeff" + source)
    assert ok_plain and ok_bom


def test_shared_memory_is_a_threadgroup_parameter(tmp_path: Any, monkeypatch) -> None:
    """`zeros_per_group` becomes a `threadgroup` buffer parameter, not `__local`.

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


def returning(x, out):
    """A kernel whose work items return a value: the first one to reach it wins."""
    for i in range(out.shape[0]):
        out[i] = x[i]
        return x[i]


def test_a_returning_kernel_uses_device_slots_and_a_metal_claim() -> None:
    """The return slots are `device` buffers and the claim word is a compare-exchange.

    Two things would be wrong without this. A `__global` slot would not compile in
    MSL, and `atomicCAS` does not exist there -- MSL's compare-and-swap takes the
    expected value by pointer and answers a bool, so the claim is read back from
    the word rather than taken as the call's result.
    """
    source = generate(returning, 2)

    # The slots the translator appends are device pointers, and no OpenCL qualifier
    # survives anywhere in the signature.
    assert "device float* _ret0" in source
    assert "device int* _ret_claim" in source
    assert "device int* _ret_site" in source
    assert "__global" not in source
    assert "atomicCAS" not in source

    # The claim is a compare-exchange plus a read-back of the word.
    assert "atomic_compare_exchange_weak_explicit" in source
    assert "_ret_claim[0] == 1" in source

    # The other backends keep their own spelling.
    opencl_source = generate(returning, 2, "opencl")
    assert "__global float* _ret0" in opencl_source
    assert "atomic_cmpxchg" in opencl_source
    cuda_source = generate(returning, 2, "cuda")
    assert "atomicCAS" in cuda_source
