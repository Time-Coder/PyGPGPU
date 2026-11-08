"""The compiles tests, split out of `test_metal_codegen.py`.

The subject is compiles.
"""
from __future__ import annotations

from _helpers.test_metal_codegen_helpers import (
    structured_sum,
    vector_scaled,
    _syntax_diagnostics,
    generate,
    returning,
    vecadd,
    vector_scale,
)

from typing import Any
import numpy as np
import pytest
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.metal_codegen import (
    prelude,
)
from pygpgpu.kernel.translator import FunctionTranslator


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
