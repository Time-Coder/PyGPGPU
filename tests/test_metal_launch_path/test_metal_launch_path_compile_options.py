"""The compile_options tests, split out of `test_metal_launch_path.py`.

The subject is compile_options.
"""
from __future__ import annotations

import pytest


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
