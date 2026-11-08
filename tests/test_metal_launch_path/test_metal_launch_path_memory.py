"""The memory tests, split out of `test_metal_launch_path.py`.

The subject is memory.
"""
from __future__ import annotations

import numpy as np
import pytest


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


    with pytest.raises(OSError) as caught:
        metal_kernel._finish_buffer(BrokenBuffer(), np.zeros(4, np.uint8), True)
    assert "device lost" in str(caught.value)
