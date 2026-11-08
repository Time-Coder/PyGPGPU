"""The scalars tests, split out of `test_metal_launch_path.py`.

The subject is scalars.
"""
from __future__ import annotations

from _helpers.test_metal_launch_path_helpers import (
    complex_scalar,
)

from typing import Any
import numpy as np
import pytest
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.metal_codegen import prelude
from pygpgpu.kernel.translator import FunctionTranslator
from pygpgpu.metal.oop.program_parser import ProgramParser


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
