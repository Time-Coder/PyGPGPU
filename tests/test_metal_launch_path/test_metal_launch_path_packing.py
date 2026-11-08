"""The packing tests, split out of `test_metal_launch_path.py`.

The subject is packing.
"""
from __future__ import annotations

import pytest


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
