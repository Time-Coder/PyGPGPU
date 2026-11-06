"""Choosing the device width a python `int` / `float` travels as.

Neither python builtin has a width of its own -- `int` is arbitrary precision and `float`
is a 64-bit double -- so the translator has to pick one. The default is 32 bits, which keeps
every existing kernel's signature unchanged; `@kernel(int=..., float=...)` and the process-
wide `set_type_map()` choose another, and this module pins what each of those does.

Three things are load-bearing and each is checked here:

* the mapping applies to the **python builtins only** -- a numpy scalar names its own width
  and keeps it, so an override cannot quietly re-type what a caller spelled out;
* a python `complex` follows `float=`, so 64-bit real arithmetic is not paired with a
  single-precision complex value in the same body;
* the mapping is part of the kernel's identity -- it is in the generated source's digest and
  in the C name -- so two mappings never share a cache entry or a compiled binary.

A python `int` outside the width it travels as used to **wrap silently** (`2 ** 31 + 5`
reached the device as a negative number). It is now refused with a `ValueError` naming the
numpy spelling that would carry it.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from pygpgpu import get_type_map, kernel, set_type_map
from pygpgpu.kernel.translator import (
    infer_ctype,
    resolve_device_type,
    validate_type_map,
)


# ---------------------------------------------------------------------------
# Type objects -> device type names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling, expected", [
    (np.int64, "long"),
    (np.int32, "int"),
    (np.int16, "short"),
    (np.int8, "char"),
    (np.uint64, "ulong"),
    (np.uint32, "uint"),
    (np.uint16, "ushort"),
    (np.uint8, "uchar"),
    (np.dtype("int64"), "long"),
    (np.dtype("uint8"), "uchar"),
    (np.dtype(np.int16), "short"),
], ids=str)
def test_a_numpy_type_object_resolves_to_a_device_type(spelling, expected):
    assert resolve_device_type(spelling, "int") == expected


@pytest.mark.parametrize("spelling, expected", [
    (np.float32, "float"),
    (np.float64, "double"),
    (np.dtype("float32"), "float"),
    (np.dtype(np.float64), "double"),
], ids=str)
def test_a_numpy_float_type_object_resolves(spelling, expected):
    assert resolve_device_type(spelling, "float") == expected


@pytest.mark.parametrize("spelling, key, expected", [
    (ctypes.c_int8, "int", "char"),
    (ctypes.c_int16, "int", "short"),
    (ctypes.c_int32, "int", "int"),
    (ctypes.c_int64, "int", "long"),
    (ctypes.c_uint8, "int", "uchar"),
    (ctypes.c_uint16, "int", "ushort"),
    (ctypes.c_uint32, "int", "uint"),
    (ctypes.c_uint64, "int", "ulong"),
    (ctypes.c_float, "float", "float"),
    (ctypes.c_double, "float", "double"),
], ids=lambda value: getattr(value, "__name__", str(value)))
def test_a_ctypes_type_object_resolves_to_a_device_type(spelling, key, expected):
    assert resolve_device_type(spelling, key) == expected


def test_a_device_type_name_is_accepted_too():
    assert resolve_device_type("double", "float") == "double"
    assert resolve_device_type("long", "int") == "long"
    assert resolve_device_type("uchar", "int") == "uchar"


def test_the_numpy_spelling_is_not_a_device_type_name():
    """`int8` names a numpy dtype; the device spelling is `char`."""
    with pytest.raises(ValueError):
        resolve_device_type("int8", "int")
    with pytest.raises(ValueError):
        resolve_device_type("float32", "float")


# ---------------------------------------------------------------------------
# The allowed set is closed, and it differs per key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    np.bool_, np.float16, np.float32, np.float64, np.complex64, np.complex128,
    "bool", "half", "float", "double", "c64", 3.5, int, complex,
], ids=["np.bool_", "np.float16", "np.float32", "np.float64", "np.complex64",
        "np.complex128", "bool", "half", "float", "double", "c64", "a-value",
        "python-int", "python-complex"])
def test_int_accepts_only_the_eight_integer_widths(bad):
    """A float, a bool and a complex value are not integer widths."""
    with pytest.raises(ValueError):
        resolve_device_type(bad, "int")


@pytest.mark.parametrize("bad", [
    np.float16, np.int8, np.uint8, np.int32, np.int64, np.complex64,
    ctypes.c_int32, ctypes.c_int64,
    "half", "int", "long", "char", 3,
], ids=["np.float16", "np.int8", "np.uint8", "np.int32", "np.int64",
        "np.complex64", "c_int32", "c_int64", "half", "int", "long", "char",
        "a-value"])
def test_float_accepts_only_float32_and_float64(bad):
    """`half` is out of scope on CUDA/HIP, and an integer is not a float width."""
    with pytest.raises(ValueError):
        resolve_device_type(bad, "float")


@pytest.mark.parametrize("key", ["bool", "double", "int64", "np.int64", "str"])
def test_only_int_and_float_can_be_mapped(key):
    """A numpy scalar already names its width, so mapping any other name is refused."""
    with pytest.raises(ValueError):
        validate_type_map({key: np.int64})


def test_every_refusal_is_a_value_error():
    """A wrong type is a wrong *value* for the argument, never a syntax error."""
    for bad in (np.bool_, np.float16, 3.5, object):
        with pytest.raises(ValueError) as caught:
            validate_type_map({"float": bad})
        assert not isinstance(caught.value, SyntaxError)


# ---------------------------------------------------------------------------
# The mapping decides what a python builtin becomes
# ---------------------------------------------------------------------------


def test_the_default_is_32_bit_then_the_mapping_wins():
    assert infer_ctype(3) == "int"
    assert infer_ctype(3.5) == "float"
    assert infer_ctype(3, {"int": "long"}) == "long"
    assert infer_ctype(3.5, {"float": "double"}) == "double"


def test_a_mapped_float_keeps_its_64_bit_precision_on_the_device(gpu_backend):
    """The whole point: a python float is a double, and the mapping stops narrowing it."""

    @kernel(float=np.float64)
    def keeps_precision(scale, out):
        for i in range(out.shape[0]):
            out[i] = scale

    out = np.zeros(4, dtype=np.float64)
    keeps_precision[gpu_backend](1.0 / 3.0, out)
    # float32(1/3) is 0.33333334; only the double width carries the 17-digit value.
    assert out[0] == 1.0 / 3.0


def test_a_mapped_int_carries_a_64_bit_counter(gpu_backend):
    """A python int beyond 2**31 is refused by default and accepted with `int=np.int64`."""

    @kernel(int=np.int64)
    def keeps_width(step, out):
        for i in range(out.shape[0]):
            out[i] = step + i

    big = 2 ** 40 + 7
    out = np.zeros(4, dtype=np.float64)
    keeps_width[gpu_backend](big, out)
    assert out[0] == float(big)


def test_a_numpy_scalar_keeps_its_own_width_under_a_mapping():
    """The mapping is about the two python builtins, not a second type system."""
    mapping = {"int": "long", "float": "double"}
    assert infer_ctype(np.uint32(7), mapping) == "uint"
    assert infer_ctype(np.int32(7), mapping) == "int"
    assert infer_ctype(np.float32(7.0), mapping) == "float"
    assert infer_ctype(np.zeros(2, dtype=np.float32), mapping) == ("*", "float")


def test_a_python_complex_follows_the_float_mapping():
    """A python complex is chosen with the python float beside it, both ways."""
    # The 32-bit default: a python float is `float`, so a python complex is `c64`.
    assert infer_ctype(1 + 2j) == "c64"
    assert infer_ctype(1 + 2j, {"float": "float"}) == "c64"
    # 64-bit reals must not be paired with a single-precision complex value.
    assert infer_ctype(1 + 2j, {"float": "double"}) == "c128"
    # The numpy spellings still name their own width, mapping or no mapping.
    assert infer_ctype(np.complex64(1 + 2j)) == "c64"
    assert infer_ctype(np.complex128(1 + 2j)) == "c128"
    assert infer_ctype(np.complex64(1 + 2j), {"float": "double"}) == "c64"
    assert infer_ctype(np.complex128(1 + 2j), {"float": "float"}) == "c128"


# ---------------------------------------------------------------------------
# An integer too wide for its type is refused, not wrapped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [2 ** 31, 2 ** 31 + 5, -(2 ** 31) - 1, 2 ** 40],
                         ids=["2**31", "2**31+5", "-(2**31)-1", "2**40"])
def test_a_python_int_too_wide_for_the_mapping_raises(value):
    """It used to wrap silently: `2 ** 31 + 5` reached the device as a negative number."""
    with pytest.raises(ValueError) as caught:
        infer_ctype(value)
    message = str(caught.value)
    assert str(value) in message
    assert "int64" in message or "uint64" in message


def test_the_error_is_a_value_error_not_a_syntax_error():
    """The source is legal; the *value* is what does not fit."""
    with pytest.raises(ValueError) as caught:
        infer_ctype(2 ** 31)
    assert not isinstance(caught.value, SyntaxError)


@pytest.mark.parametrize("value, mapping, expected", [
    (2 ** 31, {"int": "long"}, "long"),
    (2 ** 63 - 1, {"int": "long"}, "long"),
    (-(2 ** 63), {"int": "long"}, "long"),
    (2 ** 32 - 1, {"int": "uint"}, "uint"),
    (2 ** 64 - 1, {"int": "ulong"}, "ulong"),
], ids=["2**31-long", "2**63-1-long", "-2**63-long", "2**32-1-uint", "2**64-1-ulong"])
def test_a_value_that_fits_the_chosen_width_is_accepted(value, mapping, expected):
    assert infer_ctype(value, mapping) == expected


def test_an_integer_beyond_64_bits_has_no_device_type():
    with pytest.raises(ValueError) as caught:
        infer_ctype(2 ** 70, {"int": "ulong"})
    assert "no integer type that wide" in str(caught.value)


# ---------------------------------------------------------------------------
# An unsigned width refuses a negative value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", ["uchar", "ushort", "uint", "ulong"],
                         ids=["uchar", "ushort", "uint", "ulong"])
@pytest.mark.parametrize("value", [-1, -3, -(2 ** 40)], ids=["-1", "-3", "-2**40"])
def test_a_negative_python_int_is_refused_by_an_unsigned_width(device_type, value):
    """An unsigned type cannot hold it, and wrapping would answer a huge positive number."""
    with pytest.raises(ValueError) as caught:
        infer_ctype(value, {"int": device_type})
    message = str(caught.value)
    assert str(value) in message
    assert device_type in message


@pytest.mark.parametrize("device_type", ["uchar", "ushort", "uint", "ulong"],
                         ids=["uchar", "ushort", "uint", "ulong"])
def test_a_non_negative_value_is_accepted_by_an_unsigned_width(device_type):
    """The refusal is about the sign, not about unsigned widths in general."""
    assert infer_ctype(0, {"int": device_type}) == device_type
    assert infer_ctype(7, {"int": device_type}) == device_type


@pytest.mark.parametrize("device_type", ["char", "short", "int", "long"],
                         ids=["char", "short", "int", "long"])
def test_a_signed_width_accepts_a_negative_value(device_type):
    assert infer_ctype(-3, {"int": device_type}) == device_type
    assert infer_ctype(3, {"int": device_type}) == device_type


@pytest.mark.parametrize("device_type, value", [
    ("char", 128), ("char", -129),
    ("uchar", 256),
    ("short", 2 ** 15), ("ushort", 2 ** 16),
    ("int", 2 ** 31), ("uint", 2 ** 32),
], ids=["char-hi", "char-lo", "uchar-hi", "short-hi", "ushort-hi", "int-hi", "uint-hi"])
def test_a_narrow_unsigned_or_signed_width_refuses_out_of_range(device_type, value):
    """Every mapped width enforces its own bounds, not only the 32-bit default."""
    with pytest.raises(ValueError) as caught:
        infer_ctype(value, {"int": device_type})
    assert device_type in str(caught.value)


def test_an_unsigned_mapping_refuses_a_negative_before_the_launch(gpu_backend):
    """The check runs at the call, so the kernel never sees a wrapped value."""

    @kernel(int=np.uint8)
    def unsigned_step(step, out):
        for i in range(out.shape[0]):
            out[i] = step

    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError) as caught:
        unsigned_step[gpu_backend](-5, out)
    assert "uchar" in str(caught.value)


# ---------------------------------------------------------------------------
# The decorator, and the process-wide default
# ---------------------------------------------------------------------------


def test_the_decorator_records_the_mapping():
    @kernel(float=np.float64, int=np.int64)
    def mapped(scale, step, out):
        for i in range(out.shape[0]):
            out[i] = scale + step

    assert mapped._type_map == {"float": "double", "int": "long"}


def test_set_type_map_is_a_snapshot_at_decoration_time():
    """A kernel keeps the mapping it was declared with, even after the default changes."""

    @kernel
    def before(scale, out):
        for i in range(out.shape[0]):
            out[i] = scale

    set_type_map(float=np.float64)

    @kernel
    def after(scale, out):
        for i in range(out.shape[0]):
            out[i] = scale

    assert before._type_map == {}
    assert after._type_map == {"float": "double"}
    set_type_map()


def test_a_decorator_mapping_wins_over_the_global_one():
    set_type_map(int=np.int64, float=np.float64)

    @kernel(float=np.float32)
    def mixed(scale, out):
        for i in range(out.shape[0]):
            out[i] = scale

    assert mixed._type_map == {"int": "long", "float": "float"}
    set_type_map()


def test_the_global_setter_normalizes_its_entries():
    set_type_map(float=np.dtype("float64"), int=ctypes.c_int64)
    assert get_type_map() == {"float": "double", "int": "long"}
    set_type_map()


def test_setting_no_arguments_restores_the_defaults():
    set_type_map(float=np.float64)
    assert get_type_map() == {"float": "double"}
    set_type_map()
    assert get_type_map() == {}


@pytest.fixture(autouse=True)
def reset_global_mapping():
    """Every test starts and ends with the built-in widths, whatever it does in between."""
    set_type_map()
    yield
    set_type_map()
