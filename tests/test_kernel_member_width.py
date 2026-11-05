"""A python value's width under a mapping, wherever it sits.

Three places choose a device width for a python `int`/`float`, and they must agree:

* a **structure member** the scan synthesizes from an object (`obj.n`), including a nested
  one -- the width follows `int=`/`float=`, and a value the chosen width cannot hold is
  refused instead of wrapping;
* an **imported handwritten kernel's declared scalar** (`uchar step`), which has its own
  width in the source and must refuse a value outside it -- as a `ValueError`, not the
  `OverflowError` numpy 2 raises from a conversion probe;
* the **declared `ctypes` member** of a structure the caller wrote, which keeps its own
  width and is not affected by the mapping at all.

Two structural facts are also pinned, because they caused real silent corruption:

* the C spelling of a structure field follows the backend. `long` is 32-bit in CUDA C on
  Windows, so an `int64` member has to be declared `long long`; spelled `long`, the device
  read only the low four bytes and `0x1122334455` came back `0x22334455`;
* the mapping reaches a *member* through the same table a parameter does.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.structs import field_type
from pygpgpu.kernel.translator import (
    check_integer_range,
    ctypes_device_type_name,
)


# ---------------------------------------------------------------------------
# The shared helpers the three paths now use
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls, expected", [
    (ctypes.c_bool, "bool"),
    (ctypes.c_int8, "char"), (ctypes.c_int16, "short"),
    (ctypes.c_int32, "int"), (ctypes.c_int64, "long"),
    (ctypes.c_uint8, "uchar"), (ctypes.c_uint16, "ushort"),
    (ctypes.c_uint32, "uint"), (ctypes.c_uint64, "ulong"),
    (ctypes.c_float, "float"), (ctypes.c_double, "double"),
], ids=lambda value: getattr(value, "__name__", str(value)))
def test_a_ctypes_scalar_class_names_its_device_type(cls, expected):
    assert ctypes_device_type_name(cls) == expected


@pytest.mark.parametrize("other", [ctypes.c_void_p, ctypes.c_char_p, int, None, 3],
                         ids=["c_void_p", "c_char_p", "int", "None", "a-value"])
def test_something_that_is_not_a_scalar_class_names_nothing(other):
    assert ctypes_device_type_name(other) is None


@pytest.mark.parametrize("value, device_type", [
    (255, "uchar"), (0, "uchar"),
    (127, "char"), (-128, "char"),
    (2 ** 63 - 1, "long"), (-(2 ** 63), "long"),
    (2 ** 64 - 1, "ulong"),
], ids=["uchar-255", "uchar-0", "char-max", "char-min", "long-max",
        "long-min", "ulong-max"])
def test_the_shared_check_accepts_a_value_that_fits(value, device_type):
    check_integer_range(value, device_type)


@pytest.mark.parametrize("value, device_type", [
    (256, "uchar"), (-1, "uchar"), (200, "char"), (-129, "char"),
    (2 ** 63, "long"), (2 ** 32, "uint"),
], ids=["uchar-256", "uchar-neg", "char-200", "char-neg", "long-too-big", "uint-big"])
def test_the_shared_check_refuses_a_value_that_does_not_fit(value, device_type):
    with pytest.raises(ValueError):
        check_integer_range(value, device_type)


def test_the_shared_check_names_the_member_it_was_given():
    with pytest.raises(ValueError) as caught:
        check_integer_range(256, "uchar", "step")
    assert str(caught.value).startswith("step:")


# ---------------------------------------------------------------------------
# The C spelling of a structure field follows the backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend, expected", [
    ("cuda", "long long"), ("hip", "long long"), ("opencl", "long"),
], ids=["cuda", "hip", "opencl"])
def test_an_int64_field_is_declared_in_the_backends_64_bit_spelling(backend, expected):
    """`long` is 32-bit in CUDA C on Windows; a member spelled that way loses the high half."""
    base, suffix = field_type(np.dtype("int64"), backend)
    assert (base, suffix) == (expected, "")


@pytest.mark.parametrize("backend, expected", [
    ("cuda", "unsigned long long"), ("hip", "unsigned long long"), ("opencl", "ulong"),
], ids=["cuda", "hip", "opencl"])
def test_a_uint64_field_keeps_its_width_too(backend, expected):
    base, _ = field_type(np.dtype("uint64"), backend)
    assert base == expected


def test_a_40_bit_member_survives_a_round_trip(gpu_backend):
    """The regression this fix exists for: the top byte used to be silently dropped."""
    value = 0x1122334455

    class Rec(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int64)]

    @kernel
    def read_member(rec, out):
        for i in range(out.shape[0]):
            out[i] = rec.n

    out = np.zeros(2, dtype=np.float64)
    read_member[gpu_backend](Rec(n=value), out)
    assert int(out[0]) == value


# ---------------------------------------------------------------------------
# An object member follows the mapping
# ---------------------------------------------------------------------------


class Holder:
    def __init__(self, n: int, v: float) -> None:
        self.n = n
        self.v = v


class Nested:
    def __init__(self, inner: object) -> None:
        self.inner = inner


def test_an_object_int_member_follows_the_mapping(gpu_backend):
    """`o.n = 2 ** 31` under `int=np.int64` used to wrap: the member was fixed at 32 bits."""

    @kernel(int=np.int64)
    def read_member(o, out):
        for i in range(out.shape[0]):
            out[i] = o.n

    out = np.zeros(2, dtype=np.float64)
    read_member[gpu_backend](Holder(2 ** 31, 0.0), out)
    assert out[0] == 2147483648.0


def test_an_object_float_member_follows_the_mapping(gpu_backend):
    @kernel(float=np.float64)
    def read_member(o, out):
        for i in range(out.shape[0]):
            out[i] = o.v

    out = np.zeros(2, dtype=np.float64)
    read_member[gpu_backend](Holder(0, 1.0 / 3.0), out)
    assert out[0] == 1.0 / 3.0


def test_a_nested_object_member_follows_the_mapping(gpu_backend):
    @kernel(int=np.int64)
    def read_member(o, out):
        for i in range(out.shape[0]):
            out[i] = o.inner.n

    out = np.zeros(2, dtype=np.float64)
    read_member[gpu_backend](Nested(Holder(2 ** 31, 0.0)), out)
    assert out[0] == 2147483648.0


def test_an_object_member_refuses_a_value_the_chosen_width_cannot_hold(gpu_backend):
    """The default 32-bit width must refuse `2 ** 31` rather than wrap it."""

    @kernel
    def read_member(o, out):
        for i in range(out.shape[0]):
            out[i] = o.n

    out = np.zeros(2, dtype=np.float64)
    with pytest.raises(ValueError) as caught:
        read_member[gpu_backend](Holder(2 ** 31, 0.0), out)
    assert "n:" in str(caught.value)


def test_a_small_object_member_is_still_accepted(gpu_backend):
    @kernel
    def read_member(o, out):
        for i in range(out.shape[0]):
            out[i] = o.n

    out = np.zeros(2, dtype=np.float64)
    read_member[gpu_backend](Holder(5, 1.5), out)
    assert out[0] == 5.0


# ---------------------------------------------------------------------------
# A declared ctypes member keeps its own width
# ---------------------------------------------------------------------------


def test_a_declared_member_ignores_the_mapping(gpu_backend):
    """The caller wrote `c_int64` down, so that is the width -- not the kernel's `int=`."""

    class Wide(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int64)]

    @kernel(int=np.uint8)          # a mapping that would be far too narrow for the value
    def read_member(rec, out):
        for i in range(out.shape[0]):
            out[i] = rec.n

    out = np.zeros(2, dtype=np.float64)
    read_member[gpu_backend](Wide(n=2 ** 40), out)
    assert out[0] == float(2 ** 40)


# ---------------------------------------------------------------------------
# An imported handwritten kernel's declared scalar
# ---------------------------------------------------------------------------


NARROW_SOURCE = """
__kernel void narrow(const uchar step, __global float* out, const int n)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    out[i] = (float)step;
}
"""


@pytest.fixture
def narrow_kernel(tmp_path, monkeypatch):
    """Import a handwritten `.cl` declaring a `uchar` scalar."""
    from pygpgpu import kernel_import

    (tmp_path / "narrow_width.cl").write_text(NARROW_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    kernel_import.install()
    import importlib

    importlib.invalidate_caches()
    module = importlib.import_module("narrow_width")
    yield module.narrow
    sys_modules_cleanup("narrow_width")


def sys_modules_cleanup(name: str) -> None:
    import sys

    sys.modules.pop(name, None)


@pytest.mark.parametrize("value", [3, 255, 0], ids=["3", "255", "0"])
def test_a_handwritten_scalar_accepts_a_value_that_fits(narrow_kernel, value):
    out = np.zeros(4, dtype=np.float32)
    narrow_kernel["opencl"](value, out, np.int32(4))
    assert out[0] == float(value)


@pytest.mark.parametrize("value", [256, 300, -1], ids=["256", "300", "-1"])
def test_a_handwritten_scalar_refuses_a_value_outside_its_declared_width(
        narrow_kernel, value):
    """It used to wrap to 0/255, or raise numpy's `OverflowError` from a conversion probe."""
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError) as caught:
        narrow_kernel["opencl"](value, out, np.int32(4))
    assert "uchar" in str(caught.value)
    assert not isinstance(caught.value, OverflowError)
