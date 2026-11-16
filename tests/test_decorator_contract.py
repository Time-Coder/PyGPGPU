"""Bugs in the decorator's own contract: brackets, device selectors, return slots.

These are the parts a caller touches directly, so each is reachable without a device:

* ``k[None]`` states "automatic work size" -- the same thing ``k[(None,)]`` and
  ``k[(4, None)]`` state, and what the parser's own error message recommends -- yet the
  plain spelling raised a `TypeError` that told the caller to use it;
* a device selector of the form ``name:ordinal`` is documented to disambiguate among
  matches ("``name:ordinal`` disambiguates"), and the ordinal was dropped whenever the name
  part was a plain index, so ``"0:1"`` and ``"0:0"`` resolved to the same device;
* a structure value in a return slot was looked up in the vector table *before* the
  "is this a record" test, and the table is keyed on the dtype's string form -- so a user
  structure whose fields happen to spell a vector layout came back as a ``pygf`` vector and
  ``result["x"]`` stopped working.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu.kernel.decorator.devices import _parse_work_item, _resolve_device
from pygpgpu.kernel.decorator.slots import _return_value


# ---------------------------------------------------------------------------
# ``k[None]`` -- the automatic work size
# ---------------------------------------------------------------------------


def test_a_bare_none_bracket_means_automatic():
    """``k[None]`` is documented as the automatic work size and must parse as one.

    ``(None,)`` and ``(4, None)`` already parse; only the plain spelling fell through to
    the parser's own `raise`, whose message tells the caller to spell it this way.
    """
    globals_size, local_size = _parse_work_item(None)
    assert globals_size is None
    assert local_size is None


@pytest.mark.parametrize("item", [(None,), (4, None)])
def test_the_other_automatic_spellings_still_parse(item):
    """The spellings that already worked keep working, so the fix is not a rewrite."""
    globals_size, local_size = _parse_work_item(item)
    assert globals_size is None or isinstance(globals_size, tuple)


# ---------------------------------------------------------------------------
# ``name:ordinal`` on a numeric device selector
# ---------------------------------------------------------------------------


def test_a_numeric_selector_keeps_its_ordinal(gpu_backend):
    """``"0:1"`` must name match #1, not match #0.

    With one device the ordinal is out of range, so the honest answers are "device 1" (when
    it exists) or a refusal -- never silently device 0. The test asserts the two spellings
    do not resolve to the same index, which is what "the ordinal was dropped" produced.
    """
    devices = __import__("pygpgpu.kernel.decorator.devices", fromlist=["devices"])
    count = len(devices._backend_devices(gpu_backend))
    if count < 2:
        with pytest.raises(ValueError):
            _resolve_device(gpu_backend, "0:1")
        return
    first, _ = _resolve_device(gpu_backend, "0:0")
    second, _ = _resolve_device(gpu_backend, "0:1")
    assert first is not second


# ---------------------------------------------------------------------------
# A structure return slot that spells a vector layout
# ---------------------------------------------------------------------------


def test_a_structure_return_slot_is_not_mistaken_for_a_vector():
    """A user structure whose layout coincides with a vector is still the user's record.

    The vector lookup was keyed on the dtype's string form and ran before the "is this a
    record" test. A structure's layout genuinely *can* be a vector's layout -- `{float x;
    float y;}` is `float2` byte for byte -- so the two are told apart by alignment, which is
    what `structs.is_record`/`is_declared_structure` already do: `np.dtype(ctypes.Structure)`
    is an *aligned* record while the shipped vector dtypes are not. A plain
    `np.dtype([('x','<f4'),('y','<f4')])` is **not** aligned, i.e. it *is* the vector
    spelling, so the record has to be declared the way a real structure is: `align=True`.
    """
    dtype = np.dtype([("x", "<f4"), ("y", "<f4")], align=True)
    slot = np.zeros(1, dtype=dtype)
    slot[0] = (1.25, 2.5)

    value = _return_value(slot[0], dtype, "struct")

    assert isinstance(value, np.void), f"a record came back as {type(value).__name__}"
    assert value["x"] == pytest.approx(1.25)
    assert value["y"] == pytest.approx(2.5)


def test_a_differently_named_record_still_comes_back_as_a_record():
    """The control: a layout that does not collide with a vector already worked."""
    dtype = np.dtype([("i", "<i4"), ("j", "<i4")])
    slot = np.zeros(1, dtype=dtype)
    slot[0] = (3, 4)

    value = _return_value(slot[0], dtype, "struct")

    assert isinstance(value, np.void)
    assert value["i"] == 3
    assert value["j"] == 4


def test_a_real_vector_slot_still_comes_back_as_a_vector():
    """The control the fix must not break: a genuine vector slot is still a vector.

    A structure-only rule ("always return the record") would answer a `numpy.void` where
    the body returned a `float2`, so both directions are pinned.
    """
    from pygf import float2

    dtype = np.dtype(float2.numpy_dtype())
    slot = np.zeros(1, dtype=dtype)
    slot[0] = (1.5, -2.5)

    value = _return_value(slot[0], dtype, "float2")

    assert isinstance(value, float2), f"a vector came back as {type(value).__name__}"
    assert value.x == pytest.approx(1.5)
    assert value.y == pytest.approx(-2.5)
