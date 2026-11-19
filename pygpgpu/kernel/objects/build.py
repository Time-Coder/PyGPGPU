# The recursive builder: a synthesized structure carrying a plain python object's
# values, field by field, and the record-field fill that lands them without touching
# the array constructor the vector registry hooks.
from __future__ import annotations

import ctypes
from typing import Dict

from .binding import _complex_member_value, _is_complex_member_class


def _fill_record_field(target:object, value:object)->None:
    """One record field (a structured sub-array view) from a ctypes structure, recursively.

    A whole-object assignment onto the view reaches numpy's array constructor -- which
    the vector registry hooks (pygf owns the np.asarray name process-wide) and that
    hook answers "could not convert string to float" for a synthesized record. The
    field-by-field fill touches no hook, and a nested structure field recurses the
    same way, because assigning a whole `ObjectLeaf` onto `mid`'s sub-field view is
    the same hooked constructor again.
    """
    for field_name, field_kind in type(value)._fields_:
        source_value:object = getattr(value, field_name)
        if isinstance(source_value, ctypes.Structure):
            _fill_record_field(target[field_name], source_value)
        elif isinstance(source_value, ctypes.Array):
            for index in range(len(source_value)):
                _fill_record_field_element(target[field_name][index],
                                           source_value[index])
        else:
            target[field_name] = source_value


def _fill_record_field_element(target:object, value:object)->None:
    """One array element: a scalar stores directly, a structure recurses."""
    if isinstance(value, ctypes.Structure):
        _fill_record_field(target, value)
    else:
        target = value


def _build_nested(source:object, structure:type)->object:
    """An instance of ``structure`` carrying ``source``'s values, recursively.

    The walk answers one question per field of the synthesized layout: the caller's
    object may not carry that name at all (a vector member's alignment slot) -- the pad
    keeps the zero it was declared with -- and a field whose type is a synthesized
    *structure* while the caller's value is a plain python object recurses, because
    ctypes' keyword constructor accepts a plain object for a scalar field but refuses
    one for a structure field ("expected ObjectX instance, got Leaf" -- a two-level
    plain-object chain, `bag.mid.leaf.x`, is exactly that shape).
    """
    values:Dict[str, object] = {}
    for member, kind in structure._fields_:
        if not hasattr(source, member):
            continue
        value:object = getattr(source, member)
        if _is_complex_member_class(kind) and not isinstance(value, kind):
            # `ctypes` has no complex field, so the value is split into the record's
            # two parts at the class's own width.
            value = _complex_member_value(value, kind)
        elif (isinstance(kind, type) and issubclass(kind, ctypes.Structure)
                and not isinstance(value, kind)):
            value = _build_nested(value, kind)
        elif isinstance(value, list) and isinstance(kind, type) \
                and issubclass(kind, ctypes.Array):
            # A list member: the field is a fixed-length array of the element
            # structure, filled from the caller's list in order.
            filled:object = kind()
            element_kind:object = kind._type_
            for position, item in enumerate(value):
                if isinstance(element_kind, type) \
                        and issubclass(element_kind, ctypes.Structure) \
                        and not isinstance(item, element_kind):
                    item = _build_nested(item, element_kind)
                filled[position] = item
            value = filled
        values[member] = value
    return structure(**values)
