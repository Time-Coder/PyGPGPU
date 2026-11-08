# The object-binding record a launch keeps, and how a member reads back.
from __future__ import annotations
import ctypes
from typing import Dict, List
import numpy as np



class ObjectBinding:
    """The pairing a write-back needs: the caller's object and the instance that replaced it.

    written_arg: the path.member descriptions the body assigns to; values[path] is
    the caller object the writes land back on, and instances[path] the structure that
    travelled to the device.
    """

    def __init__(self)->None:
        self.written:List[str] = []
        self.values:Dict[str, object] = {}
        self.instances:Dict[str, object] = {}

    def write_back(self)->None:
        """Copy the final member values home, for every object the body wrote to."""
        for description in self.written:
            path, member = description.rsplit('.', 1)
            instance:object = self.instances.get(path)
            source:object = self.values.get(path)
            if instance is None or source is None:
                continue
            if isinstance(instance, np.ndarray):
                # The object travelled as a one-element record array; the device wrote its
                # member stores there, and the host read the array back after the launch.
                try:
                    value:object = instance[0][member]
                except (IndexError, ValueError):
                    continue
                setattr(source, member, _member_home_value(value))
                continue
            try:
                value:object = getattr(instance, member)
            except AttributeError:
                continue
            setattr(source, member, _member_home_value(value))



def _complex_member_value(value:object, kind:type)->ctypes.Structure:
    """One complex value as the record `ctypes` can store for it."""
    from ..complex import to_member  # noqa: PLC0415 (import cycle)

    return to_member(value, kind)



def _is_complex_member_class(kind:object)->bool:
    """True when a member's ctypes class stores a complex value's two parts."""
    from ..complex import Complex128Record, Complex64Record  # noqa: PLC0415 (import cycle)

    return kind is Complex64Record or kind is Complex128Record



def _member_home_value(value:object)->object:
    """One member read back from the device, in the form the caller's object holds.

    A complex member comes back either as the record `ctypes` stored or as the numpy record
    the array view produced, and the caller's object holds a plain `complex`; any other
    numpy scalar comes back as a 0-d value, so it is unwrapped the way it always was.
    """
    from ..complex import (Complex64Record, Complex128Record,  # noqa: PLC0415 (import cycle)
                          is_complex_record, member_value)

    if isinstance(value, (Complex64Record, Complex128Record)):
        return member_value(value)
    if isinstance(value, np.void) and is_complex_record(value.dtype):
        return complex(value["re"], value["im"])
    return value.item() if hasattr(value, "item") else value
