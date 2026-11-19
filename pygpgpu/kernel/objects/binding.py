# The object-binding record a launch keeps, and how a member reads back.
from __future__ import annotations
import ctypes
from typing import Dict, List, Tuple
import numpy as np



class ObjectBinding:
    """The pairing a write-back needs: the caller's object and the instance that replaced it.

    written_arg: the path.member descriptions the body assigns to; values[path] is
    the caller object the writes land back on, and instances[path] the structure that
    travelled to the device.

    array_members: the ndarray members an object argument carries that travel as
    their own kernel parameters (`_<param>_<member>`, the same shape the receiver's
    `_self_<attr>` arguments spell) -- the launch passes the caller's array itself,
    so a store through it lands in the caller's buffer and needs no write-back. The
    translator reads this map to bind `item.arr` to that parameter, and the launch
    reads it to order the arguments.
    """

    def __init__(self)->None:
        self.written:List[str] = []
        self.values:Dict[str, object] = {}
        self.instances:Dict[str, object] = {}
        self.array_members:Dict[str, List[Tuple[str, "np.ndarray"]]] = {}

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
            if isinstance(instance, np.void):
                # A NESTED route (`bag.inner.x`): the write-back walks the record's own
                # fields -- `instances['bag.inner']` is the `inner` field of the record
                # the launch carried, a structured scalar the device's store landed in.
                # Its members read like a record's (`['x']`), not like attributes.
                try:
                    value:object = instance[member]
                except (IndexError, ValueError, KeyError):
                    continue
                setattr(source, member, _member_home_value(value))
                continue
            try:
                value:object = getattr(instance, member)
            except AttributeError:
                continue
            setattr(source, member, _member_home_value(value))

    def record_nested_routes(self, root_name:str, root_value:object,
                             record_element:object, written:List[str])->None:
        """Register the intermediate objects a nested write route reaches through.

        A description like `bag.inner.x` names the caller-side objects the write-back
        has to reach (`bag.inner` is an Inner) and the *record* side the device's
        stores landed in (the record's `inner` field). Each hop of each description is
        registered here, so `write_back` can resolve any depth without re-deriving it:
        the caller side drills the object graph with `getattr`, the record side drills
        the same hops through the record's own fields -- a dotted name is not a field
        of any single level, so each hop indexes one level down.
        """
        for description in written:
            segments:List[str] = description.split(".")
            if segments[0] != root_name or len(segments) < 2:
                continue
            hop_path:str = root_name
            hop_value:object = root_value
            record_field:object = record_element
            for hop in segments[1:-1]:
                hop_path:str = f"{hop_path}.{hop}"
                hop_value:object = getattr(hop_value, hop, None)
                if hop_value is None:
                    break
                record_field:object = record_field[hop]
                self.values.setdefault(hop_path, hop_value)
                self.instances.setdefault(hop_path, record_field)



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
