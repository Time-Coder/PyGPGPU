"""Building the ctypes class a Metal structure argument is carried as.

MSL's structures are C++ records with MSL's type names in them, so the layout rules
are the ones the OpenCL builder applies (a padded three-component vector is
16-byte aligned, a structure takes its strictest member's alignment). The type
names, however, are MSL's -- which is why the base type lookup goes through
`ProgramParser._device_type_name`, the one place that maps an MSL spelling onto the
table key the driver packages use.

The result is what `kernel_import` publishes as a module attribute and what
`ArgInfo.check_type` compares an argument's dtype against, which is why this cannot
be left empty the way the declaration scanner left it: an empty table answered
"cannot tell" for every structure parameter, and a wrong dtype then failed on the
device instead of at the call.
"""
from __future__ import annotations

import ctypes
from ctypes import Structure, c_uint64
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ....kernel import structs as kernel_structs
from ..program_info import StructInfo
from ....cparser.struct_layout import align_struct

if TYPE_CHECKING:
    pass


# MSL's fixed-width spellings of the scalar types, mapped onto the names the
# package's driver tables and the rest of its type machinery are keyed by. MSL
# follows C++'s standard spellings, so `unsigned int` and `uint32_t` both mean the
# same `uint` a kernel parameter is declared with.
_SCALAR_ALIASES: Dict[str, str] = {
    "unsigned char": "uchar",
    "unsigned short": "ushort",
    "unsigned int": "uint",
    "unsigned long": "ulong",
    "unsigned long long": "ulong",
    "long long": "long",
    "int8_t": "char",
    "uint8_t": "uchar",
    "int16_t": "short",
    "uint16_t": "ushort",
    "int32_t": "int",
    "uint32_t": "uint",
    "int64_t": "long",
    "uint64_t": "ulong",
    "size_t": "ulong",
    "ptrdiff_t": "long",
    "bfloat": "half",
}


class _MetalStructBuilderMixin:

    @staticmethod
    def _make_getter(name: str):
        def getter(self):
            return getattr(self, f"_{self.__class__.__name__}_data_{name}")
        return getter

    @staticmethod
    def _make_setter(name: str):
        def setter(self, value):
            setattr(self, f"_{self.__class__.__name__}_data_{name}", value)
        return setter

    @staticmethod
    def _structure_bytes(value) -> bytes:
        """The raw bytes of one record value, from any of the shapes a caller passes."""
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, np.void):
            return value.tobytes()
        return bytes(value)

    @staticmethod
    def _apply_structure_pointers(value, cmd_queue) -> None:
        if isinstance(value, Structure) and hasattr(value, "apply_pointers"):
            value.apply_pointers(cmd_queue)
            return
        if isinstance(value, (list, tuple)):
            for sub_value in value:
                _MetalStructBuilderMixin._apply_structure_pointers(sub_value, cmd_queue)

    @staticmethod
    def _member_numpy_dtype(base_type: object) -> object:
        """The numpy dtype describing one member's storage."""
        candidate: object = getattr(base_type, "dtype", None)
        if isinstance(candidate, np.dtype):
            return candidate
        if isinstance(base_type, type) and issubclass(base_type, ctypes.Structure):
            return np.dtype(base_type)
        return base_type

    def _device_type_name(self, spelling: str) -> str:
        """One MSL type spelling reduced to the name the driver tables use."""
        normalized: str = " ".join(spelling.split())
        if normalized in _SCALAR_ALIASES:
            return _SCALAR_ALIASES[normalized]
        return normalized

    def _member_base_type(self, spelling: str) -> object:
        """The ctypes class one member's declared type denotes."""
        name: str = self._device_type_name(spelling)
        fixed: Optional[type] = kernel_structs.fixed_member_class(name)
        if fixed is not None:
            # A quaternion or a complex value: its declaration lives in a fixed header,
            # so the class the header describes is the member's type.
            return fixed
        # The vector tables are the shared ones `kernel_import` derives for Metal: a
        # vector is a layout of its components and does not depend on the device, so
        # only the *names* and widths are MSL's (2/3/4, never 8/16).
        from ....kernel_import.tables import _metal_scalar_tables

        vectors: Dict[str, type] = _metal_scalar_tables()["vec"]
        if name in vectors:
            return vectors[name]
        if name in ("float", "double", "half", "bool",
                    "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"):
            return _scalar_class(name)
        if name in self._struct_infos:
            return self._create_struct_type(self._struct_infos[name])
        raise ValueError(
            f"the structure member type {spelling!r} is not one MSL declares in a "
            f"record, and the source declares no struct by that name")

    def _create_struct_type(self, struct_info: StructInfo) -> type:
        struct_name: str = struct_info.name
        if struct_name in self._struct_types:
            return self._struct_types[struct_name]

        fields: List[object] = []
        spec: List[object] = []
        pointer_types: Dict[str, object] = {}
        for member in struct_info.members.values():
            base_type = self._member_base_type(member.base_type_str)
            member_type = base_type
            for n_elements in reversed(member.array_shape):
                member_type *= n_elements

            if member.is_ptr:
                fields.append((f"_{struct_name}_ptr_{member.name}", c_uint64))
                spec.append((member.name, np.uint64))
                pointer_types[member.name] = member_type
            else:
                fields.append((member.name, member_type))
                spec.append((member.name, self._member_numpy_dtype(base_type),
                             member.array_shape))

        fields, dtype = align_struct(fields, spec, is_union=struct_info.is_union)
        ptr_prefix: str = f"_{struct_name}_ptr_"

        def __init__(self, *args, **kwargs):
            for field in self._fields_:
                if field[0].startswith(ptr_prefix):
                    ori_name: str = field[0].replace(ptr_prefix, "")
                    setattr(self, f"_{self.__class__.__name__}_data_{ori_name}", None)

            used_kwargs = {}
            for index in range(len(args)):
                used_kwargs[fields[index][0]] = args[index]
            used_kwargs.update(kwargs)
            for key in list(used_kwargs.keys()):
                if self._members[key].is_ptr:
                    setattr(self, f"_{self.__class__.__name__}_data_{key}", used_kwargs.pop(key))

            Structure.__init__(self, **used_kwargs)

        def apply_pointers(self, cmd_queue, member_infos: Optional[Dict[str, object]] = None):
            """Metal has no structure-pointer binding yet: it refuses rather than guesses."""
            raise NotImplementedError(
                "a Metal structure member declared as a pointer is not supported: the "
                "device reads it as an 8-byte address the host cannot hand over. Pass "
                "the array as its own kernel argument instead")

        type_body = {
            '__module__': __name__,
            "_fields_": fields,
            "dtype": dtype,
            "_members": struct_info.members,
            "_pointer_types": pointer_types,
            "apply_pointers": apply_pointers,
            "__init__": __init__,
            "__repr__": lambda self: f"{struct_name}(" + ", ".join(
                f"{member.name}={getattr(self, member.name)}"
                for member in self._members.values()) + ")",
        }
        for field in fields:
            if field[0].startswith(ptr_prefix):
                ori_name = field[0].replace(ptr_prefix, "")
                type_body[ori_name] = property(self._make_getter(ori_name),
                                               self._make_setter(ori_name))

        struct_type = type(struct_name, (Structure,), type_body)
        self._struct_types[struct_name] = struct_type
        return struct_type


# The ctypes classes MSL's scalar type names denote. MSL defines them all, including
# the fixed-width spellings, so an imported source may declare any of them.
_SCALAR_CLASSES: Dict[str, type] = {
    "bool": ctypes.c_bool,
    "char": ctypes.c_int8,
    "uchar": ctypes.c_uint8,
    "short": ctypes.c_int16,
    "ushort": ctypes.c_uint16,
    "int": ctypes.c_int32,
    "uint": ctypes.c_uint32,
    "long": ctypes.c_int64,
    "ulong": ctypes.c_uint64,
    "half": ctypes.c_uint16,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
}


def _scalar_class(name: str) -> type:
    return _SCALAR_CLASSES[name]


__all__ = ["_MetalStructBuilderMixin"]

