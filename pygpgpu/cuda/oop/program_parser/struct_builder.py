"""Building the ctypes class a structure argument is carried as: the dtype, the
getters/setters the host reads members through, and the byte layout the launch
uploads. `_apply_structure_pointers` and `_structure_bytes` are called as CLASS
attributes from the backend's kernel module, so they must stay reachable on
ProgramParser -- inheritance provides that."""
from __future__ import annotations
from ctypes import c_void_p, Structure, _Pointer, pointer
from ctypes import Union as ctypes_union
from typing import Tuple, TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    pass

from ...driver import (
    CUInfo
)
from ....kernel import structs as kernel_structs
from ..program_info import StructInfo, VarInfo

import ctypes

import numpy as np
from .... import numpy as gnp

# Prefer the dual-version tree-sitter-cuda37 distribution: it returns a native
# tree_sitter.Language on both the legacy tree-sitter 0.20.x line (Python 3.7)
# and modern 0.21+, whereas the original tree-sitter-cuda's language() yields a
# raw PyCapsule that newer tree-sitter's Parser() rejects. Fall back to it when
# the cuda37 package is not installed.


class _StructBuilderMixin:

    @staticmethod
    def _make_getter(name:str):
        def getter(self):
            return getattr(self, f"_{self.__class__.__name__}_data_{name}")
        
        return getter

    
    @staticmethod
    def _make_setter(name:str):
        def setter(self, value):
            setattr(self, f"_{self.__class__.__name__}_data_{name}", value)

        return setter

    
    @staticmethod
    def _structure_bytes(value)->bytes:
        """The raw bytes of one record value, from any of the shapes a caller passes.

        A `ctypes` instance converts through `bytes()`, which is what backs it. A numpy
        record does not: `bytes(record)` goes through the buffer protocol, and a dtype whose
        members overlap (a union layout) or whose offsets do not ascend is refused there with
        "dtypes with overlapping or out-of-order fields are not representable as buffers".
        `tobytes()` copies the element's raw memory instead, so it answers for every layout.
        """
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, np.void):
            return value.tobytes()
        return bytes(value)


    @staticmethod
    def _apply_structure_pointers(value, cmd_queue):
        if isinstance(value, Structure) and hasattr(value, "apply_pointers"):
            value.apply_pointers(cmd_queue)
            return
        
        if isinstance(value, (list, tuple)):
            for sub_value in value:
                _StructBuilderMixin._apply_structure_pointers(sub_value, cmd_queue)


    @staticmethod
    def _member_numpy_dtype(base_type:object)->object:
        """The numpy dtype describing one member's storage.

        A class-level `dtype` is the answer only when it really is a dtype: a class that
        declares `dtype` as a property (pygf's quaternion and complex records do) hands back
        the descriptor on attribute access, and numpy rejects it. The ctypes class itself
        describes the layout, so that is the fallback.
        """
        candidate:object = getattr(base_type, "dtype", None)
        if isinstance(candidate, np.dtype):
            return candidate
        if isinstance(base_type, type) and issubclass(base_type, ctypes.Structure):
            return np.dtype(base_type)
        return base_type


    @staticmethod
    def _record_dtype(entries:List[object], is_union:bool)->np.dtype:
        """The numpy record for one parsed declaration, laid out by its kind.

        A structure's members follow one another, which is what `np.dtype(entries)` already
        does. A union's do not: they all sit at offset zero and the record is as wide as the
        widest member. `np.dtype` sorts a field list by member name, so the offsets cannot be
        inferred from it -- they have to be stated, which is the only way a hand-built record
        can express a union (see `structs.is_union_dtype`, which reads them back).
        """
        if not is_union:
            return np.dtype(entries)
        names:List[str] = []
        formats:List[object] = []
        widest:int = 0
        for entry in entries:
            name, base = entry[0], np.dtype(entry[1])
            shape:Tuple[int, ...] = entry[2] if len(entry) > 2 else ()
            count:int = 1
            for extent in shape or ():
                count *= int(extent)
            names.append(name)
            formats.append((base, shape) if shape else base)
            widest = max(widest, int(base.itemsize) * count)
        return np.dtype({"names": names, "formats": formats,
                         "offsets": [0] * len(names), "itemsize": widest})


    def _create_struct_type(self, struct_info:StructInfo)->type:
        struct_name:str = struct_info.name
        if struct_name in self._struct_types:
            return self._struct_types[struct_name]
        
        _fields_ = []
        dtype = []
        pointer_types = {}
        for member in struct_info.members.values():
            fixed_class:Optional[type] = kernel_structs.fixed_member_class(
                member.base_type_str)
            if fixed_class is not None:
                # A quaternion or a complex value: its declaration lives in a fixed header,
                # so it is not one of the parsed structures and has no entry in the type
                # tables -- the class the header's layout describes is the member's type.
                base_type = fixed_class
            elif member.base_type_str in CUInfo.basic_types:
                base_type = CUInfo.basic_types[member.base_type_str]
            elif member.base_type_str in self._struct_infos:
                # A structure the source declared wins over a vector of the same
                # layout: the dtypes are equal, but the alignments are not, and a
                # padded structure no longer matches the host's packed bytes.
                base_type = self._create_struct_type(
                    self._struct_infos[member.base_type_str])
            elif member.base_type_str in CUInfo.vec_types:
                # A vector member is a type of its own, not a registered structure: its
                # numpy dtype describes it for the host-side field list.
                base_type = CUInfo.vec_types[member.base_type_str]
            else:
                base_type = self._create_struct_type(self._struct_infos[member.base_type_str])

            member_type = base_type
            for n_elements in reversed(member.array_shape):
                member_type *= n_elements

            if member.is_ptr:
                _fields_.append((f"_{struct_info.name}_ptr_{member.name}", c_void_p))
                dtype.append((member.name, np.uint64))
                pointer_types[member.name] = member_type
            else:
                _fields_.append((member.name, member_type))
                # A vector member contributes its structured dtype (a ctypes class has no
                # numpy dtype of its own), so the record layout matches the C structure.
                dtype.append((member.name, self._member_numpy_dtype(base_type),
                              member.array_shape))

        dtype = self._record_dtype(dtype, struct_info.is_union)

        def __init__(self, *args, **kwargs):
            for field in self._fields_:
                if issubclass(field[1], _Pointer):
                    ori_name:str = field[0].replace(f"_{self.__class__.__name__}_ptr_", "")
                    setattr(self, f"_{self.__class__.__name__}_data_{ori_name}", None)

            used_kwargs = {}
            for i in range(len(args)):
                used_kwargs[_fields_[i][0]] = args[i]

            used_kwargs.update(kwargs)
            keys = list(used_kwargs.keys())
            for key in keys:
                if self._members[key].is_ptr:
                    setattr(self, f"_{self.__class__.__name__}_data_{key}", used_kwargs[key])
                    del used_kwargs[key]

            # `ctypes.Structure.__init__` only accepts a Structure instance, so a class
            # built on the Union base has to store the members through attribute sets --
            # one member per keyword, which is exactly what the structure branch does
            # through its constructor.
            if issubclass(type(self), Structure):
                Structure.__init__(self, **used_kwargs)
            else:
                for key, item in used_kwargs.items():
                    setattr(self, key, item)

        def apply_pointers(self, cmd_queue)->List[Dict[str, object]]:
            result = []
            for field in self._fields_:
                if issubclass(field[1], _Pointer):
                    ori_name:str = field[0].replace(f"_{self.__class__.__name__}_ptr_", "")
                    data_name:str = f"_{self.__class__.__name__}_data_{ori_name}"
                    data = getattr(self, data_name)
                    var_info:VarInfo = self._members[ori_name]
                    content_type = self._pointer_types[ori_name]
                    
                    if isinstance(data, np.ndarray):
                        used_value = data
                        if data.dtype != content_type:
                            used_value = data.astype(content_type)
                    else:
                        _StructBuilderMixin._apply_structure_pointers(data, cmd_queue)
                        used_value = np.array(data, dtype=content_type)

                    if not used_value.flags['C_CONTIGUOUS']:
                        used_value = np.ascontiguousarray(used_value)

                    buffer, event = var_info.use_buffer(cmd_queue, used_value)
                    if var_info.need_read_back and not isinstance(data, gnp.ndarray):
                        result.append({
                            "event": event,
                            "arg_info": var_info,
                            "value": data,
                            "mem_obj": buffer
                        })
                    else:
                        result.append({
                            "event": event
                        })

                    setattr(self, f"_{self.__class__.__name__}_ptr_{ori_name}", pointer(buffer.id))
                    
                if issubclass(field[1], Structure) and hasattr(field[1], "apply_pointers"):
                    result.extend(getattr(self, field[0]).apply_pointers(cmd_queue))

            return result

        type_body = {
            '__module__': self._module_path,
            "_fields_": _fields_,
            "dtype": dtype,
            "_members": struct_info.members,
            "_pointer_types": pointer_types,

            "apply_pointers": apply_pointers,
            "__init__": __init__,
            "__repr__": lambda self: f"{struct_name}({', '.join([member.name + '=' + str(getattr(self, member.name)) for member in self._members.values()])})"
        }

        for field in _fields_:
            if issubclass(field[1], _Pointer):
                ori_name:str = field[0].replace(f"_{self.__class__.__name__}_ptr_", "")
                type_body[ori_name] = property(self._make_getter(ori_name), self._make_setter(ori_name))

        # A union gives every member offset zero, and a type() over a Structure base
        # would lay them out one after another -- the device type would then be twice
        # as wide as the value the caller hands over. The dtype already says which it
        # is (see structs.is_union_dtype), so the base follows it.
        from ....kernel.structs import is_union_dtype  # noqa: PLC0415 (import cycle)

        base_type = ctypes_union if is_union_dtype(dtype) else Structure
        struct_type = type(struct_name, (base_type,), type_body)
        self._struct_types[struct_name] = struct_type

        return struct_type
