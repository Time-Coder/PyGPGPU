# Binding one argument: buffers, images, structures, scalars, vectors.
from __future__ import annotations
from ._kernel_support import complex_of_dtype
from ctypes import pointer, Structure, c_void_p
from ctypes import Union as ctypes_union
from typing import Dict
import numpy as np
from .... import numpy as gnp
from ...driver import (
    HIPInfo
)
from ..program_info import ArgInfo
from ..stream import Stream
from ..program_parser import ProgramParser


class _KernelArgsMixin:

    def _process_arg(self, stream:Stream, arg_info:ArgInfo, value:object, bound:Dict[str, object])->object:
        arg_type_str = arg_info.type_str        
        arg_type_str = arg_info.type_str
        base_type_str = arg_info.base_type_str

        if base_type_str in HIPInfo.basic_types or base_type_str in self._program.structs or base_type_str in HIPInfo.vec_types:
            if base_type_str in HIPInfo.basic_types:
                content_type = HIPInfo.basic_types[base_type_str]
                is_struct = False
            elif base_type_str in HIPInfo.vec_types:
                content_type = HIPInfo.vec_types[base_type_str]
                is_struct = False
            else:
                content_type = self._program.structs[arg_info.base_type_str]
                is_struct = True

            is_vec3:bool = (len(getattr(content_type, "_fields_", ())) == 3 and base_type_str in HIPInfo.vec_types)

            if arg_info.is_ptr and is_struct and isinstance(
                    value, (Structure, ctypes_union, np.void)):
                # A pointer-to-structure parameter: the caller hands over the structure
                # value itself, and the launch needs a device address. Allocate a buffer,
                # upload the bytes, and pass that pointer; a body that writes the members
                # reads the buffer back like any written array.
                # The bytes come through `_structure_bytes`, because `bytes(record)` is
                # refused for a dtype whose members overlap -- which is exactly a union.
                packed:object = value if isinstance(value, np.void) else \
                    content_type.from_buffer_copy(ProgramParser._structure_bytes(value))
                data:np.ndarray = np.frombuffer(ProgramParser._structure_bytes(packed),
                                                dtype=np.uint8)
                buffer = stream.context.get_buffer(data.nbytes)
                buffer.kernel_flags = arg_info.kernel_flags
                event = buffer.set_data(stream, data)
                bound[arg_info.name] = (buffer, value, packed)
                return pointer(c_void_p(buffer.id.value))

            if not arg_info.is_ptr:
                if is_vec3:
                    # HIP like CUDA: 3-component vectors are passed in a compact layout, and
                    # the launch wants the address of the value, so it must be a ctypes buffer.
                    packed:object = self._pack_vec3_value(value, content_type)
                    return pointer(packed)

                if is_struct and isinstance(value, (np.void, bytes, bytearray, Structure, ctypes_union)):
                    # A numpy record holds exactly the bytes the dtype described, which is the
                    # layout the parsed structure has, so it can back the ctypes value directly.
                    # A ctypes instance is the caller's own class, which is a
                    # different type from the one the parser generated for the same layout,
                    # so it goes through its bytes as well -- through `_structure_bytes`,
                    # because `bytes(record)` is refused for an overlapping (union) dtype.
                    value = content_type.from_buffer_copy(
                        ProgramParser._structure_bytes(value))
                elif is_struct and isinstance(value, (complex, np.complexfloating)):
                    # A complex scalar is a two-real value, which is exactly the record the
                    # structure declares: the value is converted through that layout, never
                    # through the structure's own constructor -- the constructor assigns each
                    # member by name and would narrow the complex into its first real field,
                    # silently dropping the imaginary part. The value is cast to the record's
                    # own width first: a python complex is a complex128, and reinterpreting
                    # its sixteen bytes as an eight-byte record is not a conversion -- numpy
                    # refuses to view a 0-d array at a different itemsize. So a `c64`
                    # parameter receives the `float` pair a python complex narrows to,
                    # exactly as a python float reaches a `float` parameter.
                    # The cast is `astype`, not an `asarray(..., dtype=...)` call: pygf owns
                    # the `np.asarray` name process-wide (it adds a view for vector dtypes),
                    # and a conversion must not depend on that wrapper forwarding the dtype.
                    element:object = complex_of_dtype(np.dtype(content_type))
                    if element is None:
                        raise TypeError(
                            f"type of argument '{arg_info.name}' need"
                            f" {content_type.__name__}, got {value.__class__.__name__}: a"
                            f" complex value is not a {content_type.__name__}")
                    value = np.asarray(value).astype(element, copy=False)
                    value = value.view(np.dtype(content_type))[()]
                    value = content_type.from_buffer_copy(
                        ProgramParser._structure_bytes(value))

                if isinstance(value, content_type):
                    used_value = value
                else:
                    success = True
                    try:
                        if base_type_str in HIPInfo.alter_types:
                            used_value = content_type(HIPInfo.alter_types[base_type_str][1](value))
                        else:
                            used_value = content_type(value)
                    except BaseException:
                        success = False

                    if not success:
                        raise TypeError(f"type of argument '{arg_info.name}' need {content_type.__name__}, got {value.__class__.__name__}, and cannot convert {value.__class__.__name__} to {content_type.__name__}")

                if is_struct:
                    used_value.apply_pointers(stream)

                return pointer(used_value)
            else:
                if is_vec3:
                    # HIP like CUDA: types such as float3* use a compact layout (12B elements) while the
                    # host structured array is 16B (with a padding field) -> pack/unpack conversion
                    used_value:object = self._pack_vec3(value, content_type)
                elif isinstance(value, np.ndarray):
                    used_value = value
                    if not isinstance(value, gnp.ndarray) and value.dtype != content_type:
                        # A complex structure binds a complex array: the bytes of the two
                        # layouts are identical, so the dtype view is the whole conversion.
                        complex_dtype:object = complex_of_dtype(np.dtype(content_type))
                        if complex_dtype is not None and value.dtype == complex_dtype:
                            used_value = value
                        else:
                            used_value = value.astype(content_type)
                else:
                    ProgramParser._apply_structure_pointers(value, stream)
                    used_value = np.array(value, dtype=content_type)

                # A gnp array owns its device copy and its layout handling (see the CUDA
                # backend); converting it here would orphan that copy.
                if not isinstance(used_value, gnp.ndarray) and \
                        not used_value.flags['C_CONTIGUOUS']:
                    used_value = np.ascontiguousarray(used_value)

                buffer, event = arg_info.use_buffer(stream, used_value)
                bound[arg_info.name] = (buffer, value, used_value)
                return pointer(c_void_p(buffer.id.value))
        else:
            raise NotImplementedError(f"arg type {arg_type_str} is legal, but not implemented.")
