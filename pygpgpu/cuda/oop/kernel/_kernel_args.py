# Binding one argument: buffers, images, structures, scalars, vectors.
from __future__ import annotations
from ._kernel_support import complex_of_dtype, _handle_types
from ctypes import c_uint64, pointer, Structure, c_void_p
from ctypes import Union as ctypes_union
from typing import Dict, Tuple, Optional
import numpy as np
from .... import numpy as gnp
from ...driver import (
    CUInfo,
    cudaTextureObject_t,
    cudaSurfaceObject_t
)
from ..program_info import ArgInfo
from ..stream import Stream
from ..mem_object import MemObject
from ..program_parser import ProgramParser


class _KernelArgsMixin:

    def _process_arg(self, stream:Stream, arg_info:ArgInfo, value:object, bound:Dict[str, object],
                     seen_arrays:Optional[Dict[int, Tuple[MemObject, object]]]=None)->object:
        arg_type_str = arg_info.type_str        
        base_type_str = arg_info.base_type_str

        # A backend-neutral Image was bound before compilation (see
        # PyKernel._bind_images); pick the view this argument's type needs.
        from ....image import Image as _Image
        if isinstance(value, _Image):
            # Only CLAMP is portable: the device samplers read the edge texel for wrap and
            # mirror under texel-space coordinates, so refuse rather than launch a kernel
            # whose numbers would not match the python backend.
            from ....kernel.decorator import _check_image_address
            _check_image_address(value, "cuda")
            array, texture, surface = value.bind_cuda()
            written = (base_type_str == "cudaSurfaceObject_t") or arg_info.need_read_back
            handle = surface if written else texture
            handle_value = c_uint64(handle.id.value if hasattr(handle.id, "value")
                                    else handle.id)
            # Note which backend touched the image last, so Image.read() reads it back
            # from the right device.
            value._last_backend = "cuda"
            bound[arg_info.name] = (None, value, (array, texture, surface, handle_value))
            return pointer(handle_value)

        # A cudaTextureObject_t is a host-side description with no context of its own; the
        # kernel binds it here, where the context is known, and keeps the resulting
        # objects alive for the launch.
        if isinstance(value, (cudaTextureObject_t, cudaSurfaceObject_t)):
            value = self._bind_image(value)

        # A texture (or surface) handle is passed by value as an opaque 64-bit
        # object, which the kernel samples through tex2D. cuLaunchKernel expects a
        # pointer to each argument's value, so the handle is stored on the call's
        # binding dict (keeping it alive for the launch) and its address returned.
        if isinstance(value, _handle_types()):
            handle = c_uint64(value.id.value if hasattr(value.id, "value") else value.id)
            bound[arg_info.name] = (None, value, handle)
            return pointer(handle)
        if base_type_str in CUInfo.basic_types or base_type_str in self._program.structs or base_type_str in CUInfo.vec_types:
            if base_type_str in CUInfo.basic_types:
                content_type = CUInfo.basic_types[base_type_str]
                is_struct = False
            elif base_type_str in CUInfo.vec_types:
                content_type = CUInfo.vec_types[base_type_str]
                is_struct = False
            else:
                content_type = self._program.structs[arg_info.base_type_str]
                is_struct = True

            is_vec3:bool = (len(getattr(content_type, "_fields_", ())) == 3 and base_type_str in CUInfo.vec_types)

            if arg_info.is_ptr and is_struct and isinstance(
                    value, (Structure, ctypes_union, np.void)):
                # A pointer-to-structure parameter: the caller hands over the structure
                # value itself, and the launch needs a device address. Allocate a buffer
                # for it, upload the bytes, and pass that pointer; a body that writes the
                # members reads the buffer back like any written array.
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
                    # CUDA passes 3-component vectors in a compact layout (no 4th-component
                    # padding), and cuLaunchKernel wants the address of the argument's value,
                    # so the packed element is a ctypes buffer rather than an array.
                    packed:object = self._pack_vec3_value(value, content_type)
                    return pointer(packed)

                if is_struct and isinstance(value, (np.void, bytes, bytearray,
                                                    Structure, ctypes_union)):
                    # A numpy record holds exactly the bytes the dtype described, which is the
                    # layout the parsed structure has, so it can back the ctypes value directly.
                    # A ctypes.Structure instance is the caller's own class, which is a
                    # different type from the one the parser generated for the same layout,
                    # so it goes through its bytes as well.
                    value = content_type.from_buffer_copy(bytes(value))
                elif is_struct and isinstance(value, (complex, np.complexfloating)):
                    # A complex scalar is a two-real value, which is exactly the record the
                    # structure declares: the value is converted through a one-element array
                    # of that layout, never through the structure's own constructor -- the
                    # constructor assigns each member by name and would narrow the complex
                    # into its first real field, silently dropping the imaginary part.
                    # `np.complexfloating` is checked as well as `complex`: a numpy
                    # `complex64` is *not* a python `complex`, which is why the plain
                    # isinstance test missed it and the narrowing went unnoticed.
                    # The value is cast to the record's own width first: a python complex is
                    # a complex128, and reinterpreting its sixteen bytes as an eight-byte
                    # record is not a conversion -- numpy refuses to view a 0-d array at a
                    # different itemsize. So a `c64` parameter receives the `float` pair a
                    # python complex narrows to, exactly as a python float reaches a `float`.
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
                    value = content_type.from_buffer_copy(ProgramParser._structure_bytes(value))

                if isinstance(value, content_type):
                    used_value = value
                else:
                    success = True
                    try:
                        if base_type_str in CUInfo.alter_types:
                            used_value = content_type(CUInfo.alter_types[base_type_str][1](value))
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
                    # CUDA types like float3* use a compact layout (12B elements etc.) while the host
                    # structured array is 16B (with a padding field) -> pack/unpack conversion
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

                # A gnp array owns its device copy and its layout handling: it uploads
                # through `_to_device`, which stages a non-contiguous view itself.
                # Converting it to a plain ndarray here would orphan that device copy,
                # and the kernel's writes would never reach the caller.
                if not isinstance(used_value, gnp.ndarray) and \
                        not used_value.flags['C_CONTIGUOUS']:
                    used_value = np.ascontiguousarray(used_value)

                if isinstance(used_value, gnp.ndarray) and seen_arrays is not None:
                    root_key = id(used_value._root())
                    if root_key in seen_arrays:
                        # The same array travels under two names (`k(b, b)`): one buffer backs
                        # both, so the body reads and writes the one location it names.
                        shared_buffer, _ = seen_arrays[root_key]
                        bound[arg_info.name] = (shared_buffer, value, used_value)
                        return pointer(c_void_p(shared_buffer.id.value))
                    buffer, event = arg_info.use_buffer(stream, used_value)
                    seen_arrays[root_key] = (buffer, used_value)
                else:
                    buffer, event = arg_info.use_buffer(stream, used_value)
                bound[arg_info.name] = (buffer, value, used_value)
                return pointer(c_void_p(buffer.id.value))
        else:
            raise NotImplementedError(f"arg type {arg_type_str} is legal, but not implemented.")
