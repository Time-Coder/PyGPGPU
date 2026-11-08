# Binding one argument: buffers, images, structures, scalars, vectors.
from __future__ import annotations
from ._kernel_support import complex_of_dtype, _element_dtype
from ctypes import pointer, sizeof, Structure
from ctypes import Union as ctypes_union
from typing import Dict, Union
import numpy as np
from ...driver import (
    CL,
    CLInfo,
    cl_kernel_arg_address_qualifier,
    cl_mem,
    sampler_t,
    queue_t,
    cl_kernel_arg_type_qualifier
)
from ..command_queue import CommandQueue
from ..pipe import pipe
from ..program_parser import ProgramParser
from .... import numpy as gnp


class _KernelArgsMixin:

    def _set_arg(self, cmd_queue:CommandQueue, index_or_name:Union[int, str], value:object, bound:Dict[str, object]):
        if isinstance(index_or_name, int):
            arg_info = self._arg_list[index_or_name]
            index = index_or_name
        else:
            arg_info = self._args[index_or_name]
            index = self._arg_names.index(index_or_name)

        arg_type_str = arg_info.type_str        
        arg_type_str = arg_info.type_str
        type_qualifiers = arg_info.type_qualifiers
        base_type_str = arg_info.base_type_str

        if type_qualifiers & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_PIPE:
            arg_type = CLInfo.basic_types[arg_type_str]
            
            if not isinstance(value, pipe):
                raise TypeError(f"type of argument '{arg_info.name}' must be pipe<{arg_type.__name__}>, got {value.__class__.__name__}.")
            
            if value.packet_type is not None and value.packet_type != arg_type:
                raise TypeError(f"type of argument '{arg_info.name}' must be pipe<{arg_type.__name__}>, got pipe<{value.packet_type.__name__}>.")

            if value.packet_type is None:
                value.packet_type = arg_type

            used_value = self.context.get_pipe(value)
            CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
        elif arg_type_str == "queue_t":
            if not isinstance(value, queue_t):
                raise TypeError(f"type of argument '{arg_info.name}' must be queue_t, got {value.__class__.__name__}.")
            
            used_value = self.context.get_queue(value)
            CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
        elif arg_type_str == "sampler_t":
            if not isinstance(value, sampler_t):
                raise TypeError(f"type of argument '{arg_info.name}' must be sampler_t, got {value.__class__.__name__}.")
            
            used_value = self.context.get_sampler(value)
            CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
        elif base_type_str in CLInfo.basic_types or base_type_str in self._program.structs or base_type_str in CLInfo.vec_types:
            if base_type_str in CLInfo.basic_types:
                content_type = CLInfo.basic_types[base_type_str]
                is_struct = False
            elif base_type_str in CLInfo.vec_types:
                content_type = CLInfo.vec_types[base_type_str]
                is_struct = False
            else:
                content_type = self._program.structs[arg_info.base_type_str]
                is_struct = True

            if not arg_info.is_ptr:
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
                    # silently dropping the imaginary part. `np.complexfloating` is checked as
                    # well as `complex`: a numpy `complex64` is *not* a python `complex`.
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
                    value = content_type.from_buffer_copy(
                        ProgramParser._structure_bytes(value))
                if isinstance(value, content_type):
                    used_value = value
                else:
                    success = True
                    try:
                        if base_type_str in CLInfo.alter_types:
                            used_value = content_type(CLInfo.alter_types[base_type_str][1](value))
                        else:
                            used_value = content_type(value)
                    except BaseException:
                        success = False

                    if not success:
                        raise TypeError(f"type of argument '{arg_info.name}' need {content_type.__name__}, got {value.__class__.__name__}, and cannot convert {value.__class__.__name__} to {content_type.__name__}")

                if is_struct:
                    bound.setdefault("__member_readbacks__", []).extend(
                        used_value.apply_pointers(cmd_queue, arg_info.member_infos or None))

                CL.clSetKernelArg(self.id, index, sizeof(used_value), pointer(used_value))
            else:
                if arg_info.address_qualifier == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_LOCAL:
                    if isinstance(value, int):
                        bytes_per_group = value
                    elif isinstance(value, np.ndarray):
                        bytes_per_group = value.size
                    elif isinstance(value, (list, tuple)):
                        bytes_per_group = np.array(value).size

                    bytes_per_group *= sizeof(content_type)                    
                    CL.clSetKernelArg(self.id, index, bytes_per_group, None)
                    bound[arg_info.name] = (None, value, None)
                else:
                    if is_struct and isinstance(value, (Structure, ctypes_union, np.void)):
                        # A pointer-to-structure parameter: the caller hands over the
                        # structure value itself, and the launch needs a device address.
                        # Allocate a buffer, upload the bytes, and pass that pointer; a
                        # body that writes the members reads the buffer back like any
                        # written array.
                        packed:object = (value if isinstance(value, np.void)
                                      else content_type.from_buffer_copy(
                                          ProgramParser._structure_bytes(value)))
                        data:np.ndarray = np.frombuffer(
                            ProgramParser._structure_bytes(packed), dtype=np.uint8)
                        buffer, event = arg_info.use_buffer(cmd_queue, data)
                        CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(buffer.id))
                        bound[arg_info.name] = (buffer, value, packed)
                    elif isinstance(value, np.ndarray):
                        used_value = value
                        content_dtype = (getattr(content_type, "dtype", None)
                                         if is_struct else None)
                        if isinstance(value, gnp.ndarray):
                            # A gnp array owns its device copy and its layout handling: it
                            # uploads through `_to_device`, which stages a non-contiguous
                            # view itself. Converting it here would hand the launch a
                            # *plain* ndarray copy -- the device copy would be orphaned
                            # and the kernel's writes would never reach the caller.
                            if content_dtype is not None and value.dtype != content_dtype:
                                used_value = value.astype(content_dtype)
                        else:
                            if content_dtype is not None:
                                if value.dtype != content_dtype:
                                    # A complex structure binds a complex array: the two
                                    # layouts are byte-identical, so nothing is converted --
                                    # the dtype view is the whole of it.
                                    complex_dtype:object = complex_of_dtype(content_dtype)
                                    if complex_dtype is not None:
                                        used_value = np.ascontiguousarray(value).view(
                                            complex_dtype)
                                    else:
                                        used_value = np.ascontiguousarray(value).view(
                                            content_dtype)
                            elif value.dtype != content_type:
                                used_value = value.astype(_element_dtype(content_type))

                            if not used_value.flags['C_CONTIGUOUS']:
                                used_value = np.ascontiguousarray(used_value)

                        buffer, event = arg_info.use_buffer(cmd_queue, used_value)
                        CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(buffer.id))
                        bound[arg_info.name] = (buffer, value, used_value)
                    else:
                        ProgramParser._apply_structure_pointers(value, cmd_queue)
                        used_value = np.array(value, dtype=content_type)

                        if not used_value.flags['C_CONTIGUOUS']:
                            used_value = np.ascontiguousarray(used_value)

                        buffer, event = arg_info.use_buffer(cmd_queue, used_value)
                        CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(buffer.id))
                        bound[arg_info.name] = (buffer, value, used_value)
        elif arg_type_str in CLInfo.image_types:
            image_type = CLInfo.image_types[arg_type_str]
            # A backend-neutral Image becomes the descriptor this backend binds, with
            # its channel count and element type taken from the image itself.
            from ....image import Image as _Image
            original_image:object = value if isinstance(value, _Image) else None
            if original_image is not None:
                # Only CLAMP is portable, and the sampler is built during translation, so
                # this repeats the check to fail with the same message if that path was
                # skipped (a cached kernel, for instance).
                from ....kernel.decorator import _check_image_address
                _check_image_address(original_image, "opencl")
                value = image_type(
                    (original_image.data if original_image.data is not None
                     else np.zeros(original_image.shape, original_image.dtype)),
                    dtype=original_image.dtype)
            if not isinstance(value, (np.ndarray, image_type)):
                raise TypeError(f"type of argument '{arg_info.name}' must be {image_type.__name__}, np.ndarray or Image, got {value.__class__.__name__}.")

            used_value = value
            if isinstance(value, np.ndarray):
                used_value = image_type(value)

            image, event = arg_info.use_image(cmd_queue, used_value)
            CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(image.id))
            bound[arg_info.name] = (image, value, used_value)
            if original_image is not None:
                # Record where the result lives so Image.read() can fetch it, and note
                # which backend wrote it last.
                original_image._bindings["opencl"] = (image, cmd_queue)
                original_image._last_backend = "opencl"
        else:
            raise NotImplementedError(f"arg type {arg_type_str} is legal, but not implemented.")
