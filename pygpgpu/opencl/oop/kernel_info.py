from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Optional, Dict, Set, List, Union
from collections import defaultdict

import numpy as np

from ..runtime import (
    cl_kernel_arg_address_qualifier,
    cl_kernel_arg_access_qualifier,
    cl_kernel_arg_type_qualifier,
    cl_mem_flags,
    cl_context,
    CLInfo,
    imagend_t,
    CL
)

if TYPE_CHECKING:
    from .buffer import Buffer
    from .command_queue import CommandQueue

from .mem_object import MemObject
from .event import Event
from .imagend import imagend
from .pipe import pipe


class VarInfo:

    def __init__(self, name: str, type_str: str, array_shape: Tuple[int, ...], address_qualifier: cl_kernel_arg_address_qualifier, access_qualifier: cl_kernel_arg_access_qualifier, type_qualifiers: cl_kernel_arg_type_qualifier):
        self.name = name
        self.type_str = type_str
        self.array_shape = array_shape
        self.address_qualifier = address_qualifier
        self.access_qualifier = access_qualifier
        self.type_qualifiers = type_qualifiers

    @property
    def is_ptr(self)->bool:
        return (self.type_str[-1] == "*")
    
    @property
    def base_type_str(self)->str:
        return (self.type_str[:-1] if self.is_ptr else self.type_str)
    
    @property
    def type_annotation(self)->str:
        base_type_str = self.base_type_str
        if self.type_qualifiers & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_PIPE:
            return 'pipe'
        elif self.is_ptr:
            dtype_name = base_type_str
            if base_type_str in CLInfo.scalar_types:
                dtype_name = "np." + np.dtype(CLInfo.scalar_types[base_type_str]).name
            return f"NDArray[{dtype_name}]"
        else:
            if base_type_str in ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong']:
                return 'int'
            elif base_type_str in ['float', 'double']:
                return 'float'
            else:
                return base_type_str
    
    @property
    def readonly(self)->bool:
        return (
            self.address_qualifier == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_CONSTANT or
            self.access_qualifier == cl_kernel_arg_access_qualifier.CL_KERNEL_ARG_ACCESS_READ_ONLY or
            self.type_qualifiers & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_CONST
        )
    
    @property
    def writeonly(self)->bool:
        return self.access_qualifier == cl_kernel_arg_access_qualifier.CL_KERNEL_ARG_ACCESS_WRITE_ONLY
    
    @property
    def need_read_back(self)->bool:
        return ((
            self.is_ptr and
            self.address_qualifier == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_GLOBAL and
            self.access_qualifier != cl_kernel_arg_access_qualifier.CL_KERNEL_ARG_ACCESS_READ_ONLY and
            not (self.type_qualifiers & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_CONST)
        ) or (
            self.type_str in CLInfo.image_types and self.access_qualifier != cl_kernel_arg_access_qualifier.CL_KERNEL_ARG_ACCESS_READ_ONLY
        ))
    

class ArgInfo(VarInfo):

    def __init__(self, parent:KernelInfo, name: str, type_str: str, array_shape:Tuple[int, ...], address_qualifier: cl_kernel_arg_address_qualifier, access_qualifier: cl_kernel_arg_access_qualifier, type_qualifiers: cl_kernel_arg_type_qualifier):
        VarInfo.__init__(self, name, type_str, array_shape, address_qualifier, access_qualifier, type_qualifiers)
        self.parent = parent
        self.value: Any = None
        self.mem_obj: Optional[MemObject] = None

        self.__buffers: Dict[Tuple[cl_context, int, cl_mem_flags], List[Buffer]] = defaultdict(list)
        self.__images: Dict[Tuple[cl_context, Tuple[int, ...], type], List[imagend]] = defaultdict(list)
        self.__busy_mems: Set[MemObject] = set()
    
    def check_type(self, value:Any)->None:
        base_type_str = self.base_type_str
        if self.type_qualifiers & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_PIPE:
            if not isinstance(value, pipe):
                raise TypeError(f"{self.parent.signature()}'s argument '{self.name}' must be pipe<{base_type_str}>, got {value.__class__.__name__}.")
            
            if value.packet_type.__name__ != base_type_str:
                raise TypeError(f"{self.parent.signature()}'s argument '{self.name}' must be pipe<{base_type_str}>, got pipe<{value.packet_type.__name__}>.")
        elif self.is_ptr:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{self.parent.signature()}'s argument '{self.name}' must be np.ndarray, got {value.__class__.__name__}.")
            
            dtype = None
            dtype_name = base_type_str
            if base_type_str in CLInfo.scalar_types:
                dtype = np.dtype(CLInfo.scalar_types[base_type_str])
                dtype_name = "np." + dtype.name
            elif base_type_str in CLInfo.vec_types:
                dtype = CLInfo.vec_types[base_type_str]

            if dtype is not None:
                if value.dtype != dtype:
                    raise TypeError(f"{self.parent.signature()}'s argument '{self.name}' must be NDArray[{dtype_name}], got NDArray[np.{value.dtype.name}].")
        else:
            base_type = None
            if base_type_str in CLInfo.basic_types:
                base_type = CLInfo.basic_types[base_type_str]

            if base_type_str in CLInfo.alter_types:
                base_type = CLInfo.alter_types[base_type_str]

            same_type = (value.__class__.__name__ == base_type_str if base_type is None else isinstance(value, base_type))

            if not same_type:
                raise TypeError(f"{self.parent.signature()}'s argument '{self.name}' must be {base_type_str}, got {value.__class__.__name__}.")

    def use_buffer(self, cmd_queue:CommandQueue, data:np.ndarray)->Tuple[Buffer, Event]:
        if self.readonly:
            flags = cl_mem_flags.CL_MEM_READ_ONLY
        else:
            flags = cl_mem_flags.CL_MEM_READ_WRITE

        buffer_key = (cmd_queue.context.id.value, data.nbytes, flags)

        used_buffer = None
        for buffer in self.__buffers[buffer_key]:
            if buffer not in self.__busy_mems:
                used_buffer = buffer
                break

        if used_buffer is None:
            used_buffer = cmd_queue.context.create_buffer(size=data.nbytes, flags=flags)
            self.__buffers[buffer_key].append(used_buffer)

        event = used_buffer.set_data(cmd_queue, data)
        if event is not None and CL.print_info:
            print(f"copy data to device for argument '{self.name}'")

        self.__busy_mems.add(used_buffer)
        
        return used_buffer, event
    
    def use_image(self, cmd_queue:CommandQueue, image:imagend_t)->Tuple[imagend, Event]:
        if self.readonly:
            image.flags = cl_mem_flags.CL_MEM_READ_ONLY
        elif self.writeonly:
            image.flags = cl_mem_flags.CL_MEM_WRITE_ONLY
        else:
            image.flags = cl_mem_flags.CL_MEM_READ_WRITE

        image_key = (cmd_queue.context.id.value, image.__class__.__name__, image.shape, image.dtype, image.flags)

        used_image:Optional[imagend] = None
        for image in self.__images[image_key]:
            if image not in self.__busy_mems:
                used_image:imagend = image
                break

        image_data = image.data
        image.data = None
        if used_image is None:
            used_image:imagend = cmd_queue.context.create_image(image)
            self.__images[image_key].append(used_image)
        
        event = used_image.set_data(cmd_queue, image_data)
        if event is not None and CL.print_info:
            print(f"copy data to device for argument '{self.name}'")
            
        self.__busy_mems.add(used_image)
        
        return used_image, event

    def unuse(self, mem_obj:Union[MemObject])->None:
        self.__busy_mems.remove(mem_obj)


class StructInfo:

    def __init__(self, name:str):
        self.name: str = name
        self.members: Dict[str, VarInfo] = {}
        self._fields_: List[Tuple[str, type]] = []
        self.dtype: Optional[np.dtype] = None

    def declare(self)->str:
        result = f"""
class {self.name}(ctypes.Structure):

    _fields_ = [
{'\n'.join(["        ('" + field[0] + "', ctypes." + field[1].__name__ + ")" for field in self._fields_])}
    ]

    def __init__(self, {", ".join([member.name + ": " + member.type_annotation for member in self.members.values()])})->None: ...

"""
        for member in self.members.values():
            result += f"    {member.name}: {member.type_annotation}\n"

        result += "\n"

        return result


class KernelInfo:

    def __init__(self, name:str):
        self.name: str = name
        self.args: Dict[str, ArgInfo] = {}

    def args_declare(self, with_annotation:bool=False)->str:
        if with_annotation:
            return ", ".join([arg.name + ": " + arg.type_annotation for arg in self.args.values()])
        else:
            return ", ".join(list(self.args.keys()))

    def signature(self, with_annotation:bool=False)->str:
        return self.name + "(" + self.args_declare(with_annotation) + ")"
