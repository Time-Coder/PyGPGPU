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
    CLInfo
)

if TYPE_CHECKING:
    from .buffer import Buffer
    from .command_queue import CommandQueue
    from .context import Context

from .mem import Mem
from .event import Event
from .sampler import sampler
from .sampler_t import sampler_t
from .image2d import image2d
from .image2d_t import image2d_t


class ArgInfo:

    def __init__(self, name: str, type_str: str, address_qualifier: cl_kernel_arg_address_qualifier, access_qualifier: cl_kernel_arg_access_qualifier, type_qualifiers: cl_kernel_arg_type_qualifier):
        self.name = name
        self.type_str = type_str
        self.address_qualifier = address_qualifier
        self.access_qualifier = access_qualifier
        self.type_qualifiers = type_qualifiers
        self.value: Any = None
        self.mem: Optional[Mem] = None

        self.__buffers: Dict[Tuple[cl_context, int, cl_mem_flags], List[Buffer]] = defaultdict(list)
        self.__image2ds: Dict[Tuple[cl_context, Tuple[int, ...], type], List[image2d]] = defaultdict(list)
        self.__busy_mems: Set[Mem] = set()

        self.__samplers: Dict[Tuple[cl_context, str], sampler] = {}

    @property
    def is_ptr(self)->bool:
        return (self.type_str[-1] == "*")
    
    @property
    def base_type_str(self)->str:
        return (self.type_str[:-1] if self.is_ptr else self.type_str)
    
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
    
    def check_type(self, value:Any)->None:
        base_type_str = self.base_type_str
        if self.is_ptr:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"type of argument '{self.name}' must be np.ndarray, got {value.__class__.__name__}.")
            
            if base_type_str in CLInfo.dtypes:
                dtype = CLInfo.dtypes[base_type_str]
                if value.dtype != dtype:
                    raise TypeError(f"type of argument '{self.name}' must be NDArray[{dtype.name}], got NDArray[{value.dtype.name}].")
        else:
            base_type = None
            if base_type_str in CLInfo.basic_types:
                base_type = CLInfo.basic_types[base_type_str]

            if base_type_str in CLInfo.alter_types:
                base_type = CLInfo.alter_types[base_type_str]

            same_type = (value.__class__.__name__ == base_type_str if base_type is None else isinstance(value, base_type))

            if not same_type:
                raise TypeError(f"type of argument '{self.name}' must be {base_type_str}, got {value.__class__.__name__}.")

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
        self.__busy_mems.add(used_buffer)
        
        return used_buffer, event

    def use_sampler(self, context:Context, sampler_t_:sampler_t)->sampler:
        sampler_key = (context.id.value, str(sampler_t_))

        if sampler_key not in self.__samplers:
            self.__samplers[sampler_key] = sampler(context, sampler_t_)
        
        return self.__samplers[sampler_key]
    
    def use_image2d(self, cmd_queue:CommandQueue, image:image2d_t)->Tuple[image2d, Event]:
        if self.readonly:
            image.flags = cl_mem_flags.CL_MEM_READ_ONLY
        elif self.writeonly:
            image.flags = cl_mem_flags.CL_MEM_WRITE_ONLY
        else:
            image.flags = cl_mem_flags.CL_MEM_READ_WRITE

        image_key = (cmd_queue.context.id.value, image.shape, image.dtype, image.flags)

        used_image = None
        for image in self.__image2ds[image_key]:
            if image not in self.__busy_mems:
                used_image = image
                break

        image_data = image.data
        image.data = None
        if used_image is None:
            used_image = cmd_queue.context.create_image2d(image)
            self.__image2ds[image_key].append(used_image)
        
        event = used_image.set_data(cmd_queue, image_data)
        self.__busy_mems.add(used_image)
        
        return used_image, event

    def unuse(self, mem:Union[Mem])->None:
        self.__busy_mems.remove(mem)


class KernelInfo:

    def __init__(self, name:str):
        self.name: str = name
        self.args: Dict[str, ArgInfo] = {}