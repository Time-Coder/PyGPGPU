from __future__ import annotations
import math
from ctypes import c_size_t, pointer, sizeof, Structure, byref, memmove, c_void_p
from functools import partial
import concurrent.futures
import asyncio
from typing import Dict, Any, List, TYPE_CHECKING, Union, Tuple, Optional, override

import numpy as np

from ..driver import (
    CUDA,
    CUInfo,
    CUfunction,
    CUresult
)
from .cuobject import CUObject
from .program_info import ArgInfo
from .stream import Stream
from .event import Event
from .mem_object import MemObject
from .program_parser import ProgramParser
from ...utils import detect_work_size, join_with_and
from ... import numpy as gnp

if TYPE_CHECKING:
    from .program import Program
    from .context import Context
    from .device import Device


class Kernel(CUObject):

    def __init__(self, program:Program, name:str)->None:
        kernel_id = CUfunction()
        ptr_kernel_id = pointer(kernel_id)
        CUDA.cuModuleGetFunction(ptr_kernel_id, program.id, name.encode("utf-8"))
        CUObject.__init__(self, kernel_id)

        self._name:str = name
        self._program:Program = program
        self._args:Dict[str, ArgInfo] = {}
        self._arg_list:List[ArgInfo] = []
        self._arg_names:List[str] = []

        self.type_checked:bool = False
        self._grid_dims:Optional[Tuple[int]] = None
        self._block_dims:Optional[Tuple[int]] = None

    def __getitem__(self, sizes_or_device:Tuple[Union[int, Tuple[int,...], str], ...])->Kernel:
        if not isinstance(sizes_or_device, tuple):
            sizes_or_device = (sizes_or_device,)

        if len(sizes_or_device) == 0:
            raise TypeError("Kernel.__getitem__ takes at least 1 argument, 0 were given")
        elif len(sizes_or_device) == 2:
            grid_dims = sizes_or_device[0]
            block_dims = sizes_or_device[1]
            if isinstance(grid_dims, int):
                grid_dims = (grid_dims,)

            if isinstance(block_dims, int):
                block_dims = (block_dims,)

            work_dim = len(grid_dims)
            if work_dim > 3:
                raise ValueError("dimension over 3 is not supported")
            
            if work_dim <= 0:
                raise ValueError("dimension should be at least 1")
            
            if work_dim != len(block_dims):
                raise ValueError(f"grid_dims={grid_dims} and block_dims={block_dims} do not have same dimension")
        
            self._grid_dims = grid_dims
            self._block_dims = block_dims
        else:
            raise TypeError(f"Kernel.__getitem__ takes at most 2 argument, {len(sizes_or_device)} were given")
        
        return self

    def __call__(self, *args, **kwargs)->None:
        stream = self._call(args, kwargs)
        stream.sync()

    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        stream = self._call(args, kwargs)
        return stream.create_future(concurrent.futures.Future)
    
    def async_call(self, *args, **kwargs)->asyncio.Future:
        stream = self._call(args, kwargs)
        return stream.create_future(asyncio.Future)

    def _call(self, args, kwargs)->Stream:
        stream, grid_dims, block_dims, value_ptrs = self.__before_call(args, kwargs)
        CUDA.cuLaunchKernel(
            self.id,
            grid_dims[0], grid_dims[1], grid_dims[2],
            block_dims[0], block_dims[1], block_dims[2],
            0, stream.id, value_ptrs, None
        )
        if CUDA.print_info:
            print(f"launch {self} on {stream.device} with grid_dim={grid_dims}, block_dim={block_dims}", flush=True)

        self.__after_call(stream)

        return stream

    def __before_call(self, args, kwargs):
        global_work_size = self._global_work_size
        local_work_size = self._local_work_size

        self._global_work_size = None
        self._local_work_size = None

        used_kwargs = self.__check_args(args, kwargs)

        stream = Stream()

        value_ptrs = (c_void_p * len(self._args))()
        i = 0
        for key, arg_info in self._args.items():
            value = used_kwargs[key]
            value_ptrs[i] = self._process_arg(stream, arg_info, value)
            i += 1

        if global_work_size is None or local_work_size is None:
            global_work_size, local_work_size = self.__detect_work_size()

        work_dim = len(global_work_size)
        global_work_size = (c_size_t * work_dim)(*global_work_size)
        local_work_size = (c_size_t * work_dim)(*local_work_size)
        return stream, global_work_size, local_work_size, value_ptrs

    @staticmethod
    def __copy_to_host(arg_value:Union[np.ndarray, List[Any]], buffer:MemObject):
        host_data = arg_value
        if host_data is not buffer.data:
            if isinstance(host_data, list):
                host_data[:] = buffer.data.tolist()
            elif isinstance(host_data, np.ndarray):
                host_data[:] = buffer.data
            elif isinstance(host_data, Structure):
                memmove(byref(host_data), buffer.data.ctypes.data_as(c_void_p), sizeof(host_data))

    def __after_call(self, stream:Stream)->List[Event]:
        
        def on_completed(arg_value:Union[np.ndarray, List[Any]], mem_obj:MemObject, arg_info:ArgInfo, event:Event, error_code:CUresult):
            if error_code == CUresult.CUDA_SUCCESS:
                self.__copy_to_host(arg_value, mem_obj)
                if CUDA.print_info:
                    print(f"copy data to host for argument '{arg_info.name}'")

            mem_obj.unuse()
        
        events:List[Event] = []
        for arg_info in need_read_back_mem_objs:
            event:Event = mem_obj.read(stream)
            call_back = partial(on_completed, arg_value, mem_obj, arg_info)
            if event.finished:
                call_back(event, event.status)
            else:
                event.on_completed_callbacks.append(call_back)
                events.append(event)

        return events

    def __check_args(self, args, kwargs)->Dict[str, Any]:
        for key in kwargs:
            if key not in self._args:
                raise TypeError(f"{self._func_head} got an unexpected keyword argument '{key}'")

        used_kwargs = {}
        for i in range(min(len(args), len(self._args))):
            arg_name = self._arg_names[i]
            used_kwargs[arg_name] = args[i]

        for key, value in kwargs.items():
            if key in used_kwargs:
                raise TypeError(f"{self._func_head} got multiple values for argument '{key}'")
            
            used_kwargs[key] = value

        if len(args) > len(self._args):
            raise TypeError(f"{self._func_head} takes {len(self._args)} positional arguments but {len(args)} were given")
        
        if len(used_kwargs) < len(self._args):
            missed_keys = []
            for arg_name in self._args:
                if arg_name not in used_kwargs:
                    missed_keys.append(f"'{arg_name}'")

            if len(missed_keys) == 1:
                word = "argument"
            else:
                word = "arguments"

            raise TypeError(f"{self._func_head} missing {len(missed_keys)} required positional {word}: {join_with_and(missed_keys)}")

        if self.type_checked:
            for key, value in used_kwargs.items():
                arg_info = self._args[key]
                arg_info.check_type(value)

        return used_kwargs
    
    def __detect_work_size(self):
        max_shape = None
        for arg in self._arg_list:
            if arg.need_read_back:
                arg_shape = (arg.mem_obj.data.shape if arg.mem_obj.data is not None else arg.mem_obj.shape)
                if ((
                        arg.base_type_str in CUInfo.vec_types and
                        len(arg_shape) >= 2 and
                        arg_shape[-1] == CUInfo.vec_types[arg.base_type_str].__len__(None)
                    )
                ):
                    arg_shape = arg_shape[:-1]

                if not arg_shape:
                    arg_shape = (1,)
                    
                if max_shape is None:
                    max_shape = arg_shape
                elif len(arg_shape) > len(max_shape):
                    max_shape = arg_shape
                elif len(arg_shape) == len(max_shape):
                    for n1, n2 in zip(arg_shape, max_shape):
                        if n1 > n2:
                            max_shape = arg_shape
                            break

        if len(max_shape) > 3:
            max_shape = (math.prod(max_shape),)

        total_local_size:int = 128
        global_work_size, block_dims = detect_work_size(total_local_size, max_shape)
        grid_dims = []
        for value1, value2 in zip(global_work_size, block_dims):
            grid_dims.append(value1 // value2)

        return grid_dims, block_dims

    def _process_arg(self, stream:Stream, arg_info:ArgInfo, value:Any)->List[Dict[str, Any]]:
        arg_type_str = arg_info.type_str        
        arg_type_str = arg_info.type_str
        base_type_str = arg_info.base_type_str

        if base_type_str in CUInfo.basic_types or base_type_str in self._program.structs:
            if base_type_str in CUInfo.basic_types:
                content_type = CUInfo.basic_types[base_type_str]
                is_struct = False
            else:
                content_type = self._program.structs[arg_info.base_type_str]
                is_struct = True

            if not arg_info.is_ptr:
                if isinstance(value, content_type):
                    used_value = value
                else:
                    success = True
                    try:
                        if base_type_str in CUInfo.alter_types:
                            used_value = content_type(CUInfo.alter_types[base_type_str][1](value))
                        else:
                            used_value = content_type(value)
                    except BaseException as e:
                        success = False

                    if not success:
                        raise TypeError(f"type of argument '{arg_info.name}' need {content_type.__name__}, got {value.__class__.__name__}, and cannot convert {value.__class__.__name__} to {content_type.__name__}")

                if is_struct:
                    result = used_value.apply_pointers(stream)

                ptr_value = pointer(used_value)

                if is_struct:
                    return ptr_value, result
                else:
                    return ptr_value, []
            else:
                if isinstance(value, np.ndarray):
                    used_value = value
                    if value.dtype != content_type:
                        used_value = value.astype(content_type)
                else:
                    ProgramParser._apply_structure_pointers(value, stream)
                    used_value = np.array(value, dtype=content_type)

                if not used_value.flags['C_CONTIGUOUS']:
                    used_value = np.ascontiguousarray(used_value)

                buffer, event = arg_info.use_buffer(stream, used_value)
                ptr_value = pointer(buffer.id)
                arg_info.mem_obj = buffer

                arg_info.value = value
                if arg_info.need_read_back and not isinstance(value, gnp.ndarray):
                    return ptr_value, [{
                        "arg_info": arg_info,
                        "value": value,
                        "mem_obj": buffer,
                        "event": event
                    }]
                else:
                    return ptr_value, [{
                        "event": event
                    }]
        else:
            raise NotImplementedError(f"arg type {arg_type_str} is legal, but not implemented.")

    @property
    def _func_head(self)->str:
        return self.name + "(" + ", ".join(self._arg_names) + ")"

    @property
    def program(self)->Program:
        return self._program
    
    @property
    def context(self)->Context:
        return self._program.context

    @property
    def name(self)->str:
        return self._name
    
    @property
    def args(self)->Dict[str, ArgInfo]:
        return self._args
    
    @args.setter
    def args(self, args:Dict[str, ArgInfo]):
        self._args = args
        self._arg_list = list(self._args.values())
        self._arg_names = list(self._args.keys())

    @override    
    @staticmethod
    def _prefix()->str:
        return ""

    @override    
    @staticmethod
    def _get_info_func()->CUDA.Func:
        return None

    @override    
    @staticmethod
    def _info_enum()->type:
        return None

    @override    
    @staticmethod
    def _release_func():
        return None