from __future__ import annotations
import math
from ctypes import c_size_t, pointer
from functools import partial
import concurrent.futures
import asyncio
from typing import Dict, Any, List, TYPE_CHECKING, Union, Tuple, Optional

import numpy as np

from ..runtime import (
    cl_kernel,
    CL,
    IntEnum,
    CLInfo,
    cl_kernel_info,
    cl_kernel_arg_address_qualifier,
    cl_mem_flags,
    cl_mem,
    cl_device_type,
    cl_event,
    ErrorCode,
    cl_command_queue_properties
)
from .clobject import CLObject
from .kernel_parser import ArgInfo
from .command_queue import CommandQueue
from .event import Event
from ...vectorization import sizeof, value_ptr
from ...utils import detect_work_size

if TYPE_CHECKING:
    from .program import Program
    from .context import Context
    from .device import Device
    from .buffer import Buffer


class Kernel(CLObject):

    def __init__(self, kernel_id:cl_kernel, program:Program)->None:
        CLObject.__init__(self, kernel_id)
        self._program:Program = program
        self._args:Dict[str, ArgInfo] = {}
        self._arg_list:List[ArgInfo] = []
        self._arg_names:List[str] = []

        self._used_device:Optional[Device] = None
        self._global_work_size:Optional[Tuple[int]] = None
        self._local_work_size:Optional[Tuple[int]] = None

    def __getitem__(self, work_sizes:Tuple[Union[int, Tuple[int]]])->Kernel:
        if not isinstance(work_sizes, tuple):
            work_sizes = (work_sizes,)

        if len(work_sizes) == 0:
            raise TypeError("Kernel.__getitem__ takes at least 1 argument, 0 were given")
        elif len(work_sizes) == 1:
            global_work_size = None
            local_work_size = None
            device = work_sizes[0]
        elif len(work_sizes) == 2:
            global_work_size = work_sizes[0]
            local_work_size = work_sizes[1]
            device = self._program.context.devices[0]
        elif len(work_sizes) == 3:
            global_work_size = work_sizes[0]
            local_work_size = work_sizes[1]
            device = work_sizes[2]
        else:
            raise TypeError(f"Kernel.__getitem__ takes at most 3 argument, {len(work_sizes)} were given")
        
        if isinstance(global_work_size, int):
            global_work_size = (global_work_size,)

        if isinstance(local_work_size, int):
            local_work_size = (local_work_size,)

        if global_work_size is not None and local_work_size is not None:
            work_dim = len(global_work_size)
            if work_dim > 3:
                raise ValueError("dimension over 3 is not supported")
            
            if work_dim <= 0:
                raise ValueError("dimension should be at least 1")
            
            if work_dim != len(local_work_size):
                raise ValueError(f"global_work_size={global_work_size} and local_work_size={local_work_size} do not have same dimension")
        
        self._used_device = device
        self._global_work_size = global_work_size
        self._local_work_size = local_work_size
        
        return self

    def __call__(self, *args, **kwargs)->None:
        all_events = self._call(args, kwargs)
        return Event.wait_all(all_events)

    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        all_events = self._call(args, kwargs)
        return Event.create_future(all_events, concurrent.futures.Future)
    
    def create_task(self, *args, **kwargs)->asyncio.Future:
        all_events = self._call(args, kwargs)
        return Event.create_future(all_events, asyncio.Future)

    def _call(self, args, kwargs)->List[Event]:
        cmd_queue, global_work_size, local_work_size, need_read_back_buffers, copy_to_device_events = self.__before_call(args, kwargs)
        work_dim = len(global_work_size)

        event_id = cl_event()
        CL.clEnqueueNDRangeKernel.record_call_stack()
        CL.clEnqueueNDRangeKernel(cmd_queue.id, self.id, work_dim, None, global_work_size, local_work_size, len(copy_to_device_events), Event.wait_list(copy_to_device_events), pointer(event_id))
        if CL.print_info:
            print(f"launch {self} on {cmd_queue.device} with global_work_size={tuple(global_work_size)}, local_work_size={tuple(local_work_size)}", flush=True)

        call_event = Event(self.context, event_id, CL.clEnqueueNDRangeKernel)
        copy_to_host_events = self.__after_call(cmd_queue, need_read_back_buffers, [call_event])

        return [*copy_to_device_events, call_event, *copy_to_host_events]

    def __before_call(self, args, kwargs):
        used_device = self._used_device
        global_work_size = self._global_work_size
        local_work_size = self._local_work_size

        self._used_device = None
        self._global_work_size = None
        self._local_work_size = None

        used_kwargs = self.__check_args(args, kwargs)

        if used_device is None:
            used_device = self._program.context.devices[0]

        cmd_queue = self.context.create_command_queue(used_device, cl_command_queue_properties.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
        need_read_back_buffers:List[Tuple[ArgInfo, Any, Buffer]] = []
        copy_to_device_events:List[Event] = []
        for key, value in used_kwargs.items():
            buffer_tuple = self._set_arg(key, value, cmd_queue)
            if buffer_tuple:
                need_read_back_buffers.append(buffer_tuple[0:-1])
                copy_to_device_events.append(buffer_tuple[-1])

        if global_work_size is None or local_work_size is None:
            global_work_size, local_work_size = self.__detect_work_size(used_device)

        work_dim = len(global_work_size)
        global_work_size = (c_size_t * work_dim)(*global_work_size)
        local_work_size = (c_size_t * work_dim)(*local_work_size)
        return cmd_queue, global_work_size, local_work_size, need_read_back_buffers, copy_to_device_events

    @staticmethod
    def __copy_to_host(arg_value:Union[np.ndarray, List[Any]], buffer:Buffer):
        if arg_value is not buffer.data:
            if isinstance(arg_value, list):
                arg_value[:] = buffer.data.tolist()
            else:
                arg_value[:] = buffer.data

    def __after_call(self, cmd_queue:CommandQueue, need_read_back_buffers:List[Tuple[ArgInfo, Any, Buffer]], after_events:List[Event])->List[Event]:
        
        def on_completed(arg_value:Union[np.ndarray, List[Any]], buffer:Buffer, arg_info:ArgInfo, event:Event, error_code:ErrorCode):
            if error_code == ErrorCode.CL_SUCCESS:
                self.__copy_to_host(arg_value, buffer)

            arg_info.unuse_buffer(buffer)
        
        events:List[Event] = []
        for arg_info, arg_value, buffer in need_read_back_buffers:
            event:Event = buffer.read(cmd_queue, after_events=after_events)
            event.on_completed_callbacks.append(partial(on_completed, arg_value, buffer, arg_info))
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

            raise TypeError(f"{self._func_head} missing {len(missed_keys)} required positional {word}: {Kernel.__join_with_and(missed_keys)}")

        return used_kwargs

    @staticmethod
    def __join_with_and(items):
        if not items:
            return ''
        
        if len(items) == 1:
            return items[0]
        
        if len(items) == 2:
            return ' and '.join(items)
        
        return ', '.join(items[:-1]) + ', and ' + items[-1]
    
    def __detect_work_size(self, device:Device):
        max_shape = None
        for arg in self._arg_list:
            if arg.buffer is not None and ((arg.buffer.flags & cl_mem_flags.CL_MEM_WRITE_ONLY) or (arg.buffer.flags & cl_mem_flags.CL_MEM_READ_WRITE)) and not (arg.buffer.flags & cl_mem_flags.CL_MEM_READ_ONLY):
                arg_shape = arg.buffer.data.shape
                if arg.base_type_str in CLInfo.vec_types:
                    arg_shape = arg_shape[:-1]
                    
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

        total_local_size:int = 256
        if device.type == cl_device_type.CL_DEVICE_TYPE_GPU:
            if "nvidia" in device.vendor.lower():
                total_local_size:int = 128
            elif "amd" in device.vendor.lower():
                total_local_size:int = 256
            elif "intel" in device.vendor.lower():
                total_local_size:int = 64
        elif device.type == cl_device_type.CL_DEVICE_TYPE_CPU:
            total_local_size:int = 64

        global_work_size, local_work_size = detect_work_size(total_local_size, max_shape)
        return global_work_size, local_work_size

    def _set_arg(self, index_or_name:Union[int, str], value:Any, cmd_queue:CommandQueue)->Optional[Tuple[ArgInfo, Any, Buffer, Event]]:
        if isinstance(index_or_name, int):
            arg_info = self._arg_list[index_or_name]
            index = index_or_name
        else:
            arg_info = self._args[index_or_name]
            index = self._arg_names.index(index_or_name)

        arg_type_str = arg_info.type_str
        is_ptr = (arg_type_str[-1] == "*")
        if not is_ptr and arg_info.value is not None and arg_info.value == value:
            return
        
        arg_type_str = arg_info.type_str
        content_type_str = arg_type_str
        if is_ptr:
            content_type_str = arg_type_str[:-1]

        if arg_type_str in CLInfo.basic_types:
            arg_type = CLInfo.basic_types[arg_type_str]
            used_value = (value if isinstance(value, arg_type) else arg_type(value))
            CL.clSetKernelArg(self.id, index, sizeof(used_value), value_ptr(used_value))
            arg_info.value = value
        elif is_ptr and content_type_str in CLInfo.basic_types:
            content_type = CLInfo.basic_types[content_type_str]
            if arg_info.address_qualifier == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_LOCAL:
                if isinstance(value, int):
                    bytes_per_group = value
                elif isinstance(value, np.ndarray):
                    bytes_per_group = value.size
                elif isinstance(value, (list,tuple)):
                    bytes_per_group = np.array(value).size

                bytes_per_group *= sizeof(content_type)

                if arg_info.value is not None and arg_info.value == bytes_per_group:
                    return
                
                CL.clSetKernelArg(self.id, index, bytes_per_group, None)
                arg_info.value = bytes_per_group
            else:
                if isinstance(value, np.ndarray):
                    if value.dtype == content_type:
                        used_value = value
                    else:
                        used_value = value.astype(content_type)
                else:
                    used_value = np.array(value, dtype=content_type)

                if not used_value.flags['C_CONTIGUOUS']:
                    used_value = np.ascontiguousarray(used_value)

                buffer, event = arg_info.use_buffer(used_value, cmd_queue)
                if arg_info.buffer != buffer:
                    CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(buffer.id))

                arg_info.value = value
                arg_info.buffer = buffer
                if arg_info.need_read_back:
                    return arg_info, value, buffer, event

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
        return self.function_name
    
    @property
    def args(self)->Dict[str, ArgInfo]:
        return self._args
    
    @args.setter
    def args(self, args:Dict[str, ArgInfo]):
        self._args = args
        self._arg_list = list(self._args.values())
        self._arg_names = list(self._args.keys())

    @staticmethod
    def _prefix()->str:
        return "CL_KERNEL"

    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetKernelInfo

    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.kernel_info_types

    @staticmethod
    def _info_enum()->type:
        return cl_kernel_info

    @staticmethod
    def _release_func():
        return CL.clReleaseKernel