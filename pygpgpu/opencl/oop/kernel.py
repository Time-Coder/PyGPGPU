from __future__ import annotations
import math
from ctypes import c_size_t, pointer, sizeof, Structure, byref, memmove, c_void_p
from functools import partial
import concurrent.futures
import asyncio
from typing import Dict, Any, List, TYPE_CHECKING, Union, Tuple, Optional, override

import numpy as np

from ..driver import (
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
    sampler_t,
    queue_t,
    image2d_t,
    cl_kernel_arg_type_qualifier
)
from .clobject import CLObject
from .program_info import ArgInfo
from .command_queue import CommandQueue
from .event import Event
from .image2d import image2d
from .mem_object import MemObject
from .pipe import pipe
from .program_parser import ProgramParser
from ...utils import detect_work_size, join_with_and
from ... import numpy as gnp

if TYPE_CHECKING:
    from .program import Program
    from .context import Context
    from .device import Device


class Kernel(CLObject):

    def __init__(self, program:Program, kernel_id:cl_kernel)->None:
        CLObject.__init__(self, kernel_id)
        self._program:Program = program
        self._args:Dict[str, ArgInfo] = {}
        self._arg_list:List[ArgInfo] = []
        self._arg_names:List[str] = []

        self.type_checked:bool = False
        self._used_device:Optional[Device] = None
        self._global_work_size:Optional[Tuple[int]] = None
        self._local_work_size:Optional[Tuple[int]] = None

    def __getitem__(self, sizes_or_device:Tuple[Union[int, Tuple[int,...], str], ...])->Kernel:
        if not isinstance(sizes_or_device, tuple):
            sizes_or_device = (sizes_or_device,)

        if len(sizes_or_device) == 0:
            raise TypeError("Kernel.__getitem__ takes at least 1 argument, 0 were given")
        elif len(sizes_or_device) == 1:
            from .platforms import Platforms
            device = Platforms.device(sizes_or_device[0])
            if device not in self._program.context.devices:
                raise ValueError(f"device {device} is not in current context")
            self._used_device = device
        elif len(sizes_or_device) == 2:
            global_work_size = sizes_or_device[0]
            local_work_size = sizes_or_device[1]
            if isinstance(global_work_size, int):
                global_work_size = (global_work_size,)

            if isinstance(local_work_size, int):
                local_work_size = (local_work_size,)

            work_dim = len(global_work_size)
            if work_dim > 3:
                raise ValueError("dimension over 3 is not supported")
            
            if work_dim <= 0:
                raise ValueError("dimension should be at least 1")
            
            if work_dim != len(local_work_size):
                raise ValueError(f"global_work_size={global_work_size} and local_work_size={local_work_size} do not have same dimension")
        
            self._global_work_size = global_work_size
            self._local_work_size = local_work_size
        else:
            raise TypeError(f"Kernel.__getitem__ takes at most 2 argument, {len(sizes_or_device)} were given")
        
        return self

    def __call__(self, *args, **kwargs)->None:
        all_events = self._call(args, kwargs)
        Event.wait(all_events)

    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        all_events = self._call(args, kwargs)
        return Event.create_future(all_events, concurrent.futures.Future)
    
    def async_call(self, *args, **kwargs)->asyncio.Future:
        all_events = self._call(args, kwargs)
        return Event.create_future(all_events, asyncio.Future)

    def _call(self, args, kwargs)->List[Event]:
        cmd_queue, global_work_size, local_work_size, need_read_back_mem_objs, copy_to_device_events = self.__before_call(args, kwargs)
        work_dim = len(global_work_size)

        event_id = cl_event()
        CL.clEnqueueNDRangeKernel(cmd_queue.id, self.id, work_dim, None, global_work_size, local_work_size, len(copy_to_device_events), Event.events_ptr(copy_to_device_events), pointer(event_id))
        if CL.print_info:
            print(f"launch {self} on {cmd_queue.device} with global_work_size={tuple(global_work_size)}, local_work_size={tuple(local_work_size)}", flush=True)

        call_event = Event(self.context, event_id, CL.clEnqueueNDRangeKernel)
        copy_to_host_events = self.__after_call(cmd_queue, need_read_back_mem_objs, [call_event])

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

        cmd_queue = self.context.create_command_queue(used_device)

        need_read_back_mem_objs:List[Tuple[ArgInfo, Any, MemObject]] = []
        copy_to_device_events:List[Event] = []
        for key, value in used_kwargs.items():
            results = self._set_arg(cmd_queue, key, value)
            for result in results:
                if "event" in result and result["event"] is not None:
                    copy_to_device_events.append(result["event"])

                if "arg_info" in result:
                    need_read_back_mem_objs.append((result["arg_info"], result["value"], result["mem_obj"]))

        if global_work_size is None or local_work_size is None:
            global_work_size, local_work_size = self.__detect_work_size(used_device)

        work_dim = len(global_work_size)
        global_work_size = (c_size_t * work_dim)(*global_work_size)
        local_work_size = (c_size_t * work_dim)(*local_work_size)
        return cmd_queue, global_work_size, local_work_size, need_read_back_mem_objs, copy_to_device_events

    @staticmethod
    def __copy_to_host(arg_value:Union[np.ndarray, image2d_t, List[Any]], buffer:MemObject):
        host_data = arg_value
        if isinstance(arg_value, image2d_t):
            host_data = arg_value.data

        if host_data is not buffer.data:
            if isinstance(host_data, list):
                host_data[:] = buffer.data.tolist()
            elif isinstance(host_data, np.ndarray):
                host_data[:] = buffer.data
            elif isinstance(host_data, Structure):
                memmove(byref(host_data), buffer.data.ctypes.data_as(c_void_p), sizeof(host_data))

    def __after_call(self, cmd_queue:CommandQueue, need_read_back_mem_objs:List[Tuple[ArgInfo, Any, MemObject]], after_events:List[Event])->List[Event]:
        
        def on_completed(arg_value:Union[np.ndarray, List[Any]], mem_obj:MemObject, arg_info:ArgInfo, event:Event, error_code:ErrorCode):
            if error_code == ErrorCode.CL_SUCCESS:
                self.__copy_to_host(arg_value, mem_obj)
                if CL.print_info:
                    print(f"copy data to host for argument '{arg_info.name}'")

            mem_obj.unuse()
        
        events:List[Event] = []
        for arg_info, arg_value, mem_obj in need_read_back_mem_objs:
            event:Event = mem_obj.read(cmd_queue, after_events=after_events)
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
    
    def __detect_work_size(self, device:Device):
        max_shape = None
        for arg in self._arg_list:
            if arg.mem_obj is not None and ((arg.mem_obj.flags & cl_mem_flags.CL_MEM_WRITE_ONLY) or (arg.mem_obj.flags & cl_mem_flags.CL_MEM_READ_WRITE)) and not (arg.mem_obj.flags & cl_mem_flags.CL_MEM_READ_ONLY):
                arg_shape = (arg.mem_obj.data.shape if arg.mem_obj.data is not None else arg.mem_obj.shape)
                if ((
                        arg.base_type_str in CLInfo.vec_types and
                        len(arg_shape) >= 2 and
                        arg_shape[-1] == CLInfo.vec_types[arg.base_type_str].__len__(None)
                    ) or (
                        len(arg_shape) == 3 and isinstance(arg.mem_obj, image2d)
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

    def _set_arg(self, cmd_queue:CommandQueue, index_or_name:Union[int, str], value:Any)->List[Dict[str, Any]]:
        if isinstance(index_or_name, int):
            arg_info = self._arg_list[index_or_name]
            index = index_or_name
        else:
            arg_info = self._args[index_or_name]
            index = self._arg_names.index(index_or_name)

        arg_type_str = arg_info.type_str
        if not arg_info.is_ptr and arg_info.value is not None and arg_info.value == value:
            return []
        
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
            if arg_info.value != used_value.id:
                CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
                arg_info.value = used_value.id
        elif arg_type_str == "queue_t":
            if not isinstance(value, queue_t):
                raise TypeError(f"type of argument '{arg_info.name}' must be queue_t, got {value.__class__.__name__}.")
            
            used_value = self.context.get_queue(value)
            if arg_info.value != used_value.id:
                CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
                arg_info.value = used_value.id
        elif arg_type_str == "sampler_t":
            if not isinstance(value, sampler_t):
                raise TypeError(f"type of argument '{arg_info.name}' must be sampler_t, got {value.__class__.__name__}.")
            
            used_value = self.context.get_sampler(value)
            if arg_info.value != used_value.id:
                CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
                arg_info.value = used_value.id
        elif base_type_str in CLInfo.basic_types or base_type_str in self._program.structs:
            if base_type_str in CLInfo.basic_types:
                content_type = CLInfo.basic_types[base_type_str]
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
                        if base_type_str in CLInfo.alter_types:
                            used_value = content_type(CLInfo.alter_types[base_type_str][1](value))
                        else:
                            used_value = content_type(value)
                    except BaseException as e:
                        success = False

                    if not success:
                        raise TypeError(f"type of argument '{arg_info.name}' need {content_type.__name__}, got {value.__class__.__name__}, and cannot convert {value.__class__.__name__} to {content_type.__name__}")

                if is_struct:
                    result = used_value.apply_pointers(cmd_queue)

                CL.clSetKernelArg(self.id, index, sizeof(used_value), pointer(used_value))
                arg_info.value = value

                if is_struct:
                    return result
            else:
                if arg_info.address_qualifier == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_LOCAL:
                    if isinstance(value, int):
                        bytes_per_group = value
                    elif isinstance(value, np.ndarray):
                        bytes_per_group = value.size
                    elif isinstance(value, (list, tuple)):
                        bytes_per_group = np.array(value).size

                    bytes_per_group *= sizeof(content_type)

                    if arg_info.value is not None and arg_info.value == bytes_per_group:
                        return []
                    
                    CL.clSetKernelArg(self.id, index, bytes_per_group, None)
                    arg_info.value = bytes_per_group
                else:
                    if isinstance(value, np.ndarray):
                        used_value = value
                        if value.dtype != content_type:
                            used_value = value.astype(content_type)
                    else:
                        ProgramParser._apply_structure_pointers(value, cmd_queue)
                        used_value = np.array(value, dtype=content_type)

                    if not used_value.flags['C_CONTIGUOUS']:
                        used_value = np.ascontiguousarray(used_value)

                    buffer, event = arg_info.use_buffer(cmd_queue, used_value)
                    if arg_info.mem_obj != buffer:
                        CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(buffer.id))
                        arg_info.mem_obj = buffer

                    arg_info.value = value
                    if arg_info.need_read_back and not isinstance(value, gnp.ndarray):
                        return [{
                            "arg_info": arg_info,
                            "value": value,
                            "mem_obj": buffer,
                            "event": event
                        }]
                    else:
                        return [{
                            "event": event
                        }]
        elif arg_type_str in CLInfo.image_types:
            image_type = CLInfo.image_types[arg_type_str]
            if not isinstance(value, (np.ndarray, image_type)):
                raise TypeError(f"type of argument '{arg_info.name}' must be {image_type.__name__} or np.ndarray, got {value.__class__.__name__}.")
            
            used_value = value
            if isinstance(value, np.ndarray):
                used_value = image_type(value)
            
            image, event = arg_info.use_image(cmd_queue, used_value)

            if arg_info.mem_obj != image:
                CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(image.id))
                arg_info.mem_obj = image

            arg_info.value = value
            if arg_info.need_read_back and not isinstance(used_value.data, gnp.ndarray):
                return [{
                    "arg_info": arg_info,
                    "value": value,
                    "mem_obj": image,
                    "event": event
                }]
            else:
                return [{
                    "event": event
                }]
        else:
            raise NotImplementedError(f"arg type {arg_type_str} is legal, but not implemented.")

        return []

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

    @override    
    @staticmethod
    def _prefix()->str:
        return "CL_KERNEL"

    @override    
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetKernelInfo

    @override    
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.kernel_info_types

    @override    
    @staticmethod
    def _info_enum()->type:
        return cl_kernel_info

    @override    
    @staticmethod
    def _release_func():
        return CL.clReleaseKernel