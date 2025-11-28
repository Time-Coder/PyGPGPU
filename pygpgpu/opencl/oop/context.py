from __future__ import annotations
from ctypes import pointer
from collections import defaultdict
from typing import Tuple, Dict, Optional, Any, List, Union, TYPE_CHECKING, override

import numpy as np

from ..driver import (
    cl_device_id,
    cl_int,
    CL,
    CLInfo,
    IntEnum,
    cl_context_info,
    cl_mem_flags,
    CL_CONTEXT_NOTIFY_CALLBACK,
    cl_command_queue_properties,
    sampler_t,
    imagend_t,
    image2d_t,
    image1d_t,
    image3d_t,
    image2d_array_t,
    image1d_array_t,
    queue_t
)

if TYPE_CHECKING:
    from .platform import Platform

from .device import Device
from .clobject import CLObject
from .program import Program
from .build_options import BuildOptions
from .program_parser import ProgramParser
from .buffer import Buffer
from .command_queue import CommandQueue
from .event import Event
from .sampler import sampler
from .imagend import imagend
from .image2d import image2d
from .image1d import image1d
from .image3d import image3d
from .image2d_array import image2d_array
from .image1d_array import image1d_array
from .pipe import Pipe, pipe


class Context(CLObject):

    def __init__(self, *devices):
        if not devices:
            raise ValueError("Context can only be created arround at least one device")

        platform = None
        for device in devices:
            if not isinstance(device, Device):
                raise TypeError(f"Context.__init__ only accept 'Device' type arguments, '{device.__class__.__name__}' type were given")

            if platform is None:
                platform = device.platform

            if device.platform != platform:
                raise ValueError("devices come from different platforms")
            
        self._platform:Platform = platform
        self._devices:Tuple[Device] = devices
        self._programs:Dict[str, Program] = {}

        n_devices = len(self._devices)
        self._devices_ids = (cl_device_id * n_devices)()
        for i in range(n_devices):
            self._devices_ids[i] = self._devices[i].id
        
        properties = None
        device_ids = (cl_device_id * n_devices)()
        for i in range(n_devices):
            device_ids[i] = self._devices[i].id

        user_data = None
        errcode = cl_int()

        context_id = CL.clCreateContext(properties, n_devices, device_ids, Context._pfn_notify, user_data, pointer(errcode))
        CLObject.__init__(self, context_id)

        self.__pipes:Dict[pipe, Pipe] = {}
        self.__queues:Dict[queue_t, CommandQueue] = {}
        self.__buffers: Dict[Tuple[int, cl_mem_flags], List[Buffer]] = defaultdict(list)
        self.__images: Dict[Tuple[str, Tuple[int, ...], type, cl_mem_flags], List[imagend]] = defaultdict(list)
        self.__samplers:Dict[str, sampler] = {}

    @staticmethod
    def _release_func()->CL.Func:
        return CL.clReleaseContext

    @property
    def devices(self)->Tuple[Device]:
        return self._devices
    
    @property
    def device_ids(self):
        return self._devices_ids
    
    @property
    def n_devices(self)->int:
        return len(self._devices)
    
    @property
    def platform(self)->Platform:
        return self._platform
    
    def create_command_queue(self, device:Device, properties:Optional[cl_command_queue_properties]=None)->CommandQueue:
        if device not in self.devices:
            raise KeyError(device)
        
        return CommandQueue(self, device, properties)

    def create_buffer(self, data:Union[bytes, bytearray, np.ndarray, None]=None, size:int=0, flags:Optional[cl_mem_flags]=None)->Buffer:
        return Buffer(self, data=data, size=size, flags=flags)
    
    def create_event(self)->Event:
        return Event(self)
    
    def create_image(self, image:imagend_t)->imagend:
        if isinstance(image, image2d_t):
            return self.create_image2d(image)
        elif isinstance(image, image3d_t):
            return self.create_image3d(image)
        elif isinstance(image, image1d_t):
            return self.create_image1d(image)
        elif isinstance(image, image2d_array_t):
            return self.create_image2d_array(image)
        elif isinstance(image, image1d_array_t):
            return self.create_image1d_array(image)

    def create_image2d(self, image:image2d_t)->image2d:
        return image2d(self, image)
    
    def create_image1d(self, image:image1d_t)->image1d:
        return image1d(self, image)
    
    def create_image3d(self, image:image3d_t)->image3d:
        return image3d(self, image)
    
    def create_image2d_array(self, image:image2d_array_t)->image2d_array:
        return image2d_array(self, image)
    
    def create_image1d_array(self, image:image1d_array_t)->image1d_array:
        return image1d_array(self, image)
    
    def create_sampler(self, sampler_t_:sampler_t)->sampler:
        return sampler(self, sampler_t_)
    
    def create_pipe(self, pipe_:pipe)->Pipe:
        return Pipe(self, pipe_)
    
    def get_sampler(self, sampler_t_:sampler_t)->sampler:
        sampler_key:str = str(sampler_t_)
        if sampler_key not in self.__samplers:
            self.__samplers[sampler_key] = self.create_sampler(sampler_t_)

        return self.__samplers[sampler_key]

    def get_pipe(self, pipe_:pipe)->Pipe:
        if pipe_ not in self.__pipes:
            self.__pipes[pipe_] = self.create_pipe(pipe_)

        return self.__pipes[pipe_]
    
    def get_queue(self, device:Device, queue:queue_t)->CommandQueue:
        if queue not in self.__queues:
            self.__queues[queue] = self.create_command_queue(device, queue.properties)

        return self.__queues[queue]
    
    def get_buffer(self, size:int=0, flags:cl_mem_flags=cl_mem_flags.CL_MEM_READ_WRITE)->Buffer:
        key = (size, flags)

        available_buffer = None
        for buffer in self.__buffers[key]:
            if not buffer.using:
                available_buffer = buffer
                break

        if available_buffer is None:
            available_buffer = self.create_buffer(size=size, flags=flags)
            self.__buffers[key].append(available_buffer)

        available_buffer.use()
        return available_buffer
    
    def get_image(self, image:imagend_t)->image2d:
        key = (image.__class__.__name__, image.shape, image.dtype, image.flags)
        
        available_image = None
        for image_ in self.__images[key]:
            if not image_.using:
                available_image = image
                break

        if available_image is None:
            available_image = self.create_image(image)
            self.__images[key].append(available_image)

        available_image.use()
        return available_image

    def compile(self,
        file_name:str,
        includes:Optional[List[str]] = None,
        defines:Optional[Dict[str, Any]] = None,
        single_precision_constant:bool=False,
        denorms_are_zero:bool=False,
        fp32_correctly_rounded_divide_sqrt:bool=False,
        opt_disable:bool=False,
        strict_aliasing:bool=False,
        uniform_work_group_size:bool=False,
        no_subgroup_ifp:bool=False,
        mad_enable:bool=False,
        no_signed_zeros:bool=False,
        unsafe_math_optimizations:bool=False,
        finite_math_only:bool=False,
        fast_relaxed_math:bool=False,
        w:bool=False,
        Werror:bool=False,
        cl_std:Optional[float]=None,
        kernel_arg_info:bool=False,
        g:bool=False,
        create_library:bool=False,
        enable_link_options:bool=False,
        x_spir:bool=False,
        spir_std:Optional[float]=None,
        type_checked:bool=False,
        options:Optional[BuildOptions]=None
    )->Program:
        if options is None:
            options:BuildOptions = BuildOptions(
                single_precision_constant,
                denorms_are_zero,
                fp32_correctly_rounded_divide_sqrt,
                opt_disable,
                strict_aliasing,
                uniform_work_group_size,
                no_subgroup_ifp,
                mad_enable,
                no_signed_zeros,
                unsafe_math_optimizations,
                finite_math_only,
                fast_relaxed_math,
                w,
                Werror,
                cl_std,
                kernel_arg_info,
                g,
                create_library,
                enable_link_options,
                x_spir,
                spir_std
            )
        key:str = ProgramParser.md5_of(includes, defines, options, file_name=file_name)
        if key not in self._programs:
            program = Program(self, file_name, includes, defines, options, type_checked)
            self._programs[key] = program

        return self._programs[key]

    @CL_CONTEXT_NOTIFY_CALLBACK
    def _pfn_notify(errinfo:bytes, private_info, cb, user_data):
        if CL.print_info:
            if errinfo:
                message = errinfo.decode('utf-8', errors='replace')
                print(f"[OpenCL Context Notify] {message}")
            else:
                print("[OpenCL Context Notify] (no message)")

    @override
    @staticmethod
    def _prefix()->str:
        return "CL_CONTEXT"

    @override
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetContextInfo

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.context_info_types

    @override
    @staticmethod
    def _info_enum()->type:
        return cl_context_info