from __future__ import annotations
from ctypes import pointer
from typing import Tuple, Dict, Optional, Any, List, Union, TYPE_CHECKING, override

import numpy as np

from ..runtime import (
    cl_device_id,
    cl_int,
    CL,
    CLInfo,
    IntEnum,
    cl_context_info,
    cl_mem_flags,
    CL_CONTEXT_NOTIFY_CALLBACK,
    cl_command_queue_properties
)

if TYPE_CHECKING:
    from .platform import Platform

from .device import Device
from .clobject import CLObject
from .program import Program
from .build_options import BuildOptions
from .kernel_parser import KernelParser
from .buffer import Buffer
from .command_queue import CommandQueue
from .event import Event


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

    def create_buffer(self, data_or_size:Union[bytes, bytearray, np.ndarray, int], flags:Optional[cl_mem_flags]=None)->Buffer:
        return Buffer(self, data_or_size, flags)
    
    def create_event(self)->Event:
        return Event(self)

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
        type_checked:bool=False
    )->Program:
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
        key:str = KernelParser.md5_of(file_name, includes, defines, options)
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