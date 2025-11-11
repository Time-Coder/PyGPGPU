from ctypes import pointer
from typing import Tuple, Dict, Optional, Any, List, Union

import numpy as np

from ..runtime import (
    cl_device_id,
    cl_int,
    CL,
    CLInfo,
    IntEnum,
    cl_context,
    cl_context_info,
    cl_mem_flags,
    CL_CONTEXT_NOTIFY_CALLBACK
)
from .clobject import CLObject
from .device import Device
from .platform import Platform
from .program import Program
from .build_options import BuildOptions
from .kernel_parser import KernelParser
from .buffer import Buffer
from .command_queue import CommandQueue


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
        self._default_command_queues:Dict[Device, CommandQueue] = {}

        n_devices = len(self._devices)
        self._devices_ids = (cl_device_id * n_devices)()
        for i in range(n_devices):
            self._devices_ids[i] = self._devices[i].id
        
        pfn_notify = CL_CONTEXT_NOTIFY_CALLBACK(Context._pfn_notify)

        properties = None
        device_ids = (cl_device_id * n_devices)()
        for i in range(n_devices):
            device_ids[i] = self._devices[i].id

        user_data = None
        errcode = cl_int()

        context_id = CL.clCreateContext(properties, n_devices, device_ids, pfn_notify, user_data, pointer(errcode))
        CLObject.__init__(self, context_id)

    @staticmethod
    def _release(context_id:cl_context):
        if not context_id:
            return
        
        CL.clReleaseContext(context_id)

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
    
    def default_cmd_queue(self, device:Device)->CommandQueue:
        if device in self._default_command_queues:
            return self._default_command_queues[device]

        if device not in self.devices:
            raise KeyError(str(device))
        
        cmd_queue = CommandQueue(self, device)
        self._default_command_queues[device] = cmd_queue
        return cmd_queue

    def create_buffer(self, data_or_size:Union[bytes, bytearray, np.ndarray, int], flags:Optional[cl_mem_flags]=None, auto_share:bool=True)->Buffer:
        return Buffer(self, data_or_size, flags, auto_share)

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
        spir_std:Optional[float]=None
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
            program = Program(self, file_name, includes, defines, options)
            self._programs[key] = program

        return self._programs[key]

    @staticmethod
    def _pfn_notify(errinfo:bytes, private_info, cb, user_data):
        if CL.print_info:
            if errinfo:
                message = errinfo.decode('utf-8', errors='replace')
                print(f"[OpenCL Context Notify] {message}")
            else:
                print("[OpenCL Context Notify] (no message)")

    @property
    def _prefix(self)->str:
        return "CL_CONTEXT"

    @property
    def _get_info_func(self)->CL.Func:
        return CL.clGetContextInfo

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        return CLInfo.context_info_types

    @property
    def _info_enum(self)->type:
        return cl_context_info