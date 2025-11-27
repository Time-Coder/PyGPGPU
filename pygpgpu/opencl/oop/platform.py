from ctypes import pointer
from typing import List, Dict, Tuple, Optional, Any, override

from ..driver import (
    CL, CLInfo, IntEnum,
    cl_platform_id,
    cl_device_type,
    cl_uint,
    cl_device_id,
    cl_platform_info
)

from .clobject import CLObject
from .device import Device
from .context import Context
from .program import Program
from .build_options import BuildOptions


class Platform(CLObject):

    def __init__(self, id:cl_platform_id):
        CLObject.__init__(self, id)
        self.__n_devices:int = 0
        self.__devices:Tuple[Device] = ()
        self.__default_context:Optional[Context] = None
    
    @property
    def extensions(self)->List[str]:
        return self.__getattr__("extensions").split(" ")
    
    @property
    def n_devices(self)->int:
        if self.__n_devices == 0:
            n_devices = cl_uint()
            CL.clGetDeviceIDs(self._id, cl_device_type.CL_DEVICE_TYPE_ALL, 0, None, pointer(n_devices))
            self.__n_devices = n_devices.value

        return self.__n_devices

    def __fetch_devices(self):
        if self.__devices:
            return
        
        devices = []
        device_ids = (cl_device_id * self.n_devices)()
        CL.clGetDeviceIDs(self.id, cl_device_type.CL_DEVICE_TYPE_ALL, self.n_devices, device_ids, None)
        for device_id in device_ids:
            device = Device(cl_device_id(device_id), self)
            devices.append(device)

        self.__devices = tuple(devices)

    @property
    def devices(self)->Tuple[Device]:
        self.__fetch_devices()
        return self.__devices
    
    @property
    def default_context(self)->Context:
        if self.__default_context is None:
            self.__default_context = Context(*self.devices)

        return self.__default_context
    
    def create_context(self, *devices)->Context:
        if not devices:
            devices = self.devices

        for device in devices:
            if device.platform != self:
                raise ValueError(f"{device} is not in {self}")

        return Context(*devices)
    
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
        return self.default_context.compile(
            file_name, includes, defines,
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
            spir_std,
            type_checked,
            options
        )
    
    @override
    @staticmethod
    def _prefix()->str:
        return "CL_PLATFORM"

    @override
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetPlatformInfo

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.platform_info_types

    @override
    @staticmethod
    def _info_enum()->type:
        return cl_platform_info
    
    @override
    @staticmethod
    def _release_func()->CL.Func:
        return None
