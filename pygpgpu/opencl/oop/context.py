from ctypes import pointer, c_void_p
from typing import Tuple, Dict, Optional, Any, List, Set
import weakref

from ..runtime import (
    cl_device_id,
    cl_int,
    CL,
    CLInfo,
    IntEnum,
    cl_context,
    cl_context_info,
    CL_CONTEXT_NOTIFY_CALLBACK
)
from .clobject import CLObject
from .device import Device
from .platform import Platform
from ...kernel_parser import CPreprocessor


class Context(CLObject):

    def __init__(self, *devices):
        CLObject.__init__(self, 0)

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
        self._clean_code:str = ""
        self._line_map:Dict[str, str] = {}
        self._related_files:Set[str] = set()

    def _init(self):
        if self._id:
            return
        
        n_devices = len(self._devices)
        
        pfn_notify = CL_CONTEXT_NOTIFY_CALLBACK(Context._pfn_notify)

        properties = None
        device_ids = (cl_device_id * n_devices)()
        for i in range(n_devices):
            device_ids[i] = self._devices[i].id

        user_data = None
        errcode = cl_int()

        self._id = CL.clCreateContext(properties, n_devices, device_ids, pfn_notify, user_data, pointer(errcode))
        self._finalizer = weakref.finalize(self, Context._release, self._id)

    @property
    def id(self)->c_void_p:
        self._init()
        return self._id

    def _release(context_id:cl_context):
        if not context_id:
            return
        
        CL.clReleaseContext(context_id)

    @property
    def devices(self)->Tuple[Device]:
        return self._devices
    
    @property
    def platform(self)->Platform:
        return self._platform
    
    def compile(self, file_name:str, includes:Optional[List[str]] = None, defines:Optional[Dict[str, Any]] = None):
        (
            self.__clean_code,
            self.__line_map,
            self.__related_files
        ) = CPreprocessor.macros_expand_file(file_name, includes, defines)

    def _pfn_notify(errinfo, private_info, cb, user_data):
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