from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, override

if TYPE_CHECKING:
    from .platform import Platform
    from .context import Context

from ..driver import CL, CLInfo, IntEnum, cl_device_id, cl_device_info
from .clobject import CLObject
from ...utils import sanitize_filename


class Device(CLObject):

    def __init__(self, device_id:cl_device_id, platform:Platform):
        CLObject.__init__(self, device_id)
        self.__platform:Platform = platform

    @property
    def platform(self)->Platform:
        return self.__platform
    
    @property
    def extensions(self)->List[str]:
        return self.__getattr__("extensions").split(" ")
    
    def create_context(self)->Context:
        from .context import Context
        return Context(self)
    
    @property
    def default_context(self)->Context:
        return self.platform.default_context
    
    @property
    def unique_key(self)->str:
        return sanitize_filename(f"{self.name} {self.version} {self.driver_version}")
    
    @override
    @staticmethod
    def _prefix()->str:
        return "CL_DEVICE"

    @override
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetDeviceInfo

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.device_info_types

    @override
    @staticmethod
    def _info_enum()->type:
        return cl_device_info
    
    @override
    @staticmethod
    def _release_func()->CL.Func:
        return None