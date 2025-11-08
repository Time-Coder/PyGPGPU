from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from .platform import Platform
    from .context import Context

from ..runtime import CL, CLInfo, IntEnum, cl_device_id, cl_device_info
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
    def unique_key(self)->str:
        return sanitize_filename(f"{self.name} {self.version} {self.driver_version}")
    
    @property
    def _prefix(self)->str:
        return "CL_DEVICE"

    @property
    def _get_info_func(self)->CL.Func:
        return CL.clGetDeviceInfo

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        return CLInfo.device_info_types

    @property
    def _info_enum(self)->type:
        return cl_device_info