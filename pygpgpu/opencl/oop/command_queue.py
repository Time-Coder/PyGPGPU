from __future__ import annotations
from ctypes import pointer
from typing import TYPE_CHECKING, Dict, Optional, override

if TYPE_CHECKING:
    from .context import Context
    from .device import Device

from ..driver import (
    CL,
    cl_command_queue,
    cl_int,
    IntEnum,
    CLInfo,
    cl_command_queue_info,
    cl_command_queue_properties,
    cl_ulong
)
from .clobject import CLObject


class CommandQueue(CLObject):

    def __init__(self, context:Context, device:Device, properties:Optional[cl_command_queue_properties]=None):
        self._context:Context = context
        self._device:Device = device

        error_code = cl_int(0)

        try:
            if properties is None:
                properties = 0

            cmd_queue_id:cl_command_queue = CL.clCreateCommandQueue(context.id, device.id, properties, pointer(error_code))
        except:
            if properties is None:
                used_properties = None
            else:
                used_properties = (cl_ulong * 3)(cl_command_queue_info.CL_QUEUE_PROPERTIES, properties, 0)
            cmd_queue_id:cl_command_queue = CL.clCreateCommandQueueWithProperties(context.id, device.id, used_properties, pointer(error_code))

        CLObject.__init__(self, cmd_queue_id)

    def wait(self):
        CL.clFinish(self.id)

    @property
    def context(self)->Context:
        return self._context
    
    @property
    def device(self)->Device:
        return self._device
    
    def __len__(self)->int:
        return self.size

    @override
    @staticmethod
    def _prefix()->str:
        return "CL_QUEUE"

    @override
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetCommandQueueInfo

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.command_queue_info_types

    @override
    @staticmethod
    def _info_enum()->type:
        return cl_command_queue_info

    @override
    @staticmethod
    def _release_func():        
        return CL.clReleaseCommandQueue