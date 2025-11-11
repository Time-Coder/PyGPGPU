from ctypes import pointer
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .context import Context
    from .device import Device

from ..runtime import (
    CL,
    cl_command_queue,
    cl_int,
    IntEnum,
    CLInfo,
    cl_command_queue_info
)
from .clobject import CLObject


class CommandQueue(CLObject):

    def __init__(self, context:Context, device:Device):
        self._context:Context = context
        self._device:Device = device

        error_code = cl_int(0)

        try:
            cmd_queue_id:cl_command_queue = CL.clCreateCommandQueue(context.id, device.id, 0, pointer(error_code))
        except:
            cmd_queue_id:cl_command_queue = CL.clCreateCommandQueueWithProperties(context.id, device.id, 0, pointer(error_code))

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

    @property
    def _prefix(self)->str:
        return "CL_QUEUE"

    @property
    def _get_info_func(self)->CL.Func:
        return CL.clGetCommandQueueInfo

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        return CLInfo.command_queue_info_types

    @property
    def _info_enum(self)->type:
        return cl_command_queue_info

    @staticmethod
    def _release(command_queue_id):
        if not command_queue_id:
            return
        
        CL.clReleaseCommandQueue(command_queue_id)