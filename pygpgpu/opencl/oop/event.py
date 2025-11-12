from .clobject import CLObject
from ctypes import pointer
from typing import Dict, Optional

from ..runtime import (
    CL,
    cl_event,
    IntEnum,
    CLInfo,
    cl_event_info,
    cl_int,
    ptr_cl_int
)
from .context import Context


class Event(CLObject):

    def __init__(self, context:Context, event_id:Optional[cl_event]=None):
        error_code = cl_int(0)
        self._is_user_event:bool = False
        if not event_id:
            event_id = CL.clCreateUserEvent(context, pointer(error_code))
            self._is_user_event:bool = True

        CLObject.__init__(self, event_id)

    @property
    def _prefix(self)->str:
        return "CL_EVENT"

    @property
    def _get_info_func(self)->CL.Func:
        return CL.clGetEventInfo

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        return CLInfo.event_info_types

    @property
    def _info_enum(self)->type:
        return cl_event_info

    @staticmethod
    def _release(event_id:cl_event):
        if not event_id:
            return
        
        CL.clReleaseEvent(event_id)