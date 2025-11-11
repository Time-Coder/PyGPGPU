from .clobject import CLObject
from typing import Dict

from ..runtime import (
    CL,
    cl_event,
    IntEnum
)


class Event(CLObject):

    def __init__(self):
        CLObject.__init__(self, 0)

    @property
    def _prefix(self)->str:
        pass

    @property
    def _get_info_func(self)->CL.Func:
        pass

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        pass

    @property
    def _info_enum(self)->type:
        pass

    @staticmethod
    def _release(id_):
        pass