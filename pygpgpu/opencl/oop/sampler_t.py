from ctypes import pointer
from typing import override, Dict

from ..runtime import (
    cl_addressing_mode,
    cl_filter_mode,
    cl_sampler,
    CL,
    cl_int,
    CLInfo,
    IntEnum,
    cl_sampler_info
)

from .clobject import CLObject
from .context import Context


class sampler_t(CLObject):

    def __init__(self, context:Context, normalized_coords:bool, addressing_mode:cl_addressing_mode, filter_mode:cl_filter_mode)->None:
        error_code = cl_int(0)
        sampler_id:cl_sampler = CL.clCreateSampler(context.id, normalized_coords, addressing_mode, filter_mode, pointer(error_code))
        self._context:Context = context
        self._normalized_coords:bool = normalized_coords
        self._addressing_mode:cl_addressing_mode = addressing_mode
        self._filter_mode:cl_filter_mode = filter_mode
        CLObject.__init__(self, sampler_id)

    @override
    @staticmethod
    def _prefix()->str:
        return "CL_SAMPLER"

    @override
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetSamplerInfo

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.sampler_info_types

    @override
    @staticmethod
    def _info_enum()->type:
        return cl_sampler_info

    @override
    @staticmethod
    def _release_func()->CL.Func:
        return CL.clReleaseSampler