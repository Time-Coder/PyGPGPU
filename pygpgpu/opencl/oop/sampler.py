from __future__ import annotations
from ctypes import pointer
from typing import override, Dict, TYPE_CHECKING

from ..runtime import (
    cl_addressing_mode,
    cl_filter_mode,
    cl_sampler,
    CL,
    cl_int,
    CLInfo,
    IntEnum,
    cl_sampler_info,
    cl_ulong,
    sampler_t
)

if TYPE_CHECKING:
    from .context import Context

from .clobject import CLObject


class sampler(CLObject):

    def __init__(self, context:Context, sampler_t_:sampler_t)->None:
        error_code = cl_int(0)
        try:
            sampler_id:cl_sampler = CL.clCreateSampler(context.id, sampler_t_.normalized_coords, sampler_t_.addressing_mode, sampler_t_.filter_mode, pointer(error_code))
        except RuntimeError:
            props = (cl_ulong * 7)(
                cl_sampler_info.CL_SAMPLER_NORMALIZED_COORDS, sampler_t_.normalized_coords,
                cl_sampler_info.CL_SAMPLER_ADDRESSING_MODE,   sampler_t_.addressing_mode,
                cl_sampler_info.CL_SAMPLER_FILTER_MODE,       sampler_t_.filter_mode,
                0
            )
            sampler_id:cl_sampler = CL.clCreateSamplerWithProperties(context.id, props, pointer(error_code))

        self._context:Context = context
        self._sampler_t:sampler_t = sampler_t_
        CLObject.__init__(self, sampler_id)

    @property
    def context(self)->Context:
        return self._context
    
    @property
    def normalized_coords(self)->bool:
        return self._sampler_t.normalized_coords
    
    @property
    def addressing_mode(self)->cl_addressing_mode:
        return self._sampler_t._addressing_mode
    
    @property
    def filter_mode(self)->cl_filter_mode:
        return self._sampler_t._filter_mode

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