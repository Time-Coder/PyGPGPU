from __future__ import annotations

from ctypes import pointer
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ..runtime import (
    cl_mem,
    CL,
    cl_int,
    cl_image_format,
    cl_image_desc,
    cl_channel_order,
    cl_channel_type
)

if TYPE_CHECKING:
    from .context import Context

from .mem_object import MemObject
from .imagend_t import imagend_t


class imagend(MemObject):

    def __init__(self, context:Context, image:imagend_t):
        self._image:imagend_t = image
        error_code = cl_int(0)
        image_id:cl_mem = CL.clCreateImage(context.id, self._image.flags, pointer(self._image.format), pointer(self._image.desc), self._image.host_ptr, pointer(error_code))
        MemObject.__init__(self, context, image_id, self._image.data, self._image.host_ptr, self._image.size, self._image.flags)
    
    @property
    def format(self)->cl_image_format:
        return self._image.format
    
    @property
    def desc(self)->cl_image_desc:
        return self._image.desc
    
    @property
    def channel_order(self)->cl_channel_order:
        return self._image.channel_order
    
    @property
    def channel_type(self)->cl_channel_type:
        return self._image.channel_type
    
    @property
    def shape(self)->Tuple[int, ...]:
        return self._image.shape
    
    @property
    def dtype(self)->np.dtype:
        return self._image.dtype