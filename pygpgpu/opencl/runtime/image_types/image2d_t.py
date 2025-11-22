from typing import override

from ..cltypes import (
    cl_image_format,
    cl_image_desc,
    cl_channel_order,
    cl_mem_object_type
)
from .imagend_t import imagend_t


class image2d_t(imagend_t):

    @override
    def _update_format(self)->None:
        if len(self._shape) == 2:
            self._channel_order = cl_channel_order.CL_LUMINANCE
        elif len(self._shape) == 3:
            self._channel_order = self._get_channel_order(self._shape[2])
        else:
            raise ValueError(f"{self.__class__.__name__} data shape {self._shape} is not supported")
        
        self._channel_type = self._get_channel_type(self._dtype)
        self._format:cl_image_format = cl_image_format(self._channel_order, self._channel_type)
        self._desc:cl_image_desc = cl_image_desc(cl_mem_object_type.CL_MEM_OBJECT_IMAGE2D, self._shape[1], self._shape[0], 0, 0, 0, 0, 0, 0)
