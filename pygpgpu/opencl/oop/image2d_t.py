from __future__ import annotations

from ctypes import c_void_p, pointer
from typing import Optional, Tuple, Union

import numpy as np
import imageio.v3 as iio

from ..runtime import (
    cl_mem_flags,
    cl_image_format,
    cl_image_desc,
    cl_channel_order,
    cl_channel_type,
    cl_mem_object_type
)


class image2d_t:

    def __init__(self, data:Union[str, np.ndarray, None]=None, shape:Optional[Tuple[int,...]]=None, dtype:Optional[type]=None, flags:Optional[cl_mem_flags]=None):
        if isinstance(data, str):
            data = iio.imread(data)

        self.data = data
        if data is None:
            self._shape = shape
            self._dtype = dtype
            self._update_format()

        if flags is None:
            flags = cl_mem_flags.CL_MEM_READ_WRITE

        self.flags = flags
    
    @property
    def data(self)->np.ndarray:
        return self._data
    
    @data.setter
    def data(self, data:Optional[np.ndarray]):
        if data is not None:
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            self._data = data
            self._host_ptr = data.ctypes.data_as(c_void_p)
            self._size = data.nbytes
            self._shape = data.shape
            self._dtype = data.dtype
            self._update_format()
        else:
            self._data = None
            self._host_ptr = None
    
    @property
    def size(self)->int:
        return self._size
    
    @property
    def host_ptr(self)->c_void_p:
        return self._host_ptr

    @property
    def format(self)->cl_image_format:
        return self._format
    
    @property
    def desc(self)->cl_image_desc:
        return self._desc
    
    @property
    def channel_order(self)->cl_channel_order:
        return self._channel_order
    
    @property
    def channel_type(self)->cl_channel_type:
        return self._channel_type
    
    @property
    def shape(self)->Tuple[int, ...]:
        return self._shape
    
    @shape.setter
    def shape(self, shape:Tuple[int, ...]):
        if self._shape != shape:
            self._shape = shape
            self._data = None
            self._host_ptr = None
            self._update_format()
    
    @property
    def dtype(self)->np.dtype:
        return self._dtype
    
    @dtype.setter
    def dtype(self, dtype:type)->None:
        if self._dtype != dtype:
            self._dtype = dtype
            self._data = None
            self._host_ptr = None
            self._update_format()
    
    @property
    def flags(self)->cl_mem_flags:
        return self._flags
    
    @flags.setter
    def flags(self, flags:cl_mem_flags)->None:
        self._flags = flags

    def load(self, file_name:str)->None:
        self.data = iio.imread(file_name)

    def save(self, file_name:str)->None:
        iio.imwrite(file_name, self.data)

    def _update_format(self)->None:
        if len(self._shape) == 2:
            self._channel_order = cl_channel_order.CL_LUMINANCE
        elif len(self._shape) == 3:
            if self._shape[2] == 1:
                self._channel_order = cl_channel_order.CL_LUMINANCE
            elif self._shape[2] == 2:
                self._channel_order = cl_channel_order.CL_RG
            elif self._shape[2] == 3:
                self._channel_order = cl_channel_order.CL_RGB
            elif self._shape[2] == 4:
                self._channel_order = cl_channel_order.CL_RGBA
            else:
                raise ValueError(f"image shape {self._shape} is not supported")
        else:
            raise ValueError(f"image shape {self._shape} is not supported")
        
        if self._dtype == np.int8:
            self._channel_type = cl_channel_type.CL_SIGNED_INT8
        elif self._dtype == np.uint8:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT8
        elif self._dtype == np.int16:
            self._channel_type = cl_channel_type.CL_SIGNED_INT16
        elif self._dtype == np.uint16:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT16
        elif self._dtype == np.int32:
            self._channel_type = cl_channel_type.CL_SIGNED_INT32
        elif self._dtype == np.uint32:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT32
        elif self._dtype == np.float16:
            self._channel_type = cl_channel_type.CL_HALF_FLOAT
        elif self._dtype == np.float32:
            self._channel_type = cl_channel_type.CL_FLOAT
        else:
            raise ValueError("channel type is not supported")
        
        self._format:cl_image_format = cl_image_format(self._channel_order, self._channel_type)
        self._desc:cl_image_desc = cl_image_desc(cl_mem_object_type.CL_MEM_OBJECT_IMAGE2D, self._shape[1], self._shape[0], 0, 0, 0, 0, 0, 0)

    def __repr__(self)->str:
        return f"image2d_t(format={self._format}, desc={self._desc}, flags={self._flags})"
    
    def __str__(self)->str:
        return f"image2d_t(format={self._format}, desc={self._desc}, flags={self._flags})"