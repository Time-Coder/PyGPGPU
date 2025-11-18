from __future__ import annotations

from ctypes import c_void_p
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

    def __init__(self, data:Union[str, np.ndarray], flags:Optional[cl_mem_flags]=None):
        if isinstance(data, str):
            data = iio.imread(data)

        self.data = data

        if flags is None:
            flags = cl_mem_flags.CL_MEM_READ_WRITE

        self.flags = flags
    
    @property
    def data(self)->np.ndarray:
        return self._data
    
    @data.setter
    def data(self, data:np.ndarray):
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        host_ptr = data.ctypes.data_as(c_void_p)
        shape = data.shape
        dtype = data.dtype
        size = data.nbytes

        if len(shape) == 2:
            self._channel_order = cl_channel_order.CL_LUMINANCE
        elif len(shape) == 3:
            if shape[2] == 1:
                self._channel_order = cl_channel_order.CL_LUMINANCE
            elif shape[2] == 2:
                self._channel_order = cl_channel_order.CL_RG
            elif shape[2] == 3:
                self._channel_order = cl_channel_order.CL_RGB
            elif shape[2] == 4:
                self._channel_order == cl_channel_order.CL_RGBA
            else:
                raise ValueError(f"image shape {shape} is not supported")
        else:
            raise ValueError(f"image shape {shape} is not supported")
        
        if dtype == np.int8:
            self._channel_type = cl_channel_type.CL_SIGNED_INT8
        elif dtype == np.uint8:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT8
        elif dtype == np.int16:
            self._channel_type = cl_channel_type.CL_SIGNED_INT16
        elif dtype == np.uint16:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT16
        elif dtype == np.int32:
            self._channel_type = cl_channel_type.CL_SIGNED_INT32
        elif dtype == np.uint32:
            self._channel_type = cl_channel_type.CL_UNSIGNED_INT32
        elif dtype == np.float16:
            self._channel_type = cl_channel_type.CL_HALF_FLOAT
        elif dtype == np.float32:
            self._channel_type = cl_channel_type.CL_FLOAT
        else:
            raise ValueError("channel type is not supported")
        
        self._data = data
        self._size = size
        self._host_ptr = host_ptr
        self._shape = shape
        self._dtype = dtype
        self._format:cl_image_format = cl_image_format(self._channel_order, self._channel_type)
        self._desc:cl_image_desc = cl_image_desc(cl_mem_object_type.CL_MEM_OBJECT_IMAGE2D, shape[1], shape[0], 0, 0, 0, 0, 0, 0)
    
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
    def shape(self)->Tuple[int, int, int]:
        return self._shape
    
    @property
    def dtype(self)->np.dtype:
        return self._dtype
    
    @property
    def flags(self)->cl_mem_flags:
        return self._flags
    
    @flags.setter
    def flags(self, flags:cl_mem_flags)->None:
        if (
            not (flags & cl_mem_flags.CL_MEM_USE_HOST_PTR) and 
            not (flags & cl_mem_flags.CL_MEM_COPY_HOST_PTR)
        ):
            if self._size < 4*1024*1024:
                flags |= cl_mem_flags.CL_MEM_COPY_HOST_PTR
            else:
                flags |= cl_mem_flags.CL_MEM_USE_HOST_PTR

        self._flags = flags

    def load(self, file_name:str)->None:
        self.data = iio.imread(file_name)

    def save(self, file_name:str)->None:
        iio.imwrite(file_name, self.data)