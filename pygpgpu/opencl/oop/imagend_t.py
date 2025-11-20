from __future__ import annotations

from ctypes import c_void_p
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import imageio.v3 as iio

from ..runtime import (
    cl_mem_flags,
    cl_image_format,
    cl_image_desc,
    cl_channel_order,
    cl_channel_type
)


class imagend_t(ABC):

    def __init__(self, data:Union[str, np.ndarray, None]=None, shape:Optional[Tuple[int,...]]=None, dtype:Optional[type]=None, flags:Optional[cl_mem_flags]=None):
        self._format:Optional[cl_image_format] = None
        self._desc:Optional[cl_image_desc] = None
        self._channel_order:Optional[cl_channel_order] = None
        self._channel_type:Optional[cl_channel_type] = None
        
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

    @staticmethod
    def _get_channel_type(dtype)->cl_channel_type:
        if dtype == np.int8:
            return cl_channel_type.CL_SIGNED_INT8
        elif dtype == np.uint8:
            return cl_channel_type.CL_UNSIGNED_INT8
        elif dtype == np.int16:
            return cl_channel_type.CL_SIGNED_INT16
        elif dtype == np.uint16:
            return cl_channel_type.CL_UNSIGNED_INT16
        elif dtype == np.int32:
            return cl_channel_type.CL_SIGNED_INT32
        elif dtype == np.uint32:
            return cl_channel_type.CL_UNSIGNED_INT32
        elif dtype == np.float16:
            return cl_channel_type.CL_HALF_FLOAT
        elif dtype == np.float32:
            return cl_channel_type.CL_FLOAT
        else:
            raise ValueError(f"dtype {dtype} is not supported")
        
    @staticmethod
    def _get_channel_order(n_channels:int)->cl_channel_order:
        if n_channels == 1:
            return cl_channel_order.CL_LUMINANCE
        elif n_channels == 2:
            return cl_channel_order.CL_RG
        elif n_channels == 3:
            return cl_channel_order.CL_RGB
        elif n_channels == 4:
            return cl_channel_order.CL_RGBA
        else:
            raise ValueError(f"{n_channels} channels is not supported")

    @abstractmethod
    def _update_format(self)->None:
        pass
