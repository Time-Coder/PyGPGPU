from ctypes import c_uint16
import numpy as np

from .genVec4 import genVec4


class ushort4(genVec4):

    _fields_ = [
        ('x', c_uint16),
        ('y', c_uint16),
        ('z', c_uint16),
        ('w', c_uint16),
    ]
    
    dtype = np.dtype(_fields_)