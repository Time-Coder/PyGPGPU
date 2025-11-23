from ctypes import c_uint8
import numpy as np

from .genVec4 import genVec4


class uchar4(genVec4):

    _fields_ = [
        ('x', c_uint8),
        ('y', c_uint8),
        ('z', c_uint8),
        ('w', c_uint8),
    ]
    
    dtype = np.dtype(_fields_)