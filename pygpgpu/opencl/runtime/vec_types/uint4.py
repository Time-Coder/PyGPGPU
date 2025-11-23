from ctypes import c_uint32
import numpy as np

from .genVec4 import genVec4


class uint4(genVec4):

    _fields_ = [
        ('x', c_uint32),
        ('y', c_uint32),
        ('z', c_uint32),
        ('w', c_uint32),
    ]
    
    dtype = np.dtype(_fields_)