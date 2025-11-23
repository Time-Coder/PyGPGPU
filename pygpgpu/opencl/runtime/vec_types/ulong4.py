from ctypes import c_uint64
import numpy as np

from .genVec4 import genVec4


class ulong4(genVec4):

    _fields_ = [
        ('x', c_uint64),
        ('y', c_uint64),
        ('z', c_uint64),
        ('w', c_uint64),
    ]
    
    dtype = np.dtype(_fields_)