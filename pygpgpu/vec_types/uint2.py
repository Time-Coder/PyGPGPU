from ctypes import c_uint32
import numpy as np

from .genVec2 import genVec2


class uint2(genVec2):

    _fields_ = [
        ('x', c_uint32),
        ('y', c_uint32)
    ]
    
    dtype = np.dtype(_fields_)