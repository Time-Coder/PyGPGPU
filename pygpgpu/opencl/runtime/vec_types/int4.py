from ctypes import c_int32
import numpy as np

from .genVec4 import genVec4


class int4(genVec4):

    _fields_ = [
        ('x', c_int32),
        ('y', c_int32),
        ('z', c_int32),
        ('w', c_int32),
    ]
    
    dtype = np.dtype(_fields_)