from ctypes import c_int8
import numpy as np

from .genVec4 import genVec4


class char4(genVec4):
    
    _fields_ = [
        ('x', c_int8),
        ('y', c_int8),
        ('z', c_int8),
        ('w', c_int8),
    ]
    
    dtype = np.dtype(_fields_)