from ctypes import c_float
import numpy as np

from .genVec4 import genVec4


class float4(genVec4):

    _fields_ = [
        ('x', c_float),
        ('y', c_float),
        ('z', c_float),
        ('w', c_float),
    ]
    
    dtype = np.dtype(_fields_)