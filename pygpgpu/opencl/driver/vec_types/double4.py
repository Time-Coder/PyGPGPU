from ctypes import c_double
import numpy as np

from .genVec4 import genVec4


class double4(genVec4):

    _fields_ = [
        ('x', c_double),
        ('y', c_double),
        ('z', c_double),
        ('w', c_double),
    ]
    
    dtype = np.dtype(_fields_)