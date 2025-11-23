from ctypes import c_int16
import numpy as np

from .genVec4 import genVec4


class short4(genVec4):

    _fields_ = [
        ('x', c_int16),
        ('y', c_int16),
        ('z', c_int16),
        ('w', c_int16),
    ]
    
    dtype = np.dtype(_fields_)