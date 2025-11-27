from ctypes import c_int8
import numpy as np

from .genVec2 import genVec2


class char2(genVec2):
    
    _fields_ = [
        ('x', c_int8),
        ('y', c_int8),
    ]
    
    dtype = np.dtype(_fields_)
