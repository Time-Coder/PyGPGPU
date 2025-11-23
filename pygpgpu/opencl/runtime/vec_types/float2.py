from ctypes import c_float
import numpy as np

from .genVec2 import genVec2


class float2(genVec2):

    _fields_ = [
        ('x', c_float),
        ('y', c_float)
    ]
    
    dtype = np.dtype(_fields_)
    