from ctypes import c_int32
import numpy as np

from .genVec2 import genVec2


class int2(genVec2):

    _fields_ = [
        ('x', c_int32),
        ('y', c_int32)
    ]
    
    dtype = np.dtype(_fields_)