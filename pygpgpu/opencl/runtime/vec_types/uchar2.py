from ctypes import c_uint8
import numpy as np

from .genVec2 import genVec2


class uchar2(genVec2):

    _fields_ = [
        ('x', c_uint8),
        ('y', c_uint8)
    ]
    
    dtype = np.dtype(_fields_)