from ctypes import c_uint8
import numpy as np

from .genVec3 import genVec3


class uchar3(genVec3):

    _fields_ = [
        ('x', c_uint8),
        ('y', c_uint8),
        ('z', c_uint8),
        ('_', c_uint8),
    ]
    
    dtype = np.dtype(_fields_)