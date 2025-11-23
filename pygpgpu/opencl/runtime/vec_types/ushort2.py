from ctypes import c_uint16
import numpy as np

from .genVec2 import genVec2


class ushort2(genVec2):

    _fields_ = [
        ('x', c_uint16),
        ('y', c_uint16)
    ]
    
    dtype = np.dtype(_fields_)