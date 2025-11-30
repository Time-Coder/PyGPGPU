from ctypes import c_uint16
import numpy as np

from ....vec_types import genVec3


class ushort3(genVec3):

    _fields_ = [
        ('x', c_uint16),
        ('y', c_uint16),
        ('z', c_uint16)
    ]
    
    dtype = np.dtype(_fields_)