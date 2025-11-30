from ctypes import c_uint32
import numpy as np

from ....vec_types import genVec3


class uint3(genVec3):

    _fields_ = [
        ('x', c_uint32),
        ('y', c_uint32),
        ('z', c_uint32)
    ]
    
    dtype = np.dtype(_fields_)