from ctypes import c_int16
import numpy as np

from ....vec_types import genVec3


class short3(genVec3):

    _fields_ = [
        ('x', c_int16),
        ('y', c_int16),
        ('z', c_int16),
        ('_', c_int16),
    ]
    
    dtype = np.dtype(_fields_)