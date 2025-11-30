from ctypes import c_double
import numpy as np

from ....vec_types import genVec3


class double3(genVec3):

    _fields_ = [
        ('x', c_double),
        ('y', c_double),
        ('z', c_double),
        ('_', c_double),
    ]
    
    dtype = np.dtype(_fields_)