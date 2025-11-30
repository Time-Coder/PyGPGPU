from ctypes import c_float
import numpy as np

from ....vec_types import genVec3


class float3(genVec3):

    _fields_ = [
        ('x', c_float),
        ('y', c_float),
        ('z', c_float)
    ]
    
    dtype = np.dtype(_fields_)