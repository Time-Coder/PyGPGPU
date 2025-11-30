from ctypes import c_int32
import numpy as np

from ....vec_types import genVec3


class int3(genVec3):

    _fields_ = [
        ('x', c_int32),
        ('y', c_int32),
        ('z', c_int32)
    ]
    
    dtype = np.dtype(_fields_)