from ctypes import c_uint64
import numpy as np

from ....vec_types import genVec3


class ulong3(genVec3):

    _fields_ = [
        ('x', c_uint64),
        ('y', c_uint64),
        ('z', c_uint64),
        ('_', c_uint64),
    ]
    
    dtype = np.dtype(_fields_)