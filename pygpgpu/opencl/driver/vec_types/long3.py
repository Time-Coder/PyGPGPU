from ctypes import c_int64
import numpy as np

from ....vec_types import genVec3


class long3(genVec3):

    _fields_ = [
        ('x', c_int64),
        ('y', c_int64),
        ('z', c_int64),
        ('_', c_int64),
    ]
    
    dtype = np.dtype(_fields_)