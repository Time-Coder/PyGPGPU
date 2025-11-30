from ctypes import c_int64
import numpy as np

from ....vec_types import genVec4


class long4(genVec4):

    _fields_ = [
        ('x', c_int64),
        ('y', c_int64),
        ('z', c_int64),
        ('w', c_int64),
    ]
    
    dtype = np.dtype(_fields_)