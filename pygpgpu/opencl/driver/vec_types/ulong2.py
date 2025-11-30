from ctypes import c_uint64
import numpy as np

from ....vec_types import genVec2


class ulong2(genVec2):

    _fields_ = [
        ('x', c_uint64),
        ('y', c_uint64)
    ]
    
    dtype = np.dtype(_fields_)