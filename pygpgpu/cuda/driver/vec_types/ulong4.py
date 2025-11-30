from ctypes import c_ulong
import numpy as np

from ....vec_types import genVec4


class ulong4(genVec4):

    _fields_ = [
        ('x', c_ulong),
        ('y', c_ulong),
        ('z', c_ulong),
        ('w', c_ulong),
    ]
    
    dtype = np.dtype(_fields_)