from ctypes import c_long
import numpy as np

from ....vec_types import genVec4


class long4(genVec4):

    _fields_ = [
        ('x', c_long),
        ('y', c_long),
        ('z', c_long),
        ('w', c_long),
    ]
    
    dtype = np.dtype(_fields_)