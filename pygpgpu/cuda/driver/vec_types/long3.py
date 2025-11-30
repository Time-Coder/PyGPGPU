from ctypes import c_long
import numpy as np

from ....vec_types import genVec3


class long3(genVec3):

    _fields_ = [
        ('x', c_long),
        ('y', c_long),
        ('z', c_long)
    ]
    
    dtype = np.dtype(_fields_)