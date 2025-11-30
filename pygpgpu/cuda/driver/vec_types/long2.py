from ctypes import c_long
import numpy as np

from ....vec_types import genVec2


class long2(genVec2):

    _fields_ = [
        ('x', c_long),
        ('y', c_long)
    ]
    
    dtype = np.dtype(_fields_)