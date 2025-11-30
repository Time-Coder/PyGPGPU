from ctypes import c_ulong
import numpy as np

from ....vec_types import genVec2


class ulong2(genVec2):

    _fields_ = [
        ('x', c_ulong),
        ('y', c_ulong)
    ]
    
    dtype = np.dtype(_fields_)