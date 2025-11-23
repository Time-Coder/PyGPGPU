from ctypes import c_double
import numpy as np

from .genVec2 import genVec2


class double2(genVec2):

    _fields_ = [
        ('x', c_double),
        ('y', c_double)
    ]
    
    dtype = np.dtype(_fields_)