from ctypes import c_int16
import numpy as np

from .genVec2 import genVec2


class short2(genVec2):

    _fields_ = [
        ('x', c_int16),
        ('y', c_int16)
    ]
    
    dtype = np.dtype(_fields_)