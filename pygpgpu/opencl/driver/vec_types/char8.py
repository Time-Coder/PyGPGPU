from ctypes import c_int8
import numpy as np

from .genVec8 import genVec8


class char8(genVec8):

    _fields_ = [
        ('s0', c_int8),
        ('s1', c_int8),
        ('s2', c_int8),
        ('s3', c_int8),
        ('s4', c_int8),
        ('s5', c_int8),
        ('s6', c_int8),
        ('s7', c_int8),
    ]
    
    dtype = np.dtype(_fields_)