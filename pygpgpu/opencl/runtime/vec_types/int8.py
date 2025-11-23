from ctypes import c_int32
import numpy as np

from .genVec8 import genVec8


class int8(genVec8):

    _fields_ = [
        ('s0', c_int32),
        ('s1', c_int32),
        ('s2', c_int32),
        ('s3', c_int32),
        ('s4', c_int32),
        ('s5', c_int32),
        ('s6', c_int32),
        ('s7', c_int32),
    ]
    
    dtype = np.dtype(_fields_)