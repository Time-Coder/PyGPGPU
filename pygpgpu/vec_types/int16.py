from ctypes import c_int32
import numpy as np

from .genVec16 import genVec16


class int16(genVec16):

    _fields_ = [
        ('s0', c_int32),
        ('s1', c_int32),
        ('s2', c_int32),
        ('s3', c_int32),
        ('s4', c_int32),
        ('s5', c_int32),
        ('s6', c_int32),
        ('s7', c_int32),
        ('s8', c_int32),
        ('s9', c_int32),
        ('sA', c_int32),
        ('sB', c_int32),
        ('sC', c_int32),
        ('sD', c_int32),
        ('sE', c_int32),
        ('sF', c_int32),
    ]
    
    dtype = np.dtype(_fields_)