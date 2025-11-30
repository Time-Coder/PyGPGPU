from ctypes import c_int8
import numpy as np

from .genVec16 import genVec16


class char16(genVec16):

    _fields_ = [
        ('s0', c_int8),
        ('s1', c_int8),
        ('s2', c_int8),
        ('s3', c_int8),
        ('s4', c_int8),
        ('s5', c_int8),
        ('s6', c_int8),
        ('s7', c_int8),
        ('s8', c_int8),
        ('s9', c_int8),
        ('sA', c_int8),
        ('sB', c_int8),
        ('sC', c_int8),
        ('sD', c_int8),
        ('sE', c_int8),
        ('sF', c_int8),
    ]
    
    dtype = np.dtype(_fields_)