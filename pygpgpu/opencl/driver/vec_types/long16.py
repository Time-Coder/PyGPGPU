from ctypes import c_int64
import numpy as np

from .genVec16 import genVec16


class long16(genVec16):

    _fields_ = [
        ('s0', c_int64),
        ('s1', c_int64),
        ('s2', c_int64),
        ('s3', c_int64),
        ('s4', c_int64),
        ('s5', c_int64),
        ('s6', c_int64),
        ('s7', c_int64),
        ('s8', c_int64),
        ('s9', c_int64),
        ('sA', c_int64),
        ('sB', c_int64),
        ('sC', c_int64),
        ('sD', c_int64),
        ('sE', c_int64),
        ('sF', c_int64),
    ]
    
    dtype = np.dtype(_fields_)