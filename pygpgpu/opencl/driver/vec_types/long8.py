from ctypes import c_int64
import numpy as np

from ....vec_types import genVec8


class long8(genVec8):

    _fields_ = [
        ('s0', c_int64),
        ('s1', c_int64),
        ('s2', c_int64),
        ('s3', c_int64),
        ('s4', c_int64),
        ('s5', c_int64),
        ('s6', c_int64),
        ('s7', c_int64),
    ]
    
    dtype = np.dtype(_fields_)