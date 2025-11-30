from ctypes import c_uint64
import numpy as np

from ....vec_types import genVec16


class ulong16(genVec16):

    _fields_ = [
        ('s0', c_uint64),
        ('s1', c_uint64),
        ('s2', c_uint64),
        ('s3', c_uint64),
        ('s4', c_uint64),
        ('s5', c_uint64),
        ('s6', c_uint64),
        ('s7', c_uint64),
        ('s8', c_uint64),
        ('s9', c_uint64),
        ('sA', c_uint64),
        ('sB', c_uint64),
        ('sC', c_uint64),
        ('sD', c_uint64),
        ('sE', c_uint64),
        ('sF', c_uint64),
    ]
    
    dtype = np.dtype(_fields_)