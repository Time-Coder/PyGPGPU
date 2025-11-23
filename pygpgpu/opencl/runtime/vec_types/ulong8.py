from ctypes import c_uint64
import numpy as np

from .genVec8 import genVec8


class ulong8(genVec8):

    _fields_ = [
        ('s0', c_uint64),
        ('s1', c_uint64),
        ('s2', c_uint64),
        ('s3', c_uint64),
        ('s4', c_uint64),
        ('s5', c_uint64),
        ('s6', c_uint64),
        ('s7', c_uint64),
    ]
    
    dtype = np.dtype(_fields_)