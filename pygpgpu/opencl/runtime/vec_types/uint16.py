from ctypes import c_uint32
import numpy as np

from .genVec16 import genVec16


class uint16(genVec16):

    _fields_ = [
        ('s0', c_uint32),
        ('s1', c_uint32),
        ('s2', c_uint32),
        ('s3', c_uint32),
        ('s4', c_uint32),
        ('s5', c_uint32),
        ('s6', c_uint32),
        ('s7', c_uint32),
        ('s8', c_uint32),
        ('s9', c_uint32),
        ('sA', c_uint32),
        ('sB', c_uint32),
        ('sC', c_uint32),
        ('sD', c_uint32),
        ('sE', c_uint32),
        ('sF', c_uint32),
    ]
    
    dtype = np.dtype(_fields_)