from ctypes import c_uint32
import numpy as np

from .genVec8 import genVec8


class uint8(genVec8):

    _fields_ = [
        ('s0', c_uint32),
        ('s1', c_uint32),
        ('s2', c_uint32),
        ('s3', c_uint32),
        ('s4', c_uint32),
        ('s5', c_uint32),
        ('s6', c_uint32),
        ('s7', c_uint32),
    ]
    
    dtype = np.dtype(_fields_)