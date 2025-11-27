from ctypes import c_uint8
import numpy as np

from .genVec16 import genVec16


class uchar16(genVec16):

    _fields_ = [
        ('s0', c_uint8),
        ('s1', c_uint8),
        ('s2', c_uint8),
        ('s3', c_uint8),
        ('s4', c_uint8),
        ('s5', c_uint8),
        ('s6', c_uint8),
        ('s7', c_uint8),
        ('s8', c_uint8),
        ('s9', c_uint8),
        ('sA', c_uint8),
        ('sB', c_uint8),
        ('sC', c_uint8),
        ('sD', c_uint8),
        ('sE', c_uint8),
        ('sF', c_uint8),
    ]
    
    dtype = np.dtype(_fields_)