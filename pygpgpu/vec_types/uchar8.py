from ctypes import c_uint8
import numpy as np

from .genVec8 import genVec8


class uchar8(genVec8):

    _fields_ = [
        ('s0', c_uint8),
        ('s1', c_uint8),
        ('s2', c_uint8),
        ('s3', c_uint8),
        ('s4', c_uint8),
        ('s5', c_uint8),
        ('s6', c_uint8),
        ('s7', c_uint8),
    ]
    
    dtype = np.dtype(_fields_)