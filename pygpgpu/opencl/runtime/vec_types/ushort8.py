from ctypes import c_uint16
import numpy as np

from .genVec8 import genVec8


class ushort8(genVec8):

    _fields_ = [
        ('s0', c_uint16),
        ('s1', c_uint16),
        ('s2', c_uint16),
        ('s3', c_uint16),
        ('s4', c_uint16),
        ('s5', c_uint16),
        ('s6', c_uint16),
        ('s7', c_uint16),
    ]
    
    dtype = np.dtype(_fields_)