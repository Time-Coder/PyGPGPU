from ctypes import c_uint16
import numpy as np

from .genVec16 import genVec16


class ushort16(genVec16):

    _fields_ = [
        ('s0', c_uint16),
        ('s1', c_uint16),
        ('s2', c_uint16),
        ('s3', c_uint16),
        ('s4', c_uint16),
        ('s5', c_uint16),
        ('s6', c_uint16),
        ('s7', c_uint16),
        ('s8', c_uint16),
        ('s9', c_uint16),
        ('sA', c_uint16),
        ('sB', c_uint16),
        ('sC', c_uint16),
        ('sD', c_uint16),
        ('sE', c_uint16),
        ('sF', c_uint16),
    ]
    
    dtype = np.dtype(_fields_)