from ctypes import c_int16
import numpy as np

from .genVec16 import genVec16


class short16(genVec16):

    _fields_ = [
        ('s0', c_int16),
        ('s1', c_int16),
        ('s2', c_int16),
        ('s3', c_int16),
        ('s4', c_int16),
        ('s5', c_int16),
        ('s6', c_int16),
        ('s7', c_int16),
        ('s8', c_int16),
        ('s9', c_int16),
        ('sA', c_int16),
        ('sB', c_int16),
        ('sC', c_int16),
        ('sD', c_int16),
        ('sE', c_int16),
        ('sF', c_int16),
    ]
    
    dtype = np.dtype(_fields_)