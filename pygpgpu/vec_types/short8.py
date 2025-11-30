from ctypes import c_int16
import numpy as np

from .genVec8 import genVec8


class short8(genVec8):

    _fields_ = [
        ('s0', c_int16),
        ('s1', c_int16),
        ('s2', c_int16),
        ('s3', c_int16),
        ('s4', c_int16),
        ('s5', c_int16),
        ('s6', c_int16),
        ('s7', c_int16),
    ]
    
    dtype = np.dtype(_fields_)