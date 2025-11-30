from ctypes import c_double
import numpy as np

from .genVec8 import genVec8


class double8(genVec8):

    _fields_ = [
        ('s0', c_double),
        ('s1', c_double),
        ('s2', c_double),
        ('s3', c_double),
        ('s4', c_double),
        ('s5', c_double),
        ('s6', c_double),
        ('s7', c_double),
    ]
    
    dtype = np.dtype(_fields_)