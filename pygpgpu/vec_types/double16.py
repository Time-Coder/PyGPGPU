from ctypes import c_double
import numpy as np

from .genVec16 import genVec16


class double16(genVec16):

    _fields_ = [
        ('s0', c_double),
        ('s1', c_double),
        ('s2', c_double),
        ('s3', c_double),
        ('s4', c_double),
        ('s5', c_double),
        ('s6', c_double),
        ('s7', c_double),
        ('s8', c_double),
        ('s9', c_double),
        ('sA', c_double),
        ('sB', c_double),
        ('sC', c_double),
        ('sD', c_double),
        ('sE', c_double),
        ('sF', c_double),
    ]
    
    dtype = np.dtype(_fields_)