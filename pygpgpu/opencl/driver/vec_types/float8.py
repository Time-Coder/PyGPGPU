from ctypes import c_float
import numpy as np

from .genVec8 import genVec8


class float8(genVec8):

    _fields_ = [
        ('s0', c_float),
        ('s1', c_float),
        ('s2', c_float),
        ('s3', c_float),
        ('s4', c_float),
        ('s5', c_float),
        ('s6', c_float),
        ('s7', c_float),
    ]
    
    dtype = np.dtype(_fields_)