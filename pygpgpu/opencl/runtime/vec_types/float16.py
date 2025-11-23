from ctypes import c_float
import numpy as np

from .genVec16 import genVec16


class float16(genVec16):

    _fields_ = [
        ('s0', c_float),
        ('s1', c_float),
        ('s2', c_float),
        ('s3', c_float),
        ('s4', c_float),
        ('s5', c_float),
        ('s6', c_float),
        ('s7', c_float),
        ('s8', c_float),
        ('s9', c_float),
        ('sA', c_float),
        ('sB', c_float),
        ('sC', c_float),
        ('sD', c_float),
        ('sE', c_float),
        ('sF', c_float),
    ]
    
    dtype = np.dtype(_fields_)