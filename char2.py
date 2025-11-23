from ctypes import c_int8, Structure
import numpy as np


class char2(Structure):

    _fields_ = [
        ('x', c_int8),
        ('y', c_int8)
    ]

    dtype = np.dtype(_fields_)

    