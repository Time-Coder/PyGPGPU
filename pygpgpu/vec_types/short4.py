from ctypes import c_int16

from .genVec4 import genVec4


class short4(genVec4):

    _fields_ = [
        ('x', c_int16),
        ('y', c_int16),
        ('z', c_int16),
        ('w', c_int16),
    ]
    
