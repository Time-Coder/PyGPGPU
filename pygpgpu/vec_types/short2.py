from ctypes import c_int16

from .genVec2 import genVec2


class short2(genVec2):

    _fields_ = [
        ('x', c_int16),
        ('y', c_int16)
    ]
    
