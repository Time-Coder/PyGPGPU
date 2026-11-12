from ctypes import c_uint16

from .genVec4 import genVec4


class ushort4(genVec4):

    _fields_ = [
        ('x', c_uint16),
        ('y', c_uint16),
        ('z', c_uint16),
        ('w', c_uint16),
    ]
    
