from ctypes import c_uint16

from ....vec_types import genVec3


class ushort3(genVec3):

    _fields_ = [
        ('x', c_uint16),
        ('y', c_uint16),
        ('z', c_uint16),
        ('_', c_uint16),
    ]
    
