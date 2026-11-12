from ctypes import c_uint32

from ....vec_types import genVec3


class uint3(genVec3):

    _fields_ = [
        ('x', c_uint32),
        ('y', c_uint32),
        ('z', c_uint32),
        ('_', c_uint32),
    ]
    
