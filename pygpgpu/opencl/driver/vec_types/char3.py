from ctypes import c_int8

from ....vec_types import genVec3


class char3(genVec3):
    
    _fields_ = [
        ('x', c_int8),
        ('y', c_int8),
        ('z', c_int8),
        ('_', c_int8),
    ]
    
