from ctypes import c_uint64

from ....vec_types import genVec3


class ulong3(genVec3):

    _fields_ = [
        ('x', c_uint64),
        ('y', c_uint64),
        ('z', c_uint64),
        ('_', c_uint64),
    ]
    
