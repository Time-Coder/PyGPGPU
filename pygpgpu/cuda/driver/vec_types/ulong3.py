from ctypes import c_ulong

from ....vec_types import genVec3


class ulong3(genVec3):

    _fields_ = [
        ('x', c_ulong),
        ('y', c_ulong),
        ('z', c_ulong)
    ]
    
