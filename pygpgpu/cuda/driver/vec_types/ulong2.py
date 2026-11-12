from ctypes import c_ulong

from ....vec_types import genVec2


class ulong2(genVec2):

    _fields_ = [
        ('x', c_ulong),
        ('y', c_ulong)
    ]
    
