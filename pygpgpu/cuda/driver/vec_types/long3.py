from ctypes import c_long

from ....vec_types import genVec3


class long3(genVec3):

    _fields_ = [
        ('x', c_long),
        ('y', c_long),
        ('z', c_long)
    ]
    
