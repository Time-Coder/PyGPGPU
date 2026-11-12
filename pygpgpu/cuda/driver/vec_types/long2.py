from ctypes import c_long

from ....vec_types import genVec2


class long2(genVec2):

    _fields_ = [
        ('x', c_long),
        ('y', c_long)
    ]
    
