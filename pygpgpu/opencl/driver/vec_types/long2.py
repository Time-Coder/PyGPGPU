from ctypes import c_int64

from ....vec_types import genVec2


class long2(genVec2):

    _fields_ = [
        ('x', c_int64),
        ('y', c_int64)
    ]
    
