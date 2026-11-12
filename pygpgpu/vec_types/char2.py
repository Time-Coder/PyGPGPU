from ctypes import c_int8

from pygf.genVec2 import genVec2


class char2(genVec2):
    
    _fields_ = [
        ('x', c_int8),
        ('y', c_int8),
    ]
    
