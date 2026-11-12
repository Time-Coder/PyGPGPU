from ctypes import c_uint16

from pygf.genVec2 import genVec2


class ushort2(genVec2):

    _fields_ = [
        ('x', c_uint16),
        ('y', c_uint16)
    ]
    
