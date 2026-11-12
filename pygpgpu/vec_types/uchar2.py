from ctypes import c_uint8

from pygf.genVec2 import genVec2


class uchar2(genVec2):

    _fields_ = [
        ('x', c_uint8),
        ('y', c_uint8)
    ]
    
