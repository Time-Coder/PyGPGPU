from .. import genVec16, Flavor

import ctypes


class ushort16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_uint16
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 32