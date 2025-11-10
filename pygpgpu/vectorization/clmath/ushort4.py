from .. import genVec4, Flavor

import ctypes


class ushort4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_uint16
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL