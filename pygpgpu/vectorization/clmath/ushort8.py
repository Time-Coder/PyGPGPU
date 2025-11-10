from .. import genVec8, Flavor

import ctypes


class ushort8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_uint16
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL