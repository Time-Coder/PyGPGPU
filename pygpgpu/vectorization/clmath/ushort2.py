from .. import genVec2, Flavor

import ctypes


class ushort2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_uint16
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL