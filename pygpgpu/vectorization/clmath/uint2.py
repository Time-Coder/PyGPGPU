from .. import genVec2, Flavor

import ctypes


class uint2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_uint32
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL