from .. import genVec16, Flavor

import ctypes


class uint16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_uint32
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL