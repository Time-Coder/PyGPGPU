from .. import genVec8, Flavor

import ctypes


class int8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_int32
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL