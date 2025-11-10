from .. import genVec4, Flavor

import ctypes


class int4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_int32
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL