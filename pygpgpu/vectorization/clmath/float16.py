from .. import genVec16, Flavor

import ctypes


class float16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL