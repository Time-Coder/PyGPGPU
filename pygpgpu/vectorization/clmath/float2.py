from .. import genVec2, Flavor

import ctypes


class float2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL