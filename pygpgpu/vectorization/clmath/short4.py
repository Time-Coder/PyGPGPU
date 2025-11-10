from .. import genVec4, Flavor

import ctypes


class short4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_int16
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL