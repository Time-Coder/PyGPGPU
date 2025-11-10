from .. import genVec2, Flavor

import ctypes


class short2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_int16
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL