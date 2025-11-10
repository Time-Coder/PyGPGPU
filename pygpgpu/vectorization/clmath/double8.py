from .. import genVec8, Flavor

import ctypes


class double8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL