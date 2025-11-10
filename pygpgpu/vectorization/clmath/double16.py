from .. import genVec16, Flavor

import ctypes


class double16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL