from .. import genVec8, Flavor

import ctypes


class long8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_int64
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL