from .. import genVec8, Flavor

import ctypes


class char8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_char
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL