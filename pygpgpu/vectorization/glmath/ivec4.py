from .. import genVec4, Flavor

import ctypes


class ivec4(genVec4):
    
    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL