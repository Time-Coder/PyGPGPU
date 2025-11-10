from .. import genVec2, Flavor

import ctypes


class ivec2(genVec2):
    
    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL