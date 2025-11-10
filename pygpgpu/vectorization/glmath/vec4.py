from .. import genVec4, Flavor

import ctypes


class vec4(genVec4):
    
    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL