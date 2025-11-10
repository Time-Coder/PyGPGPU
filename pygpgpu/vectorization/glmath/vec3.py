from .. import genVec3, Flavor

import ctypes


class vec3(genVec3):
    
    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    