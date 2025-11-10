from .. import genVec3, Flavor

import ctypes


class uvec3(genVec3):
    
    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL