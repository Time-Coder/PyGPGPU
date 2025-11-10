from .. import genVec2, Flavor

import ctypes


class bvec2(genVec2):
    
    @property
    def dtype(self)->type:
        return ctypes.c_bool
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL