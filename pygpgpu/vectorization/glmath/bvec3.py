from .. import genVec3, Flavor

import ctypes


class bvec3(genVec3):
    
    @property
    def dtype(self)->type:
        return ctypes.c_bool
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 3