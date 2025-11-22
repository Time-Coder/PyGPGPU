from .. import genVec4, Flavor

import ctypes


class dvec4(genVec4):
    
    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 32