from .. import genVec2, Flavor

import ctypes


class dvec2(genVec2):
    
    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 16