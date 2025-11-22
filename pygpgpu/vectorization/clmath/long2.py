from .. import genVec2, Flavor

import ctypes


class long2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_int64
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 16