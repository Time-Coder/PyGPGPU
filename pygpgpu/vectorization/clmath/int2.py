from .. import genVec2, Flavor

import ctypes


class int2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_int32
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 8