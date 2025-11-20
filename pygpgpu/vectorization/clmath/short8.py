from .. import genVec8, Flavor

import ctypes


class short8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_int16
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 16