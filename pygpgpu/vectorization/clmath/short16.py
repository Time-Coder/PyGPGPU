from .. import genVec16, Flavor

import ctypes


class short16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_int16
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 32