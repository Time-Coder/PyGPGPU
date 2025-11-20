from .. import genVec16, Flavor

import ctypes


class long16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_int64
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 128