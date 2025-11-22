from .. import genVec4, Flavor

import ctypes


class char4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_char
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 4