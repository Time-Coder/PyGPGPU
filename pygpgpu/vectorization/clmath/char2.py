from .. import genVec2, Flavor

import ctypes


class char2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_char
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL