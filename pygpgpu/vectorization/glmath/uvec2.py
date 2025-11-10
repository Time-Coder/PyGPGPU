from .. import genVec2, Flavor

import ctypes


class uvec2(genVec2):
    
    @property
    def dtype(self)->type:
        return ctypes.c_uint