from .. import genVec4, Flavor

import ctypes


class uchar4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_ubyte
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL