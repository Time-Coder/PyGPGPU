from .. import genVec16, Flavor

import ctypes


class uchar16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_ubyte
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL