from .. import genVec8, Flavor

import ctypes


class uchar8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_ubyte
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL