from .. import genVec2, Flavor

import ctypes


class uchar2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_ubyte
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 2