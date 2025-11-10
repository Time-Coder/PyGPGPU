from .. import genVec2, Flavor

import ctypes


class ulong2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_uint64
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL