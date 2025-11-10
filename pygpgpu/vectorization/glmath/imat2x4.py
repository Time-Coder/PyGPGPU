from .. import genMat2x4, Flavor

import ctypes


class imat2x4(genMat2x4):

    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL