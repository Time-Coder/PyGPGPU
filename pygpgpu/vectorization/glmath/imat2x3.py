from .. import genMat2x3, Flavor

import ctypes


class imat2x3(genMat2x3):

    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL