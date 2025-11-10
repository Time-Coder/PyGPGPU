from .. import genMat2x4, Flavor

import ctypes


class dmat2x4(genMat2x4):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL