from .. import genMat4x2, Flavor

import ctypes


class dmat4x2(genMat4x2):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL