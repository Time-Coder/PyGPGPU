from .. import genMat3x2, Flavor

import ctypes


class imat3x2(genMat3x2):

    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL