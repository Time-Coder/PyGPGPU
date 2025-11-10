from .. import genMat4x2, Flavor

import ctypes


class bmat4x2(genMat4x2):

    @property
    def dtype(self)->type:
        return ctypes.c_bool
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL