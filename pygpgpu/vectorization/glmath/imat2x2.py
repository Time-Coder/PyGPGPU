from .. import genMat2x2, Flavor

import ctypes


class imat2x2(genMat2x2):

    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
imat2 = imat2x2