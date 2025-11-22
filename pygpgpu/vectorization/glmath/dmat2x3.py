from .. import genMat2x3, Flavor

import ctypes


class dmat2x3(genMat2x3):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 48