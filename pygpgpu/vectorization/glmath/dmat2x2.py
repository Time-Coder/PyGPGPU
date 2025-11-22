from .. import genMat2x2, Flavor

import ctypes


class dmat2x2(genMat2x2):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 32
    
dmat2 = dmat2x2