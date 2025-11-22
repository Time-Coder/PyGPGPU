from .. import genMat2x4, Flavor

import ctypes


class bmat2x4(genMat2x4):

    @property
    def dtype(self)->type:
        return ctypes.c_bool
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 8