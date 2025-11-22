from .. import genMat4x4, Flavor

import ctypes


class bmat4x4(genMat4x4):

    @property
    def dtype(self)->type:
        return ctypes.c_bool
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 16
    
bmat4 = bmat4x4