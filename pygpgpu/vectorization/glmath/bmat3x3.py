from .. import genMat3x3, Flavor

import ctypes


class bmat3x3(genMat3x3):

    @property
    def dtype(self)->type:
        return ctypes.c_bool
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 9
    
bmat3 = bmat3x3