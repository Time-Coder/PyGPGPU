from .. import genMat3x4, Flavor

import ctypes


class imat3x4(genMat3x4):

    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 48