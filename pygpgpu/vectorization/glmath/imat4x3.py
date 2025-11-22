from .. import genMat4x3, Flavor

import ctypes


class imat4x3(genMat4x3):

    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 48