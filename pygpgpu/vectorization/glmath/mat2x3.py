from .. import genMat2x3, Flavor

import ctypes


class mat2x3(genMat2x3):

    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 24