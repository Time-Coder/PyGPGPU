from .. import genMat2x2, Flavor

import ctypes


class mat2x2(genMat2x2):

    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 16
    
mat2 = mat2x2