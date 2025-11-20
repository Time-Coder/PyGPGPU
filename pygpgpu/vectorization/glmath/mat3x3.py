from .. import genMat3x3, Flavor

import ctypes


class mat3x3(genMat3x3):

    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 36
    
mat3 = mat3x3