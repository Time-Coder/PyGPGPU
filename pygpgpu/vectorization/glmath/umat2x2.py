from .. import genMat2x2, Flavor

import ctypes


class umat2x2(genMat2x2):

    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 16
    
umat2 = umat2x2