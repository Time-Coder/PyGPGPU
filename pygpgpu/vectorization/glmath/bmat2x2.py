from .. import genMat2x2, Flavor

import ctypes


class bmat2x2(genMat2x2):

    @property
    def dtype(self)->type:
        return ctypes.c_bool
    
bmat2 = bmat2x2