import ctypes

from ...vectorization import genQuat


class quat(genQuat):

    @property
    def dtype(self)->type:
        return ctypes.c_float