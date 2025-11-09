import ctypes

from ...vectorization import genQuat


class dquat(genQuat):

    @property
    def dtype(self)->type:
        return ctypes.c_double