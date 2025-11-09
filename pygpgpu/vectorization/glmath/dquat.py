import ctypes

from .. import genQuat


class dquat(genQuat):

    @property
    def dtype(self)->type:
        return ctypes.c_double