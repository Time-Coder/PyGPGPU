import ctypes

from .. import genQuat, Flavor


class dquat(genQuat):

    @property
    def dtype(self)->type:
        return ctypes.c_double