import ctypes

from .. import genQuat, Flavor


class quat(genQuat):

    @property
    def dtype(self)->type:
        return ctypes.c_float