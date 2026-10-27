from __future__ import annotations

import ctypes
from ctypes import byref
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from ..driver import (
    CUDA,
    CUDA_RESOURCE_DESC,
    CUresourcetype,
    CUsurfObject,
)
from .cuobject import CUObject

if TYPE_CHECKING:
    from .array import Array


class SurfaceObject(CUObject):
    """A writable view over an :class:`Array`, written in a kernel as ``surf2Dwrite``.

    Whereas a :class:`TextureObject` fixes sampling rules and is read-only, a surface
    exposes the same storage for writing: a kernel stores through ``surf2Dwrite`` and
    the result can be read back on the host. CUDA keeps the two as separate object
    kinds over the same array, so both wrap one ``Array``.

    The array must be created with the surface flag set, which ``Array(...,
    surface=True)`` does.
    """

    def __init__(self, array:Array):
        if not array.is_surface:
            raise ValueError(
                "a SurfaceObject needs an array created with surface=True")
        self._array:Array = array

        resource = CUDA_RESOURCE_DESC()
        ctypes.memset(byref(resource), 0, ctypes.sizeof(resource))
        resource.resType = CUresourcetype.CU_RESOURCE_TYPE_ARRAY
        resource.res.array.hArray = array.id

        surf_id = CUsurfObject()
        CUDA.cuSurfObjectCreate(byref(surf_id), byref(resource))
        CUObject.__init__(self, surf_id)

    @property
    def array(self)->Array:
        return self._array

    @staticmethod
    @override
    def _prefix()->str:
        return ""

    @staticmethod
    @override
    def _get_info_func()->Any:
        return None

    @staticmethod
    @override
    def _info_enum()->type|None:
        return None

    @staticmethod
    @override
    def _release_func()->Any:
        return CUDA.cuSurfObjectDestroy
