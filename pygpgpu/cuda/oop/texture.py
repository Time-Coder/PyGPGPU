from __future__ import annotations

import ctypes
from ctypes import byref
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from ..driver import (
    CUDA,
    CUarray_format,
    CUDA_RESOURCE_DESC,
    CUDA_TEXTURE_DESC,
    CUaddress_mode,
    CUfilter_mode,
    CUresourcetype,
    CUtexObject,
    CU_TRSF_NORMALIZED_COORDINATES,
)
from .cuobject import CUObject

if TYPE_CHECKING:
    from .array import Array


def _format_of(array:Array)->CUarray_format:
    from .array import _FORMAT_BY_DTYPE

    return _FORMAT_BY_DTYPE[array.dtype]


class TextureObject(CUObject):
    """A sampled view over an :class:`Array`, read in a kernel as ``tex2D``.

    A texture object bundles the storage with its sampling rules: how coordinates
    outside the image behave (address mode) and how samples are filtered (point or
    linear). Unlike OpenCL, where the sampler is a separate kernel argument, CUDA
    bakes the sampler into the texture object, so the kernel takes a single handle.
    """

    def __init__(self, array:Array, normalized:bool=True,
                 address_mode:CUaddress_mode=CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
                 filter_mode:CUfilter_mode=CUfilter_mode.CU_TR_FILTER_MODE_POINT):
        self._array:Array = array
        self._normalized:bool = normalized
        self._address_mode:CUaddress_mode = address_mode
        self._filter_mode:CUfilter_mode = filter_mode

        # Resource: the array this texture samples.
        resource = CUDA_RESOURCE_DESC()
        ctypes.memset(byref(resource), 0, ctypes.sizeof(resource))
        resource.resType = CUresourcetype.CU_RESOURCE_TYPE_ARRAY
        resource.res.array.hArray = array.id
        resource.res.array.Format = _format_of(array)
        resource.res.array.NumChannels = array.num_channels

        # Sampling rules. maxAnisotropy must be 1 (the driver rejects 0), and the
        # mipmap fields must be sane even though no mipmaps are used here.
        texture = CUDA_TEXTURE_DESC()
        ctypes.memset(byref(texture), 0, ctypes.sizeof(texture))
        for index in range(3):
            texture.addressMode[index] = address_mode
        texture.filterMode = filter_mode
        texture.flags = CU_TRSF_NORMALIZED_COORDINATES if normalized else 0
        texture.maxAnisotropy = 1
        texture.mipmapFilterMode = filter_mode
        texture.mipmapLevelBias = 0.0
        texture.minMipmapLevelClamp = 0.0
        texture.maxMipmapLevelClamp = 0.0
        for index in range(4):
            texture.borderColor[index] = 0.0

        tex_id = CUtexObject()
        CUDA.cuTexObjectCreate(byref(tex_id), byref(resource), byref(texture), None)
        CUObject.__init__(self, tex_id)

    @property
    def array(self)->Array:
        return self._array

    @property
    def normalized(self)->bool:
        return self._normalized

    @property
    def address_mode(self)->CUaddress_mode:
        return self._address_mode

    @property
    def filter_mode(self)->CUfilter_mode:
        return self._filter_mode

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
        return CUDA.cuTexObjectDestroy
