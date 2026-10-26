from __future__ import annotations

import ctypes
from ctypes import byref
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import override

from ..driver import (
    CUDA,
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


# How many times to retry a refused creation directly. A cold driver refuses a few
# times before accepting, so a couple of tries distinguishes "not yet warm" from
# "rejected outright" without burning time.
_CREATE_ATTEMPTS:int = 3

# Warm-up rounds allowed, each creating a texture over a throwaway array.
_WARM_ATTEMPTS:int = 8

# Warm-up handles kept alive so the priming survives until the process exits.
_warm_keep:list = []


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

        # Resource: the array this texture samples. Only the handle belongs here;
        # the format and channel count are properties of the array's own
        # descriptor, and cuda.h gives this union member no other field.
        resource = CUDA_RESOURCE_DESC()
        ctypes.memset(byref(resource), 0, ctypes.sizeof(resource))
        resource.resType = CUresourcetype.CU_RESOURCE_TYPE_ARRAY
        resource.res.array.hArray = array.id

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

        # A cold driver rejects the first few creations with
        # CUDA_ERROR_INVALID_VALUE and then starts accepting them. Repeated attempts
        # on the same array were not enough on their own, but creating a texture
        # over a throwaway array does make the wanted one succeed, so a refusal is
        # answered by warming on a scratch image.
        tex_id = self._create(resource, texture)
        CUObject.__init__(self, tex_id)

    def _create(self, resource:CUDA_RESOURCE_DESC, texture:CUDA_TEXTURE_DESC)->CUtexObject:
        """Create the texture object, warming on a scratch array if refused."""
        last_error:Optional[BaseException] = None

        for _ in range(_CREATE_ATTEMPTS):
            tex_id = CUtexObject()
            try:
                CUDA.cuTexObjectCreate(byref(tex_id), byref(resource), byref(texture), None)
                return tex_id
            except BaseException as error:
                last_error = error

        for _ in range(_WARM_ATTEMPTS):
            if not self._warm_once(resource, texture):
                continue
            tex_id = CUtexObject()
            try:
                CUDA.cuTexObjectCreate(byref(tex_id), byref(resource), byref(texture), None)
                return tex_id
            except BaseException as error:
                last_error = error

        raise last_error if last_error is not None else RuntimeError(
            "cuTexObjectCreate did not succeed after warming")

    @staticmethod
    def _warm_once(resource:CUDA_RESOURCE_DESC, texture:CUDA_TEXTURE_DESC)->bool:
        """Create and keep a texture over a scratch array, to prime the driver."""
        from .array import Array

        scratch = Array(8, 4, "float32")
        warm_resource = CUDA_RESOURCE_DESC.from_buffer_copy(bytes(resource))
        warm_resource.res.array.hArray = scratch.id

        warm = CUtexObject()
        try:
            CUDA.cuTexObjectCreate(byref(warm), byref(warm_resource), byref(texture), None)
        except BaseException:
            return False

        # Retain both: dropping them was seen to undo the effect.
        _warm_keep.extend([scratch, warm])
        return True

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

