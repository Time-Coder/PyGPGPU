from __future__ import annotations

from ctypes import byref
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from typing_extensions import override

from ..driver import (
    CUDA,
    CUarray,
    CUarray_format,
    CU_ARRAY_DESC,
    CUDA_MEMCPY2D,
    CUmemorytype,
)
from .cuobject import CUObject

if TYPE_CHECKING:
    from .stream import Stream


# element dtype -> CUDA array format. Only the formats CUDA can store in an array
# are listed; the float32 path is the one the sampling tests use.
_FORMAT_BY_DTYPE = {
    np.dtype(np.uint8): CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8,
    np.dtype(np.uint16): CUarray_format.CU_AD_FORMAT_UNSIGNED_INT16,
    np.dtype(np.uint32): CUarray_format.CU_AD_FORMAT_UNSIGNED_INT32,
    np.dtype(np.int8): CUarray_format.CU_AD_FORMAT_SIGNED_INT8,
    np.dtype(np.int16): CUarray_format.CU_AD_FORMAT_SIGNED_INT16,
    np.dtype(np.int32): CUarray_format.CU_AD_FORMAT_SIGNED_INT32,
    np.dtype(np.float16): CUarray_format.CU_AD_FORMAT_HALF,
    np.dtype(np.float32): CUarray_format.CU_AD_FORMAT_FLOAT,
}

# CUDA_ARRAY_FLAGS values used here.
_CUDA_ARRAY_FLAG_SURFACE_LDST = 0x02


class Array(CUObject):
    """A 2D device-side image, the CUDA counterpart of OpenCL's image2d_t.

    An array is the storage the texture hardware samples; it is created with a
    fixed width/height/channel layout and filled or read back through a 2D copy.
    Sampling itself goes through a TextureObject built on top of it.

    Only 2D single-channel arrays are modelled for now, which covers the common
    "upload an image, sample it in a kernel" case.
    """

    def __init__(self, width:int, height:int, dtype:Any=np.float32, num_channels:int=1):
        dtype = np.dtype(dtype)
        if dtype not in _FORMAT_BY_DTYPE:
            raise TypeError(
                "CUDA arrays do not support dtype %s; supported: %s"
                % (dtype, ", ".join(str(d) for d in _FORMAT_BY_DTYPE)))
        if num_channels not in (1, 2, 4):
            raise ValueError(
                "CUDA arrays support 1, 2 or 4 channels per element, got %d" % num_channels)
        if width <= 0 or height <= 0:
            raise ValueError("array width and height must be positive")

        self._width:int = int(width)
        self._height:int = int(height)
        self._dtype:np.dtype = dtype
        self._num_channels:int = int(num_channels)

        desc = CU_ARRAY_DESC()
        desc.Width = width
        desc.Height = height
        desc.Format = _FORMAT_BY_DTYPE[dtype]
        desc.NumChannels = num_channels

        array_id = CUarray()
        CUDA.cuArrayCreate_v2(byref(array_id), byref(desc))
        CUObject.__init__(self, array_id)

    @property
    def width(self)->int:
        return self._width

    @property
    def height(self)->int:
        return self._height

    @property
    def dtype(self)->np.dtype:
        return self._dtype

    @property
    def num_channels(self)->int:
        return self._num_channels

    @property
    def shape(self)->Tuple[int, int]:
        return (self._height, self._width)

    @property
    def row_bytes(self)->int:
        return self._width * self._dtype.itemsize * self._num_channels

    def _copy2d(self, host_ptr:int, host_pitch:int, to_device:bool, stream:Optional[Stream])->None:
        """Move whole rows between host memory and the array."""
        copy = CUDA_MEMCPY2D()
        copy.WidthInBytes = self.row_bytes
        copy.Height = self._height
        copy.dstXInBytes = 0
        copy.dstY = 0
        copy.srcXInBytes = 0
        copy.srcY = 0
        if to_device:
            copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST
            copy.srcHost = host_ptr
            copy.srcPitch = host_pitch
            copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY
            copy.dstArray = self.id
            copy.dstPitch = 0
        else:
            copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY
            copy.srcArray = self.id
            copy.srcPitch = 0
            copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST
            copy.dstHost = host_ptr
            copy.dstPitch = host_pitch

        if stream is None:
            CUDA.cuMemcpy2D_v2(byref(copy))
        else:
            CUDA.cuMemcpy2DAsync_v2(byref(copy), stream.id)

    def set_data(self, data:np.ndarray, stream:Optional[Stream]=None)->None:
        """Upload a 2D host array.

        The host layout is expected to be ``(height, width[, channels])`` and
        C-contiguous; it is made contiguous here rather than silently producing a
        wrong pitch.
        """
        source = self._as_host_layout(data)
        self._copy2d(source.ctypes.data, source.strides[0], True, stream)

    def to_numpy(self, stream:Optional[Stream]=None)->np.ndarray:
        """Read the whole array back into a host numpy array."""
        shape:Tuple[int, ...] = (self._height, self._width)
        if self._num_channels > 1:
            shape = shape + (self._num_channels,)
        out = np.empty(shape, dtype=self._dtype)
        self._copy2d(out.ctypes.data, out.strides[0], False, stream)
        return out

    def _as_host_layout(self, data:np.ndarray)->np.ndarray:
        array = np.asarray(data)
        expected:Tuple[int, ...] = (self._height, self._width)
        if self._num_channels > 1:
            expected = expected + (self._num_channels,)
        if array.shape != expected:
            raise ValueError(
                "data shape %s does not match the array layout %s"
                % (array.shape, expected))
        if array.dtype != self._dtype:
            array = array.astype(self._dtype)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return array

    @staticmethod
    @override
    def _prefix()->str:
        return ""

    @staticmethod
    @override
    def _get_info_func()->Any:
        return CUDA.cuArrayGetDescriptor_v2

    @staticmethod
    @override
    def _info_enum()->type|None:
        return None

    @staticmethod
    @override
    def _release_func()->Any:
        return CUDA.cuArrayDestroy
