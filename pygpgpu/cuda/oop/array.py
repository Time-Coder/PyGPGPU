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
    CUDA_ARRAY3D_DESCRIPTOR,
    CUDA_MEMCPY2D,
    CUDA_MEMCPY3D,
    CUDA_ARRAY_FLAG_SURFACE_LDST,
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


class Array(CUObject):
    """A 1D, 2D or 3D device-side image, the CUDA counterpart of OpenCL's image types.

    An array is the storage the texture or surface hardware reads; it is created with
    a fixed width/height/depth/channel layout and filled or read back through a
    copy. Sampling goes through a TextureObject (read-only) or writing through a
    SurfaceObject, both built on top of it.

    A single-element dimension is what makes a 1D or 2D array, so ``Array(1024)`` is
    a 1D image of 1024 elements and ``Array(16, 8)`` a 16x8 image.
    """

    def __init__(self, width:int, height:int=1, depth:int=1,
                 dtype:Any=np.float32, num_channels:int=1,
                 surface:bool=False):
        dtype = np.dtype(dtype)
        if dtype not in _FORMAT_BY_DTYPE:
            raise TypeError(
                "CUDA arrays do not support dtype %s; supported: %s"
                % (dtype, ", ".join(str(d) for d in _FORMAT_BY_DTYPE)))
        if num_channels not in (1, 2, 4):
            raise ValueError(
                "CUDA arrays support 1, 2 or 4 channels per element, got %d" % num_channels)
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError("array dimensions must be positive")

        self._width:int = int(width)
        self._height:int = int(height)
        self._depth:int = int(depth)
        self._dtype:np.dtype = dtype
        self._num_channels:int = int(num_channels)
        # A surface needs CU_ARRAY_FLAG_SURFACE_LDST set when the array is created;
        # a texture reads an ordinary array.
        self._surface:bool = bool(surface)

        flags:int = CUDA_ARRAY_FLAG_SURFACE_LDST if surface else 0

        array_id = CUarray()
        if depth > 1:
            desc3d = CUDA_ARRAY3D_DESCRIPTOR()
            desc3d.Width = width
            desc3d.Height = height
            desc3d.Depth = depth
            desc3d.Format = _FORMAT_BY_DTYPE[dtype]
            desc3d.NumChannels = num_channels
            desc3d.Flags = flags
            CUDA.cuArray3DCreate_v2(byref(array_id), byref(desc3d))
        elif height > 1:
            desc = CU_ARRAY_DESC()
            desc.Width = width
            desc.Height = height
            desc.Format = _FORMAT_BY_DTYPE[dtype]
            desc.NumChannels = num_channels
            CUDA.cuArrayCreate_v2(byref(array_id), byref(desc))
        else:
            # A one-row 2D array is how CUDA represents a 1D image.
            desc = CU_ARRAY_DESC()
            desc.Width = width
            desc.Height = 1
            desc.Format = _FORMAT_BY_DTYPE[dtype]
            desc.NumChannels = num_channels
            CUDA.cuArrayCreate_v2(byref(array_id), byref(desc))

        CUObject.__init__(self, array_id)

    @property
    def width(self)->int:
        return self._width

    @property
    def height(self)->int:
        return self._height

    @property
    def depth(self)->int:
        return self._depth

    @property
    def dtype(self)->np.dtype:
        return self._dtype

    @property
    def num_channels(self)->int:
        return self._num_channels

    @property
    def is_surface(self)->bool:
        return self._surface

    @property
    def ndim(self)->int:
        """How many dimensions the array actually uses."""
        if self._depth > 1:
            return 3
        if self._height > 1:
            return 2
        return 1

    @property
    def shape(self)->Tuple[int, ...]:
        """Host layout: (height[, width]) without trailing 1s, plus channels."""
        shape:Tuple[int, ...]
        if self._depth > 1:
            shape = (self._depth, self._height, self._width)
        elif self._height > 1:
            shape = (self._height, self._width)
        else:
            shape = (self._width,)
        if self._num_channels > 1:
            shape = shape + (self._num_channels,)
        return shape

    @property
    def row_bytes(self)->int:
        return self._width * self._dtype.itemsize * self._num_channels

    def _copy2d(self, host_ptr:int, host_pitch:int, to_device:bool,
                stream:Optional[Stream])->None:
        """Move whole rows between host memory and a 1D/2D array."""
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

    def _copy3d(self, host_ptr:int, host_pitch:int, to_device:bool,
                stream:Optional[Stream])->None:
        """Move whole slices between host memory and a 3D array.

        ``srcPitch``/``dstPitch`` is the row stride in bytes and
        ``srcHeight``/``dstHeight`` is the slice stride *in rows*, not in bytes --
        giving the driver a byte count there makes it walk off the end of the host
        buffer after the first slice, which shows up as garbage in later slices.
        """
        copy = CUDA_MEMCPY3D()
        copy.WidthInBytes = self.row_bytes
        copy.Height = self._height
        copy.Depth = self._depth
        copy.srcXInBytes = 0
        copy.srcY = 0
        copy.srcZ = 0
        copy.dstXInBytes = 0
        copy.dstY = 0
        copy.dstZ = 0
        if to_device:
            copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST
            copy.srcHost = host_ptr
            copy.srcPitch = host_pitch
            copy.srcHeight = self._height
            copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY
            copy.dstArray = self.id
        else:
            copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY
            copy.srcArray = self.id
            copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST
            copy.dstHost = host_ptr
            copy.dstPitch = host_pitch
            copy.dstHeight = self._height

        if stream is None:
            CUDA.cuMemcpy3D_v2(byref(copy))
        else:
            CUDA.cuMemcpy3DAsync_v2(byref(copy), stream.id)

    def set_data(self, data:np.ndarray, stream:Optional[Stream]=None)->None:
        """Upload a host array shaped like ``self.shape``."""
        source = self._as_host_layout(data)
        if self._depth > 1:
            # strides[0] spans a whole slice; the row pitch is the next axis in.
            self._copy3d(source.ctypes.data, source.strides[-2], True, stream)
        else:
            # A one-row array stores its whole width in a single row, so the host
            # pitch is the row length rather than the stride of a 1D array (which
            # would be one element).
            pitch = self.row_bytes if self._height == 1 else source.strides[0]
            self._copy2d(source.ctypes.data, pitch, True, stream)

    def to_numpy(self, stream:Optional[Stream]=None)->np.ndarray:
        """Read the whole array back into a host numpy array."""
        out = np.empty(self.shape, dtype=self._dtype)
        if self._depth > 1:
            self._copy3d(out.ctypes.data, out.strides[-2], False, stream)
        else:
            pitch = self.row_bytes if self._height == 1 else out.strides[0]
            self._copy2d(out.ctypes.data, pitch, False, stream)
        return out

    def _as_host_layout(self, data:np.ndarray)->np.ndarray:
        array = np.asarray(data)
        expected = self.shape
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
