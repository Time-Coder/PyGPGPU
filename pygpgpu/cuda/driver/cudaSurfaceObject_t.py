from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import imageio.v3 as iio

from ...constants import KernelFlags
from .image_shape import extents_from_data, extents_from_shape, split_channels


class cudaSurfaceObject_t:
    """A 1D, 2D or 3D image a kernel can write to, named exactly as the device type.

    A handwritten kernel takes ``cudaSurfaceObject_t`` and stores through
    ``surf1Dwrite`` / ``surf2Dwrite`` / ``surf3Dwrite``; this is the host object that
    produces one:

        surface = cudaSurfaceObject_t(shape=(height, width))
        kernel["cuda"](surface, width, height)
        image = surface.read()          # bring the result back

    It accepts the same shapes as :class:`cudaTextureObject_t` -- one, two or three
    spatial dimensions, optionally with 1, 2 or 4 channels -- and can be built from a
    file path, a numpy array, or just a shape.

    Unlike a texture, which CUDA makes read-only, a surface is both readable and
    writable, and the kernel's read/write analysis decides which copies happen.
    Storage is a device array created with the surface flag set, which CUDA requires
    before a surface object can bind to it.
    """

    def __init__(self, data:Union[str, np.ndarray, None]=None,
                 shape:Optional[Tuple[int, ...]]=None,
                 dtype:Any=np.float32,
                 address_mode:Optional[Any]=None):
        self._address_mode:Optional[Any] = address_mode
        # A surface is both read and written by kernels, which is what the static
        # analysis reports for it; declaring it here keeps that intent explicit.
        self._kernel_flags:KernelFlags = KernelFlags.Readed | KernelFlags.Writed

        if isinstance(data, str):
            data = iio.imread(data)

        if data is not None:
            image = np.asarray(data)
            if not image.flags["C_CONTIGUOUS"]:
                image = np.ascontiguousarray(image)
            self._channels = split_channels(image.shape)[1]
            self._width, self._height, self._depth = extents_from_data(image.shape)
            self._image:Optional[np.ndarray] = image
            self._dtype:np.dtype = image.dtype
        elif shape is None:
            raise ValueError(
                "cudaSurfaceObject_t needs either an image or a "
                "(width[, height[, depth]]) shape")
        else:
            shape = tuple(int(v) for v in shape)
            self._channels = split_channels(shape)[1]
            self._width, self._height, self._depth = extents_from_shape(shape)
            self._dtype = np.dtype(dtype)
            self._image = None

        self.bind()

    @property
    def image(self)->Optional[np.ndarray]:
        """The host image the surface started from, or None."""
        return self._image

    @property
    def shape(self)->Tuple[int, ...]:
        if self._depth > 1:
            return (self._depth, self._height, self._width)
        if self._height > 1:
            return (self._height, self._width)
        return (self._width,)

    @property
    def ndim(self)->int:
        if self._depth > 1:
            return 3
        if self._height > 1:
            return 2
        return 1

    @property
    def height(self)->int:
        return self._height

    @property
    def width(self)->int:
        return self._width

    @property
    def depth(self)->int:
        return self._depth

    @property
    def channels(self)->int:
        return self._channels

    @property
    def dtype(self)->np.dtype:
        return self._dtype

    @property
    def address_mode(self)->Optional[Any]:
        return self._address_mode

    @property
    def kernel_flags(self)->KernelFlags:
        return self._kernel_flags

    @kernel_flags.setter
    def kernel_flags(self, flags:KernelFlags)->None:
        self._kernel_flags = flags

    def bind(self)->Any:
        """Create the device surface now, and keep returning the same one.

        The context comes from the current thread, or from the first device when none
        is current yet, so the caller never has to pick one.
        """
        device = getattr(self, "_device", None)
        if device is not None:
            return device[1]

        try:
            from ..oop.context import Context
            from ..oop.devices import Devices
        except ImportError:  # pragma: no cover - package layout guard
            return None

        context = Context.current()
        if context is None and len(Devices) > 0:
            context = Devices[0].default_context
        if context is None:
            return None

        array = context.create_array(self._width, self._height, self._depth,
                                     self._dtype, self._channels, surface=True)
        if self._image is not None:
            array.set_data(self._image)

        from ..oop.surface import SurfaceObject  # imported lazily to avoid a cycle
        surface = SurfaceObject(array)
        self._device = (array, surface)
        return surface

    def read(self)->np.ndarray:
        """Read the surface's current contents back into a host array."""
        surface = self.bind()
        if surface is None:
            raise RuntimeError(
                "the surface is not bound to a device; nothing ran yet")
        return surface.array.to_numpy()

    def __repr__(self)->str:
        return "cudaSurfaceObject_t(%s, %d channel(s), %s)" % (
            "x".join(str(v) for v in self.shape), self._channels, self._dtype)
