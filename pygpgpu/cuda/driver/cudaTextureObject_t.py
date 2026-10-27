from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import imageio.v3 as iio

from ...constants import KernelFlags
from .image_shape import extents_from_data, extents_from_shape, split_channels


class cudaTextureObject_t:
    """A 2D image to sample in a kernel, named exactly as the device type.

    A handwritten kernel takes ``cudaTextureObject_t``; this is the host object that
    produces one, so the signature and the call site use the same name:

        texture = cudaTextureObject_t("photo.png")        # read with imageio
        texture = cudaTextureObject_t(numpy_image)        # (height, width[, channels])
        texture = cudaTextureObject_t(shape=(height, width))   # allocate, fill later

        kernel["cuda"](texture, out, width, height)

    This is the CUDA analogue of OpenCL's ``image2d_t``: a plain description of an
    image with no device or context attached, which the backend binds to whatever
    context the launch uses. Constructing it binds immediately -- taking the current
    thread's context, or the first device's when none is current -- so the caller
    never has to choose a device.

    The sampling rules travel with it, because CUDA bakes the sampler into the
    texture object rather than passing it as a separate kernel argument:

        cudaTextureObject_t(image, normalized=True,
                            address_mode=CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
                            filter_mode=CUfilter_mode.CU_TR_FILTER_MODE_POINT)
    """

    def __init__(self, data:Union[str, np.ndarray, None]=None,
                 shape:Optional[Tuple[int, ...]]=None,
                 dtype:Any=np.float32,
                 normalized:bool=True,
                 address_mode:Optional[Any]=None,
                 filter_mode:Optional[Any]=None):
        self._normalized:bool = normalized
        self._address_mode:Optional[Any] = address_mode
        self._filter_mode:Optional[Any] = filter_mode
        self._kernel_flags:KernelFlags = KernelFlags.Readed

        if isinstance(data, str):
            data = iio.imread(data)

        if data is not None:
            image = np.asarray(data, dtype=None)
            if not image.flags["C_CONTIGUOUS"]:
                image = np.ascontiguousarray(image)
            # A trailing axis of size 1, 2 or 4 is read as channels; anything else
            # means the array is purely spatial (1D, 2D or 3D).
            self._channels = split_channels(image.shape)[1]
            self._width, self._height, self._depth = extents_from_data(image.shape)
            self._image:Optional[np.ndarray] = image
            self._dtype:np.dtype = image.dtype
        elif shape is None:
            raise ValueError(
                "cudaTextureObject_t needs either an image or a (width[, height"
                "[, depth]]) shape")
        else:
            shape = tuple(int(v) for v in shape)
            self._channels = split_channels(shape)[1]
            self._width, self._height, self._depth = extents_from_shape(shape)
            self._dtype = np.dtype(dtype)
            self._image = None

        # Bind eagerly. Besides keeping the common case to a single step for the
        # caller, it means the device image exists before any module containing
        # tex2D is loaded -- a driver this was developed against stops accepting new
        # texture objects once such a module is in the process.
        self.bind()

    @property
    def image(self)->Optional[np.ndarray]:
        """The host image, or None when only a shape was given."""
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
    def depth(self)->int:
        return self._depth

    @property
    def width(self)->int:
        return self._width

    @property
    def channels(self)->int:
        return self._channels

    @property
    def dtype(self)->np.dtype:
        return self._dtype

    @property
    def normalized(self)->bool:
        return self._normalized

    @property
    def address_mode(self)->Optional[Any]:
        return self._address_mode

    @property
    def filter_mode(self)->Optional[Any]:
        return self._filter_mode

    @property
    def kernel_flags(self)->KernelFlags:
        return self._kernel_flags

    @kernel_flags.setter
    def kernel_flags(self, flags:KernelFlags)->None:
        self._kernel_flags = flags

    def bind(self)->Any:
        """Create the device texture now, and keep returning the same one.

        A launch binds the image automatically, but binding early matters on a driver
        that stops accepting new textures once a module containing ``tex2D`` has been
        loaded: constructing the image before importing such a module keeps it usable.

        The context comes from the current thread, or from the first device when none
        is current yet, so the caller never has to pick one. Returns the
        ``TextureObject``, whose id is the handle a kernel receives.
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
            # Nothing has run yet on this thread; the first device's context is the
            # one a default launch would have picked anyway.
            context = Devices[0].default_context
        if context is None:
            return None

        array = context.create_array(self._width, self._height, self._depth,
                                     self._dtype, self._channels)
        if self._image is not None:
            array.set_data(self._image)

        options:dict = {"normalized": self._normalized}
        if self._address_mode is not None:
            options["address_mode"] = self._address_mode
        if self._filter_mode is not None:
            options["filter_mode"] = self._filter_mode

        texture = context.create_texture(array, **options)
        self._device = (array, texture)
        return texture

    def __repr__(self)->str:
        return "cudaTextureObject_t(%s, %d channel(s), %s)" % (
            "x".join(str(v) for v in self.shape[:self.ndim]).replace(
                str(self._width), str(self._width)),
            self._channels, self._dtype)
