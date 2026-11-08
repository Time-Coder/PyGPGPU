from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING, Union


if TYPE_CHECKING:
    # numpy is imported inside the methods that use it; the annotation needs the name.
    import numpy as np

    from ..surface import SurfaceObject

    from ..array import Array
    from ..texture import TextureObject


# Upper bound on the streams a Context keeps alive. Synchronous calls share one
# cached stream per thread and asynchronous calls draw from a separate pool of
# non-blocking streams; capping both keeps a process that creates and drops
# threads (or keeps launching asynchronously) from growing streams forever.
STREAM_POOL_SIZE:int = 16

# How many times create_texture rebuilds an image before giving up. The driver
# rejects texture creation intermittently, and a refused array stays refused, so
# each attempt allocates a replacement rather than retrying the same handle.
TEXTURE_RETRY_ATTEMPTS:int = 40


# How many times create_texture rebuilds an image before giving up. The driver
# rejects texture creation intermittently, and a refused array stays refused, so
# each attempt allocates a replacement rather than retrying the same handle.
TEXTURE_RETRY_ATTEMPTS:int = 40

if TYPE_CHECKING:
    import numpy as np

    from ..array import Array
    from ..surface import SurfaceObject
    from ..texture import TextureObject

class _ContextImageMixin:

    def create_array(self, width:int, height:int=1, depth:int=1,
                     dtype:object=None, num_channels:int=1,
                     surface:bool=False)->Array:
        """Allocate a device-side image: 1D, 2D or 3D depending on the dimensions.

        ``surface=True`` marks the storage writable, which a SurfaceObject requires
        before it can bind to it.
        """
        import numpy as np
        from ..array import Array

        self.make_current()
        return Array(width, height, depth, np.float32 if dtype is None else dtype,
                     num_channels, surface)


    def create_texture(self, array:Array, **options:object)->TextureObject:
        """Build a sampled view over ``array`` for use as a kernel argument.

        A refusal is retried by rebuilding the image from the array's contents: the
        driver drops refusals once it accepts one, and a freshly allocated image
        tends to be the one it accepts. The caller gets a texture that owns the
        replacement, so the retry is invisible from outside.
        """
        from ..texture import TextureObject

        self.make_current()
        try:
            return TextureObject(array, **options)
        except BaseException as first_error:
            last_error:BaseException = first_error

        contents = array.to_numpy()
        for _ in range(TEXTURE_RETRY_ATTEMPTS):
            replacement = self.create_array(array.width, array.height, array.depth,
                                             array.dtype, array.num_channels,
                                             array.is_surface)
            replacement.set_data(contents)
            try:
                texture = TextureObject(replacement, **options)
            except BaseException as error:
                last_error = error
                continue
            texture._array = replacement
            return texture

        raise last_error


    def texture(self, data:object, **options:object)->TextureObject:
        """Upload a host image and return a texture that samples it.

        The short form of ``create_texture(create_array(...))`` for the common case
        where the image is a numpy array nothing else needs:

            texture = Devices[0].default_context.texture(image)
            kernel["cuda"](texture, out, width, height)

        ``data`` is interpreted as ``(height, width)`` or ``(height, width,
        channels)``; its dtype and channel count carry over to the device image.
        """
        import numpy as np

        image = np.asarray(data)
        if image.ndim not in (2, 3):
            raise ValueError(
                "texture() expects a (height, width) or (height, width, channels) "
                "image, got %d dimensions" % image.ndim)

        height, width = image.shape[0], image.shape[1]
        channels = image.shape[2] if image.ndim == 3 else 1
        array = self.create_array(width, height, image.dtype, channels)
        array.set_data(image)
        return self.create_texture(array, **options)


    def create_surface(self, array_or_shape:"Union[Array, np.ndarray, Tuple[int, ...]]", **options:object)->SurfaceObject:
        """Build a writable view for a kernel to store through.

        Pass either an array created with ``surface=True``, or a numpy image, or a
        (width[, height[, depth]]) shape, in which case the storage is created here:

            surface = ctx.create_surface(np.zeros((8, 16), np.float32))
            surface = ctx.create_surface(shape=(16, 8))
        """
        import numpy as np
        from ..array import Array
        from ..surface import SurfaceObject
        from ...driver.image_shape import (extents_from_data, extents_from_shape,
                                    split_channels)

        self.make_current()
        target:Array
        if isinstance(array_or_shape, Array):
            target = array_or_shape
        elif isinstance(array_or_shape, np.ndarray):
            image = array_or_shape
            if not image.flags["C_CONTIGUOUS"]:
                image = np.ascontiguousarray(image)
            channels = split_channels(image.shape)[1]
            width, height, depth = extents_from_data(image.shape)
            target = self.create_array(width, height, depth, image.dtype, channels,
                                       surface=True)
            target.set_data(image)
        else:
            shape = tuple(int(v) for v in array_or_shape)
            channels = split_channels(shape)[1]
            width, height, depth = extents_from_shape(shape)
            dtype = options.pop("dtype", np.float32)
            target = self.create_array(width, height, depth, dtype, channels,
                                       surface=True)
        return SurfaceObject(target)


    def create_image(self, image:object, writable:bool=False)->object:
        """Bind a backend-neutral :class:`pygpgpu.Image` to this CUDA context.

        The device needs different objects depending on how the kernel uses the
        image: reading samples a texture object, writing needs a surface object,
        and a kernel that does both needs the array created with the surface flag.
        ``writable`` comes from the kernel's read/write analysis.

        Returns ``(array, texture, surface)`` with the unused entries None; the
        caller keeps them alive for the launch.
        """
        from ..array import Array
        from ..surface import SurfaceObject
        from ...driver import CUaddress_mode, CUfilter_mode

        self.make_current()

        array = Array(image.width, image.height, image.depth, image.dtype,
                      image.channels, surface=bool(writable))
        if image.data is not None:
            array.set_data(image.data)

        texture:Optional[object] = None
        surface:Optional[object] = None

        if writable:
            surface = SurfaceObject(array)
        else:
            filter_mode = (CUfilter_mode.CU_TR_FILTER_MODE_LINEAR
                           if image.filter == "linear"
                           else CUfilter_mode.CU_TR_FILTER_MODE_POINT)
            address_mode = {
                "clamp": CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
                "wrap": CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP,
                "mirror": CUaddress_mode.CU_TR_ADDRESS_MODE_MIRROR,
            }[image.address]
            # Coordinates are texel-space (see pygpgpu.image), so the texture is
            # built without coordinate normalization and the device sees the same
            # numbers the host path uses.
            texture = self.create_texture(array, normalized=False,
                                          filter_mode=filter_mode,
                                          address_mode=address_mode)
        return array, texture, surface
