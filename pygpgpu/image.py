"""The backend-neutral image type used by kernels.

A CUDA kernel samples through a texture object and writes through a surface object,
and its coordinates live in texel space; an OpenCL kernel takes an image plus a
separate sampler and addresses pixels as integers. Writing one kernel that runs on
both would otherwise mean picking one of those spellings and faking the other.

``Image`` removes that choice. A kernel indexes it exactly like an array:

    @kernel
    def brighten(src, dst):
        for y, x in np.ndindex(dst.shape):
            dst[y, x] = src[y, x] * 0.5

The subscript is translated into the sampling call each backend provides, and the
same subscript works on the ``python`` backend, where it runs this module's host
implementation instead -- so one kernel produces the same numbers on every backend.

Coordinates follow CUDA's texel space: texel ``i`` spans ``[i, i+1)`` and its centre
is at ``i + 0.5``. An integer subscript therefore means "that texel" (the centre is
sampled exactly), while a fractional subscript addresses a position between texels:

* with ``filter="point"`` the texel covering the coordinate is returned,
  ``floor(u)``;
* with ``filter="linear"`` the two neighbouring texel centres are blended by
  distance.

Those are the same rules the device applies, which is what keeps the host path and
the GPU paths in agreement.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple, Union

import numpy as np


class StrEnum(str, Enum):
    """A string-valued enum (the stdlib one is 3.11+, and this package supports 3.7).

    Subclassing str keeps the members comparable with plain strings, so an existing
    ``filter="linear"`` call site keeps working alongside ``filter=ImageFilter.LINEAR``.
    """

    def __str__(self)->str:
        return str(self.value)

    @classmethod
    def parse(cls, value:Any, what:str)->"StrEnum":
        """Accept a member, its value, or its name (case-insensitive)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                pass
            try:
                return cls[value.strip().upper()]
            except KeyError:
                pass
        options:str = ", ".join(member.value for member in cls)
        raise ValueError("%s must be one of %s, got %r" % (what, options, value))


# Sampling rules are enums so the class stays backend-neutral while the choices stay
# discoverable. The names are the neutral ones: CUDA spells the filters POINT/LINEAR and
# OpenCL spells them NEAREST/LINEAR, and the address modes differ just as much
# (CLAMP_TO_EDGE vs CLAMP, REPEAT vs WRAP), so neither backend's vocabulary is used here.
class ImageFilter(StrEnum):
    """How a sample is computed from the texels around its coordinate."""

    POINT = "point"      # the one texel the coordinate falls in
    LINEAR = "linear"    # the neighbouring texel centres, blended by distance


class ImageAddress(StrEnum):
    """What a coordinate outside the image reads.

    Only CLAMP is portable today. WRAP and MIRROR are implemented on the python backend
    and have device objects that request them, but the GPU sampling paths read the edge
    texel instead of wrapping: with texel-space coordinates the drivers ignore
    CL_ADDRESS_REPEAT / CL_ADDRESS_MIRRORED_REPEAT and CU_TR_ADDRESS_MODE_WRAP /
    _MIRROR (measured on the NVIDIA and AMD OpenCL devices and the CUDA device available
    while developing this). Launching such an image on a GPU backend raises rather than
    returning different numbers from the same kernel on the python backend.
    """

    CLAMP = "clamp"      # the nearest edge texel
    WRAP = "wrap"        # continue from the opposite edge (python backend only)
    MIRROR = "mirror"    # reflect back inside the image (python backend only)


FILTERS:Tuple[str, ...] = tuple(member.value for member in ImageFilter)
ADDRESS_MODES:Tuple[str, ...] = tuple(member.value for member in ImageAddress)

# Channel counts an image may have. The device formats a sample can carry are 1, 2 or 4
# wide; 3-channel data is rejected explicitly (see _normalise_channels) rather than being
# guessed at, because no hardware format matches it without padding.
CHANNELS:Tuple[int, ...] = (1, 2, 4)


class Image:
    """A 1D, 2D or 3D image that kernels can read and write.

    Construct it from a file path, a numpy array, or just a shape::

        Image("photo.png")                        # read with imageio
        Image(numpy_image)                        # (height, width)
        Image(numpy_image, channels=4)            # (height, width, 4)
        Image(shape=(width, height))              # allocate, fill later

    The channel count is never inferred from the array shape: every axis of ``shape`` is a
    spatial extent, and the trailing axis of ``data`` is a spatial extent too unless
    ``channels`` says otherwise. That keeps a 4x4 grey image and a 4-pixel RGBA row from
    being confused, which guessing used to do.

    Sampling rules are fixed for the whole image, so a kernel needs no sampler
    argument::

        Image(..., filter="linear", address="clamp")

    The same object can be handed to a kernel on any backend. It is bound to a device
    lazily, on first use, so one Image may serve a cuda launch and an opencl launch in
    the same process.
    """

    def __init__(self, data:Union[str, np.ndarray, None]=None,
                 shape:Optional[Tuple[int, ...]]=None,
                 dtype:Any=np.float32,
                 channels:Optional[int]=None,
                 filter:Union[ImageFilter, str]=ImageFilter.POINT,
                 address:Union[ImageAddress, str]=ImageAddress.CLAMP):
        self._filter:ImageFilter = ImageFilter.parse(filter, "filter")
        self._address:ImageAddress = ImageAddress.parse(address, "address")
        self._channels:int = 1 if channels is None else _normalise_channels(channels)
        # Whether the caller named a channel axis at all. channels=None means every axis of
        # the given shape is spatial; an explicit value (even 1) makes the last axis the
        # channel axis, so the two forms can never disagree about what a shape means.
        self._channels_given:bool = channels is not None

        if isinstance(data, str):
            import imageio.v3 as iio

            data = iio.imread(data)

        if data is not None:
            image = np.ascontiguousarray(np.asarray(data))
            spatial = _spatial_shape(image.shape, self._channels if self._channels_given
                                     else None, "data")
            self._data:Optional[np.ndarray] = image
            self._extents:Tuple[int, int, int] = _extents(spatial)
            self._dtype:np.dtype = image.dtype
        elif shape is None:
            raise ValueError("Image needs either an image or a shape")
        else:
            spatial = _spatial_shape(tuple(int(v) for v in shape),
                                     self._channels if self._channels_given else None,
                                     "shape")
            self._extents = _extents(spatial)
            self._dtype = np.dtype(dtype)
            self._data = None

        # Device bindings, keyed by backend name: one Image can be used by several.
        self._bindings:dict = {}

        # Bind to CUDA immediately: loading a module with texture or surface calls
        # stops the driver from accepting new textures and surfaces, so an image
        # created after any kernel has compiled could never get device storage. An
        # eagerly-bound image is created while nothing has been compiled yet.
        try:
            self.bind_cuda()
        except BaseException:
            # No CUDA runtime, no device, or no context: nothing to bind yet, and a
            # launch will bind it (see PyKernel._bind_images).
            pass

    # ---------- description ----------

    @property
    def filter(self)->ImageFilter:
        return self._filter

    @property
    def address(self)->ImageAddress:
        return self._address

    @property
    def dtype(self)->np.dtype:
        return self._dtype

    @property
    def channels(self)->int:
        return self._channels

    @property
    def width(self)->int:
        return self._extents[0]

    @property
    def height(self)->int:
        return self._extents[1]

    @property
    def depth(self)->int:
        return self._extents[2]

    @property
    def ndim(self)->int:
        if self.depth > 1:
            return 3
        if self.height > 1:
            return 2
        return 1

    @property
    def shape(self)->Tuple[int, ...]:
        """The host shape: (height[, width]) without trailing 1s, plus channels."""
        if self.depth > 1:
            shape:Tuple[int, ...] = (self.depth, self.height, self.width)
        elif self.height > 1:
            shape = (self.height, self.width)
        else:
            shape = (self.width,)
        if self._channels > 1:
            shape = shape + (self._channels,)
        return shape

    @property
    def data(self)->Optional[np.ndarray]:
        """The host mirror, or None when only a shape was given."""
        return self._data

    @property
    def nbytes(self)->int:
        return int(np.prod(self.shape)) * self._dtype.itemsize

    # ---------- host-side sampling (the python backend runs this) ----------

    def __getitem__(self, index:Any)->Any:
        """Sample the image, with the same rules the device applies."""
        coords = self._coords(index)
        return self._sample(coords)

    def __setitem__(self, index:Any, value:Any)->None:
        """Store one element.

        Only the python backend reaches this method; a GPU launch replaces the
        subscript with a device store. Marking the backend here makes :meth:`read`
        return the host mirror rather than a device array that was never written.
        """
        self._last_backend:str = "python"
        coords = self._coords(index)
        self._store(coords, value)

    def _coords(self, index:Any)->Tuple[float, ...]:
        """Normalise a subscript into image coordinates.

        The host shape is (depth, height, width) -- numpy order -- while the kernel
        addresses (x, y, z), so the tuple is reversed here. An integer subscript names
        a texel and is moved to that texel's centre (``i + 0.5``), which is what the
        device does for an integer coordinate; a fractional subscript is used as given
        and addresses a position between texels.
        """
        if not isinstance(index, tuple):
            index = (index,)
        values = []
        for item in index:
            if isinstance(item, np.ndarray) and item.ndim == 0:
                item = item.item()
            if isinstance(item, (bool, np.bool_)):
                raise TypeError("image coordinates must be numbers, got a bool")
            if not isinstance(item, (int, float, np.integer, np.floating)):
                raise TypeError(
                    "image coordinates must be numbers, got %s"
                    % type(item).__name__)
            integral = isinstance(item, (int, np.integer))
            values.append(float(item) + 0.5 if integral else float(item))

        expected = self.ndim
        if len(values) < expected:
            raise IndexError(
                "this image needs %d coordinate(s), got %d" % (expected, len(values)))
        # Reverse to (x, y, z); the leading axes of a taller subscript are ignored.
        return tuple(reversed(values[:expected]))

    def _sample(self, coords:Tuple[float, ...])->Any:
        if self._data is None:
            raise RuntimeError("this image has no data; fill it before sampling")
        taps = self._taps(coords)
        if len(taps) == 1:
            weight, position = taps[0]
            return self._read_texel(position)
        return self._blend(taps)

    def _taps(self, coords:Tuple[float, ...])->list:
        """The (weight, texel) pairs one sample reads.

        Coordinates live in texel space: texel ``i`` covers ``[i, i+1)`` and its centre
        is ``i + 0.5``. Point filtering takes the texel the coordinate falls in, so a
        centre coordinate ``i + 0.5`` returns texel ``i`` exactly. Linear filtering
        blends the two texel centres nearest the coordinate.
        """
        per_axis = []
        for axis, value in enumerate(coords):
            limit = self._extents[axis]
            if self._filter is ImageFilter.POINT:
                per_axis.append([(1.0, self._clamp(int(np.floor(value)), limit))])
                continue

            # Linear: the two centres bracketing the coordinate. A coordinate that
            # lands exactly on a centre reads only that texel.
            base = int(np.floor(value - 0.5))
            fraction = (value - 0.5) - base
            if fraction <= 0.0:
                per_axis.append([(1.0, self._clamp(base, limit))])
                continue
            per_axis.append([
                (1.0 - fraction, self._clamp(base, limit)),
                (fraction, self._clamp(base + 1, limit)),
            ])

        # Cartesian product across axes.
        taps:list = [(1.0, ())]
        for axis_pairs in per_axis:
            expanded = []
            for weight, positions in taps:
                for tap_weight, texel in axis_pairs:
                    expanded.append((weight * tap_weight, positions + (texel,)))
            taps = expanded
        return taps

    def _clamp(self, texel:int, limit:int)->int:
        """Apply the address mode to one axis."""
        if 0 <= texel < limit:
            return texel
        if self._address is ImageAddress.CLAMP:
            return 0 if texel < 0 else limit - 1
        if self._address is ImageAddress.WRAP:
            return texel % limit
        # mirror
        period = 2 * limit - 1 if limit > 1 else 1
        position = texel % period
        return position if position < limit else period - position

    def _read_texel(self, texel:Tuple[int, ...])->Any:
        """Read one texel as an ndarray of channels (host (z, y, x) order)."""
        index = tuple(reversed(texel))
        if self._channels == 1:
            return np.asarray(self._data[index])
        return np.asarray(self._data[index])

    def _blend(self, taps:list)->Any:
        total = None
        for weight, texel in taps:
            value = self._read_texel(texel) * weight
            total = value if total is None else total + value
        return total

    def _store(self, coords:Tuple[float, ...], value:Any)->None:
        """Write one texel; writes use point addressing (there is nothing to blend)."""
        if self._data is None:
            # A shape-only image becomes writable on demand: allocate the mirror so a
            # kernel that only writes still has storage to describe.
            self._data = np.zeros(self.shape, dtype=self._dtype)
        position = tuple(self._clamp(int(np.floor(v)), limit)
                         for v, limit in zip(coords, self._extents))
        index = tuple(reversed(position))
        if self._channels == 1:
            self._data[index] = value
        else:
            self._data[index] = np.asarray(value).reshape(self._channels)

    # ---------- device binding ----------

    def bind_cuda(self)->Any:
        """Create both device views this image might need, and cache them.

        A kernel reads through a texture object and writes through a surface object,
        and which one a parameter needs is only known once the kernel is compiled.
        On the tested driver, loading a module that samples or stores makes any later
        texture/surface creation fail, so both are created here -- before compilation
        -- and the launch picks the one its argument type calls for.
        """
        binding = self._bindings.get("cuda")
        if binding is not None:
            return binding

        from .cuda.oop.context import Context
        from .cuda.oop.devices import Devices
        from .cuda.oop.array import Array
        from .cuda.oop.surface import SurfaceObject
        from .cuda.driver import CUaddress_mode, CUfilter_mode

        context = Context.current()
        if context is None:
            # Nothing has run on this thread yet; the first device's context is the one
            # a default launch would pick anyway.
            if len(Devices) == 0:
                raise RuntimeError("no CUDA device is available for this image")
            context = Devices[0].default_context

        # One array backs both views, and it carries the surface flag so a surface
        # object can bind to it.
        array = Array(self.width, self.height, self.depth, self.dtype,
                      self.channels, surface=True)
        if self._data is not None:
            array.set_data(self._data)

        # The surface is created first: on the tested driver, binding a surface to an
        # array that already has a texture over it fails, while a texture over an
        # array that already has a surface is accepted.
        surface = SurfaceObject(array)

        filter_mode = {
            ImageFilter.POINT: CUfilter_mode.CU_TR_FILTER_MODE_POINT,
            ImageFilter.LINEAR: CUfilter_mode.CU_TR_FILTER_MODE_LINEAR,
        }[self._filter]
        address_mode = {
            ImageAddress.CLAMP: CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
            ImageAddress.WRAP: CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP,
            ImageAddress.MIRROR: CUaddress_mode.CU_TR_ADDRESS_MODE_MIRROR,
        }[self._address]

        # Texel-space coordinates, so the device sees the numbers the host path uses.
        texture = context.create_texture(array, normalized=False,
                                         filter_mode=filter_mode,
                                         address_mode=address_mode)

        binding = (array, texture, surface)
        self._bindings["cuda"] = binding
        return binding

    def read(self)->np.ndarray:
        """The image's current contents, read back from the device if needed.

        A kernel that writes an image leaves the result on the device; this brings it
        back so the host mirror reflects it. An image never bound to a device simply
        returns what it already holds.

        One Image may hold bindings for more than one backend (it is created eagerly
        for CUDA, and OpenCL binds it at launch), so the source is read from whichever
        backend most recently wrote it -- tracked in ``_last_backend``. The python
        backend writes the host mirror and nothing on a device, so it is read from
        here directly.
        """
        if getattr(self, "_last_backend", None) == "python":
            if self._data is None:
                raise RuntimeError("this image has no data to return")
            return self._data

        for backend in self._read_order():
            binding = self._bindings.get(backend)
            if binding is None:
                continue
            if backend == "cuda":
                data = binding[0].to_numpy()
            else:
                image, queue = binding
                data = np.empty(self.shape, dtype=self._dtype)
                pointer = data.ctypes.data_as(__import__("ctypes").c_void_p)
                image.read(queue, host_ptr=pointer)
                queue.wait()
            self._data = data
            return data

        if self._data is None:
            raise RuntimeError(
                "this image has neither host data nor a device binding")
        return self._data

    def _read_order(self)->list:
        """Backends to consult, starting with the one that wrote the image last."""
        order = ["cuda", "opencl"]
        last = getattr(self, "_last_backend", None)
        if last in order:
            order.remove(last)
            order.insert(0, last)
        return order

    def to_numpy(self)->np.ndarray:
        """Alias for :meth:`read`, matching the numpy layer's spelling."""
        return self.read()

    # ---------- python protocol ----------

    def __repr__(self)->str:
        size = "x".join(str(v) for v in self._extents[:self.ndim])
        return "Image(%s, %d channel(s), %s, filter=%r, address=%r)" % (
            size, self._channels, self._dtype, self._filter, self._address)

    __str__ = __repr__


def _normalise_channels(channels:Any)->int:
    """Validate a requested channel count."""
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise TypeError(
            "channels must be %s, got %s"
            % (" or ".join(str(v) for v in CHANNELS), type(channels).__name__))
    if channels == 3:
        # No device image format is 3 channels wide: a sample is returned as a 1, 2 or 4
        # component value, so RGB data has to be padded. Saying that is more useful than
        # letting the caller discover it from a driver error.
        raise ValueError(
            "channels=3 is not a device image format: samples are 1, 2 or 4 components "
            "wide. Pad the data to 4 channels and pass channels=4")
    if channels not in CHANNELS:
        raise ValueError(
            "channels must be %s, got %d" % (" or ".join(str(v) for v in CHANNELS),
                                             channels))
    return channels


def _spatial_shape(shape:Tuple[int, ...], channels:Any, what:str)->Tuple[int, ...]:
    """Strip the trailing channel axis when channels was given, and validate the rank.

    The channel count is never inferred: a trailing axis is a spatial extent unless the
    caller says the array carries channel data. That is what keeps (4, 4) a 4x4 grey
    image instead of a 4-pixel RGBA row, and (8, 16, 3) an RGB image instead of a volume.

    ``channels=None`` means "not given": every axis is spatial. Passing any explicit value
    -- including 1 -- makes the trailing axis the channel axis, so the two forms never
    disagree about what a shape means.
    """
    if channels is not None:
        if not shape or shape[-1] != channels:
            raise ValueError(
                "Image %s %s does not end with the %d channel(s) it was given; pass "
                "channels=%d only for data whose last axis is the channel axis"
                % (what, shape, channels, channels))
        shape = shape[:-1]

    if len(shape) not in (1, 2, 3):
        raise ValueError(
            "Image %s must have 1, 2 or 3 spatial dimensions%s, got %s"
            % (what, " plus an optional channel axis" if channels is not None else "",
               tuple(shape)))
    if any(int(v) <= 0 for v in shape):
        raise ValueError("Image %s must have positive extents, got %s"
                         % (what, tuple(shape)))
    return tuple(int(v) for v in shape)


def _extents(spatial:Tuple[int, ...])->Tuple[int, int, int]:
    """Host spatial shape -> (width, height, depth); missing axes are 1."""
    if len(spatial) == 1:
        return int(spatial[0]), 1, 1
    if len(spatial) == 2:
        return int(spatial[1]), int(spatial[0]), 1
    return int(spatial[2]), int(spatial[1]), int(spatial[0])
