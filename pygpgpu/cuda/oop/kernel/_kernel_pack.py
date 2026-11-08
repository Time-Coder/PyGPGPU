# Images and the 3-component vector padding the CUDA ABI needs.
from __future__ import annotations
from typing import Dict, List
import numpy as np
from ...driver import (
    CUInfo,
    cudaSurfaceObject_t
)
from ..program_info import ArgInfo


class _KernelPackMixin:

    def _bind_image(self, image:object)->object:
        """Turn a host-side image description into a device object for this launch.

        The ``cudaTextureObject_t`` / ``cudaSurfaceObject_t`` descriptions carry no
        context, so binding happens here, where the kernel knows which context it
        runs in. A description bound earlier (see its ``bind``) returns what it
        already holds, so repeated launches reuse the same device storage.
        """
        bound = image.bind()
        if bound is not None:
            return bound

        from ..context import Context

        context = Context.current()
        if context is None:
            raise RuntimeError(
                "no CUDA context is current; launching the kernel normally "
                "establishes one")

        if isinstance(image, cudaSurfaceObject_t):
            surface = context.create_surface(image.shape)
            # The surface owns an array created with the surface flag; seed it from
            # the host image when one was supplied.
            if image.image is not None:
                surface.array.set_data(image.image)
            image._device = (surface.array, surface)
            return surface

        array = context.create_array(image.width, image.height, 1,
                                     image.dtype, image.channels)
        if image.image is not None:
            array.set_data(image.image)

        options:Dict[str, object] = {"normalized": image.normalized}
        if image.address_mode is not None:
            options["address_mode"] = image.address_mode
        if image.filter_mode is not None:
            options["filter_mode"] = image.filter_mode

        texture = context.create_texture(array, **options)
        # Keep both alive alongside the description the caller passed in, so a
        # second launch reuses them instead of allocating again.
        image._device = (array, texture)
        return texture


    def _is_vec3_arg(self, arg_info:ArgInfo)->bool:
        """Whether this argument's element type is a 3-component vector.

        The driver table also maps the scalar spellings (`float`, `int`, ...), so the
        `_fields_` test has to be defensive: a scalar type has none, and asking for it
        turns every scalar pointer argument into an AttributeError.
        """
        base:str = arg_info.base_type_str
        cls:object = CUInfo.vec_types.get(base)
        fields:object = getattr(cls, "_fields_", None)
        return (fields is not None and len(fields) == 3)


    @staticmethod
    def _pack_vec3(value:object, content_type:object)->object:
        """Pack 3-component data such as float3/int3 into the CUDA compact layout (drop the padding component)."""
        fields:List[str] = [f for f, _ in content_type._fields_][:3]
        elem_dtype:object = content_type.dtype.fields[fields[0]][0]

        if isinstance(value, np.ndarray) and value.dtype.fields is not None:
            count:int = int(np.prod(value.shape)) if value.ndim > 0 else 1
            flat:object = value.reshape(-1)
            packed:object = np.empty((count, 3), dtype=elem_dtype)
            for i, f in enumerate(fields):
                packed[:, i] = flat[f]
            return packed.reshape(value.shape + (3,)) if value.ndim > 1 else packed

        if isinstance(value, np.void):
            packed = np.empty(3, dtype=elem_dtype)
            for i, f in enumerate(fields):
                packed[i] = value[f]
            return packed

        if hasattr(value, "x"):
            packed = np.empty(3, dtype=elem_dtype)
            for i, f in enumerate(fields):
                packed[i] = getattr(value, f)
            return packed

        raise TypeError(f"cannot pack {type(value).__name__} as 3-component vector")


    @staticmethod
    def _pack_vec3_value(value:object, content_type:object)->object:
        """Pack one 3-component *value* into CUDA's compact layout, as a ctypes buffer.

        This is the by-value counterpart of :meth:`_pack_vec3`, which packs whole arrays for
        upload: a kernel argument is passed as the address of its value, so the buffer has to
        be a ctypes object, and CUDA's `float3` is three floats with no padding.
        """
        fields:List[str] = [f for f, _ in content_type._fields_][:3]
        element:object = content_type._fields_[0][1]
        packed:object = (element * 3)()
        for i, f in enumerate(fields):
            packed[i] = value[f] if isinstance(value, np.void) else getattr(value, f)
        return packed
