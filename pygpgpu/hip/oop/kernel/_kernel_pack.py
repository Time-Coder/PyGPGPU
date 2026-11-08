# The 3-component vector padding the HIP ABI needs.
from __future__ import annotations
from typing import List
import numpy as np
from ...driver import (
    HIPInfo
)
from ..program_info import ArgInfo


class _KernelPackMixin:

    def _is_vec3_arg(self, arg_info:ArgInfo)->bool:
        """Whether this argument's element type is a 3-component vector.

        The driver table also maps the scalar spellings, so the `_fields_` test has to be
        defensive: a scalar type has none, and asking for it turns every scalar pointer
        argument into an AttributeError.
        """
        base:str = arg_info.base_type_str
        cls:object = HIPInfo.vec_types.get(base)
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
        """Pack one 3-component *value* into the compact layout, as a ctypes buffer.

        The by-value counterpart of :meth:`_pack_vec3`, which packs arrays for upload: a
        kernel argument is passed as the address of its value, so it has to be ctypes.
        """
        fields:List[str] = [f for f, _ in content_type._fields_][:3]
        element:object = content_type._fields_[0][1]
        packed:object = (element * 3)()
        for i, f in enumerate(fields):
            packed[i] = value[f] if isinstance(value, np.void) else getattr(value, f)
        return packed
