"""OpenCL C structure layout for parsed kernels.

ctypes aligns a nested structure by its members' scalar types, so a structure holding a
`float3` comes out 20 bytes where the compiler's ABI -- a padded vector is 16-byte aligned --
gives 32. The parsed type has to match the compiler exactly, because the argument is passed
by value (`clSetKernelArg` with `sizeof`) and an array of such structures is mapped onto the
host buffer by viewing it with the parsed dtype.

These helpers compute the OpenCL offsets and size, and insert the padding members so both the
ctypes type and the numpy record match.
"""
from __future__ import annotations

from ctypes import c_char
from typing import Any, List, Optional

import numpy as np


def np_alignment(dtype:Any) -> int:
    """The OpenCL C alignment of a field type.

    A three- or four-component vector is 16-byte aligned (a padded vec3 occupies 16 bytes), a
    two-component vector 8, and a scalar its own size; a structure takes the largest
    alignment of its members.
    """
    from ...vec_types.layout import _vector_dtype_name

    field:np.dtype = np.dtype(dtype)
    if field.fields is not None:
        vector:Optional[str] = _vector_dtype_name(field)
        if vector is not None:
            first:str = (field.names or ("x",))[0]
            components:int = int(field.itemsize) // max(1, int(field[first].itemsize))
            return 16 if components >= 3 else 8
        return max([np_alignment(np.dtype(field.fields[name][0]))
                    for name in field.names or ()] or [1])
    return int(field.itemsize) or 1


def align_struct(fields:List[Any], spec:List[Any]) -> tuple:
    """The aligned ctypes fields and numpy dtype for one parsed structure.

    ``fields`` are the ``(name, ctype)`` members and ``spec`` the matching
    ``(name, numpy_dtype[, shape])`` entries, in declaration order.
    """
    layout:List[tuple] = []
    cursor:int = 0
    alignment:int = 1
    for (member_name, ctype), entry in zip(fields, spec):
        base:np.dtype = np.dtype(entry[1]) if len(entry) > 1 else np.dtype("u1")
        shape:Any = entry[2] if len(entry) > 2 else ()
        element_align:int = np_alignment(base)
        count:int = 1
        for extent in shape or ():
            count *= int(extent)
        alignment = max(alignment, element_align)
        cursor = -(-cursor // element_align) * element_align
        layout.append((entry[0], ctype, base, shape, cursor, count))
        cursor += int(base.itemsize) * count

    itemsize:int = -(-cursor // alignment) * alignment

    aligned_fields:List[Any] = []
    names:List[str] = []
    formats:List[Any] = []
    offsets:List[int] = []
    cursor = 0
    for index, (_member_name, ctype, base, shape, offset, count) in enumerate(layout):
        if offset > cursor:
            aligned_fields.append((f"_pad{index}", c_char * (offset - cursor)))
            cursor = offset
        aligned_fields.append((layout[index][0], ctype))
        names.append(layout[index][0])
        formats.append((base, shape) if shape else base)
        offsets.append(offset)
        cursor = offset + int(base.itemsize) * count
    if itemsize > cursor:
        aligned_fields.append(("_pad_tail", c_char * (itemsize - cursor)))

    dtype:np.dtype = np.dtype({"names": names, "formats": formats,
                               "offsets": offsets, "itemsize": itemsize})
    return aligned_fields, dtype
