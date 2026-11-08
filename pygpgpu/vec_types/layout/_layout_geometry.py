# The geometry of a record and of a vector: the field list the device declares,
# the alignment, and the components a vector really has (a 3-component vector
# carries a trailing padding field).
from __future__ import annotations
from ._layout_classes import _device_class, _host_classes

from typing import Callable, List, Optional, Tuple
import numpy as np



def _component_size(dtype:np.dtype)->int:
    """The size of one scalar or vector value in the device layout."""
    vector:Optional[str] = None
    if dtype.fields is not None:
        vector = _vector_dtype_name(dtype)
    if vector is not None:
        return int(_device_class(vector, "opencl").dtype.itemsize)
    return int(dtype.itemsize)


def _alignment_of(dtype:np.dtype)->int:
    """The OpenCL C alignment of a field type.

    A vector of three or four components is 16-byte aligned (a padded vec3 is 16 bytes), a
    two-component vector 8, and a scalar its own size. A structure takes the largest
    alignment of its members.
    """
    if dtype.fields is not None:
        vector:Optional[str] = _vector_dtype_name(dtype)
        if vector is not None:
            size:int = int(_device_class(vector, "opencl").dtype.itemsize)
            return size if size in (2, 4, 8, 16) else 16
        # A nested structure: the maximum alignment of its members.
        return max([_alignment_of(np.dtype(dtype.fields[name][0]))
                    for name in dtype.names or ()] or [1])
    return int(dtype.itemsize) or 1


# One field of a record as the packers describe it: its name, its dtype (or the ctypes
# type that stands for it), and the shape it is an array of when it is one.
RecordField = Tuple[str, object, Optional[Tuple[int, ...]]]

# What every packer hands back: the value in the device's own layout, plus the callable
# that copies a moved value back into the host's form (None when nothing moved).
Packed = Tuple[object, Optional[Callable[[object], None]]]


def _opencl_record_layout(fields:List[RecordField])->np.dtype:
    """A record dtype with the offsets and total size OpenCL C would give the structure."""
    names:List[str] = []
    formats:List[object] = []
    offsets:List[int] = []
    cursor:int = 0
    alignment:int = 1
    for name, field_dtype, shape in fields:
        base:np.dtype = np.dtype(field_dtype)
        element_align:int = _alignment_of(base)
        element_size:int = _component_size(base)
        count:int = 1
        if shape:
            for extent in shape:
                count *= int(extent)
            element_align = max(element_align, 1)
        field_align:int = element_align
        alignment = max(alignment, field_align)
        cursor = -(-cursor // field_align) * field_align
        names.append(name)
        formats.append((base, shape) if shape else base)
        offsets.append(cursor)
        cursor += element_size * count
    itemsize:int = -(-cursor // alignment) * alignment
    return np.dtype({"names": names, "formats": formats,
                     "offsets": offsets, "itemsize": itemsize})


def _record_device_dtype(dtype:np.dtype, backend:str)->Tuple[np.dtype, bool]:
    """The device-layout dtype for a record, and whether anything had to change.

    Every field is looked at on its own: a vector field of width three is replaced by the
    padded device type, a nested record is rebuilt, and an array of either keeps its extents.
    """
    fields:List[RecordField] = []
    changed:bool = False
    for name in dtype.names or ():
        field_dtype:np.dtype = np.dtype(dtype.fields[name][0])
        shape:Optional[Tuple[int, ...]] = None
        if field_dtype.subdtype is not None:
            base_dtype, shape = field_dtype.subdtype
            field_dtype = base_dtype
        vector:Optional[str] = _vector_dtype_name(field_dtype)
        if vector is not None and _needs_padding(vector, backend):
            field_dtype = _device_class(vector, "opencl").numpy_dtype()
            changed = True
        elif vector is None and field_dtype.fields is not None:
            replaced, nested_changed = _record_device_dtype(field_dtype, backend)
            if nested_changed:
                field_dtype = replaced
                changed = True
        fields.append((name, field_dtype, shape))
    if backend != "opencl":
        return np.dtype([(name, dtype_) if shape is None else (name, dtype_, shape)
                         for name, dtype_, shape in fields]), changed
    return _opencl_record_layout(fields), changed


def vector_name_of(value:object)->Optional[str]:
    """The vector type name of a vector instance, or None."""
    if not isinstance(value, type(value)):
        return None
    # A numpy scalar is not a vector, even when its class name collides with a vector type
    # name (`np.uint8` is `numpy.uint8`, and the registry has an eight-component `uint8`).
    if isinstance(value, np.generic):
        return None
    for name, cls in _host_classes().items():
        if type(value) is cls or isinstance(value, cls):
            return name
    return None


def _is_vector_class(cls:object)->Optional[str]:
    """The vector type name a class stands for, or None."""
    if not isinstance(cls, type):
        return None
    classes:dict[str, type] = _host_classes()
    if cls.__name__ in classes:
        return cls.__name__
    for base in cls.__mro__:
        if base.__name__ in classes:
            return base.__name__
    return None


def _needs_padding(name:str, backend:str)->bool:
    """True when the device layout has more components than the host's (a padded vec3)."""
    if backend != "opencl":
        return False
    return len(_vec_device_components(name)) > len(_host_components(name))


def _vec_device_components(name:str)->List[str]:
    """The component names the device type actually has (a vec3 has a padding slot)."""
    return [field for field, _ in _device_class(name, "opencl")._fields_]


def _host_components(name:str)->List[str]:
    return [field for field, _ in _host_classes()[name]._fields_ if field != "_"]


def _vector_dtype_name(dtype:np.dtype)->Optional[str]:
    """The vector type an ndarray dtype denotes, or None.

    The comparison is on the whole dtype, not on the field names: `float3` and `int3` share
    the names x/y/z and differ only in the component type. A device-layout dtype (the padded
    OpenCL form) maps back to the same vector, so results arriving from a kernel are
    recognized too.
    """
    wanted:np.dtype = np.dtype(dtype)
    for name, cls in _host_classes().items():
        if wanted == cls.numpy_dtype():
            return name
    for name in _host_classes():
        if wanted == _device_class(name, "opencl").numpy_dtype():
            return name
    return None
