"""Host <-> device memory layout for vectors, and for containers that hold them.

The host library (`pygf`) stores a 3-component vector in **three** components (12 bytes for
float3) because that is what a Python object naturally is. The device ABI is not uniform:

* CUDA and HIP define `float3` as three floats, so the host layout is already correct;
* OpenCL defines `cl_float3` as four slots (16 bytes, the fourth unused), because every
  vector type there is sized for `vload4` alignment.

So a 3-component vector -- and any structure or ndarray that contains one -- has to be
re-laid-out when it crosses into an OpenCL kernel, and re-laid-out again when data comes
back. This module is that boundary: `pack` produces the buffer the device expects,
`unpack` writes the device's result back into the caller's own object, so nothing above the
boundary ever sees the padded form.

Everything here is a no-op for CUDA/HIP and for vector widths other than three, which is
what keeps the common case free.
"""
from __future__ import annotations

import ctypes
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


_HOST_CLASSES: Optional[dict[str, type]] = None


def _host_classes()->dict[str, type]:
    """name -> the class the host uses for that vector type (`pygf`'s classes)."""
    global _HOST_CLASSES
    if _HOST_CLASSES is not None:
        return _HOST_CLASSES
    import pygf
    from ..kernel.translator import _vec_registry

    classes:dict[str, type] = {}
    for name in _vec_registry():
        host:Any = getattr(pygf, name, None)
        classes[name] = host if isinstance(host, type) else _vec_registry()[name]
    _HOST_CLASSES = classes
    return classes


def _device_class(name:str, backend:str)->type:
    """The class whose memory layout the device uses for this vector type."""
    if backend != "opencl":
        return _host_classes()[name]
    from ..kernel.translator import _vec_registry
    return _vec_registry()[name]


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


def _opencl_record_layout(fields:List[Any])->np.dtype:
    """A record dtype with the offsets and total size OpenCL C would give the structure."""
    names:List[str] = []
    formats:List[Any] = []
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
    fields:List[Any] = []
    changed:bool = False
    for name in dtype.names or ():
        field_dtype:np.dtype = np.dtype(dtype.fields[name][0])
        shape:Any = None
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


def _copy_record(host:Any, target:Any, backend:str)->None:
    """Copy a record's fields into a device-layout record (or back, when reversed)."""
    for name in host.dtype.names or ():
        field_dtype:np.dtype = np.dtype(host.dtype.fields[name][0])
        shape:Any = None
        if field_dtype.subdtype is not None:
            base_dtype, shape = field_dtype.subdtype
            field_dtype = base_dtype
        vector:Optional[str] = _vector_dtype_name(field_dtype)
        if vector is not None and _needs_padding(vector, backend):
            for component in _host_components(vector):
                target[name][component] = host[name][component]
        elif vector is None and field_dtype.fields is not None:
            _copy_record(host[name], target[name], backend)
        else:
            target[name] = host[name]


def _pack_record_array(array:np.ndarray, backend:str)->Tuple[Any, Optional[Callable[[Any], None]]]:
    """A record array in the device layout, with a callable that copies results home."""
    if backend != "opencl":
        return array, None
    device_dtype, changed = _record_device_dtype(array.dtype, backend)
    if not changed:
        return array, None
    packed:np.ndarray = np.zeros(array.shape, dtype=device_dtype)
    _copy_record(array, packed, backend)

    def unpack(source:Any)->None:
        # The kernel wrote the device layout; the caller's array keeps the host one.
        _copy_record(np.asarray(source), array, backend)

    return packed, unpack


def vector_name_of(value:Any)->Optional[str]:
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


def _is_vector_class(cls:Any)->Optional[str]:
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


# ---------------------------------------------------------------------------
# A single vector value
# ---------------------------------------------------------------------------


def _pack_vector(value:Any, backend:str)->Any:
    name:str = vector_name_of(value) or ""
    if not name or not _needs_padding(name, backend):
        return value
    target:type = _device_class(name, backend)
    packed:Any = target()
    for field in _host_components(name):
        setattr(packed, field, getattr(value, field))
    return packed


def _unpack_vector(device_value:Any, host_value:Any)->None:
    """Copy the components back; the padding slot is ignored."""
    for field in _host_components(vector_name_of(host_value) or ""):
        setattr(host_value, field, getattr(device_value, field))


# ---------------------------------------------------------------------------
# ndarrays of vectors, and structures that contain vectors
# ---------------------------------------------------------------------------


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


def _pack_array(array:np.ndarray, backend:str)->Tuple[Any, Optional[Callable[[Any], None]]]:
    """A vector-typed ndarray in the device layout, plus a way to copy results back."""
    name:Optional[str] = _vector_dtype_name(array.dtype)
    if name is None or not _needs_padding(name, backend):
        return array, None
    device_dtype:np.dtype = _device_class(name, "opencl").numpy_dtype()
    packed:np.ndarray = np.zeros(array.shape, dtype=device_dtype)
    for field in _host_components(name):
        packed[field] = array[field]

    def unpack(source:Any)->None:
        # The kernel may have written the padding slot; only real components come home.
        current:np.ndarray = source if isinstance(source, np.ndarray) else packed
        for field in _host_components(name):
            array[field] = current[field]

    return packed, unpack


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def pack(value:Any, backend:str)->Tuple[Any, Optional[Callable[[Any], None]]]:
    """The device-layout version of ``value``, and a callable that writes results back.

    ``value`` may be a vector instance, an ndarray of vectors, or a ctypes structure that
    contains either. The write-back callable returns None for values that need no
    conversion (CUDA/HIP, non-3-component vectors, everything else), so callers can keep it
    unconditionally and simply run it after the launch.
    """
    if value is None:
        return value, None
    if isinstance(value, np.void):
        # A structure value is a record: convert it through a one-element array so the same
        # field-wise rules apply as for an array of structures.
        packed, unpack = _pack_record_array(np.asarray(value).reshape(1), backend)
        if unpack is None:
            return value, None
        # A by-value argument is read-only inside the kernel, so nothing has to come home.
        return packed[0], None
    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None and _vector_dtype_name(value.dtype) is None:
            return _pack_record_array(value, backend)
        return _pack_array(value, backend)
    name:Optional[str] = vector_name_of(value)
    if name is not None:
        if not _needs_padding(name, backend):
            return value, None
        return _pack_vector(value, backend), None
    if isinstance(value, ctypes.Structure):
        return _pack_structure(value, backend)
    return value, None


def _pack_structure(value:Any, backend:str)->Tuple[Any, Optional[Callable[[Any], None]]]:
    """Re-lay a ctypes structure whose members include vectors or arrays of them.

    The device type is built on the fly: same field order, with every vector member (and
    every array of vectors, and every nested structure) replaced by its device layout. The
    write-back copies each member home.
    """
    if backend != "opencl":
        return value, None
    source_type:type = type(value)
    members:List[Tuple[str, Any, Any]] = []
    changed:bool = False
    for field_name, field_type in source_type._fields_:
        device_type:Any = field_type
        name:Optional[str] = _is_vector_class(field_type)
        if name is not None:
            device_type = _device_class(name, backend)
            changed = True
        elif issubclass(field_type, ctypes.Array):
            name = _is_vector_class(field_type._type_)
            if name is not None:
                device_type = _device_class(name, backend) * field_type._length_
                changed = True
            else:
                if issubclass(field_type._type_, ctypes.Structure):
                    packed, packed_write_back = _pack_structure(field_type._type_(), backend)
                    if packed_write_back is not None:
                        device_type = type(packed)
                        changed = True
        elif issubclass(field_type, ctypes.Structure):
            packed, packed_write_back = _pack_structure(field_type(), backend)
            if packed_write_back is not None:
                # The nested layout differs (a vector inside it): the packed instance's class
                # is the field's device type.
                device_type = type(packed)
            # When the nested layout is unchanged the original class is already right; using
            # the packed *instance* here is what made ctypes answer "type of field 'inner'
            # must be a C type" for a structure whose only members are scalars.
            changed = changed or packed_write_back is not None
        members.append((field_name, device_type, field_type))

    if not changed:
        return value, None

    device_type_new:type = type(f"{source_type.__name__}_dev",
                                (ctypes.Structure,),
                                {"_fields_": [(name, device) for name, device, _ in members]})
    packed:Any = device_type_new()
    for field_name, device_field_type, host_field_type in members:
        name = _is_vector_class(host_field_type)
        if name is not None:
            setattr(packed, field_name, _pack_vector(getattr(value, field_name), backend))
        elif issubclass(host_field_type, ctypes.Array) and _is_vector_class(host_field_type._type_):
            host_array:Any = getattr(value, field_name)
            target:Any = getattr(packed, field_name)
            for index in range(host_field_type._length_):
                target[index] = _pack_vector(host_array[index], backend)
        else:
            setattr(packed, field_name, getattr(value, field_name))

    def unpack(source:Any)->None:
        for field_name, _device_field, host_field_type in members:
            name = _is_vector_class(host_field_type)
            if name is not None:
                _unpack_vector(getattr(source, field_name), getattr(value, field_name))
            elif issubclass(host_field_type, ctypes.Array) and _is_vector_class(host_field_type._type_):
                for index in range(host_field_type._length_):
                    _unpack_vector(getattr(source, field_name)[index],
                                   getattr(value, field_name)[index])

    return packed, unpack
