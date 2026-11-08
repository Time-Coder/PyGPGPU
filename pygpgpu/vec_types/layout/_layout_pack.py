# Packing a value into the layout the device reads.
from __future__ import annotations
from ._layout_geometry import (
    Packed,
    RecordField,
    _host_components,
    _is_vector_class,
    _needs_padding,
    _opencl_record_layout,
    _record_device_dtype,
    _vector_dtype_name,
    vector_name_of,
)

from ._layout_classes import _device_class

import ctypes
from typing import Dict, List, Optional, Tuple
import numpy as np



def _copy_record(host:object, target:object, backend:str)->None:
    """Copy a record's fields into a device-layout record (or back, when reversed)."""
    for name in host.dtype.names or ():
        field_dtype:np.dtype = np.dtype(host.dtype.fields[name][0])
        shape:Optional[Tuple[int, ...]] = None
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


def _pack_record_array(array:np.ndarray, backend:str)->Packed:
    """A record array in the device layout, with a callable that copies results home."""
    if backend != "opencl":
        return array, None
    device_dtype, changed = _record_device_dtype(array.dtype, backend)
    if not changed:
        return array, None
    packed:np.ndarray = np.zeros(array.shape, dtype=device_dtype)
    _copy_record(array, packed, backend)

    def unpack(source:object)->None:
        # The kernel wrote the device layout; the caller's array keeps the host one.
        _copy_record(np.asarray(source), array, backend)

    return packed, unpack


def _pack_vector(value:object, backend:str)->object:
    name:str = vector_name_of(value) or ""
    if not name or not _needs_padding(name, backend):
        return value
    target:type = _device_class(name, backend)
    packed:object = target()
    for field in _host_components(name):
        setattr(packed, field, getattr(value, field))
    return packed


def _unpack_vector(device_value:object, host_value:object)->None:
    """Copy the components back; the padding slot is ignored."""
    for field in _host_components(vector_name_of(host_value) or ""):
        setattr(host_value, field, getattr(device_value, field))


def _pack_array(array:np.ndarray, backend:str)->Packed:
    """A vector-typed ndarray in the device layout, plus a way to copy results back."""
    name:Optional[str] = _vector_dtype_name(array.dtype)
    if name is None or not _needs_padding(name, backend):
        return array, None
    device_dtype:np.dtype = _device_class(name, "opencl").numpy_dtype()
    packed:np.ndarray = np.zeros(array.shape, dtype=device_dtype)
    for field in _host_components(name):
        packed[field] = array[field]

    def unpack(source:object)->None:
        # The kernel may have written the padding slot; only real components come home.
        current:np.ndarray = source if isinstance(source, np.ndarray) else packed
        for field in _host_components(name):
            array[field] = current[field]

    return packed, unpack


def pack(value:object, backend:str)->Packed:
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


_DEV_TYPE_CACHE:Dict[Tuple[type, str], type] = {}


def _ctypes_align(field_type:type)->int:
    """The OpenCL C alignment of a ctypes field type, which ctypes does not know.

    `ctypes` lays out an OpenCL `float3` (a 16-byte padded class) with align 4, because the
    class itself is three floats plus padding. OpenCL C aligns a three- or four-component
    vector to 16, so a structure holding one is wider than ctypes computes -- and the driver
    rejects the short buffer ("Buffer size too small (40 instead of at least 64 bytes)").

    A class the caller declared is aligned the way the *declaration* spells it, which is by
    layout: `{float x; float y;}` and `float2` are one numpy dtype, the renderer spells the
    member `float2`, and OpenCL aligns that to 8 where ctypes aligns the class to 4. Without
    this branch the rebuilt structure was still 12 bytes and the driver kept refusing it.
    """
    if issubclass(field_type, ctypes.Array):
        return _ctypes_align(field_type._type_)
    name:Optional[str] = _is_vector_class(field_type)
    if name is None and issubclass(field_type, ctypes.Structure):
        name = _vector_dtype_name(np.dtype(field_type))
    if name is not None:
        size:int = ctypes.sizeof(_device_class(name, "opencl"))
        return 16 if size >= 16 else size
    if issubclass(field_type, ctypes.Structure):
        return max([_ctypes_align(kind) for _, kind in field_type._fields_] or [1])
    return ctypes.alignment(field_type) or 1


def _opencl_field_list(members:List[RecordField])->List[Tuple[str, object]]:
    """The ctypes field list that reproduces OpenCL C's layout.

    Each field is placed at the offset OpenCL would give it, and the gaps are spelled as
    anonymous padding arrays -- ctypes has no way to set an offset or an alignment directly,
    so the padding has to be a field of its own.
    """
    fields:List[Tuple[str, object]] = []
    cursor:int = 0
    alignment:int = 1
    for index, (name, device_type, _) in enumerate(members):
        field_align:int = _ctypes_align(device_type)
        alignment = max(alignment, field_align)
        aligned:int = -(-cursor // field_align) * field_align
        if aligned > cursor:
            fields.append((f"_pad{index}", ctypes.c_ubyte * (aligned - cursor)))
        cursor = aligned + ctypes.sizeof(device_type)
        fields.append((name, device_type))
    tail:int = -(-cursor // alignment) * alignment
    if tail > cursor:
        fields.append(("_pad_tail", ctypes.c_ubyte * (tail - cursor)))
    return fields


def _pack_structure(value:object, backend:str)->Packed:
    """Re-lay a ctypes structure whose members include vectors or arrays of them.

    The device type is built on the fly: same field order, with every vector member (and
    every array of vectors, and every nested structure) replaced by its device layout. The
    write-back copies each member home.

    The generated class is cached by (source type, backend): an array of these structures
    needs ONE element class -- ctypes rejects a `T[2]` field assignment that mixes two
    distinct `T_dev` types even when their layouts are identical, which is what "incompatible
    types, T_dev instance instead of T_dev instance" was.
    """
    if backend != "opencl":
        return value, None
    source_type:type = type(value)
    members:List[RecordField] = []
    changed:bool = False
    for field_name, field_type in source_type._fields_:
        device_type:type = field_type
        name:Optional[str] = _is_vector_class(field_type)
        if name is not None:
            # Only a vector the device stores in more slots than the host does changes the
            # layout: a 2-component vector is 8 bytes on both sides and a 4-component one is
            # already four slots, so rebuilding a structure that holds either of them would
            # copy bytes to produce an identical layout.
            if _needs_padding(name, backend):
                device_type = _device_class(name, backend)
                changed = True
        elif issubclass(field_type, ctypes.Array):
            name = _is_vector_class(field_type._type_)
            if name is not None:
                if _needs_padding(name, backend):
                    device_type = _device_class(name, backend) * field_type._length_
                    changed = True
            elif issubclass(field_type._type_, ctypes.Structure):
                # An array of structures: the element's own class says whether its layout
                # differs, and a nested array is asked the same question recursively.
                packed_element:object = _pack_structure(field_type._type_(), backend)[0]
                if type(packed_element) is not field_type._type_:
                    device_type = type(packed_element) * field_type._length_
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

    # The declaration and the host can disagree on the structure's *size* even when every
    # member's size agrees: the renderer spells a member whose layout matches a vector as
    # that vector, and OpenCL aligns a `float2` to 8 while `ctypes` aligns a two-float
    # structure to 4 -- so `{float2 origin; float scale;}` is 16 bytes where the caller's
    # class is 12, and the driver rejected the short buffer. Comparing the OpenCL layout
    # against the host's is what decides whether the padded class is needed at all.
    device_bytes:int = _opencl_record_layout(
        [(name, np.dtype(device_type), ()) for name, device_type, _ in members]).itemsize
    if device_bytes != ctypes.sizeof(source_type):
        changed = True

    if not changed:
        return value, None

    device_type_new:type = _DEV_TYPE_CACHE.get((source_type, backend))
    if device_type_new is None:
        device_type_new = type(f"{source_type.__name__}_dev",
                               (ctypes.Structure,),
                               {"_fields_": _opencl_field_list(members)})
        _DEV_TYPE_CACHE[(source_type, backend)] = device_type_new
    packed:object = device_type_new()
    for field_name, device_field_type, host_field_type in members:
        name = _is_vector_class(host_field_type)
        if name is not None:
            setattr(packed, field_name, _pack_vector(getattr(value, field_name), backend))
        elif issubclass(host_field_type, ctypes.Array) and _is_vector_class(host_field_type._type_):
            host_array:object = getattr(value, field_name)
            target:object = getattr(packed, field_name)
            for index in range(host_field_type._length_):
                target[index] = _pack_vector(host_array[index], backend)
        elif issubclass(host_field_type, ctypes.Array) \
                and issubclass(host_field_type._type_, ctypes.Structure):
            # An array of structures with vector members: every element is packed into its
            # own device class, the same way a single nested structure is.
            host_array = getattr(value, field_name)
            target = getattr(packed, field_name)
            for index in range(host_field_type._length_):
                element_packed, _ = _pack_structure(host_array[index], backend)
                target[index] = element_packed
        elif issubclass(host_field_type, ctypes.Structure):
            # One nested structure. Its device class may differ from the host's, in which
            # case the field above was declared with that class -- so the host instance
            # cannot be assigned to it and has to be packed first, the way an element of an
            # array of structures is. A nested structure that needed no change comes back as
            # itself either way.
            nested_packed, _nested_write_back = _pack_structure(
                getattr(value, field_name), backend)
            setattr(packed, field_name, nested_packed)
        else:
            setattr(packed, field_name, getattr(value, field_name))

    def unpack(source:object)->None:
        for field_name, _device_field, host_field_type in members:
            name = _is_vector_class(host_field_type)
            if name is not None:
                _unpack_vector(getattr(source, field_name), getattr(value, field_name))
            elif issubclass(host_field_type, ctypes.Array) and _is_vector_class(host_field_type._type_):
                for index in range(host_field_type._length_):
                    _unpack_vector(getattr(source, field_name)[index],
                                   getattr(value, field_name)[index])
            elif issubclass(host_field_type, ctypes.Array) \
                    and issubclass(host_field_type._type_, ctypes.Structure):
                # Copy every member back through the nested structure's own write-back: its
                # vector members are padded on the device, so `source.x` is a device `float3`
                # and assigning it straight onto the caller's field answered "incompatible
                # types, float3 instance instead of point3f instance".
                source_array:object = getattr(source, field_name)
                host_array = getattr(value, field_name)
                for index in range(host_field_type._length_):
                    _packed_element, element_unpack = _pack_structure(
                        host_array[index], backend)
                    if element_unpack is not None:
                        element_unpack(source_array[index])

    return packed, unpack
