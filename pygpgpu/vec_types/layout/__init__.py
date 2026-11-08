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
what keeps the common case free."""
# This module is the facade of a split: the class registry, the record/vector
# geometry and the packing move to the module named after their layer, and every
# name stays importable from here (the kernel package and two test modules import
# `pack`, `_host_classes` and `_vector_dtype_name` by name).
from __future__ import annotations


from ._layout_classes import (
    _HOST_CLASSES,
    _device_class,
    _host_classes,
)
from ._layout_geometry import (
    Packed,
    RecordField,
    _alignment_of,
    _component_size,
    _host_components,
    _is_vector_class,
    _needs_padding,
    _opencl_record_layout,
    _record_device_dtype,
    _vec_device_components,
    _vector_dtype_name,
    vector_name_of,
)
from ._layout_pack import (
    _DEV_TYPE_CACHE,
    _copy_record,
    _ctypes_align,
    _opencl_field_list,
    _pack_array,
    _pack_record_array,
    _pack_structure,
    _pack_vector,
    _unpack_vector,
    pack,
)
