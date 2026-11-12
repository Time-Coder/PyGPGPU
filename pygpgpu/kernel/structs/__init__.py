"""Device structures built from numpy record dtypes.

A structured dtype describes a layout completely, so it is also the definition the device
needs: `np.dtype([("i", "<i4"), ("f", "<f4")])` is `struct { int i; float f; };`. Nothing on
the host has to be a `ctypes` class, and an array of that dtype is simply an array of the
structure.

A dtype whose fields all sit at offset zero is a **union**: two members share the storage, and
no C `struct` can describe that -- emitting one would place the second member after the first
and make the type twice as wide. Such a dtype is therefore declared as `union { ... }` on the
device, whichever way the host built it: `ctypes.Union` (and `ctypes.LittleEndianUnion`) yields
exactly that record shape, and so does a `np.dtype` written by hand with explicit offsets.

The generated names are derived from the dtype text, so the same dtype always yields the same
C type name -- the kernel cache is keyed by the generated source, and a name that changed
between runs would defeat it."""
# This module is the facade of a split: every name it used to define is re-exported
# from the part module that now owns it, so `structs.<name>` and
# `from pygpgpu.kernel.structs import <name>` keep working unchanged.
#
# The registries live in `state.py` and are imported as objects everywhere: a second
# copy of one would re-initialise it, and `_SETTLED_NAMES` pins the C type names the
# generated source cache is keyed on. The import direction is registry/unions ->
# layout -> state, so the two calls layout makes back into the registry are
# function-local, with the reason written beside them.
from __future__ import annotations


from .state import (
    _DECLARED,
    _DECLARED_ALL,
    _DECLARED_FILES,
    _NAME_BASES,
    _REGISTRY,
    _SETTLED_NAMES,
)
from .registry import (
    _class_source_file,
    _safe_identifier,
    declared_class,
    declared_class_carrying,
    declared_classes,
    declared_file,
    is_declared_structure,
    register,
    register_class,
    register_declared_class,
    register_name_hint,
    reregister_class,
)
from .layout import (
    _backend_type_name,
    _is_vector_alias,
    _scalar_type,
    device_member_extent,
    dtype_for,
    dtype_of,
    field_type,
    fixed_member_class,
    is_record,
    member_types,
    nested_dtypes,
    padded_member_fields,
    render_structures,
    struct_name,
    structure_dtype_of,
    structure_fields,
)
from .unions import (
    UnionProxy,
    _add_subscript_access,
    is_ctypes_record,
    is_union_dtype,
    union_needs_proxy,
)
