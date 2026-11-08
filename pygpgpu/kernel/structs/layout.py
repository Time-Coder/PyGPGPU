# Turning a dtype into the C the device compiles, member by member.
from __future__ import annotations
from .state import _NAME_BASES, _REGISTRY, _SETTLED_NAMES
import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import DTypeLike



def is_record(dtype:DTypeLike)->bool:
    """True for a structured dtype that is a *structure* or a *union*, not a vector.

    A dtype whose layout coincides with a vector type is a structure when a
    `ctypes.Structure` class declared it -- the layout alone cannot tell the two apart.
    """
    from ...vec_types.layout import _vector_dtype_name

    # Upward calls into registry and unions: both import this module, so a top-level
    # import would make the package a cycle.
    from .registry import is_declared_structure

    if not isinstance(dtype, np.dtype) or dtype.fields is None:
        return False
    if is_declared_structure(dtype):
        return True
    if _vector_dtype_name(dtype) is not None:
        return False
    return True



def struct_name(dtype:np.dtype)->str:
    """A stable C identifier for the structure a dtype describes.

    A quaternion is the one exception, and it is a deliberate one: its declaration lives in
    the fixed quaternion header, so the name has to be one the header can spell -- a content
    hash cannot be written in advance. Two quaternion layouts exist, so two fixed names cover
    every case, and `render_structures` then leaves them out of its typedef block.
    """
    from ..complex import record_complex_type_name  # noqa: PLC0415 (import cycle)
    from ..quaternion import quat_type_name  # noqa: PLC0415 (import cycle)

    dtype:np.dtype = np.dtype(dtype)
    fixed:Optional[str] = quat_type_name(dtype)
    if fixed is not None:
        return fixed
    # A complex member has the same arrangement: `c64`/`c128` are declared by the complex
    # header, so the member's declaration names that type rather than a second structure
    # with the same bytes -- which is what keeps `z.real` a complex access rather than a
    # field of an unrelated struct.
    complex_fixed:Optional[str] = record_complex_type_name(dtype)
    if complex_fixed is not None:
        return complex_fixed
    # The name is fixed the first time it is asked for. A class can register after the
    # dtype has already been named (a value is converted before the kernel that uses it is
    # translated, and another module may register its own class later), and a name that
    # changes afterwards would change the type signature of every kernel already compiled
    # against the old spelling -- the same function then appears twice in one generated file.
    settled:Optional[str] = _SETTLED_NAMES.get(dtype)
    if settled is not None:
        return settled
    digest:str = hashlib.md5(repr(dtype).encode("utf-8")).hexdigest()[:10]
    base:Optional[str] = _NAME_BASES.get(dtype)
    name:str = f"{base}_{digest}" if base is not None else f"_st_{digest}"
    _SETTLED_NAMES[dtype] = name
    return name



def _scalar_type(dtype:np.dtype)->Optional[str]:
    from ..translator import DTYPE_MAP

    return DTYPE_MAP.get(dtype.name)



def _backend_type_name(scalar:str, backend:str)->str:
    """The C spelling a structure member of this device type must use on this backend.

    A structure member is declared in the generated source, so it needs the *C* spelling of
    its device type rather than the internal name. `long` is the one that bites: it is 32-bit
    in CUDA C on Windows (where NVRTC and HIP are used here), so a member spelled `long` was
    four bytes while the host packed the dtype's eight -- reading back `0x1122334455` gave
    `0x22334455`, the top byte silently gone. `translator._CUDA_TYPE_NAMES` already carries
    that rewrite for parameters and locals; a structure field has to use the same table, or
    the host's layout and the device's disagree.

    `opencl` keeps the internal names (`uchar`, `uint`, `long`, `ulong`), which OpenCL C
    defines as 8/32/64/64-bit.
    """
    from ..translator import _CUDA_TYPE_NAMES

    if backend in ("cuda", "hip"):
        return _CUDA_TYPE_NAMES.get(scalar, scalar)
    return scalar



def field_type(field_dtype:DTypeLike, backend:str)->Tuple[str, str]:
    """(C base type, array suffix) for one field: scalar, vector, nested structure, or array."""
    from ...vec_types.layout import _vector_dtype_name

    dtype:np.dtype = np.dtype(field_dtype)
    shape:Optional[Tuple[int, ...]] = None
    if dtype.subdtype is not None:
        base_dtype, shape = dtype.subdtype
        dtype = base_dtype

    vector:Optional[str] = _vector_dtype_name(dtype)
    if vector is not None:
        base:str = vector
    elif is_record(dtype):
        base = struct_name(dtype)
    else:
        scalar:Optional[str] = _scalar_type(dtype)
        if scalar is None:
            raise SyntaxError(
                f"a structure member of dtype {dtype} has no device type; use a scalar, a "
                f"vector, a nested structure, or an array of those")
        # A bool member is a byte. OpenCL spells the unsigned byte `uchar`; CUDA C does not
        # define that alias, so it takes the full spelling there.
        if dtype.name == "bool":
            base = "uchar" if backend == "opencl" else "unsigned char"
        else:
            base = _backend_type_name(scalar, backend)

    # numpy records a sub-array member as (dtype, shape); C puts the extents on the name.
    suffix:str = ""
    if shape is not None:
        suffix = "".join(f"[{int(extent)}]" for extent in shape)
    return base, suffix



def structure_fields(dtype:np.dtype, backend:str)->List[str]:
    """One C member declaration per field, in declaration order."""
    members:List[str] = []
    for name in dtype.names or ():
        field_dtype:object = dtype.fields[name][0]
        base, suffix = field_type(field_dtype, backend)
        members.append(f"{base} {name}{suffix}")
    return members



def nested_dtypes(dtype:np.dtype)->List[np.dtype]:
    """Every structure dtype this one contains, deepest first (a typedef needs its members)."""
    from ...vec_types.layout import _vector_dtype_name

    found:List[np.dtype] = []
    for name in dtype.names or ():
        base:np.dtype = np.dtype(dtype.fields[name][0])
        if base.subdtype is not None:
            base = base.subdtype[0]
        if _vector_dtype_name(base) is not None or not is_record(base):
            continue
        for inner in nested_dtypes(base):
            if inner not in found:
                found.append(inner)
        if base not in found:
            found.append(base)
    return found



def render_structures(dtypes:List[np.dtype], backend:str)->str:
    """The typedef block for every structure the kernel uses, members before containers."""
    from ..complex import record_complex_type_name  # noqa: PLC0415 (import cycle)
    from ..quaternion import quat_type_name  # noqa: PLC0415 (import cycle)

    ordered:List[np.dtype] = []
    for dtype in dtypes:
        for inner in nested_dtypes(dtype):
            if inner not in ordered:
                ordered.append(inner)
        if dtype not in ordered:
            ordered.append(dtype)

    lines:List[str] = []
    for dtype in ordered:
        name:str = struct_name(dtype)
        # A quaternion or a complex value is declared by its fixed header, which a source
        # using one always includes; emitting it here too would define the type twice, which
        # C++ rejects.
        if quat_type_name(dtype) is not None:
            continue
        if record_complex_type_name(dtype) is not None:
            continue
        if lines:
            # One blank line between structure definitions.
            lines.append("")
        members:List[str] = structure_fields(dtype, backend)
        # A layout whose members all share one address is a union, and the declaration has to
        # say so: `struct` would put the second member after the first and hand the kernel a
        # value of the wrong width, so the rank a member sits at is part of the declaration.
        # Upward call into the union module, which imports this one.
        from .unions import is_union_dtype
        kind:str = "union" if is_union_dtype(dtype) else "struct"
        # Every structure is guarded by its own name. A generated file can be reached from
        # several others (`#include`), and two modules can declare one layout -- therefore
        # one C type -- so the guard is what makes that declaration appear once however
        # many files reach it.
        guard:str = f"__{name}__"
        lines.append(f"#ifndef {guard}")
        lines.append(f"#define {guard}")
        if backend == "opencl":
            lines.append(f"typedef {kind}")
            lines.append("{")
            body:str = "\n".join(f"    {member};" for member in members)
            lines.append(body)
            lines.append(f"}} {name};")
        else:
            lines.append(f"{kind} {name}")
            lines.append("{")
            lines.append("\n".join(f"    {member};" for member in members))
            lines.append("};")
            lines.append(f"typedef {kind} {name} {name};")
        lines.append("#endif")
    return "\n".join(lines)



def structure_dtype_of(value:object) -> Optional[np.dtype]:
    """The structure dtype a value denotes, or None, respecting how it was built.

    A vector type is a `ctypes.Structure` as well, and its dtype carries the same alignment,
    so a value that is one of the package's vectors has to be excluded first -- otherwise
    every vector argument would be read as a structure.

    An ndarray keeps the dtype object it was created with, so an aligned record array stays
    distinguishable from a vector array whose layout happens to match.
    """
    from ..translator import _vec_registry

    # Upward calls into registry and unions, for the reason recorded in `is_record`.
    from .registry import is_declared_structure, register_class
    from .unions import is_ctypes_record

    if isinstance(value, np.ndarray):
        dtype:np.dtype = value.dtype
        return dtype if is_declared_structure(dtype) else None

    if not is_ctypes_record(value):
        return None
    cls:type = type(value) if not isinstance(value, type) else value
    if cls in _vec_registry().values() or _is_vector_alias(cls):
        return None
    return register_class(cls)



def _is_vector_alias(cls:type) -> bool:
    """True when a class is a pygf vector or one of its aliases (`point3f`, `color3d`)."""
    from ...vec_types.layout import _host_classes

    try:
        return any(cls is host or issubclass(cls, host)
                   for host in _host_classes().values())
    except (ImportError, TypeError):
        return False



def dtype_for(name:str) -> Optional[np.dtype]:
    """The structure dtype behind a registered C type name, or None."""
    return _REGISTRY.get(name)



def member_types(dtype:np.dtype, backend:str) -> Dict[str, object]:
    """field name -> the translator's type for reading it.

    A scalar member is its scalar type name, a vector member its vector name, a nested
    structure its structure name, and a sub-array member a pointer marker so `s["xs"][1]`
    indexes it directly.
    """
    from ...vec_types.layout import _vector_dtype_name
    from ..complex import record_complex_type_name  # noqa: PLC0415 (import cycle)

    members:Dict[str, object] = {}
    for name in dtype.names or ():
        field_dtype:np.dtype = np.dtype(dtype.fields[name][0])
        shape:Optional[Tuple[int, ...]] = None
        if field_dtype.subdtype is not None:
            base_dtype, shape = field_dtype.subdtype
            field_dtype = base_dtype
        vector:Optional[str] = _vector_dtype_name(field_dtype)
        complex_name:Optional[str] = record_complex_type_name(field_dtype)
        if complex_name is not None:
            # A complex member reads as a complex *value*, so `.real`, `.imag`, arithmetic
            # and the transcendental functions all apply to it; treating it as a nested
            # structure would hand the body a type with two fields and no complex meaning.
            kind:object = complex_name
        elif vector is not None:
            kind = vector
        elif is_record(field_dtype):
            # Upward call into the registry, which imports this one.
            from .registry import register
            kind = register(field_dtype)
        else:
            scalar:Optional[str] = _scalar_type(field_dtype)
            kind = "uchar" if field_dtype.name == "bool" else scalar
        if shape is not None:
            # An array member is addressed with a plain index, so it carries the pointer
            # marker; the element type is the member's own.
            members[name] = ("*", kind)
        else:
            members[name] = kind
    return members



def fixed_member_class(name:object)->Optional[type]:
    """The `ctypes` class a fixed header type name stands for, or None.

    Two families of member carry a fixed C name instead of a content hash, because a header
    declares them: a quaternion (`quatf` / `quatd`) and a complex value (`c64` / `c128`). A
    backend that reads a generated structure back needs the class for the fields it finds,
    and this is the one place that maps the name to it -- the three parsers are otherwise
    copies of one another.
    """
    from ..complex import member_class_for_name  # noqa: PLC0415 (import cycle)
    from ..quaternion import fixed_type_class  # noqa: PLC0415 (import cycle)

    return member_class_for_name(name) or fixed_type_class(name)



def dtype_of(value:object)->Optional[np.dtype]:
    """The record dtype a value carries, when it is a structure or an array of them."""
    if isinstance(value, np.void):
        return value.dtype if is_record(value.dtype) else None
    if isinstance(value, np.ndarray) and is_record(value.dtype):
        return value.dtype
    return None
