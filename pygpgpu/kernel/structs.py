"""Device structures built from numpy record dtypes.

A structured dtype describes a layout completely, so it is also the definition the device
needs: `np.dtype([("i", "<i4"), ("f", "<f4")])` is `struct { int i; float f; };`. Nothing on
the host has to be a `ctypes` class, and an array of that dtype is simply an array of the
structure.

The generated names are derived from the dtype text, so the same dtype always yields the same
C type name -- the kernel cache is keyed by the generated source, and a name that changed
between runs would defeat it.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Dtypes that a `ctypes.Structure` class declared. A dtype cannot say whether it means a
# vector or a structure -- `{float x; float y;}` and `float2` compare and hash equal -- so the
# only usable evidence is the dtype *object*: numpy marks a record built from a ctypes class
# as aligned (`isalignedstruct`), which the vector dtypes this package ships are not.
# Every class that declared this layout, in registration order. Two classes can share one
# dtype, and a member call has to find the one that actually carries the method.
_DECLARED: Dict[np.dtype, type] = {}
_DECLARED_ALL: Dict[np.dtype, List[type]] = {}


def is_record(dtype:Any)->bool:
    """True for a structured dtype that is a *structure*, not a vector.

    A dtype whose layout coincides with a vector type is a structure when a
    `ctypes.Structure` class declared it -- the layout alone cannot tell the two apart.
    """
    from ..vec_types.layout import _vector_dtype_name

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
    from .quaternion import quat_type_name  # noqa: PLC0415 (import cycle)

    dtype:np.dtype = np.dtype(dtype)
    fixed:Optional[str] = quat_type_name(dtype)
    if fixed is not None:
        return fixed
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
    from .translator import DTYPE_MAP

    return DTYPE_MAP.get(dtype.name)


def field_type(field_dtype:Any, backend:str)->Tuple[str, str]:
    """(C base type, array suffix) for one field: scalar, vector, nested structure, or array."""
    from ..vec_types.layout import _vector_dtype_name

    dtype:np.dtype = np.dtype(field_dtype)
    shape:Any = None
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
            base = scalar

    # numpy records a sub-array member as (dtype, shape); C puts the extents on the name.
    suffix:str = ""
    if shape is not None:
        suffix = "".join(f"[{int(extent)}]" for extent in shape)
    return base, suffix


def structure_fields(dtype:np.dtype, backend:str)->List[str]:
    """One C member declaration per field, in declaration order."""
    members:List[str] = []
    for name in dtype.names or ():
        field_dtype:Any = dtype.fields[name][0]
        base, suffix = field_type(field_dtype, backend)
        members.append(f"{base} {name}{suffix}")
    return members


def nested_dtypes(dtype:np.dtype)->List[np.dtype]:
    """Every structure dtype this one contains, deepest first (a typedef needs its members)."""
    from ..vec_types.layout import _vector_dtype_name

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


def _class_source_file(cls:type) -> str:
    """The file a class was written in, from any function it defines."""
    import sys

    module:Any = sys.modules.get(getattr(cls, "__module__", ""))
    module_file:str = getattr(module, "__file__", "") or ""
    if module_file:
        return module_file
    # A class a test module created on the fly can name a module without a file; its own
    # method still carries the source location.
    for value in getattr(cls, "__dict__", {}).values():
        code:Any = getattr(getattr(value, "__func__", value), "__code__", None)
        if code is not None:
            return getattr(code, "co_filename", "") or ""
    return ""


def declared_file(dtype:Any) -> str:
    """The python file the class behind this structure was declared in, or "".

    A structure belongs where its python class lives: that module's generated file holds
    the declaration, and every other file that names the type includes it. A structure
    synthesized from a plain python object takes the file of the class the caller wrote --
    the object's own class -- because the generated class behind the layout lives in this
    package and would send the declaration here instead of to the caller.
    """
    recorded:Optional[str] = _DECLARED_FILES.get(np.dtype(dtype))
    if recorded:
        return recorded
    cls:Optional[type] = declared_class(dtype)
    if cls is None:
        return ""
    return _class_source_file(cls)


def render_structures(dtypes:List[np.dtype], backend:str)->str:
    """The typedef block for every structure the kernel uses, members before containers."""
    from .quaternion import quat_type_name  # noqa: PLC0415 (import cycle)

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
        # A quaternion is declared by the fixed quaternion header, which a source using one
        # always includes; emitting it here too would define the type twice, which C++ rejects.
        if quat_type_name(dtype) is not None:
            continue
        if lines:
            # One blank line between structure definitions.
            lines.append("")
        members:List[str] = structure_fields(dtype, backend)
        # Every structure is guarded by its own name. A generated file can be reached from
        # several others (`#include`), and two modules can declare one layout -- therefore
        # one C type -- so the guard is what makes that declaration appear once however
        # many files reach it.
        guard:str = f"__{name}__"
        lines.append(f"#ifndef {guard}")
        lines.append(f"#define {guard}")
        if backend == "opencl":
            lines.append("typedef struct")
            lines.append("{")
            body:str = "\n".join(f"    {member};" for member in members)
            lines.append(body)
            lines.append(f"}} {name};")
        else:
            lines.append(f"struct {name}")
            lines.append("{")
            lines.append("\n".join(f"    {member};" for member in members))
            lines.append("};")
            lines.append(f"typedef struct {name} {name};")
        lines.append("#endif")
    return "\n".join(lines)


_REGISTRY: Dict[str, np.dtype] = {}
# dtype -> the python class name its structure is spelled after. The C name is
# `<base>_<hash>`: readable, and still distinct when two modules declare one class name.
_NAME_BASES: Dict[np.dtype, str] = {}
# dtype -> the C name it was first given. Pinned, because a later class registration must
# not rename a layout an existing signature already spells.
_SETTLED_NAMES: Dict[np.dtype, str] = {}
# dtype -> the python file the class behind it was written in. Kept separately from
# `_DECLARED`, because a structure synthesized from a plain object has a *generated* class
# (built by this package) while its members were inferred from the caller's source: the
# declaration belongs to that caller's module, not to the file that built the class.
_DECLARED_FILES: Dict[np.dtype, str] = {}


def register_name_hint(dtype:Any, cls:type) -> None:
    """Spell this dtype's C structure after the python class the value came from.

    A structure describes a layout a hash can name, but the hash says nothing a reader
    recognizes -- the class the value came from does. The C name is `<class>_<hash>`: the
    readable half is the class, and the hash keeps two classes of one name apart (two
    modules can both declare `Ray`), which a bare class name cannot. A structure with no
    class behind it keeps the opaque `_st_<hash>`.
    """
    name:str = getattr(cls, "__name__", "")
    if not name:
        return
    dtype:np.dtype = np.dtype(dtype)
    _NAME_BASES.setdefault(dtype, _safe_identifier(name))
    # The class the value came from is the one that says where the declaration belongs, and
    # for a plain python object it is the only such evidence: the structure itself is built
    # here, so its own class names this package's file.
    source_file:str = _class_source_file(cls)
    if source_file:
        _DECLARED_FILES.setdefault(dtype, source_file)


def _safe_identifier(name:str) -> str:
    """A C identifier from a python class name, with leading digits prefixed."""
    cleaned:str = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    return f"_{cleaned}" if cleaned and cleaned[0].isdigit() else (cleaned or "_anon")


def register(dtype:Any) -> str:
    """Record a structure dtype and return its C type name."""
    dtype:np.dtype = np.dtype(dtype)
    name:str = struct_name(dtype)
    _REGISTRY[name] = dtype
    return name


def register_class(cls:Any) -> Optional[np.dtype]:
    """Record the dtype a `ctypes.Structure` class declares, and return it.

    The class is kept as evidence: a dtype whose layout matches a vector type is still a
    structure when the caller wrote a class for it, and member access then has to be the
    structure's string subscript rather than a vector's component name.

    The class also gains numpy's member spelling (`instance["i"]`) if it does not already
    accept one. A kernel body writes `p["i"]` -- the spelling that works on every backend --
    so the interpreter backend, which runs the body unchanged, must see the same thing: a
    `ctypes.Structure` instance answers `["i"]` exactly as a record element does.
    """
    import ctypes

    from .translator import _vec_registry

    if not (isinstance(cls, type) and issubclass(cls, ctypes.Structure)):
        return None
    # A vector type is a `ctypes.Structure` too, and it already answers a positional
    # subscript (`v[0]`); installing a member-name subscript on it would replace that.
    if cls in _vec_registry().values() or _is_vector_alias(cls):
        return None
    try:
        dtype:np.dtype = np.dtype(cls)
    except TypeError:
        return None
    if dtype.fields is None:
        return None
    _DECLARED.setdefault(dtype, cls)
    _DECLARED_ALL.setdefault(dtype, [])
    if cls not in _DECLARED_ALL[dtype]:
        _DECLARED_ALL[dtype].append(cls)
    # The class the caller wrote is the readable half of the structure's C name: the
    # declaration it generates reads `struct Ray_<hash>`, not an opaque `_st_<hash>`.
    class_name:str = getattr(cls, "__name__", "")
    if class_name:
        _NAME_BASES.setdefault(dtype, _safe_identifier(class_name))
    register(dtype)
    _add_subscript_access(cls)
    return dtype


def declared_class(dtype:Any) -> Optional[type]:
    """The `ctypes.Structure` class a registered structure dtype came from, or None.

    A caller's methods live on that class, and a dtype alone cannot name it -- two classes
    with the same layout share one dtype.
    """
    return _DECLARED.get(np.dtype(dtype)) if isinstance(dtype, (np.dtype, type)) else None


def declared_classes(dtype:Any) -> List[type]:
    """Every class registered for this layout, in registration order."""
    if not isinstance(dtype, (np.dtype, type)):
        return []
    return list(_DECLARED_ALL.get(np.dtype(dtype), ()))


def declared_class_carrying(dtype:Any, attribute:str) -> Optional[type]:
    """The registered class for this layout that defines `attribute`, or None.

    Two classes can share one dtype (two `Ray`s with the same two members, defined in two
    modules), and only the one that defines the method being called can answer it. Without
    this the call reported the other class as having no such method.
    """
    for cls in declared_classes(dtype):
        if hasattr(cls, attribute):
            return cls
    return None


def register_declared_class(dtype:np.dtype, cls:type) -> None:
    """Record the class a synthesized structure came from.

    A plain python class a kernel body builds has no `np.dtype(cls)` -- it is not a
    `ctypes.Structure` -- so its layout comes from the members the body reads, and the class
    is paired with that dtype here. The pairing is what lets a member method resolve:
    `ray.at(t)` needs the class the method is defined on, and a bare record dtype cannot say
    whose methods to compile.
    """
    _DECLARED.setdefault(np.dtype(dtype), cls)
    register(dtype)


def reregister_class(cls:Any) -> Optional[np.dtype]:
    """Register `cls` for its dtype, replacing any class recorded earlier.

    A synthesized structure is built per member set, and two member sets can produce
    the same dtype with different methods. The later, richer class is the one a kernel
    that calls a method needs to resolve against, so it replaces the earlier one
    instead of losing to `setdefault`.
    """
    dtype:Optional[np.dtype] = register_class(cls)
    if dtype is not None:
        _DECLARED[np.dtype(dtype)] = cls
    return dtype


def _add_subscript_access(cls:Any) -> None:
    """Let a structure instance read a member by name (`s["i"]`), like a record element.

    Only installed when the class has no subscript of its own, so a user's `__getitem__`
    -- or the one a pygf vector/quaternion inherits from its base -- is never shadowed.
    The check walks the whole MRO: `quatf` defines no `__getitem__` itself, but `genType`
    does, and installing ours on the subclass replaced positional indexing with a
    member-name one, so `conjugate(q)` failed inside pygf.
    """
    if any("__getitem__" in base.__dict__ for base in getattr(cls, "__mro__", ())):
        return

    def __getitem__(self:Any, key:Any)->Any:
        if isinstance(key, str):
            return getattr(self, key)
        raise TypeError(
            f"{type(self).__name__} is indexed by member name, not by position: "
            f"use s['{list(np.dtype(cls).names or ())[0]}']")

    # Marked, because this subscript is the translation's own member spelling (`s["i"]`),
    # not a method the caller wrote: a device translation must not turn `s[i]` into a call
    # to it the way a caller's own `__getitem__` becomes one.
    __getitem__.__pygpgpu_member_subscript__ = True
    cls.__getitem__ = __getitem__


def is_declared_structure(dtype:Any) -> bool:
    """True when a `ctypes.Structure` class declared this dtype's layout.

    Alignment is the evidence, not just the registry: `np.dtype(cls)` over a
    `ctypes.Structure` always produces an *aligned* record (C layout), while the vector
    dtypes this package ships are built with `np.dtype(_fields_)` and are not aligned. So a
    record whose layout coincides with a vector -- `{float x; float y;}` against `float2` --
    is a structure when it is aligned, and a vector when it is not.

    Note the argument has to be the dtype *object*, not a value compared against one: an
    aligned record and the vector's plain record compare equal and hash equal, so only the
    ``isalignedstruct`` flag on the dtype you were handed tells them apart.
    """
    if not isinstance(dtype, np.dtype) or dtype.fields is None:
        return False
    return bool(dtype.isalignedstruct)


def structure_dtype_of(value:Any) -> Optional[np.dtype]:
    """The structure dtype a value denotes, or None, respecting how it was built.

    A vector type is a `ctypes.Structure` as well, and its dtype carries the same alignment,
    so a value that is one of the package's vectors has to be excluded first -- otherwise
    every vector argument would be read as a structure.

    An ndarray keeps the dtype object it was created with, so an aligned record array stays
    distinguishable from a vector array whose layout happens to match.
    """
    import ctypes

    from .translator import _vec_registry

    if isinstance(value, np.ndarray):
        dtype:np.dtype = value.dtype
        return dtype if is_declared_structure(dtype) else None

    cls:Any = type(value) if isinstance(value, ctypes.Structure) else value
    if not (isinstance(cls, type) and issubclass(cls, ctypes.Structure)):
        return None
    if cls in _vec_registry().values() or _is_vector_alias(cls):
        return None
    return register_class(cls)


def _is_vector_alias(cls:Any) -> bool:
    """True when a class is a pygf vector or one of its aliases (`point3f`, `color3d`)."""
    from ..vec_types.layout import _host_classes

    try:
        return any(cls is host or issubclass(cls, host)
                   for host in _host_classes().values())
    except (ImportError, TypeError):
        return False


def dtype_for(name:str) -> Optional[np.dtype]:
    """The structure dtype behind a registered C type name, or None."""
    return _REGISTRY.get(name)


def member_types(dtype:np.dtype, backend:str) -> Dict[str, Any]:
    """field name -> the translator's type for reading it.

    A scalar member is its scalar type name, a vector member its vector name, a nested
    structure its structure name, and a sub-array member a pointer marker so `s["xs"][1]`
    indexes it directly.
    """
    from ..vec_types.layout import _vector_dtype_name

    members:Dict[str, Any] = {}
    for name in dtype.names or ():
        field_dtype:np.dtype = np.dtype(dtype.fields[name][0])
        shape:Any = None
        if field_dtype.subdtype is not None:
            base_dtype, shape = field_dtype.subdtype
            field_dtype = base_dtype
        vector:Optional[str] = _vector_dtype_name(field_dtype)
        if vector is not None:
            kind:Any = vector
        elif is_record(field_dtype):
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


def dtype_of(value:Any)->Optional[np.dtype]:
    """The record dtype a value carries, when it is a structure or an array of them."""
    if isinstance(value, np.void):
        return value.dtype if is_record(value.dtype) else None
    if isinstance(value, np.ndarray) and is_record(value.dtype):
        return value.dtype
    return None
