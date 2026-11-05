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
between runs would defeat it.
"""
from __future__ import annotations

import hashlib
from types import CodeType, ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import DTypeLike

# Dtypes that a `ctypes.Structure` class declared. A dtype cannot say whether it means a
# vector or a structure -- `{float x; float y;}` and `float2` compare and hash equal -- so the
# only usable evidence is the dtype *object*: numpy marks a record built from a ctypes class
# as aligned (`isalignedstruct`), which the vector dtypes this package ships are not.
# Every class that declared this layout, in registration order. Two classes can share one
# dtype, and a member call has to find the one that actually carries the method.
_DECLARED: Dict[np.dtype, type] = {}
_DECLARED_ALL: Dict[np.dtype, List[type]] = {}


def is_record(dtype:DTypeLike)->bool:
    """True for a structured dtype that is a *structure* or a *union*, not a vector.

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


def is_union_dtype(dtype:DTypeLike)->bool:
    """True when a record dtype describes a **union** rather than a structure.

    A C union gives every member the same address, and numpy records exactly that as a list
    of offsets that are all zero. Nothing else produces that shape -- a C structure's members
    follow one another, and a packed record's offsets still ascend -- so the offsets are the
    whole answer, and it is the flat one: the layout comes from the bytes, not from the way
    a caller happened to build the dtype. Two dtypes with one offset list therefore agree on
    this, which is what lets a `ctypes.Union` and a hand-written `np.dtype` with zero offsets
    share one device type.
    """
    if not isinstance(dtype, np.dtype) or dtype.fields is None:
        return False
    offsets:List[int] = [int(dtype.fields[name][1]) for name in dtype.names or ()]
    return bool(offsets) and all(offset == 0 for offset in offsets)


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
    from .translator import _CUDA_TYPE_NAMES

    if backend in ("cuda", "hip"):
        return _CUDA_TYPE_NAMES.get(scalar, scalar)
    return scalar


def field_type(field_dtype:DTypeLike, backend:str)->Tuple[str, str]:
    """(C base type, array suffix) for one field: scalar, vector, nested structure, or array."""
    from ..vec_types.layout import _vector_dtype_name

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

    module:Optional[ModuleType] = sys.modules.get(getattr(cls, "__module__", ""))
    module_file:str = getattr(module, "__file__", "") or ""
    if module_file:
        return module_file
    # A class a test module created on the fly can name a module without a file; its own
    # method still carries the source location.
    for value in getattr(cls, "__dict__", {}).values():
        code:Optional[CodeType] = getattr(getattr(value, "__func__", value), "__code__", None)
        if code is not None:
            return getattr(code, "co_filename", "") or ""
    return ""


def declared_file(dtype:DTypeLike) -> str:
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
        # A layout whose members all share one address is a union, and the declaration has to
        # say so: `struct` would put the second member after the first and hand the kernel a
        # value of the wrong width, so the rank a member sits at is part of the declaration.
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


def register_name_hint(dtype:DTypeLike, cls:type) -> None:
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


def register(dtype:DTypeLike) -> str:
    """Record a structure dtype and return its C type name."""
    dtype:np.dtype = np.dtype(dtype)
    name:str = struct_name(dtype)
    _REGISTRY[name] = dtype
    return name


def is_ctypes_record(value:object)->bool:
    """True when a value is a `ctypes` structure **or union** subclass/instance.

    A caller can declare a layout in four spellings: `ctypes.Structure`,
    `ctypes.LittleEndianStructure`, `ctypes.Union` and `ctypes.LittleEndianUnion`. The
    little-endian pair are subclasses of the native pair, so `ctypes.Structure` covers them;
    a union is a *sibling* type in `ctypes`, not a subclass, and testing only for
    `ctypes.Structure` made every union fall through to the plain-object path, where the
    members are guessed from what the body reads and a `p["i"]` spelling has nothing to
    answer it.

    A class and an instance are both accepted: `infer_ctype` sees whichever the caller
    passed, and both are `isinstance` of the abstract base of their own kind.
    """
    import ctypes

    kind:type = value if isinstance(value, type) else type(value)
    return isinstance(kind, type) and issubclass(kind, (ctypes.Structure, ctypes.Union))


def register_class(cls:type) -> Optional[np.dtype]:
    """Record the dtype a `ctypes` structure or union class declares, and return it.

    The class is kept as evidence: a dtype whose layout matches a vector type is still a
    structure when the caller wrote a class for it, and member access then has to be the
    structure's string subscript rather than a vector's component name.

    A union is registered through this same door -- `np.dtype(ctypes_union)` yields the
    record whose member offsets are all zero, which is exactly what `is_union_dtype`
    recognizes, so no separate registry is needed.

    The class also gains numpy's member spelling (`instance["i"]`) if it does not already
    accept one. A kernel body writes `p["i"]` -- the spelling that works on every backend --
    so the interpreter backend, which runs the body unchanged, must see the same thing: a
    `ctypes` record instance answers `["i"]` exactly as a record element does.
    """
    import ctypes

    from .translator import _vec_registry

    if not (isinstance(cls, type) and issubclass(cls, (ctypes.Structure, ctypes.Union))):
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


def declared_class(dtype:DTypeLike) -> Optional[type]:
    """The `ctypes.Structure` class a registered structure dtype came from, or None.

    A caller's methods live on that class, and a dtype alone cannot name it -- two classes
    with the same layout share one dtype.
    """
    return _DECLARED.get(np.dtype(dtype)) if isinstance(dtype, (np.dtype, type)) else None


def declared_classes(dtype:DTypeLike) -> List[type]:
    """Every class registered for this layout, in registration order."""
    if not isinstance(dtype, (np.dtype, type)):
        return []
    return list(_DECLARED_ALL.get(np.dtype(dtype), ()))


def declared_class_carrying(dtype:DTypeLike, attribute:str) -> Optional[type]:
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


def reregister_class(cls:type) -> Optional[np.dtype]:
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


def _add_subscript_access(cls:type) -> None:
    """Let a structure instance read a member by name (`s["i"]`), like a record element.

    Only installed when the class has no subscript of its own, so a user's `__getitem__`
    -- or the one a pygf vector/quaternion inherits from its base -- is never shadowed.
    The check walks the whole MRO: `quatf` defines no `__getitem__` itself, but `genType`
    does, and installing ours on the subclass replaced positional indexing with a
    member-name one, so `conjugate(q)` failed inside pygf.

    A `ctypes.Union` subclass is deliberately **not** patched this way. On CPython/Linux
    that type starts with no mapping slot (``tp_as_mapping`` is NULL, unlike
    ``ctypes.Structure``), so a `__getitem__` put on the class dictionary afterwards is
    visible to ``hasattr`` while ``u["i"]`` still raises "'U' object is not
    subscriptable" -- the C-level slot was never filled and cannot be filled from
    Python for a class that already exists (`PyType_Modified` only makes it intermittent,
    and putting the method on `ctypes.Union` itself leaks a global behaviour change into
    every union in the process, including ones this package never saw). The union case
    is therefore handled where it belongs: at the point the value is handed to a body,
    by `UnionProxy` (see `union_needs_proxy`).
    """
    if any("__getitem__" in base.__dict__ for base in getattr(cls, "__mro__", ())):
        return

    def __getitem__(self:object, key:object)->object:
        if isinstance(key, str):
            member:object = getattr(self, key)
            # A structure member can be a union (`s["value"]["i"]`), and on the
            # platforms where a union needs the proxy the bare value would fail the
            # second subscript. See `union_needs_proxy`.
            if union_needs_proxy(member):
                return UnionProxy(member)
            return member
        raise TypeError(
            f"{type(self).__name__} is indexed by member name, not by position: "
            f"use s['{list(np.dtype(type(self)).names or ())[0]}']")

    # Marked, because this subscript is the translation's own member spelling (`s["i"]`),
    # not a method the caller wrote: a device translation must not turn `s[i]` into a call
    # to it the way a caller's own `__getitem__` becomes one.
    __getitem__.__pygpgpu_member_subscript__ = True
    try:
        import ctypes

        if issubclass(cls, ctypes.Union):
            # See the docstring: the class-level subscript cannot be installed on a
            # union that already exists, so it is not attempted.
            return
    except Exception:
        pass
    cls.__getitem__ = __getitem__


def union_needs_proxy(value:object)->bool:
    """Whether a value handed to the interpreter must be wrapped to answer `p["i"]`.

    True for a `ctypes.Union` instance whose class has no working member subscript --
    which is every union on CPython/Linux and none on Windows, where the class-level
    assignment works. The test is on the instance, not on the platform, so a build
    that fixes the slot behaviour needs no special case here.
    """
    try:
        import ctypes
    except Exception:
        return False
    if not isinstance(value, ctypes.Union):
        return False
    try:
        value["__pygpgpu_probe__"]
    except TypeError:
        return True
    except KeyError:
        return False
    return False


class UnionProxy:
    """A `ctypes.Union` instance plus the member subscript, for the interpreter backend.

    Every read and write is forwarded to the wrapped union, so the attribute spelling a
    body may use (`p.i`) and a store through it (`p.f = 1.5`) land in the caller's object
    exactly as they do without the proxy; only the subscript is answered here.
    """

    __slots__ = ("_wrapped",)

    def __init__(self, wrapped:object)->None:
        object.__setattr__(self, "_wrapped", wrapped)

    def __getitem__(self:object, key:object)->object:
        if isinstance(key, str):
            member:object = getattr(self._wrapped, key)
            # A structure member may itself be a union (`s["value"]["i"]`), and that
            # nested value is the bare ctypes object, so it needs the same wrapping
            # or the second subscript fails on the platforms this proxy exists for.
            if union_needs_proxy(member):
                return UnionProxy(member)
            return member
        return self._wrapped[key]

    def __setitem__(self:object, key:object, value:object)->None:
        if isinstance(key, str):
            setattr(self._wrapped, key, value)
            return
        self._wrapped[key] = value

    def __getattr__(self:object, name:str)->object:
        return getattr(self._wrapped, name)

    def __setattr__(self:object, name:str, value:object)->None:
        setattr(self._wrapped, name, value)

    def __repr__(self:object)->str:
        return repr(self._wrapped)


def is_declared_structure(dtype:DTypeLike) -> bool:
    """True when a `ctypes` structure or union class produced this dtype's layout.

    Alignment is the evidence, not just the registry: `np.dtype(cls)` over a
    `ctypes.Structure` always produces an *aligned* record (C layout), while the vector
    dtypes this package ships are built with `np.dtype(_fields_)` and are not aligned. So a
    record whose layout coincides with a vector -- `{float x; float y;}` against `float2` --
    is a structure when it is aligned, and a vector when it is not.

    A `ctypes.Union` is the one exception to the alignment rule: numpy cannot mark a record
    whose members all share an address as aligned (alignment describes the stride of a
    *sequence* of them), so `np.dtype(ctypes_union)` comes back with ``isalignedstruct``
    False. Its offsets are the evidence instead, and they are all zero -- see
    `is_union_dtype`, which is the same test a hand-written dtype of that shape passes.

    Note the argument has to be the dtype *object*, not a value compared against one: an
    aligned record and the vector's plain record compare equal and hash equal, so only the
    ``isalignedstruct`` flag on the dtype you were handed tells them apart.
    """
    if not isinstance(dtype, np.dtype) or dtype.fields is None:
        return False
    return bool(dtype.isalignedstruct) or is_union_dtype(dtype)


def structure_dtype_of(value:object) -> Optional[np.dtype]:
    """The structure dtype a value denotes, or None, respecting how it was built.

    A vector type is a `ctypes.Structure` as well, and its dtype carries the same alignment,
    so a value that is one of the package's vectors has to be excluded first -- otherwise
    every vector argument would be read as a structure.

    An ndarray keeps the dtype object it was created with, so an aligned record array stays
    distinguishable from a vector array whose layout happens to match.
    """
    from .translator import _vec_registry

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
    from ..vec_types.layout import _host_classes

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
    from ..vec_types.layout import _vector_dtype_name

    members:Dict[str, object] = {}
    for name in dtype.names or ():
        field_dtype:np.dtype = np.dtype(dtype.fields[name][0])
        shape:Optional[Tuple[int, ...]] = None
        if field_dtype.subdtype is not None:
            base_dtype, shape = field_dtype.subdtype
            field_dtype = base_dtype
        vector:Optional[str] = _vector_dtype_name(field_dtype)
        if vector is not None:
            kind:object = vector
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


def dtype_of(value:object)->Optional[np.dtype]:
    """The record dtype a value carries, when it is a structure or an array of them."""
    if isinstance(value, np.void):
        return value.dtype if is_record(value.dtype) else None
    if isinstance(value, np.ndarray) and is_record(value.dtype):
        return value.dtype
    return None
