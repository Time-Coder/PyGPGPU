# What the package knows about the caller's ctypes classes, and the names they settle on.
from __future__ import annotations
from .layout import (
    _is_vector_alias,
    struct_name,
)
from .state import (
    _DECLARED,
    _DECLARED_ALL,
    _DECLARED_FILES,
    _NAME_BASES,
    _REGISTRY,
)
from .unions import _add_subscript_access, is_union_dtype
from types import CodeType, ModuleType
from typing import List, Optional
import numpy as np
from numpy.typing import DTypeLike



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

    from ..translator import _vec_registry

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
