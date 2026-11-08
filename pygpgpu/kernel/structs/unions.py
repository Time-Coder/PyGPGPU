# Union members: the proxy that makes an overlapping layout addressable.
from __future__ import annotations
from typing import List
import numpy as np
from numpy.typing import DTypeLike



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
