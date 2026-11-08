# What a receiver object is, from the classes it belongs to.
from __future__ import annotations
from .notation import Translated
from .source_text import _parse_kernel_source, get_dedented_source
import ast
import enum
import inspect
from typing import (Any, Callable, Dict, List, Optional, Sequence,
                    Set, Tuple)



def _class_attr(owner:Optional[type], name:str)->object:
    """The raw class-dictionary entry for ``owner.name``, or ``_MISSING`` when absent.

    Static methods never appear in an instance dictionary, so the raw entry is the only
    way to tell a ``staticmethod`` from the plain function it unwraps to on access.
    """
    if owner is None:
        return _MISSING
    for klass in getattr(owner, "__mro__", ()):
        if name in getattr(klass, "__dict__", {}):
            return klass.__dict__[name]
    return _MISSING


_MISSING:object = object()


def _helper_globals(func:Callable, fallback:Callable[[str], Any])->Callable[[str], Any]:
    """The name resolver a device helper is translated with.

    A helper is an ordinary python function, so a name it reads resolves in *its own* module
    first -- `ray.py`'s `hit_sphere` sees the `dot` that `ray.py` imported. Falling back to
    the caller's resolver is what keeps outer-scope variables working: a helper that reads a
    global of the *kernel's* module, or a name from the enclosing scope, still finds it and
    has it forwarded as a parameter at the call site.

    Resolving everything against the kernel's module instead used to work only because
    helpers lived beside the kernel; a method defined in the structure's own module is the
    first case where they do not.
    """
    own:Dict[str, object] = getattr(func, "__globals__", None) or {}

    def resolve(name:str)->object:
        if name in own:
            return own[name]
        return fallback(name)

    # The host has to read an outer value from the same place this resolver found it, or a
    # name that lives in the helper's module reads back as None at launch. Recorded on the
    # resolver rather than in a signature so `_bind_outer_variables` can pass it along.
    setattr(resolve, "source_globals", own)
    return resolve


def self_attr_kind(owner:Optional[type], name:str)->Translated:
    """Classify what ``self.name`` refers to on the kernel's own class.

    Returns ``(kind, target)`` where kind is one of:

    - ``"method"``: a plain function defined on the class, i.e. a device helper that
      also receives the kernel's self attributes (as extra parameters);
    - ``"staticmethod"``: a staticmethod, i.e. a plain device function with no receiver;
    - ``"classmethod"``: rejected, ``cls`` semantics do not exist on a device;
    - ``"kernel"``: a ``@kernel``-decorated method, rejected (kernels are launched from
      the host, never called from device code);
    - ``"data"``: an instance or class data attribute (array, scalar, property, ...).

    ``owner`` is None at decoration time, before the class exists; every member is then
    reported as data, which matches the decoration-time body scan that reads members as
    values.
    """
    if owner is None:
        return "data", None
    raw:object = _class_attr(owner, name)
    if isinstance(raw, staticmethod):
        func:object = raw.__func__
        return ("kernel" if getattr(func, "is_pykernel", False) else "staticmethod"), func
    if isinstance(raw, classmethod):
        return "classmethod", raw.__func__
    if getattr(raw, "is_pykernel", False):
        return "kernel", raw
    if inspect.isfunction(raw):
        return "method", raw
    return "data", (None if raw is _MISSING else raw)


def _property_of(cls:type, name:str)->Optional[object]:
    """The `property` descriptor a class defines for `name`, or None.

    A property is a member the class computes: `box.doubled` is a call to its getter, not a
    field read. Only a descriptor written in the class body counts, for the same reason a
    magic method is read from `__dict__` -- an inherited one belongs to another class.
    """
    own:object = getattr(cls, "__dict__", {}).get(name)
    return own if isinstance(own, property) else None


def _property_carrier(classes:Optional[Sequence[type]],
                     name:str)->Optional[object]:
    """The descriptor for `name` from the first of `classes` that defines one.

    One dtype can stand for several classes -- two classes with one layout share it -- so
    the property has to be looked for in every class the layout carries, not just the one
    registered first. Without this, the second class's `box.total` compiled the first
    class's getter and answered its value.
    """
    for cls in classes or []:
        descriptor:Optional[object] = _property_of(cls, name)
        if descriptor is not None:
            return descriptor
    return None


def _property_accessor(descriptor:property, writer:bool)->Optional[Callable]:
    """The getter (or setter) of a property, or None when it defines no such half."""
    accessor:object = descriptor.fset if writer else descriptor.fget
    return accessor if callable(accessor) else None


def _receiver_param_of(method:object)->str:
    """The parameter name a method binds its receiver to."""
    return (method.__code__.co_varnames[0]
            if getattr(method, "__code__", None) else "self")


def _enum_member_value(value:object)->Optional[Tuple[Any, Any]]:
    """The constant an `enum.Enum`/`enum.IntEnum` member stands for, or None.

    An enum is a *named constant* in python, not a type a device can hold: `Kind.B` in a
    kernel body means the number `B` was given, and every use of it (a comparison, an
    arithmetic expression, an argument to `np.full`) wants that number. So the member is
    replaced by its value at translation time rather than carried as a parameter.

    Returns (value, member) so a caller can name the member in an error, or None when the
    value is not an enum member. `enum.IntEnum` is a plain `int` subclass, which is why this
    is asked *before* the ordinary int/float constant branch -- otherwise an `IntEnum`
    member silently became the integer, which is the answer for an `IntEnum` but not for an
    `Enum`, whose value may be any type.
    """
    if not isinstance(value, enum.Enum):
        return None
    member:object = value
    payload:object = member.value
    if isinstance(payload, enum.Enum):
        # An alias chain (`A = B` in the class body) resolves to the member that holds the
        # actual value; python's own `_.value` stops at the first level.
        payload = payload.value
    return payload, member


def _operator_name(magic:str)->str:
    """The `ast` operator name a magic method belongs to."""
    for name, candidate in MAGIC_METHODS.items():
        if candidate == magic:
            return name
    return ""


def _magic_method_of(cls:type, name:str)->Optional[Callable]:
    """The device method a class defines for `name`, or None.

    Only a method written out in the class body qualifies. `__eq__` exists on `object`
    (and on `ctypes.Structure`) as a synthesised identity comparison, which has no device
    meaning and would silently turn `a == b` into a pointer compare, so the lookup walks the
    class's own `__dict__` rather than its `__mro__`.
    """
    own:object = getattr(cls, "__dict__", {}).get(name)
    if own is None:
        return None
    if isinstance(own, (staticmethod, classmethod)):
        return None
    function:object = getattr(own, "__func__", own)
    # The member-name subscript a structure gets (`s["i"]`) is the translation's own
    # spelling, not a method the caller wrote: `s[i]` must stay an array index.
    if getattr(function, "__pygpgpu_member_subscript__", False):
        return None
    return function if callable(function) else None


def _self_member_call_kind(owner:Optional[type], name:str)->Translated:
    """Classify ``self.name(...)``: what the attribute read yields, then called.

    A staticmethod read through an instance is a plain function, so a call site has to
    reach the underlying function; a plain function the instance carries (assigned in
    __init__) is reached through the instance too, and both are receiver-less device
    calls.
    """
    kind:str
    target:object
    kind, target = self_attr_kind(owner, name)
    if kind == "data" and owner is not None:
        attribute:object = inspect.getattr_static(owner, name, _MISSING)
        if isinstance(attribute, staticmethod):
            func:object = attribute.__func__
            return ("kernel" if getattr(func, "is_pykernel", False) else "staticmethod"), func
        if inspect.isfunction(attribute):
            return "method", attribute
    return kind, target


def structs_name(dtype:object)->str:
    """The device type name of a structure dtype, registered on the way."""
    from ..structs import register  # noqa: PLC0415

    return register(dtype)


def writes_receiver(func:Callable, self_param:str)->bool:
    """Whether a method body assigns to a member of its receiver (`self.x = ...`).

    A method that does has to receive its receiver *by pointer*: a by-value structure is a
    copy, so the store lands in the callee's local and never reaches the caller's value.
    `HitRecord.set_face_normal` writes `self.front_face` and `self.normal`, and with a
    by-value receiver every hit record's normal stayed zero -- a uniformly grey image on the
    GPU while the interpreter was right, because it runs the method unchanged.
    """
    if not self_param:
        return False
    try:
        tree:ast.Module = _parse_kernel_source(get_dedented_source(func))
    except Exception:
        return False
    for node in ast.walk(tree):
        targets:List[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == self_param):
                return True
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == self_param):
                return True
    return False


def scan_self_attrs(func:Callable, self_param:str, owner:Optional[type]=None,
                    host_lines:Optional[Set[int]]=None)->List[str]:
    """Collect the self.xxx data attribute names the method (and its helpers) reference.

    A member read as a value becomes a kernel argument. With ``owner`` given, a method
    called as ``self.helper(...)`` is followed instead: the helper's own ``self.xxx``
    references then count too, recursively, so nested helpers keep working and every
    attribute they need is uploaded with the kernel. ``owner`` is None at decoration
    time (the class does not exist yet), where only the decorated body itself is read.

    ``host_lines`` are the statements the interpreter already ran (see
    `_host_region_values`): their calls are host work, so a helper reached only from there
    contributes no arguments.
    """
    if not self_param:
        return []
    attrs:List[str] = []
    _collect_self_attrs(func, self_param, owner, attrs, set(), host_lines or set())
    return attrs


def _collect_self_attrs(func:Callable, self_param:str, owner:Optional[type],
                        attrs:List[str], visited:Set[object],
                        host_lines:Optional[Set[int]]=None)->None:
    """Walk one method body, recursing into the self-methods it calls.

    A call that sits on a line the interpreter already executed (`host_lines`) is host
    work: it ran on the host, together with everything it touched, so nothing it reads is
    a kernel argument. Skipping it is what lets a host-only helper take an
    arbitrary-precision value (`def refresh(self): self.scale * 3`) without the value
    becoming a device member.
    """
    if func in visited:
        return
    visited.add(func)
    tree:ast.Module = _parse_kernel_source(get_dedented_source(func))
    called:Set[str] = {node.func.attr for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and isinstance(node.func.value, ast.Name)
                       and node.func.value.id == self_param
                       and not (host_lines and int(getattr(node, "lineno", 0) or 0) in host_lines)}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == self_param):
            continue
        if host_lines and int(getattr(node, "lineno", 0) or 0) in host_lines:
            # The whole statement is host work, so nothing it names is a device member --
            # including a call it makes (`self.refresh()` above the loop).
            continue
        kind:str
        target:object
        kind, target = _self_member_call_kind(owner, node.attr)
        if node.attr in called:
            # A member that is called is not data, so it needs no kernel argument. An
            # instance method's own self.xxx reads do, so follow it; anything else
            # (a staticmethod, a @kernel method, a classmethod) is decided at the
            # call site, where a clear error can name the reason.
            if kind == "method":
                _collect_self_attrs(target, self_param, owner, attrs, visited, host_lines)
            continue
        if node.attr not in attrs:
            attrs.append(node.attr)


# A python class can give an operator a meaning of its own, and device code has to keep
# it: `a + b` on a class that defines `__add__` is that method, exactly as `ray.at(t)` is
# `Ray_at`. The keys are the `ast` operator class names the dispatchers already use, so one
# table serves the binary, unary and comparison paths.
MAGIC_METHODS:Dict[str, str] = {
    "Add": "__add__", "Sub": "__sub__", "Mult": "__mul__", "Div": "__truediv__",
    "FloorDiv": "__floordiv__", "Mod": "__mod__", "Pow": "__pow__",
    "LShift": "__lshift__", "RShift": "__rshift__",
    "BitOr": "__or__", "BitXor": "__xor__", "BitAnd": "__and__",
    "MatMult": "__matmul__",
    "Lt": "__lt__", "LtE": "__le__", "Gt": "__gt__", "GtE": "__ge__",
    "Eq": "__eq__", "NotEq": "__ne__",
    # `x in box` asks box.__contains__(x): the receiver is the RIGHT operand, which is why
    # the dispatch for these two reverses the pair.
    "In": "__contains__", "NotIn": "__contains__",
    "USub": "__neg__", "UAdd": "__pos__", "Invert": "__invert__",
}
# The reflected spelling is tried on the right operand when the left one has no method.
# A conversion the class defines: `int(box)` is `box.__int__()`. `bool` is here too, but a
# class that defines no `__bool__` still answers through the ordinary truth test, so it is
# tried first rather than instead.
CONVERSION_METHODS:Dict[str, str] = {
    "int": "__int__", "float": "__float__", "bool": "__bool__",
}
# A single-argument builtin whose device meaning is the class's own method.
BUILTIN_METHODS:Dict[str, str] = {"abs": "__abs__", "round": "__round__"}
# `box(args)` is `box.__call__(args)`.
CALL_METHOD:str = "__call__"

# `v += w` asks `v.__iadd__(w)` first, and falls back to `v.__add__(w)` when the class
# defines no in-place form -- python's own rule, and the reason a class that only knows
# `__add__` still supports `+=`.
MAGIC_INPLACE:Dict[str, str] = {
    "Add": "__iadd__", "Sub": "__isub__", "Mult": "__imul__", "Div": "__itruediv__",
    "FloorDiv": "__ifloordiv__", "Mod": "__imod__", "Pow": "__ipow__",
    "LShift": "__ilshift__", "RShift": "__irshift__",
    "BitOr": "__ior__", "BitXor": "__ixor__", "BitAnd": "__iand__",
    "MatMult": "__imatmul__",
}

MAGIC_REFLECTED:Dict[str, str] = {
    "Add": "__radd__", "Sub": "__rsub__", "Mult": "__rmul__", "Div": "__rtruediv__",
    "FloorDiv": "__rfloordiv__", "Mod": "__rmod__", "Pow": "__rpow__",
    "LShift": "__rlshift__", "RShift": "__rrshift__",
    "BitOr": "__ror__", "BitXor": "__rxor__", "BitAnd": "__rand__",
    "MatMult": "__rmatmul__",
}
# A comparison has its own reflected form, and python reads `a < b` as `a.__lt__` first.
MAGIC_COMPARE_REFLECTED:Dict[str, str] = {
    "Lt": "__gt__", "LtE": "__ge__", "Gt": "__lt__", "GtE": "__le__",
    "Eq": "__eq__", "NotEq": "__ne__",
}
