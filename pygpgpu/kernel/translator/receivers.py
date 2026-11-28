# What a receiver object is, from the classes it belongs to.
from __future__ import annotations
from .notation import Translated
from .source_text import parse_kernel_body
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
    - ``"classmethod"``: a classmethod, unwrapped to ``__func__`` -- a device helper
      whose first parameter names the class. A ``cls.<attr>`` read is the class
      attribute's *value*, which the caller's receiver struct already carries under
      the same name, so the call passes the receiver the way ``self.method`` does;
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
        func = raw.__func__
        if getattr(func, "is_pykernel", False):
            return "kernel", func
        return "classmethod", func
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


def _magic_method_of(cls:type, name:str)->Optional[Callable]:
    """The device method a class defines for `name`, or None.

    Only a method written out in the class body qualifies. `__eq__` exists on `object`
    (and on `ctypes.Structure`) as a synthesised identity comparison, which has no device
    meaning and would silently turn `a == b` into a pointer compare, so the lookup walks the
    class's own `__dict__` rather than its `__mro__`.

    A `@kernel`-decorated magic method is a kernel on the host and a **device helper** where
    an operator triggers it: `a + b` in device code cannot be a launch (there is no work size
    an operator could carry), so the decorated function's own body is what the operator
    translates. `PyKernel.raw_func` is that body.
    """
    own:object = getattr(cls, "__dict__", {}).get(name)
    if own is None:
        return None
    if isinstance(own, (staticmethod, classmethod)):
        return None
    function:object = getattr(own, "__func__", own)
    if getattr(function, "is_pykernel", False):
        raw:object = getattr(function, "raw_func", None)
        return raw if callable(raw) else None
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
        if isinstance(attribute, classmethod):
            func = attribute.__func__
            return ("kernel" if getattr(func, "is_pykernel", False) else "classmethod"), func
        if inspect.isfunction(attribute):
            return "method", attribute
    return kind, target


def _super_target(owner:Optional[type], attr:str)->Optional[Callable]:
    """The function `super().attr` resolves to, from a method of ``owner``, or None.

    Python's zero-argument `super()` from a method of `C` answers "the next class after
    `C` in the MRO that defines `attr`". The MRO is a fact about the python classes, all
    of them in hand at translation time, so the answer is a plain walk: every base after
    the owner is asked in order, the first carrier decides. A @kernel method is not a
    device helper, so it answers None -- the caller reports that no base implementation
    can be called from device code. A base classmethod IS one: its `__func__` is a plain
    function whose first parameter names the class, which is exactly the receiver the
    caller already holds.
    """
    if owner is None:
        return None
    for klass in owner.__mro__[1:]:
        if attr not in getattr(klass, "__dict__", {}):
            continue
        raw:object = klass.__dict__[attr]
        function:object = getattr(raw, "__func__", raw)
        if getattr(function, "is_pykernel", False) or not inspect.isfunction(function):
            return None
        return function
    return None


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
        tree:ast.Module = parse_kernel_body(func)
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


def _follow_descriptor_reads(owner:Optional[type], name:str, attrs:List[str],
                             visited:Set[object])->None:
    """Collect what a class-body descriptor's accessors read on the same instance.

    `self.total` where `total` is a descriptor is not a stored member: the device calls its
    `__get__`/`__set__`, and what *those* read (`instance.a`) has to travel with the receiver
    like any other member read. The accessor's instance parameter is what the walk follows --
    it is the same receiver, spelled by the descriptor's own parameter name.

    A descriptor whose accessor reads the descriptor itself or its owner class is refused by
    the translator when the value is computed; scanning it here only means the members it
    names are collected, which is harmless.
    """
    if owner is None:
        return
    from .descriptors import descriptor_of  # noqa: PLC0415 (import cycle)

    descriptor:object = descriptor_of(owner, name)
    if descriptor is None:
        return
    for kind in ("__get__", "__set__"):
        accessor:object = getattr(type(descriptor), kind, None)
        if not inspect.isfunction(accessor) or accessor in visited:
            continue
        code:object = getattr(accessor, "__code__", None)
        parameters:List[str] = list(code.co_varnames[:code.co_argcount]) if code else []
        if len(parameters) < 2:
            continue
        _collect_self_attrs(accessor, parameters[1], owner, attrs, visited, None)


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
    tree:ast.Module = parse_kernel_body(func)
    called:Set[str] = {node.func.attr for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and isinstance(node.func.value, ast.Name)
                       and node.func.value.id == self_param
                       and not (host_lines and int(getattr(node, "lineno", 0) or 0) in host_lines)}
    # `super().method(self, ...)` and `Base.method(self, ...)` reach the same helpers a
    # `self.method(...)` call does: the base implementation is called on THIS receiver, so
    # its own `self.xxx` reads are the instance's reads and collect the same way. The
    # targets are resolved statically -- the MRO walk for super, the method's own module
    # for a class-qualified name -- and followed with the same self_param, because the
    # callee's receiver is the caller's.
    follows:List[Callable] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if host_lines and int(getattr(node, "lineno", 0) or 0) in host_lines:
            continue
        value:object = node.func.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                and value.func.id == "super" and not value.args:
            target:Optional[Callable] = _super_target(owner, node.func.attr)
            if target is not None:
                follows.append(target)
        elif (isinstance(value, ast.Name) and value.id != self_param
                and node.func.attr not in called):
            holder:object = getattr(func, "__globals__", {}).get(value.id)
            if isinstance(holder, type):
                raw:object = getattr(holder, node.func.attr, None)
                function:object = getattr(raw, "__func__", raw)
                if inspect.isfunction(function):
                    follows.append(function)
    # `helper(self, ...)`: the receiver itself is a VALUE the body hands on. The
    # helper's own attribute reads -- and the method calls it makes on the receiver
    # (`r.other(v)`), whose callees read members too -- are this receiver's reads,
    # so each such helper is followed under the name its parameter binds the
    # receiver to. `visited` guards the recursion.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if host_lines and int(getattr(node, "lineno", 0) or 0) in host_lines:
            continue
        if not any(isinstance(argument, ast.Name) and argument.id == self_param
                   for argument in node.args):
            continue
        target:object = func.__globals__.get(node.func.id)
        if inspect.isfunction(target) and not getattr(target, "is_pykernel", False):
            parameter:str = (target.__code__.co_varnames[0]
                             if getattr(target, "__code__", None) else "r")
            # The owner rides along: a method the helper calls on the received
            # receiver (`r.other(v)`) resolves against the class that owns THIS
            # receiver -- the same class the device call will bind it from.
            _collect_self_attrs(target, parameter, owner, attrs, visited, None)
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
            # instance method's own self.xxx reads do, so follow it; a classmethod's
            # own cls.xxx reads are the receiver's reads too (the device call passes
            # this receiver as the callee's first parameter); anything else (a
            # staticmethod, a @kernel method) is decided at the call site, where a
            # clear error can name the reason.
            if kind in ("method", "classmethod"):
                # The host lines are the KERNEL's own numbering: they say which
                # statements the interpreter already ran in *this* body. A followed
                # method is device work by construction (the call into it was checked
                # against those lines above), so its own body is scanned with none --
                # reusing the kernel's numbers here would drop any read whose line
                # happens to collide with one of them. The scan follows the callee's
                # own receiver name (`cls` for a classmethod), because that is the
                # spelling its body reads its members through.
                _collect_self_attrs(target, _receiver_param_of(target), owner,
                                    attrs, visited, None)
            continue
        if node.attr not in attrs:
            attrs.append(node.attr)
        # A class-body descriptor is computed by its accessor, so the members the accessor
        # reads are this receiver's members too: `self.total` with a `__get__` that returns
        # `instance.a * 2` needs `a` uploaded even though the kernel body never names it.
        _follow_descriptor_reads(owner, node.attr, attrs, visited)
    # The super()/class-qualified calls reached on this receiver: their targets' own
    # self.xxx reads are the instance's reads, exactly as a `self.method(...)` target's
    # are. `visited` guards the recursion, so a super chain that loops stops here.
    for followed in follows:
        _collect_self_attrs(followed, self_param, owner, attrs, visited, None)
