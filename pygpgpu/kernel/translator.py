from __future__ import annotations

import ast
import builtins
import enum
import hashlib
import inspect
import math
import os
import re
import textwrap
import types
from typing import (Any, Callable, Dict, List, NoReturn, Optional, Sequence,
                    Set, Tuple, Union, cast)

# TypedDict is 3.8+ in typing; this package supports 3.7, so it comes from the
# backport the rest of the package already uses (see cparser/ts_parser.py).
from typing_extensions import TypedDict

import numpy as np

from .builtin_registry import builtin_name_of
from .device_library import library_function_name, library_functions
from .structs import dtype_for as structs_dtype_for, member_types as structs_member_types
from . import complex as complex_support
from .device_builtins import (
    ATOMIC_BUILTINS,
    CUDA_ONLY_BUILTINS,
    CUDA_ONLY_CONVERSIONS,
    GENERIC_BUILTINS,
    HIP_ROUND_TOWARD_ZERO_HELPERS,
    HIP_ROUND_TOWARD_ZERO_SOURCES,
    INDEX_BUILTINS,
    INDEX_VECTOR_AXES_CUDA,
    INDEX_VECTOR_AXES_OPENCL,
    INDEX_VECTOR_BUILTINS,
    INDEX_VECTOR_OPENCL,
    MATH_BUILTINS,
    SHARED_MATH_BUILTINS,
    SHARED_MATH_TEMPLATES,
    emit_atomic,
    emit_barrier,
    emit_index,
    float_atomic_add_source,
    python_floor_name,
    python_floor_source,
)


def get_dedented_source(func:Callable)->str:
    """Read the function source, drop decorator lines and dedent (class methods are indented; ast.parse needs a top-level def)."""
    try:
        source:str = inspect.getsource(func)
    except OSError as e:
        raise SyntaxError(f"cannot read source of {getattr(func, '__name__', '<function>')}: {e}") from e
    lines:List[str] = source.split("\n")
    start:int = next((i for i, line in enumerate(lines) if line.lstrip().startswith("def ")), 0)
    return textwrap.dedent("\n".join(lines[start:]))


def _trimmed_source(func:Callable)->str:
    """The text the translator parses: decorators dropped, dedented."""
    return get_dedented_source(func)


def _source_first_line(func:Callable)->int:
    """The file line the parsed copy's line 1 came from.

    ``get_dedented_source`` keeps everything from the ``def`` line on and drops the
    decorators above it, so the copy's line 1 is the ``def`` line. ``getsourcelines`` starts
    at the first decorator, which can be several lines earlier, so the skipped decorator
    lines have to be added back; using its number directly would report every error a few
    lines too high.
    """
    try:
        lines, first_decorator = inspect.getsourcelines(func)
    except (OSError, TypeError):
        return 1
    offset = next((i for i, line in enumerate(lines)
                   if line.lstrip().startswith("def ")), 0)
    return int(first_decorator) + offset


def _get_attr(obj:object, name:str, default:object=None)->object:
    """Attribute read that falls back to a default instead of raising."""
    return getattr(obj, name, default)


def _copy_ast(node:ast.AST, cls:type, fields:Dict[str, object])->ast.AST:
    """Build a node of ``cls`` from a field dict shared with a 3.7-era node."""
    obj:ast.AST = cls()
    for _f in getattr(cls, "_fields", ()):
        try:
            setattr(obj, _f, fields.get(_f))
        except Exception:
            pass
    try:
        ast.fix_missing_locations(obj)
    except Exception:
        pass
    return obj


def _normalize_ast_legacy_nodes(tree:ast.AST)->ast.AST:
    """Fold Python 3.7/3.8-only AST node shapes into the post-3.8 form.

    CPython 3.9 removed several AST node classes that 3.7/3.8 still emitted: the
    ``ast.Index`` slice wrapper (``Subscript(slice=Index(value=...))``) and the
    ``ast.Num`` / ``ast.Str`` / ``ast.Bytes`` / ``ast.NameConstant`` constant
    aliases, plus the ``ast.Ellipsis`` singleton, all replaced by ``ast.Constant``.
    The translator only inspects the tree structurally, so rewriting these legacy
    forms to their modern equivalents lets the same code run unchanged on 3.7 and
    3.9+. On 3.9+ the legacy classes no longer exist and ``ast.parse`` already
    returns the modern shape, so this is a no-op.

    The legacy class names are looked up dynamically (``getattr``) rather than
    referenced statically so type checkers do not flag their deprecation on new
    CPython versions.
    """
    if not hasattr(ast, "Num"):
        # CPython 3.9+: the constant aliases no longer exist and ast.parse
        # already returns the modern form, so no rewriting is needed.
        return tree
    _ast_index = getattr(ast, "Index", None)
    if _ast_index is not None and isinstance(tree, _ast_index):
        return _normalize_ast_legacy_nodes(tree.value)
    _ast_num = getattr(ast, "Num", None)
    if _ast_num is not None and isinstance(tree, _ast_num):
        return _copy_ast(tree, ast.Constant, {"value": _get_attr(tree, "n"), "kind": None})
    _ast_str = getattr(ast, "Str", None)
    if _ast_str is not None and isinstance(tree, _ast_str):
        return _copy_ast(tree, ast.Constant, {"value": _get_attr(tree, "s"), "kind": None})
    _ast_bytes = getattr(ast, "Bytes", None)
    if _ast_bytes is not None and isinstance(tree, _ast_bytes):
        return _copy_ast(tree, ast.Constant, {"value": _get_attr(tree, "s"), "kind": None})
    _ast_name_constant = getattr(ast, "NameConstant", None)
    if _ast_name_constant is not None and isinstance(tree, _ast_name_constant):
        return _copy_ast(tree, ast.Constant, {"value": _get_attr(tree, "value"), "kind": None})
    _ast_ellipsis = getattr(ast, "Ellipsis", None)
    if _ast_ellipsis is not None and isinstance(tree, _ast_ellipsis):
        return _copy_ast(tree, ast.Constant, {"value": Ellipsis, "kind": None})
    for _field, _value in ast.iter_fields(tree):
        if isinstance(_value, list):
            # A field list can hold plain strings as well as nodes (`Global.names` is
            # `list[str]`), and only nodes can be rewritten -- recursing into a string
            # made `ast.iter_fields` fail with "'str' object has no attribute '_fields'".
            _value[:] = [_normalize_ast_legacy_nodes(v) if isinstance(v, ast.AST) else v
                         for v in _value]
        elif isinstance(_value, ast.AST):
            setattr(tree, _field, _normalize_ast_legacy_nodes(_value))
    return tree


def _returns_a_value(body:List[ast.stmt])->bool:
    """True when some `return` of this body carries a value (and not just None)."""
    for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            continue
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            continue
        return True
    return False


def _has_device_return(body:List[ast.stmt])->bool:
    """True when some `return` of this body is emitted into the generated kernel.

    A body's statements fall into three regions (see PyKernel._run_host_region): the ones
    above the first magic loop and the ones after the last run on the host, so their
    `return`s never reach the device. A `return` inside a magic loop's body -- nested `if`
    blocks and serial loops included -- *is* device code, and it claims the result.

    A bare `return` counts: it claims with None, which is how python's "the first work item
    to leave decides" rule survives a value-returning site elsewhere in the body. A
    gid-style body has no host regions at all, so every statement of it is device code.
    """
    loops:List[ast.For] = [stmt for stmt in body
                           if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt)]
    if not loops:
        regions:List[ast.AST] = list(body)
    else:
        # Only the loop bodies: a statement between two loops is a gap the host runs, and a
        # `return` there answers on the host like the tail's does.
        regions = [ast.Module(body=list(loop.body), type_ignores=[]) for loop in loops]
    for region in regions:
        for node in ast.walk(region):
            if isinstance(node, ast.Return):
                return True
    return False


def _parse_kernel_source(source:str)->ast.Module:
    """Parse a kernel source string into an AST normalized across Python versions."""
    return _normalize_ast_legacy_nodes(ast.parse(source, mode="exec"))


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
    from .structs import register  # noqa: PLC0415

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


def _syntax_error(message:str, node:Optional[ast.AST]=None,
                  filename:str="<kernel>",
                  source:Optional[str]=None,
                  first_line:int=1)->SyntaxError:
    """Build a plain :class:`SyntaxError` positioned at kernel source the backend rejects.

    Callers ``raise`` the result ``from None``, so the traceback shows only the user's own
    frame and CPython's familiar rendering: the file, the line number, the offending source
    line and a caret under the column.

    The translator parses a *dedented copy* of the function with its decorator line removed,
    so an AST ``lineno`` counts from that copy. ``first_line`` converts it back to the line
    in the user's file; the copy's line 1 is the decorator line.
    """
    lineno:int = getattr(node, "lineno", 0) if node is not None else 0
    column:int = getattr(node, "col_offset", 0) if node is not None else 0
    end_column:Optional[int] = getattr(node, "end_col_offset", None) if node is not None else None

    # lineno in the parsed copy -> line in the user's file (the copy starts at the decorator).
    file_lineno:Optional[int] = (first_line + lineno - 1) if lineno else None

    text:Optional[str] = None
    # Prefer the user's own file: the parsed copy is dedented and has no decorator, so its
    # line text would show the wrong indentation and the caret would sit off by that much.
    if filename and filename not in ("<kernel>", "<string>"):
        try:
            with open(filename, encoding="utf-8") as handle:
                file_lines = handle.read().splitlines()
        except OSError:
            file_lines = []
    else:
        file_lines = []

    if file_lines:
        if file_lineno is not None and 0 < file_lineno <= len(file_lines):
            text = file_lines[file_lineno - 1]
            # Columns come from the dedented copy; add back the indentation it removed. The
            # same shift applies to the node's end column.
            if source and lineno and 0 < lineno <= len(source.splitlines()):
                shift = max(len(text) - len(source.splitlines()[lineno - 1]), 0)
                column += shift
                if end_column is not None:
                    end_column += shift
    elif source and lineno:
        lines = source.splitlines()
        if 0 < lineno <= len(lines):
            text = lines[lineno - 1]

    details:Tuple[object, ...] = (
        filename if filename else "<kernel>",
        file_lineno,
        (column + 1) if text is not None else None,
        text,
    )
    error = SyntaxError(message, details)
    if end_column is not None and text is not None:
        # A column range makes CPython underline the whole span instead of one caret.
        error.end_lineno = file_lineno
        error.end_offset = end_column + 1
    return error


def _unpack_count_message(names:int, values:int)->str:
    """What python itself says when a loop's names do not match the number of values.

    `for i, j in np.ndindex(arr.shape[0])` is an ordinary unpacking error in python -- the
    loop asks for more names than the iterator yields -- and a kernel body should say the
    same thing, in the same words, so the two spellings of one mistake read alike. CPython
    has two messages and picks by direction: too few values is "not enough values to
    unpack", too many is "too many values to unpack", both with "(expected N, got M)".
    """
    if values < names:
        return (f"not enough values to unpack (expected {names}, got {values})")
    return (f"too many values to unpack (expected {names}, got {values})")


def _unpacking_error(message:str, node:Optional[ast.AST], filename:str,
                     source:Optional[str]=None, first_line:int=1)->ValueError:
    """An unpacking :class:`ValueError`, positioned at the loop that raises it.

    ``ValueError`` has no ``filename``/``lineno`` of its own, so the position travels in the
    message exactly as the interpreter's does -- and the traceback is trimmed to the user's
    frame by the launch (see ``PyKernel._invoke``). The source line and caret come from the
    same helper a translation error uses, so the two render alike.
    """
    error:ValueError = ValueError(message)
    _position_error(error, node, filename, source, first_line)
    return error


def _position_error(error:BaseException, node:Optional[ast.AST], filename:str,
                    source:Optional[str]=None, first_line:int=1)->BaseException:
    """Attach file/line/column to an exception raised *about* a span of the user's source."""
    if node is None:
        return error
    lineno:int = getattr(node, "lineno", 0)
    file_lineno:Optional[int] = (first_line + lineno - 1) if lineno else None
    error.filename = filename  # type: ignore[attr-defined]
    error.lineno = file_lineno  # type: ignore[attr-defined]
    return error


INT_TYPES:Set[str] = {"bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"}
FLOAT_TYPES:Set[str] = {"float", "double"}
SCALAR_TYPES:Set[str] = INT_TYPES | FLOAT_TYPES

DTYPE_MAP:Dict[str, str] = {
    "bool": "bool",
    "int8": "char", "int16": "short", "int32": "int", "int64": "long",
    "uint8": "uchar", "uint16": "ushort", "uint32": "uint", "uint64": "ulong",
    "float16": "half", "float32": "float", "float64": "double",
    # A complex value is one of the fixed structures the complex header declares, not a
    # scalar: the entry here is the type *name* a signature and a subscript use, exactly as
    # a structure name is. `is_complex_type` is what tells the two apart.
    "complex64": "c64", "complex128": "c128",
}

# The ctypes scalar classes a caller may name for a python `int`/`float`, and the device type
# each stands for. The package already reads ctypes classes this way for a vector component
# (`_c_scalar_name`), so this is the same mapping given a caller-facing name: `_c_scalar_name`
# is the vector component's *name*, and this is the advice of what a scalar argument becomes.
CTYPES_SCALAR_TYPES:Dict[object, str] = {}

# The python type a width is requested for, and the only names a `type_map` may carry. Two
# names only, because these are the two python builtins whose width is chosen rather than
# carried: every numpy scalar already names its own width (`np.int64`, `np.float64`), so a
# mapping for those would be a way to override what the caller spelled, which is a different
# (and much larger) feature.
TYPE_MAP_KEYS:Set[str] = {"int", "float"}

# What `int=` may name: the eight integer widths, signed and unsigned, 8 through 64 bits. `bool`
# is not among them -- it is not an integer width a caller would pick for a counter -- and
# neither is `half`, which the device has no scalar for on CUDA and HIP.
INTEGER_MAPPING_TARGETS:Set[str] = {
    "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong",
}

# What `float=` may name: the two floating-point widths. `half` is deliberately absent
# (`DTYPE_MAP` has an entry for it, but NVRTC on this machine cannot resolve `cuda_fp16.h`
# and the CUDA/HIP path refuses it), so the mapping would only move the failure later.
FLOAT_MAPPING_TARGETS:Set[str] = {"float", "double"}

# The integer widths that cannot hold a negative value. A python `int` mapped to one of these
# is refused when it is negative, instead of wrapping to a huge positive number.
UNSIGNED_INTEGER_TYPES:Set[str] = {"uchar", "ushort", "uint", "ulong"}


def _ctypes_scalar_types()->Dict[object, str]:
    """Build the ctypes class table once: the scalar classes a caller may name."""
    global CTYPES_SCALAR_TYPES
    if CTYPES_SCALAR_TYPES:
        return CTYPES_SCALAR_TYPES
    import ctypes as _ctypes

    CTYPES_SCALAR_TYPES = {
        _ctypes.c_bool: "bool",
        _ctypes.c_int8: "char", _ctypes.c_int16: "short",
        _ctypes.c_int32: "int", _ctypes.c_int64: "long",
        _ctypes.c_uint8: "uchar", _ctypes.c_uint16: "ushort",
        _ctypes.c_uint32: "uint", _ctypes.c_uint64: "ulong",
        _ctypes.c_float: "float", _ctypes.c_double: "double",
    }
    return CTYPES_SCALAR_TYPES


def ctypes_device_type_name(expected:object)->Optional[str]:
    """The device type name a backend's scalar ctypes class stands for, or None.

    `kernel_import` parses a handwritten source and gets each parameter as the backend's own
    `ctypes` class; to range-check a value against it, the class has to name the device type
    the same way a `@kernel` argument does. Only scalar classes answer; an array, a structure
    or an unknown class answers None, and the caller then falls back to its own test.
    """
    import ctypes as _ctypes

    if not isinstance(expected, type) or not issubclass(expected, _ctypes._SimpleCData):
        return None                  # noqa: SLF001 (the documented base)
    return _ctypes_scalar_types().get(expected)


def resolve_device_type(value:object, key:str)->str:
    """The device type name a caller's type object stands for, for the given mapping key.

    A `@kernel(int=..., float=...)` mapping is given with a **type object**, because that is
    the spelling a reader of numpy already knows and it carries its width and signedness in
    the name rather than in a string this package would have to parse:

    * a numpy dtype or scalar type -- `np.int64`, `np.dtype("uint32")`, `np.dtype(np.float64)`;
    * a ctypes scalar class -- `ctypes.c_int64`, `ctypes.c_double`;
    * the device type name itself -- `"long"`, `"double"`.

    **The choice is a closed set, and it is not the same set for each key.** `int` may be
    any of the eight integer widths (signed and unsigned, 8 through 64 bits); `float` may be
    `float32` or `float64` and nothing else. The alternative -- accepting whatever the device
    happens to have -- was wrong in both directions: `bool` is not an integer width at all,
    and `float16` (`half`) is out of scope on CUDA and HIP, so a kernel broadened to one of
    them would fail to compile instead of failing here with a reason.

    A string is accepted only when it names one of those widths, so `"int8"` is refused while
    `"char"` is accepted: a string is the *device* type name, and the numpy spelling exists
    because `np.int8` is the spelling a caller actually writes. Every refusal is a
    `ValueError`: a wrong type is a wrong value for this argument, not a syntax error.
    """
    import ctypes as _ctypes

    allowed:Dict[str, Set[str]] = INTEGER_MAPPING_TARGETS if key == "int" else FLOAT_MAPPING_TARGETS

    # A numpy dtype or scalar type, given as the type (`np.int64`) or a dtype (`np.dtype("int8")`).
    if isinstance(value, np.dtype) or (isinstance(value, type)
                                       and issubclass(value, np.generic)):
        dtype:np.dtype = np.dtype(value)
        # A numpy scalar names a width; map it to the device spelling and then to the boards
        # of the closed set, so `np.int8` is accepted only when `char` is (it is).
        device_name:Optional[str] = DTYPE_MAP.get(dtype.name)
        if device_name is None or device_name not in allowed:
            raise ValueError(_mapping_refusal(key, value)) from None
        return device_name

    # A ctypes scalar class.
    if isinstance(value, type) and issubclass(value, _ctypes._SimpleCData):
        device_name = _ctypes_scalar_types().get(value)
        if device_name is not None and device_name in allowed:
            return device_name
        raise ValueError(_mapping_refusal(key, value)) from None

    # A device type name, spelled the way a signature spells it.
    if isinstance(value, str):
        if value in allowed:
            return value
        raise ValueError(_mapping_refusal(key, value)) from None

    raise ValueError(_mapping_refusal(key, value)) from None


def _mapping_refusal(key:str, value:object)->str:
    """The message for a type the mapping key does not accept."""
    if key == "int":
        np_names:str = ("np.int8, np.int16, np.int32, np.int64, "
                        "np.uint8, np.uint16, np.uint32, np.uint64")
        others:str = "A bool, a float and a complex value are not integer widths."
        widths:str = ("char (int8), short (int16), int (int32), long (int64), "
                      "uchar (uint8), ushort (uint16), uint (uint32), ulong (uint64)")
    else:
        np_names = "np.float32, np.float64"
        others = "An integer is not a floating-point width, and `half` is out of scope."
        widths = "float (float32), double (float64)"
    return (f"`{key}=` cannot be {value!r}: a python {key} travels as one of "
            f"{widths} -- written as {np_names}, or as the matching ctypes type. {others}")


def validate_type_map(type_map:Optional[Dict[str, object]])->Dict[str, str]:
    """Normalize a caller's `int=`/`float=` mapping to device type names.

    The keys are the two python builtins whose width is otherwise chosen here, and every
    entry is resolved through `resolve_device_type`, so the decorator and the global setter
    accept exactly the same spellings and reject exactly the same mistakes. Every refusal
    here is a `ValueError`, because a mapping is a value the caller supplied.
    """
    if not type_map:
        return {}
    resolved:Dict[str, str] = {}
    for key, value in type_map.items():
        if key not in TYPE_MAP_KEYS:
            raise ValueError(
                f"unknown type-mapping key: {key!r}; only 'int' and 'float' can be mapped "
                f"(a numpy scalar already names its own width)") from None
        resolved[key] = resolve_device_type(value, key)
    return resolved


# The device type names that are complex values rather than structures. Unlike a structure
# they travel by value, their arithmetic is generated (not refused), and their two members
# are read with the header's own accessors.
COMPLEX_TYPE_NAMES:Set[str] = {"c64", "c128"}

# A python `complex` has no width of its own, so its width follows the `float` mapping: a
# kernel that asks for 64-bit reals asks for 64-bit complex values beside them, and the
# default 32-bit real pairs with the single-precision complex. The two widths are the only
# ones a complex value has, so the table is written out for both of the mapping's own names
# and anything else is read as the single-precision pair.
_COMPLEX_WIDTH_PER_REAL:Dict[str, str] = {"float": "c64", "double": "c128"}


def python_complex_type(float_type:str)->str:
    """The complex width a python `complex` travels as beside a real of this width.

    A python `complex` has no width of its own -- the same rule a python `float` follows --
    so the two are always chosen together: `float=` naming a double gives `c128`, and the
    32-bit default (`float`, or no mapping at all) gives `c64`. This is the single place that
    decides it, which is what keeps a python complex value from being the one value in a body
    whose width disagrees with the reals beside it: the scalar path (`infer_ctype`), a body
    literal and `complex(re, im)` all read their width from here, so a `float=` mapping moves
    all three together.
    """
    return _COMPLEX_WIDTH_PER_REAL.get(float_type, "c64")

# The range of every integer the device can hold, as (minimum, maximum). A python `int` is
# arbitrary precision, so a value outside the width it is about to travel as used to wrap
# silently: `k(2 ** 31 + 5, out)` reached the device as -2147483648, and the kernel then
# computed with a number the caller never wrote. Refusing it is the only honest answer, and
# the message names the spelling that carries the width the caller actually needs.
_INTEGER_RANGES:Dict[str, Tuple[int, int]] = {
    "char": (-(2 ** 7), 2 ** 7 - 1),
    "uchar": (0, 2 ** 8 - 1),
    "short": (-(2 ** 15), 2 ** 15 - 1),
    "ushort": (0, 2 ** 16 - 1),
    "int": (-(2 ** 31), 2 ** 31 - 1),
    "uint": (0, 2 ** 32 - 1),
    "long": (-(2 ** 63), 2 ** 63 - 1),
    "ulong": (0, 2 ** 64 - 1),
}


def check_integer_range(value:int, device_type:str, where:str='')->None:
    """Public entry point for the width check, shared by every path that chooses a type.

    A scalar argument, a structure member, and an imported kernel's declared parameter all
    have to agree on what "does not fit" means. `where` names the member when the caller has
    one, so the message points at it rather than at the kernel as a whole.
    """
    _check_integer_fits(value, device_type, where)


def _check_integer_fits(value:int, device_type:str, where:str='')->None:
    """Refuse a python `int` the device type it travels as cannot represent."""
    bounds:Optional[Tuple[int, int]] = _INTEGER_RANGES.get(device_type)
    if bounds is None:
        return                      # `bool`: python bools are not ints for this rule
    low, high = bounds
    if low <= value <= high:
        return
    # The suggestion is the numpy spelling that holds this value, and only when one does: an
    # integer beyond 64 bits has no device type at all, so pointing at `np.int64` there would
    # be pointing at a second failure.
    if 0 <= value <= 2 ** 64 - 1 and value > 2 ** 63 - 1:
        suggestion = f" Pass it as `np.uint64({value})` to keep that width."
    elif -(2 ** 63) <= value <= 2 ** 63 - 1:
        suggestion = (f" Pass it as `np.int64({value})` to keep that width."
                      if value < 0 else
                      f" Pass it as `np.int64({value})` or `np.uint64({value})` "
                      f"to keep that width.")
    else:
        suggestion = " The device has no integer type that wide."
    raise ValueError(
        f"{where + ': ' if where else ''}the integer {value} does not fit the device "
        f"type '{device_type}' "
        f"({low}..{high}); a python int has no width of its own, so it travels as the "
        f"width this kernel chose for `int`.{suggestion}") from None

# The numpy scalars whose device type is the *narrow* one, as it always was: an 8- or
# 16-bit value is an `int` on the device, and a float64 is a `float`. A caller who wants
# a width of its own says so (`np.uint32`, `np.int64`), and that width is kept.
NARROWED_SCALARS:Dict[str, str] = {
    "int8": "int", "int16": "int", "uint8": "int", "uint16": "int",
    "float16": "float", "float64": "float",
}

PTR = "*"  # type notation: ("*", base) means an array parameter, a plain string means a scalar

# A zeros_per_group() slice: that group's storage, plus the shape it was declared with.
SHARED_SLICE = "shared_slice"

# Names a kernel body may use that C, C++ or OpenCL C reserves. A parameter or local called
# `new` is legal python and illegal in every one of those languages -- CUDA and HIP compile
# as C++, so `void f(c64 new, ...)` fails with a bare `expected a ")"` that names nothing the
# user wrote. The mapping is the prefix `_pyg_` plus the name, which is the one spelling that
# cannot collide with a user's own identifier (a python name cannot contain the prefix
# without also being mapped, so the two never meet).
#
# The set is the union of C89/C99/C11 keywords, C++ keywords as far as CUDA C++ accepts
# them, and OpenCL C's address-space and function qualifiers (`__global`, `__kernel`, ...),
# because one generated body is compiled by all three.
RESERVED_DEVICE_NAMES:Set[str] = {
    # C and C++ keywords
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
    "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct",
    "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
    "bool", "catch", "class", "constexpr", "const_cast", "delete", "dynamic_cast",
    "explicit", "export", "false", "friend", "mutable", "namespace", "new", "noexcept",
    "nullptr", "operator", "private", "protected", "public", "reinterpret_cast",
    "static_cast", "template", "this", "throw", "true", "try", "typeid", "typename",
    "using", "virtual", "wchar_t",
    # C++20 spellings CUDA/HIP toolchains accept
    "concept", "constinit", "consteval", "co_await", "co_return", "co_yield",
    "requires", "char8_t", "char16_t", "char32_t",
    # OpenCL C qualifiers and types
    "__global", "__local", "__constant", "__private", "__kernel", "__read_only",
    "__write_only", "__read_write", "__generic", "__attribute__", "__kernel_exec",
}

RESERVED_PREFIX:str = "_pyg_"


def device_name(name:str)->str:
    """The C identifier a python name is emitted as.

    A name the device language reserves is spelled with the `_pyg_` prefix, so a kernel
    parameter called `new`, `register`, `float`, `global` or `kernel` compiles. Everything
    else is emitted unchanged, so the generated source still reads like the python it came
    from. The two spellings are disjoint by construction: `_pyg_new` is only produced for
    `new`, and a python identifier `_pyg_new` is itself prefixed (`_pyg__pyg_new`).
    """
    return f"{RESERVED_PREFIX}{name}" if name in RESERVED_DEVICE_NAMES else name


# The internal type names use OpenCL's spellings, because one generator serves every backend.
# CUDA C does not define uchar/ushort/uint/ulong (NVRTC rejects them; char/short/int/long
# and the full unsigned forms are fine), so the CUDA path writes those out in full. HIP
# compiles through CUDA generation too and its clang accepts the same full spellings.
_CUDA_TYPE_NAMES:Dict[str, str] = {
    "uchar": "unsigned char",
    "ushort": "unsigned short",
    "uint": "unsigned int",
    # `long` is 32-bit in CUDA C on Windows (where NVRTC and HIP are used here), so the
    # 64-bit spelling has to be written out: leaving it as `long` made an int64 array a
    # 4-byte-stride buffer, which silently produced wrong values for every element.
    "long": "long long",
    "ulong": "unsigned long long",
}

# An image parameter is ("image", channels, scalar type): its storage is opaque to
# the kernel, which only ever samples or stores through it.
IMAGE = "image"

# The type of a translated expression. A device type is one of a closed set of shapes:
#
# * a **scalar or vector name** -- `"int"`, `"float3"`, `"quatf"`, a struct name;
# * a **pointer** -- `("*", base)`, an array parameter;
# * an **image** -- `("image", channels, scalar)`, opaque storage the body samples;
# * a **shared slice** -- `("shared_slice", ...)`, one group's zeros_per_group buffer;
#
# `is_ptr` / `is_vec` / `is_image` / `is_int` decide which, so the union says exactly what
# those predicates accept. It used to be spelled `Any` at every one of these sites, which
# threw away the one piece of type information the translator carries around most.
DeviceType = Union[str, Tuple[object, ...]]

# A translated expression is its C text plus the type above -- the pair every `_visit_*`
# returns and every emitter consumes.
Translated = Tuple[str, DeviceType]

# One value slot a `return` fills: the C type the launch declares for it, and the device
# type the body produced. Two return sites whose values share a C type share the slot.
class ReturnLayoutEntry(TypedDict):
    ctype: str
    type: DeviceType

# One position of one `return` statement, as `_visit_kernel_return` sees it. A `value`
# position travels as a one-element buffer (`code` is how the body reads it back) and an
# `array` position travels as the array parameter itself, named by `name`. `ordinal` and
# `slot` appear once the slot map has assigned them.
class ReturnPosition(TypedDict, total=False):
    kind: str
    type: DeviceType
    code: str
    name: str
    ordinal: int
    slot: int
    slot_parameter: str

# One entry of `return_site_layout`: the positions a site hands back, in return order, and
# whether it returned a tuple (which the host unpacks rather than passes through).
class ReturnSiteLayout(TypedDict):
    positions: List[ReturnSitePosition]
    is_tuple: bool

# One host-facing rebuild entry: a `value` names the slot it reads (or, when the value
# already lives in a parameter, the parameter itself through `reused`), and an `array`
# names the parameter to hand back.
class ReturnSitePosition(TypedDict, total=False):
    kind: str
    ordinal: int
    slot: int
    name: str
    reused: str

# An outer-scope name the body reads: the device type it travels as, whether the body
# writes it (which makes it a one-element buffer), and how the host reaches it -- a
# closure cell, the kernel's module globals, or a helper's own module (`scope`).
class OuterVarEntry(TypedDict, total=False):
    type: DeviceType
    writable: bool
    kind: str
    scope: Optional[types.ModuleType]

# An object the body creates above the loop and the host therefore builds: the device
# type, plus the expression the host evaluates to make it.
class HostValueEntry(TypedDict):
    expr: str
    type: DeviceType

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

_C_FIELD_TYPES:Dict[object, str] = {}

# Declared here (and filled on first use) because importing the driver tables eagerly would
# load a GPU runtime at `import pygpgpu` time.
_VEC_REGISTRY:Dict[str, object]

def _vec_registry()->Dict[str, object]:
    """vec type name -> ctypes class (e.g. float4), built lazily."""
    global _VEC_REGISTRY
    try:
        return _VEC_REGISTRY
    except NameError:
        pass
    from ..opencl.driver import CLInfo
    from . import structs

    _VEC_REGISTRY = dict(CLInfo.vec_types)
    for name, cls in _VEC_REGISTRY.items():
        for field_name, field_type in cls._fields_:
            _C_FIELD_TYPES[(name, field_name)] = _c_scalar_name(field_type)
        # A vector is a `ctypes.Structure` with an aligned dtype, which is the evidence
        # `is_declared_structure` uses for a structure; claiming the dtype first keeps the
        # vector reading when a caller's class happens to have the same layout.
        try:
            structs.claim_vector_dtype(np.dtype(cls._fields_))
        except Exception:
            pass
    return _VEC_REGISTRY

def _c_scalar_name(ctypes_type:object)->str:
    import ctypes as _ctypes
    sizes = {_ctypes.c_float: "float", _ctypes.c_double: "double",
             _ctypes.c_int8: "char", _ctypes.c_int16: "short", _ctypes.c_int32: "int",
             _ctypes.c_int64: "long", _ctypes.c_uint8: "uchar", _ctypes.c_uint16: "ushort",
             _ctypes.c_uint32: "uint", _ctypes.c_uint64: "ulong"}
    return sizes.get(ctypes_type, "float")

def is_vec(t:object)->bool:
    return isinstance(t, str) and t in _vec_registry()

# A 3-component vector is padded to 16 bytes for the device ABI, so its ctypes struct carries
# a trailing `_` field that is not a component. Everything that names components counts them
# through this helper, so the padding never reaches a constructor, a swizzle or a print.
def _vec_components(vec:str)->List[Translated]:
    """The real (field name, ctypes type) components of a vector type, in order."""
    return [field for field in _vec_registry()[vec]._fields_ if field[0] != "_"]

def _vec_field_names(cls_or_name:object)->List[str]:
    """The component names of a vector type, given the class or its name.

    Everything that expands a vector into per-component code goes through this, so the
    padding field of a 3-component vector never becomes a member the device does not have.
    """
    fields:Tuple[Tuple[str, type], ...] = (cls_or_name._fields_
                  if isinstance(cls_or_name, type)
                  else _vec_registry()[cls_or_name]._fields_)
    return [name for name, _ in fields if name != "_"]

# Colour-style component names, as in `color.rgb`. OpenCL C accepts them natively but CUDA's
# vector types only have x/y/z/w, so the translator normalizes them for every backend.
_VEC_COLOR_ALIASES:Dict[str, str] = {"r": "x", "g": "y", "b": "z", "a": "w"}

def _math_device_name(attr:str, backend:str, result_type:object=None)->str:
    """The device spelling of a `math.*` function.

    OpenCL's C dialect has float overloads under the plain name. CUDA has both the
    double and the `f`-suffixed float form, and asking for the plain name with a float
    argument is ambiguous whenever the argument's type is not exactly `float` -- which is
    how `floor` came to fail with "more than one instance of overloaded function".

    The caller's *result* type decides which of the two CUDA forms to spell: `ret` is
    already computed from the argument types (`_emit_math_call` widens it to `double`
    when any argument is one), so a double-width call gets the plain name and the
    compiler resolves the double overload. Spelling `sinf` with a double argument
    silently cut a `float=np.float64` kernel's transcendentals back to float32.
    """
    if backend != "cuda":
        return attr
    # The three predicates are already type-generic in CUDA and take no `f` suffix.
    if attr in ("isnan", "isinf", "isfinite"):
        return attr
    if result_type == "double":
        return attr
    return attr if attr.endswith("f") else attr + "f"


# The name of the compile-time switch (see `builtins.KERNEL_TRANSLATING`), and the value a
# translation sees. Python resolves the same name to False, which is what lets one body run
# differently on each side without the translator guessing at it.
KERNEL_TRANSLATING_NAME:str = "KERNEL_TRANSLATING"
KERNEL_TRANSLATING_VALUE:bool = True


def _translation_branch(node:ast.expr, body:List[ast.stmt],
                        orelse:List[ast.stmt])->Optional[List[ast.stmt]]:
    """The branch a translation keeps for an ``if`` about ``KERNEL_TRANSLATING``.

    Returns None when the condition says nothing about translation (an ordinary device
    ``if``), and otherwise the statements the device side should contain -- an empty list
    when the condition is false, so the whole ``if`` disappears.
    """
    truth:Optional[bool] = _translation_truth(node)
    if truth is None:
        return None
    return list(body) if truth else list(orelse)


def _translation_truth(node:ast.expr)->Optional[bool]:
    """The compile-time truth of a condition about `KERNEL_TRANSLATING`, or None.

    None means the condition says nothing about translation, so it is left to the device.

    **The trigger is the name, not the object it resolves to.** A body that declares its own
    `KERNEL_TRANSLATING` -- a module global, a local, anything that reads as that name --
    gets the same conditional translation, because a kernel author writing that name is
    asking for exactly this and requiring an import would only make the spelling longer. It
    is also why this works on source text rather than on a resolved value: the translator
    never has to import anything to decide the branch.

    The constant is False in python and True while translating, so a condition is decidable
    when it is built from that name with `not`, `and`, `or` and boolean literals --
    each of which python evaluates with short-circuiting. `if KERNEL_TRANSLATING:` is
    therefore the device branch and `if not KERNEL_TRANSLATING:` is python's.
    """
    if isinstance(node, ast.Name) and node.id == KERNEL_TRANSLATING_NAME:
        return KERNEL_TRANSLATING_VALUE
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        # A boolean literal says the same thing on both sides, so it is decided here and a
        # branch around it collapses (KERNEL_TRANSLATING and True is just the constant).
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner:Optional[bool] = _translation_truth(node.operand)
        return None if inner is None else not inner
    if isinstance(node, ast.BoolOp):
        values:List[Optional[bool]] = [_translation_truth(value) for value in node.values]
        if any(value is None for value in values):
            return None
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        return None
    return None


def _vec_component_name(text:str)->str:
    """`rgb` -> `xyz`: the device spelling of a component or swizzle."""
    return "".join(_VEC_COLOR_ALIASES.get(char, char) for char in text)


_MATH_FUNCS:Dict[str, Optional[str]] = {
    "sqrt": None, "exp": None, "log": None, "log2": None, "log10": None,
    "sin": None, "cos": None, "tan": None, "asin": None, "acos": None,
    "atan": None, "atan2": None, "sinh": None, "cosh": None, "tanh": None,
    "fabs": None, "pow": None, "hypot": None, "erf": None,
    "log1p": None, "expm1": None, "fmod": None, "cbrt": None,
    "copysign": None, "ldexp": None, "degrees": None, "radians": None,
    # python returns an int here; the *argument* stays a float (see _emit_math_call, where
    # casting the argument made `ceil(2.1)` answer 2).
    "floor": "int", "ceil": "int", "trunc": "int",
    # python returns a bool; the device answers 0/1, which is the same number.
    "isnan": "int", "isinf": "int", "isfinite": "int",
}

_MATH_CONSTS:Dict[str, str] = {
    "pi": "3.14159265358979323846f",
    "e": "2.71828182845904523536f",
    "tau": "6.28318530717958647692f",
    # A literal rather than the INFINITY macro: NVRTC's default headers do not define it,
    # so the CUDA translation unit failed with "identifier INFINITY is undefined". A float
    # literal is the same value on every backend and needs no header.
    "inf": "(1.0f / 0.0f)",
}

# ---------- the complex vocabulary ----------

# Which of the math vocabulary's names map to which of the complex header's functions, which
# answer a *real* (the magnitude and the phase), and which have no complex meaning at all.
# numpy is the authority here: it defines exactly this split, the interpreter backend runs
# numpy, and the device has to agree with it rather than inventing an answer.
COMPLEX_DEVICE_FUNCTIONS:Dict[str, str] = {
    "sqrt": "sqrt", "exp": "exp", "log": "log", "log2": "log2", "log10": "log10",
    "sin": "sin", "cos": "cos", "tan": "tan",
    # numpy spells the inverse trig `arcsin`/`arccos`/`arctan`, and its `np.sin`-shaped
    # aliases (`np.asin`, ...) exist too; both spellings mean the same function here.
    "asin": "asin", "acos": "acos", "atan": "atan",
    "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "square": "square", "reciprocal": "reciprocal", "conjugate": "conjugate",
    "conj": "conjugate", "exp2": "exp2",
    # `rint` answers a complex value in numpy (each part rounded to even), which is the same
    # rule the real path spells `rint`; `round` is a builtin and is refused for a complex
    # value (see COMPLEX_BUILTIN_REFUSALS), so a kernel writes `np.round(z)` for one.
    "rint": "rint",
}
# A complex argument answering a *real*: numpy's own rule (`abs` is the magnitude, `np.angle`
# the phase). An empty spelling means the plain real function of the magnitude.
COMPLEX_ABS_FUNCTIONS:Dict[str, str] = {
    "absolute": "abs", "abs": "abs", "angle": "angle", "sign": "",
}
# The real-only names, which numpy refuses for a complex input -- with the complex spelling
# that is the right counterpart, or the fact that there is none.
COMPLEX_UNSUPPORTED_FUNCTIONS:Dict[str, str] = {
    "floor": "there is none: a complex value has no ordering",
    "ceil": "there is none: a complex value has no ordering",
    "trunc": "there is none: a complex value has no ordering",
    "fmod": "use `a - b * floor(a / b)`, or np.mod for a real pair",
    "hypot": "use `np.abs(a - b)` on the difference",
    "cbrt": "use `np.pow(z, 1 / 3)`",
    "ldexp": "use `z * 2 ** exponent`",
    "atan2": "use `np.angle(a / b)`, or `np.atan(a / b)`",
}

# The *builtin* spellings -- pygf's functions, re-exported from the package top level -- that
# are a complex operation, and the header function each one calls. The interpreter answers
# these for a complex value (they are pygf's functions, which route a complex scalar through
# `cmath`), so the device answers them too: a kernel and its serial run have to agree. The
# table exists because the builtin tables are keyed by *type* and a complex value has no entry
# in them, which is how `sqrtf(z)` and `powf(z, w)` used to be emitted.
COMPLEX_BUILTIN_FUNCTIONS:Dict[str, str] = {
    "sqrt": "sqrt", "exp": "exp", "exp2": "exp2", "log": "log", "log2": "log2",
    "log10": "log10", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "conjugate": "conjugate", "conj": "conjugate", "pow": "pow",
    # `sign` answers a *complex* value (the direction on the unit circle), unlike `abs`,
    # which answers the magnitude -- so it belongs here and not beside `abs`.
    "sign": "sign",
    # `rint` rounds each part to even. This package's own host body answers that now
    # (sci_math rounds the two parts with python's `round`), which is what makes the device
    # allowed to: python's builtin `round` still refuses a complex value, and that is a
    # different name.
    "rint": "rint",
}

# The builtin names that answer a *real* for a complex argument: the magnitude, which is
# numpy's own rule for it.
COMPLEX_BUILTIN_REAL_RESULTS:Dict[str, str] = {"abs": "abs"}

# The builtin names numpy refuses for a complex value, and what to write instead. The
# interpreter refuses them as well -- they reach `math.floor` and friends through pygf -- so
# saying so is the only honest answer, and it keeps the device from emitting a call the
# compiler cannot type. `round` is here rather than in the table above because *python's*
# builtin refuses a complex value; numpy's `np.round` is the spelling that answers one.
COMPLEX_BUILTIN_REFUSALS:Dict[str, str] = {
    "floor": "there is none: a complex value has no ordering",
    "ceil": "there is none: a complex value has no ordering",
    "trunc": "there is none: a complex value has no ordering",
    "round": "use `np.round(z)`: python's round() refuses a complex value",
    "fract": "use `z - np.floor(z)`, which is a real-pair operation",
    "fmod": "use `a - b * np.floor(a / b)`, or np.mod for a real pair",
    "remainder": "there is none: numpy has no complex form for it",
    "rsqrt": "use `1 / np.sqrt(z)`",
    "cbrt": "use `np.power(z, 1 / 3)`",
    "ldexp": "use `z * 2 ** exponent`",
    "step": "there is none: a complex value has no ordering",
    "smoothstep": "there is none: a complex value has no ordering",
    "fdim": "there is none: numpy has no complex form for it",
    "copysign": "there is none: numpy has no complex form for it",
    "nextafter": "there is none: numpy has no complex form for it",
    "degrees": "there is none: numpy has no complex form for it",
    "radians": "there is none: numpy has no complex form for it",
    "hypot": "use `np.abs(a - b)` on the difference",
}

# ---------- print() support ----------

# print() is a host builtin, so the name means nothing on a device. The translator rewrites
# a print(...) statement into device printf calls that produce the same text: arguments
# separated by one space, a trailing newline, and every value rendered as str() renders it
# (True/False, decimal integers, python's shortest round-trip floats, float4(...) vectors).
#
# Two device facts shape the generated code:
#   * printf's format string must be a literal on OpenCL (a computed one is rejected);
#   * each printf call site has its own output buffer, and the buffers are drained call
#     site by call site, so a line split over several printf calls comes out reordered.
# A line is therefore emitted as one printf per leaf of its branches, with the whole line
# text repeated in each leaf, so all of its parts stay together.

# printf specifier and argument cast per integer type. Two things differ between the
# backends: what a `long` is (64-bit in OpenCL C, but only 32-bit in CUDA/HIP on Windows)
# and which 64-bit conversions the device printf implements (AMD's OpenCL printf refuses
# "%lld", so OpenCL uses "%ld" with a long cast, which is exactly as wide there).
PRINT_INT_FORMATS:Dict[str, Dict[str, str]] = {
    "opencl": {
        "char": "d", "short": "d", "int": "d",
        "uchar": "u", "ushort": "u", "uint": "u",
        "long": "ld", "ulong": "lu",
    },
    "cuda": {
        "char": "d", "short": "d", "int": "d",
        "uchar": "u", "ushort": "u", "uint": "u",
        "long": "lld", "ulong": "llu",
    },
}

PRINT_INT_CASTS:Dict[str, Dict[str, str]] = {
    "opencl": {
        "char": "int", "short": "int", "int": "int",
        "uchar": "unsigned int", "ushort": "unsigned int", "uint": "unsigned int",
        "long": "long", "ulong": "unsigned long",
    },
    "cuda": {
        "char": "int", "short": "int", "int": "int",
        "uchar": "unsigned int", "ushort": "unsigned int", "uint": "unsigned int",
        "long": "long long", "ulong": "unsigned long long",
    },
}

# Python prints a float with the fewest significant digits that still read back as the same
# value. numpy's str() of a float32 switches to the exponent form from 1e6 up, and python's
# repr of a double from 1e16 up; both switch down to it below 1e-4 (0.0001 stays fixed,
# 0.00001 becomes 1e-05, i.e. from decimal exponent -5).
PRINT_REAL_SCI_HI:Dict[bool, int] = {True: 6, False: 16}
PRINT_REAL_SCI_LO:Dict[bool, int] = {True: -5, False: -5}

# Significant digits to try for a real: 9 round-trips every float32, 17 every double.
PRINT_REAL_DIGITS:Dict[bool, int] = {True: 9, False: 17}

# The largest magnitude the exponent form is needed for, per precision: a float32 that big
# is already close to its range, a double's is the top of its range.
PRINT_REAL_BIG:Dict[bool, str] = {True: "3.4028235e38", False: "1.7976931348623157e308"}

# Whole values below this magnitude print with a trailing ".0" rather than in the exponent
# form, matching python: numpy's str() of a float32 switches to the exponent form at 1e6, so
# a whole value from there up is written "1e+06" rather than "1000000.0".
PRINT_REAL_PLAIN_MAX:Dict[bool, str] = {True: "1e6", False: "1e16"}

# The five values python spells out rather than through a conversion: nan, inf, -inf, 0.0
# and -0.0. They own the first leaves of a real's layout (see _print_real_conversion).
PRINT_REAL_SPECIAL_COUNT:int = 5

# How many printf leaves one print() may expand into. Each bool argument doubles the count
# and each real multiplies it by its digit budget, so the cap keeps a single line from
# growing without bound. Ordinary lines stay far below it; one that trips it should be
# split into several print() calls. Keep this small: a huge switch of printf call sites is
# also a lot of device code for a debug facility, and some drivers handle it badly.
PRINT_MAX_LINES:int = 512


def is_int(t:object)->bool:
    return isinstance(t, str) and t in INT_TYPES

def is_float(t:object)->bool:
    return isinstance(t, str) and t in FLOAT_TYPES

def is_complex(t:object)->bool:
    """True for a device type that is a complex value (`c64` / `c128`).

    A complex value is a *value* like a scalar -- it is passed by value, returned in a slot,
    and stored in a local -- but its arithmetic is generated rather than a C operator, and
    its two members are read through the header's accessors. So the predicates that ask
    "can this take part in an ordinary scalar operator" must answer False for it, and the
    predicates that ask "is this a pointer/aggregate" must answer False as well.
    """
    return isinstance(t, str) and t in COMPLEX_TYPE_NAMES

def is_aggregate(t:object)->bool:
    """True for a value the device carries as a *structure* (`c64`/`c128` included).

    The difference from `structs_dtype_for` is that this answers from the type name alone,
    without the registry: a complex type is a structure on the device even though it is not
    a numpy record dtype here.
    """
    return isinstance(t, str) and (t in COMPLEX_TYPE_NAMES
                                   or structs_dtype_for(t) is not None)

def is_ptr(t:object)->bool:
    return isinstance(t, tuple) and len(t) == 2 and t[0] == PTR

def is_image(t:object)->bool:
    """True for the marker a backend-neutral Image argument gets."""
    return isinstance(t, tuple) and len(t) == 3 and t[0] == IMAGE

def image_channels(t:object)->int:
    return int(t[1])

def image_scalar(t:object)->str:
    """The C scalar type of one image channel."""
    return t[2]

def promote(t1:DeviceType, t2:DeviceType)->DeviceType:
    if is_ptr(t1) or is_ptr(t2):
        raise SyntaxError("cannot mix arrays with scalars in arithmetic") from None
    if is_vec(t1) or is_vec(t2):
        raise SyntaxError("vector arithmetic is handled separately") from None
    if "double" in (t1, t2):
        return "double"
    if is_float(t1) or is_float(t2):
        return "float"
    rank:List[str] = ["bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"]
    if t1 in rank and t2 in rank:
        return t1 if rank.index(t1) >= rank.index(t2) else t2
    return "int"

def infer_ctype(value:object, type_map:Optional[Dict[str, str]]=None)->DeviceType:
    """Actual call argument -> C type.

    `type_map` is the caller's `int=`/`float=` override (already normalized to device type
    names by `validate_type_map`): a python `int`/`float` has no width of its own, so the
    decorator chooses one here. It applies to **nothing else** -- a numpy scalar names its own
    width and keeps it, which is what makes the mapping an override of the two builtins
    rather than a second, competing type system.
    """
    from ..vec_types.layout import _vector_dtype_name
    from . import structs

    registry:Dict[str, object] = _vec_registry()
    mapping:Dict[str, str] = type_map or {}
    int_type:str = mapping.get("int", "int")
    float_type:str = mapping.get("float", "float")

    # A backend-neutral Image is opaque storage: the kernel only samples or stores
    # through it, so the parameter carries its channel count and scalar type.
    from ..image import Image
    if isinstance(value, Image):
        channel_type = DTYPE_MAP.get(value.dtype.name)
        if channel_type is None:
            raise SyntaxError(
                "Image of dtype %s cannot be sampled: unsupported element type"
                % value.dtype.name) from None
        return (IMAGE, value.channels, channel_type)

    # A `ctypes.Structure` the caller declared is a structure value, before any of the
    # vector lookups: its dtype compares equal to a vector of the same layout
    # (`{float x; float y;}` *is* `float2`), and only the dtype object's alignment flag says
    # which one the caller meant.
    declared:Optional[np.dtype] = structs.structure_dtype_of(value)
    if declared is not None and not isinstance(value, np.ndarray):
        from ..vec_types.layout import _vector_dtype_name, _host_classes

        declared_class:type = type(value)
        is_vector_value:bool = (
            _vector_dtype_name(declared) is not None
            and (declared_class in _host_classes().values()
                 or any(issubclass(declared_class, host)
                        for host in _host_classes().values())))
        if is_vector_value:
            # A vector type is a ctypes.Structure with a vector layout: it travels by
            # value, because the device reads its components directly. A caller-declared
            # class with a vector-identical layout (Vec2Like) is NOT one of the package's
            # vectors, so it keeps the structure treatment.
            return structs.register(declared)
        if isinstance(value, type):
            # The class itself (a spelling for the layout, not a value): it names the
            # structure, and travels by value like any type.
            return structs.register(declared)
        # A structure argument always travels as a pointer: copying a struct per launch is
        # work the device never needs, a write through it reaches the caller's object, and
        # one spelling covers both read-only (the signature adds `const`) and written.
        return (PTR, structs.register(declared))

    if isinstance(value, np.void):
        # A structured scalar is either a vector element or a structure value: both are
        # described completely by the dtype, so both become a device type.
        if structs.is_declared_structure(value.dtype):
            # An element of a ctypes-declared array reaches here as a record, and the
            # declaration wins over the coincidental vector layout.
            return (PTR, structs.register(value.dtype))
        vec_name:Optional[str] = _vector_dtype_name(value.dtype)
        if vec_name is not None:
            return vec_name
        if structs.is_record(value.dtype):
            return (PTR, structs.register(value.dtype))
        raise SyntaxError(f"unsupported numpy structured scalar: {value.dtype}") from None

    for vec_name, cls in registry.items():
        if isinstance(value, cls):
            return vec_name

    # A vector instance from the host library (`pygf`) is not one of the driver's classes,
    # but its bases name the same device type: `point3f` -> `float3`, `color3d` -> `double3`.
    # A numpy scalar is never a vector, even where the class name collides: `np.int8` is
    # `numpy.int8`, while the registry has an eight-component `int8` vector, and a scalar
    # reaching that lookup became a vector parameter the device could not bind.
    if not isinstance(value, np.generic):
        for base in type(value).__mro__:
            if base.__name__ in registry:
                return base.__name__

    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None:
            if structs.is_declared_structure(value.dtype):
                # An array of a caller's `ctypes.Structure`: structured, whatever layout.
                return (PTR, structs.register(value.dtype))
            vec_name = _vector_dtype_name(value.dtype)
            if vec_name is not None:
                return (PTR, vec_name)
            if structs.is_record(value.dtype):
                return (PTR, structs.register(value.dtype))
            raise SyntaxError(f"unsupported numpy structured dtype: {value.dtype}") from None
        if value.dtype.name not in DTYPE_MAP:
            raise SyntaxError(f"unsupported numpy dtype: {value.dtype.name}") from None
        return (PTR, DTYPE_MAP[value.dtype.name])
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    # A numpy scalar carries its own width and signedness, and `np.uint32(x)` is how a
    # caller says "unsigned 32-bit" without an annotation the translator would ignore.
    # Collapsing every integer to `int` made an unsigned mix (`x * 0x9E3779B9`, `x >> 16`)
    # signed, so the multiply overflowed and the shift filled with ones.
    #
    # The narrow widths keep their old answer: an 8- or 16-bit numpy scalar is `int` on the
    # device, as it always was (see tests/test_kernel_narrow_arguments.py), and a float64
    # is `float`. The widths a caller reaches for *deliberately* -- `np.uint32` for an
    # unsigned mix, `np.int64`/`np.uint64` for a wide counter -- are kept as they are,
    # because there would be no reason to spell them otherwise. A python `int` has no
    # width of its own and stays `int`.
    if isinstance(value, np.integer):
        dtype_name:str = np.dtype(type(value)).name
        if dtype_name in NARROWED_SCALARS:
            return NARROWED_SCALARS[dtype_name]
        if dtype_name in DTYPE_MAP:
            return DTYPE_MAP[dtype_name]
        return int_type
    if isinstance(value, int):
        _check_integer_fits(value, int_type)
        return int_type
    if isinstance(value, np.floating):
        dtype_name = np.dtype(type(value)).name
        if dtype_name in NARROWED_SCALARS:
            return NARROWED_SCALARS[dtype_name]
        if dtype_name in DTYPE_MAP:
            return DTYPE_MAP[dtype_name]
        return float_type
    if isinstance(value, float):
        return float_type
    # A complex scalar. `np.complex64` and `np.complex128` are numpy scalars (they are
    # `np.complexfloating`, not `np.floating`), and a python `complex` has no width of its
    # own -- the same rule a python `float` follows, which is why the two widths are chosen
    # together rather than one of them staying fixed: a kernel that maps `float` to a double
    # is asking for 64-bit real arithmetic, and a python complex beside it should not be the
    # single value in the body that silently lost half its precision. The default is the
    # other way round and just as important: a python float travels as `float`, so a python
    # complex beside it travels as `c64`. A python complex is never one of the numpy scalars,
    # so `isinstance(complex(1, 2), complex)` is checked on its own.
    if isinstance(value, np.complexfloating):
        dtype_name = np.dtype(type(value)).name
        return DTYPE_MAP.get(dtype_name, "c128")
    if isinstance(value, complex):
        return python_complex_type(float_type)
    if _is_device_handle(value):
        # A texture or surface reaches the device as an opaque 64-bit handle, which
        # the kernel reads through tex2D or writes through surf2Dwrite.
        return _handle_device_type(value)
    raise SyntaxError(f"unsupported argument type: {type(value).__name__} (only numpy arrays, int/float/bool/complex scalars and texture handles are supported)") from None


TEXTURE_HANDLE_TYPES = ("TextureObject", "cudaTextureObject_t")
SURFACE_HANDLE_TYPES = ("SurfaceObject", "cudaSurfaceObject_t")


def _is_device_handle(value:object)->bool:
    """True for the objects a kernel receives as an opaque texture or surface handle."""
    return type(value).__name__ in TEXTURE_HANDLE_TYPES + SURFACE_HANDLE_TYPES


def _handle_device_type(value:object)->str:
    """The device-side type name a handle argument takes in a kernel signature."""
    if type(value).__name__ in SURFACE_HANDLE_TYPES:
        return "cudaSurfaceObject_t"
    return "cudaTextureObject_t"

# numpy names a kernel body may use as a type in a creation statement (`dtype=np.float32`).
_NUMPY_DEVICE_TYPES:Set[str] = {
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_", "complex64", "complex128",
}

# numpy functions that build or transform host data, and can therefore be executed by the
# interpreter instead of translated (see _creation_assignments). This is the whole family
# of array constructors and shape/layout transforms: any of them may appear on the
# right-hand side of a statement above the magic loop. Device-side arithmetic on the
# result is not part of a creation statement, which is why only these names are listed --
# a `np.sum(...)` or `np.dot(...)` there is a computation the kernel itself should do.
_HOST_CREATION_FUNCTIONS:Set[str] = {
    # constructors
    "zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like",
    "full_like", "arange", "linspace", "logspace", "geomspace",
    "identity", "eye", "tri", "tril", "triu", "diag", "diagflat", "vander",
    "indices", "meshgrid", "fromfunction", "frombuffer", "fromiter", "fromfile",
    "array", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray",
    "copy", "copyto", "filled",
    # shape and layout transforms
    "reshape", "ravel", "flatten", "transpose", "swapaxes", "moveaxis",
    "flip", "fliplr", "flipud", "roll", "rot90", "squeeze", "expand_dims",
    "broadcast_to", "broadcast_arrays", "atleast_1d", "atleast_2d", "atleast_3d",
    "repeat", "tile", "resize", "pad", "take", "compress", "trim_zeros",
    # combining and splitting
    "concatenate", "stack", "hstack", "vstack", "dstack", "column_stack",
    "row_stack", "block", "split", "hsplit", "vsplit", "dsplit", "append",
    "insert", "delete", "unique",
    # element-wise or ordering helpers that only reshape the data
    "where", "clip", "round", "around", "sort", "argsort", "lexsort", "cumsum",
    "cumprod", "nan_to_num", "real", "imag", "conj", "angle",
    # dtype constructors (np.float32(x) and friends)
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_",
}

# The statements a kernel body may execute on the host: assignments above the magic loop
# whose right-hand side only builds data.
def _magic_loop_end(statements:List[ast.stmt])->int:
    """The index just past the statements that belong to the magic loop's body.

    A phase is `[header..., for-loop, after...]`, so everything from the loop onwards except
    its own body is the *gap* before the next phase -- the statements a body places between
    two magic loops.
    """
    for index, stmt in enumerate(statements):
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt):
            return index + 1
    return len(statements)


def _outside_loops(body:List[ast.stmt])->List[ast.stmt]:
    """The top-level statements a body runs on the host, outside its work-item dispatch.

    A single-loop body places all of them before the loop, but a body with several loops
    may also place one *between* two of them (`mid = np.zeros(4)` in a two-pass body), and
    the pass that reads such a value needs it bound on the host just the same.

    Only statements whose meaning is *host* work belong here. A serial `for` loop and an
    `if` block are per-work-item code that the device must run (every work item iterates
    its own slice, or tests its own lane), so they are left to the translator; returning
    them to the host would execute device-only builtins there -- `zeros_per_group` used
    under such a loop reported "only available inside a @kernel function body" for a body
    that plainly was one.
    """
    loops:List[int] = [
        index for index, stmt in enumerate(body)
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt)]
    # A statement is host work only if the host can *execute* it. The body's serial loops and
    # its `if` blocks are per-work-item code: every work item iterates its own slice or tests
    # its own lane, so handing them to the host would run device-only builtins there --
    # `zeros_per_group` under such a loop reported "only available inside a @kernel function
    # body" for a body that plainly was one. A bare `for` / `If` is therefore never host code,
    # with or without a dispatch loop; everything else (a creation, a plain assignment) is,
    # because its value has to exist before the launch.
    per_item = (ast.For, ast.Return, ast.If)
    if not loops:
        # A gid-style body: all of it is per-work-item code, except the creations the host
        # still has to build (see _creation_assignments), which stay in the list.
        return [stmt for stmt in body if not isinstance(stmt, per_item)]
    # Everything before the first loop, plus the gap between consecutive loops: those are
    # the statements a body places outside its work-item dispatch, and a value bound there
    # (a created array, a scalar) is read by a later pass. Statements *after* the last loop
    # are excluded -- `return out` lives there and must keep its device meaning.
    outside:List[ast.stmt] = [stmt for stmt in body[:loops[0]]
                              if not isinstance(stmt, per_item)]
    for start, end in zip(loops, loops[1:]):
        outside.extend(stmt for stmt in body[start + 1:end]
                       if not isinstance(stmt, per_item))
    return outside


def _creation_assignments(func:Callable)->List[ast.stmt]:
    """The statements the host runs before the launch, in body order.

    Two kinds are lifted out, both from **above** the magic loop:

    * a **creation** (`arr = np.zeros(n)`) -- the object has to exist with the shape and
      dtype numpy gives it, and a device has no way to make one;
    * a plain **assignment** that a later creation reads (`n = 5` feeding `np.zeros(n)`) --
      the host has to have the name in scope when it evaluates the creation.

    The second kind is *also* still emitted as device code, because a work item reads it
    too; the host evaluation is an extra, not a replacement. Everything else above the loop
    is device code only. A magic loop is what defines "above" here: an assignment below it
    would run once per work item, so it is not lifted out (the translator refuses a creation
    there).
    """
    source:str = get_dedented_source(func)
    tree:ast.Module = _parse_kernel_source(source)
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return []
    body:List[ast.stmt] = tree.body[0].body
    magic_index:Optional[int] = next(
        (index for index, stmt in enumerate(body) if _is_magic_loop_stmt(stmt)), None)
    # A gid-style body has no magic loop, so its whole body is "above the work item": it is
    # the per-work-item code, and a top-level creation statement there still runs once per
    # launch on the host, which is what python means.
    if magic_index is None:
        magic_index = len(body)

    import builtins as _builtins

    globalns:Dict[str, object] = getattr(func, "__globals__", {})

    def is_numpy(node:ast.expr)->bool:
        return (isinstance(node, ast.Name) and globalns.get(node.id) is np)

    def is_creation(node:ast.expr)->bool:
        return _is_creation_shape(node, is_numpy)

    del _builtins
    # Which plain assignments a creation needs: the names a creation statement reads that
    # are bound above the loop but are not themselves creations. Only those are lifted, so
    # the host does not run unrelated device code -- and everything lifted here is still
    # emitted for the work items as well.
    creations:List[ast.stmt] = [
        stmt for stmt in _outside_loops(body)
        if _creation_target_of(stmt) is not None and is_creation(_creation_target_of(stmt))]
    needed:set = set()
    for stmt in creations:
        for node in ast.walk(stmt.value):
            if isinstance(node, ast.Name):
                needed.add(node.id)

    bound_above:Dict[str, ast.stmt] = {}
    for stmt in _outside_loops(body):
        name:Optional[str] = _assigned_name_of(stmt)
        if name is not None:
            bound_above.setdefault(name, stmt)

    creation_names:set = {_assigned_name_of(stmt) for stmt in creations}
    # A fixed point, because a lifted assignment can depend on another lifted one:
    # `a = 2; b = a * 2; c = np.zeros(b)` needs all three, and one pass over the creation's
    # own names would stop at `b` and leave `a` undefined when the host runs it.
    fitting:set = set()
    changed:bool = True
    while changed:
        changed = False
        for name in list(needed):
            if name in fitting or name in creation_names:
                continue
            stmt:Optional[ast.stmt] = bound_above.get(name)
            if stmt is None or stmt not in _outside_loops(body):
                continue
            if not _is_host_evaluable(stmt.value if stmt.value is not None else stmt):
                continue
            fitting.add(name)
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and node.id not in fitting:
                    needed.add(node.id)
            changed = True

    chosen:List[ast.stmt] = []
    for stmt in _outside_loops(body):
        value:Optional[ast.expr] = _creation_target_of(stmt)
        name:Optional[str] = _assigned_name_of(stmt)
        if value is not None and is_creation(value):
            chosen.append(stmt)
        elif name is not None and name in fitting and name not in creation_names:
            chosen.append(stmt)
    return chosen


def _is_host_evaluable(node:ast.expr)->bool:
    """Whether a right-hand side can be computed by the interpreter, for a creation's need.

    Only the forms that carry no device-only meaning: a literal, a plain name, attribute
    access, and arithmetic over those. A call is excluded because it may name a device
    helper whose body the host cannot run -- lifting it would either fail or, worse, run
    something different from what the device computes.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, (ast.Name, ast.Attribute))
    if isinstance(node, ast.BinOp):
        return _is_host_evaluable(node.left) and _is_host_evaluable(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_host_evaluable(node.operand)
    if isinstance(node, ast.Subscript):
        return _is_host_evaluable(node.value) and _is_host_evaluable(node.slice)
    if isinstance(node, ast.Tuple):
        return all(_is_host_evaluable(element) for element in node.elts)
    return False


def _is_creation_shape(node:ast.expr, is_numpy:Callable[[ast.expr], bool])->bool:
    """True when this expression only *builds* data (and cannot run on the device).

    The device has no numpy, so a constructor that allocates is host-only by definition.
    A scalar-type call (`np.float32(...)`) is a creation only when its arguments are pure
    constants, because `np.float32(x[i])` is a device cast instead.
    """
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.Name):
        return False
    if isinstance(node, ast.Attribute):
        return is_numpy(node.value)
    if isinstance(node, ast.Call):
        func_node:ast.expr = node.func
        if not (isinstance(func_node, ast.Attribute) and is_numpy(func_node.value)):
            return False
        name:str = func_node.attr
        if name in _NUMPY_DEVICE_TYPES or name in FunctionTranslator.RETURN_SLOT_TYPES:
            return all(isinstance(arg, ast.Constant) for arg in node.args)
        if name not in _HOST_CREATION_FUNCTIONS and not name.endswith("_like"):
            return False
        # The arguments may be ordinary values (a shape, a count, an input array); what
        # makes the statement host-only is the numpy constructor itself. Only a dtype
        # argument has to be a type name, which is why it is not evaluated on the device.
        return True
    return False


def _is_creation_expression(node:ast.expr, func:Callable)->bool:
    """Whether a right-hand side is a creation, answering for one expression alone.

    The launcher needs this to tell a creation from a plain assignment it lifted for a
    creation's sake: only the creation travels to the device as a parameter.
    """
    globalns:Dict[str, object] = getattr(func, "__globals__", {})
    return _is_creation_shape(
        node, lambda candidate: isinstance(candidate, ast.Name)
        and globalns.get(candidate.id) is np)


def _creation_target_of(stmt:ast.stmt)->Optional[ast.expr]:
    """The right-hand side of `name = expr` / `name: T = expr`, when the target is a plain name.

    An annotated assignment is the same statement to python -- the annotation is only a
    hint for readers and type checkers -- so `x: np.ndarray = np.zeros_like(y)` creates an
    array exactly like `x = np.zeros_like(y)`. Recognising only the unannotated form made
    the annotated one fall through to the device code, which has no numpy.
    """
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            return None
        target:ast.expr = stmt.targets[0]
        return stmt.value if isinstance(target, ast.Name) else None
    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is None:
            return None  # `x: np.ndarray` declares nothing, so there is nothing to create
        return stmt.value if isinstance(stmt.target, ast.Name) else None
    return None


def _constant_int_of(node:ast.expr)->Optional[int]:
    """The integer an expression is a literal for, or None.

    `-1` reaches the translator as `UnaryOp(USub, Constant(1))`, not as a negative
    constant -- the parser does not fold it -- so a rule written against `ast.Constant`
    alone silently misses every negative literal. That is exactly where it matters: a
    negative index and a reversed slice are both written that way.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner:Optional[int] = _constant_int_of(node.operand)
        return None if inner is None else -inner
    return None


def _assigned_name_of(stmt:ast.stmt)->Optional[str]:
    """The plain name an assignment binds, for either assignment spelling."""
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            return None
        target:ast.expr = stmt.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(stmt, ast.AnnAssign):
        return stmt.target.id if isinstance(stmt.target, ast.Name) else None
    return None


def _magic_loop_index(body:List[ast.stmt])->int:
    """The index of the outermost loop that becomes work-item dispatch, or len(body).

    The same detection the translator uses, reduced to "where does the body's host region
    end": statements before it run once, everything from it on runs per work item.
    """
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt):
            return index
    return len(body)


def _is_shape_access(node:ast.expr)->bool:
    """True for `<X>.shape[0]`, the only bound a work-item dispatch may take.

    This is the same shape `_shape0_range_param` insists on when it decides the loop
    is the dispatch: a plain count, a start/step, or any other expression describes a
    *serial* loop the work item runs itself, and is not the iteration domain.
    """
    return (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute) and node.value.attr == "shape"
            and isinstance(node.slice, ast.Constant) and node.slice.value == 0)


def _is_magic_loop_stmt(stmt:ast.stmt)->bool:
    """True for a loop *shaped* like the work-item dispatch.

    The test has to be as strict as the dispatch itself, because its other callers
    (`_outside_loops`, `magic_loop_phases`, `_translate_phase_kernels`) use it to decide
    which statements are host code. `for j in range(lane * 4, lane * 4 + 4)` is an ordinary
    serial loop over four elements, but it used to count as the dispatch -- which pushed
    every statement above it into the host region, where a device-only builtin such as
    `zeros_per_group` cannot run (it reported "only available inside a @kernel function
    body" for a body that plainly was one). `_is_secondary_magic_loop` already told a
    literal `range(8)` apart from a real domain; this predicate has to agree with it.
    """
    if not isinstance(stmt, ast.For):
        return False
    iterator:ast.expr = stmt.iter
    if not isinstance(iterator, ast.Call):
        return False
    func:ast.expr = iterator.func
    if isinstance(func, ast.Name) and func.id == "range":
        # One argument, and it addresses an array's first dimension: the dispatch form.
        return (len(iterator.args) == 1 and not iterator.keywords
                and _is_shape_access(iterator.args[0]))
    if isinstance(func, ast.Name) and func.id == "enumerate":
        # `enumerate([1.0, 2.0])` is a dispatch too: the domain is a literal list, whose
        # length and elements are compile-time constants.
        if len(iterator.args) == 1 and not iterator.keywords:
            argument:ast.expr = iterator.args[0]
            if isinstance(argument, (ast.List, ast.Tuple)):
                return all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                           and not isinstance(e.value, bool) for e in argument.elts)
        return len(iterator.args) == 1 and not iterator.keywords
    if (isinstance(func, ast.Attribute) and func.attr in ("ndindex", "ndenumerate")
            and isinstance(func.value, ast.Name)):
        return True
    return False


_UNRESOLVED:object = object()

# Python builtins are never kernel parameters: they resolve through the translator's own
# dispatch (print, len, range, ...) or are rejected with a message of their own.
_BUILTIN_NAMES:Set[str] = set(dir(builtins))


def _declared_scope_names(tree:ast.AST)->Set[str]:
    """Names the function declares with `nonlocal`/`global`.

    Python requires such a declaration before a nested function may assign to an outer
    name, so the declarations are exactly the writable outer variables. Both statements
    are otherwise unsupported in a kernel body, which is why this scan happens before the
    body is visited instead of in the statement dispatch.
    """
    names:Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Nonlocal):
            names.update(node.names)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _outer_scope_names(func:object)->Set[str]:
    """The names this function's body declares `nonlocal`/`global`.

    Those travel through the outer scope's own device slot: the *device* writes them, so
    the host must not run the statement that assigns them (see `_host_region_values`).
    """
    try:
        tree:ast.Module = _parse_kernel_source(get_dedented_source(func))
    except SyntaxError:
        return set()
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return set()
    return _declared_scope_names(tree.body[0])


def type_sig(types:List[object])->str:
    parts:List[str] = []
    for t in types:
        parts.append((t[1] + "[]") if is_ptr(t) else str(t))
    return "_".join(parts) or "void"


def short_hash(text:str)->str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


class FunctionTranslator:
    """Translate a single Python function into a C function (kernel or device function).

    With is_kernel=True, the outermost `for i in range(<array>.shape[0])`
    loop of the function body is translated into the work-item dispatch.
    """

    # How many elements a private array created in the work-item body may spell out in its
    # initialiser; a longer one cannot be initialised without a loop, so it is refused.
    PRIVATE_ARRAY_MAX_UNROLL = 64

    # Element types a device atomic operation can address. Both backends provide 32-bit and
    # 64-bit integer atomics; int8/int16 have none, which is why a byte-sized histogram bin
    # is rejected rather than silently miscompiled.
    ATOMIC_ELEMENT_TYPES:Set[str] = {"int", "uint", "long", "ulong"}

    def __init__(self, func:Callable, param_types:List[object], backend:str,
                 resolve_global:Callable[[str], Any], device_cache:Dict[object, object],
                 is_kernel:bool=False, self_param:str="",
                 self_attr_types:Optional[Dict[str, object]]=None,
                 self_owner:Optional[type]=None, launch_backend:Optional[str]=None,
                 host_creation_values:Optional[Dict[str, object]]=None,
                 host_executed_lines:Optional[List[int]]=None,
                 self_struct_members:Optional[Dict[str, str]]=None,
                 self_struct_type:object=None,
                 struct_arg_classes:Optional[Dict[str, type]]=None,
                 receiver_as_argument:bool=False,
                 type_map:Optional[Dict[str, str]]=None):
        self.func:Callable = func
        self.filename:str = getattr(getattr(func, "__code__", None), "co_filename", "<kernel>")
        self.backend:str = backend
        # The backend the kernel is actually launched on. hip reuses cuda code generation, so
        # backend is "cuda" for a hip kernel; decisions that depend on the runtime rather than
        # on the generated syntax (for example whether device printf is safe) must use this.
        self.launch_backend:str = (launch_backend or backend)
        # The code-generation shape, as opposed to the runtime: "opencl" and "metal" share
        # the C-family emission (kernels, pointers into buffers, atomics, the same vector
        # types), so every site that asks "is this the OpenCL-looking syntax" tests this
        # rather than the backend name -- and a site that needs Metal's own spelling tests
        # `metal_syntax`. Two properties rather than a name comparison keep the two
        # questions apart: "does this backend spell a kernel `__kernel void`" is not the
        # same question as "is this backend Metal's attributes".
        self._opencl_syntax:bool = backend in ("opencl", "metal")
        self._metal_syntax:bool = backend == "metal"
        self._c_syntax:bool = backend in ("cuda", "hip")
        self.resolve_global:Callable[[str], Any] = resolve_global
        self.device_cache:Dict[object, object] = device_cache
        self.is_kernel:bool = is_kernel
        # The argument types the kernel was specialized for; kept so a phase translator can
        # be built with the same signature (see _translate_phase_kernels).
        self.param_types:List[object] = list(param_types)
        # Kernels launched from this kernel's device code (`inner[(n,)](args)`). Each entry is
        # one rendered `__global__` function, emitted ahead of the outer kernel because C
        # requires a callee before its caller.
        self.nested_kernels:List[str] = []
        # callee -> the name its definition was given, so two launches share one definition.
        self.nested_seen:Dict[Callable, str] = {}
        # Device helpers a nested kernel needs, emitted once for the whole source.
        self.nested_helpers:List[str] = []
        # Device helpers this kernel's own body needs, emitted once for the whole source.
        # A built-in that a backend spells with a helper (the round-toward-zero forms, which
        # HIP has no intrinsic for) contributes one of these; C requires the definition
        # before the kernel that calls it.
        self.device_helpers:List[str] = []
        # Set when this function's body reads a work-item index (`get_global_id`, or a
        # `blockIdx`/`threadIdx` name that resolves to one). Metal needs it to decide
        # whether a helper has to receive the six prologue parameters; every other
        # backend ignores the flag (see `_needs_prologue_parameters`).
        self._uses_index_builtins:bool = False
        # One entry per magic loop when a body has several; each becomes its own kernel.
        self.phases:List[List[ast.stmt]] = []
        # Locals initialized above the magic loop and written inside it: one location for the
        # whole launch, carried as a one-element buffer (see _bind_carried_scalars).
        self.carried_scalars:List[str] = []
        # Names a statement between two magic loops computed on the host: each travels to the
        # pass that reads it as its own parameter (see _translate_phase_kernels).
        self.gap_names:List[str] = []
        # The statements of the gap that follows this phase's loop (set by
        # _translate_phase_kernels before the body is visited).
        self.phase_gap:List[ast.stmt] = []
        # The statements after the last magic loop of a single-phase body: host code, run
        # once after the launch (see _translate_magic_loop and PyKernel._run_host_region).
        self.body_tail:List[ast.stmt] = []
        # The whole-body carried set the main translator computed, so a phase that writes
        # one of these names keeps the shared slot (see _bind_carried_scalars).
        self.phase_carried_hint:List[str] = []
        # By-value arguments the body returns, promoted to pointers so the return reuses the
        # parameter instead of allocating a slot for a copy (see _promote_returned_parameters).
        self.promoted_parameters:Set[str] = set()
        # The promoted parameters a return site lands in, so no slot is allocated for them.
        self.promoted_returns:Set[str] = set()
        # Local names that are only another name for an array (`src = a`). The device reads
        # the same pointer, so the alias asks for its target's extents (see _shape_var).
        self._array_pointer_sources:Dict[str, str] = {}
        # A local bound to a slice (`part = a[lo:hi]`) keeps the length here: it is a
        # pointer into the same buffer plus an offset, and `part.shape[0]` answers this.
        self._array_slice_extents:Dict[str, str] = {}
        # The array parameters a local was sliced from (`part = a[lo:hi]`). A slice is a
        # pointer at an offset, which matches python's view only on a 1-D array, so the
        # launch checks the rank of each of these (see PyKernel._compile's sliced_arrays).
        self.sliced_arrays:List[str] = []
        # `enumerate([1.0, 2.0, ...])`: a literal list is a compile-time table, so its
        # elements are recorded here and the dispatch reads them (see the enumerate branch).
        self._enumerate_constants:List[float] = []
        # A name bound to a literal list that a loop enumerates (`weights = [1.0, 2.0]`):
        # the loop reads the elements themselves, so the binding is not device code.
        self._constant_list_names:Set[str] = set()
        self._constant_lists:Dict[str, List[float]] = {}
        self.phase_translators:List["FunctionTranslator"] = []
        self.self_param:str = self_param
        self.self_attr_types:Dict[str, object] = dict(self_attr_types or {})
        # The class the translated method belongs to, for deciding whether `self.name`
        # is a data attribute, an instance method, or a staticmethod. None for the
        # decoration-time body scan, where the class does not exist yet.
        self.self_owner:Optional[type] = self_owner
        # The class each STRUCTURE argument was actually built from. One dtype can stand
        # for several classes -- two classes with one layout share it -- so the dtype alone
        # cannot say whose method `p.total` is. The launch knows: the caller handed over an
        # instance, and its type is recorded here under the parameter's name.
        self.struct_arg_classes:Dict[str, type] = dict(struct_arg_classes or {})
        # The receiver travels as one structure: member name -> the attribute it holds.
        # A member read of `self.<attr>` becomes a field read of the struct parameter, and
        # an ndarray attribute stays a separate `_self_<attr>` argument (the read/write
        # analysis and the shape plumbing need it apart).
        self.self_struct_members:Dict[str, str] = dict(self_struct_members or {})
        self.self_struct_type:object = self_struct_type
        # True when this function's receiver is its own first positional parameter
        # (`def at(self, t)` called as `ray.at(t)`), rather than a field-carrying struct
        # the caller packs separately (see _emit_device_call).
        self.receiver_as_argument:bool = receiver_as_argument
        # Objects the host already built for statements above the magic loop, keyed by the
        # name the body assigns them to (see _host_created_assignment).
        self.host_creation_values:Dict[str, object] = dict(host_creation_values or {})
        # The values every *host region* bound, created or not, keyed the same way, and the
        # line numbers of the statements the interpreter actually ran. A statement in that
        # set is host work -- it runs once, on the host, before the launch -- so the device
        # must not run it too (see _host_statement_lines and _bind_host_statement).
        self.host_region_values:Dict[str, object] = dict(host_creation_values or {})
        # The plain values a host-region statement bound, by name -> device type (see
        # _bind_host_statement). They are parameters like any other, but they are not
        # *creations*, so the launch must not treat them as objects.
        self.host_scalars:Dict[str, DeviceType] = {}
        self.host_statement_lines:Set[int] = set(host_executed_lines or [])
        # The width a python `int`/`float` travels as (device type names, already resolved).
        # Every type query the translator makes on a *host* value -- a creation above the
        # loop, a constant it lifts -- goes through it, so a kernel and the objects it builds
        # agree on the width.
        self.type_map:Dict[str, str] = dict(type_map or {})

        try:
            # The dedented copy starts at the decorator line, so remember where that was in
            # the user's file: error line numbers are reported against the file, not the copy.
            self.first_line:int = _source_first_line(func)
            self.source_text:str = _trimmed_source(func)
            tree:ast.Module = _parse_kernel_source(self.source_text)
            if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
                raise SyntaxError("decorated object must be a plain function") from None
            self.tree:ast.FunctionDef = tree.body[0]
        except SyntaxError as e:
            raise SyntaxError(f"cannot parse function source: {e}") from None

        self.params:List[Translated] = []
        self.shapes_used:List[Tuple[str, int]] = []
        # Structure parameters whose members the body assigns: a parameter in this set is
        # emitted without `const` (see render_kernel / render_device).
        self.member_writes:Set[str] = set()
        self.gid_param:Optional[str] = None
        self.work_size_dims:List[Tuple[str, int]] = []
        self.work_size_members:List[str] = []
        # One (array, dimension) source per dimension of the ndindex magic loop, in the
        # order the loop unpacks them (see _ndindex_extents). None for the other forms.
        self._nd_loop_extents:Optional[List[Tuple[str, int]]] = None
        # The array named by np.ndindex(arr.shape), whose rank is the loop's unpacking.
        self._nd_shape_holder:str = ""
        # name -> how many indices the body subscripts that array with (see
        # _record_subscript_indices); checked against the real shape at launch.
        self.subscript_indices:Dict[str, int] = {}
        # name -> the line of the widest such subscript, so the launch can blame that
        # statement rather than the loop when the count exceeds the array's rank.
        self.subscript_lines:Dict[str, int] = {}
        # name -> the column that subscript starts at, so the caret lands on the indexed
        # expression (`a[i, i]`), the way numpy's own IndexError points at it.
        self.subscript_columns:Dict[str, int] = {}
        # name -> the subscript as the user wrote it (`image[i, j, 2]`). The launch hands
        # it to the frame stand-in, which re-runs that very expression against an array of
        # the wrong rank so CPython draws its own caret instead of a synthetic one.
        self.subscript_texts:Dict[str, str] = {}
        # The magic loop's index variables, in unpack order. A subscript formed from
        # exactly these names addresses a different element per work item, which is what
        # makes a plain read-modify-write safe (see _augassign_is_race_free).
        self.work_item_indices:List[str] = []
        # Structure dtypes a private array created in the body mentions: their typedefs
        # have to be emitted even though they are not parameters.
        self.private_structs:List[object] = []
        # Dtypes synthesized from a plain class the body constructs (`ray = Ray(...)` on a
        # class that is not a ctypes.Structure). Their members are what the body reads, so a
        # construction may hand over more than that and the unread arguments are dropped --
        # a declared structure, whose layout is fixed, still refuses them.
        self.plain_class_dtypes:Set[np.dtype] = set()
        # Generated quaternion device functions, keyed by (name, body) so one definition is
        # emitted however many call sites ask for it.
        self.quaternion_helpers:Dict[str, str] = {}
        # The quaternion blocks a generated source needs, as the `QUAT_NEED_*` macro names. One
        # header serves both widths; only the blocks in use have to be compiled.
        self.quaternion_needs:Set[str] = set()
        # The complex blocks this source needs, as the `COMPLEX_NEED_*` macro names -- the
        # same contract as the quaternion widths: one header, and only the widths in use are
        # compiled (a `double` block does not compile where fp64 is absent).
        self.complex_needs:Set[str] = set()
        # Set when the body reads or writes a complex value at all, which is what decides
        # whether the generated source includes the complex header.
        self.uses_complex:bool = False
        # The headers a registered device-side library function asked for: what the registry
        # spells for them (a path, or a full `#include` line). The compile step turns them
        # into the include lines the generated source carries.
        self.device_library_headers:List[str] = []
        # The array whose whole shape a `np.ndindex(arr.shape)` loop iterates, when that is
        # how the domain was spelled: the launch checks its rank against the loop's names.
        self.nd_whole_shape_param:str = ""
        # The array whose shape declares the iteration domain, recorded from
        # `@kernel(domain=...)`; this is the gid-style counterpart of a magic loop's implied
        # domain. The rank comes from that array's runtime shape (a pointer type carries no
        # rank), so PyKernel expands it at call time.
        self.declared_work_arrays:List[str] = []
        # Set when the body calls gpu.get_global_id()/... , i.e. it dispatches on the launch
        # index and therefore needs an iteration domain from somewhere.
        self.uses_global_id:bool = False
        # Set once a magic loop is detected: it fixes the iteration domain, so a
        # `@kernel(domain=...)` declaration would be redundant.
        self.magic_loop_present:bool = False
        # The line of that loop (in the parsed copy's numbering), so a launch-time rank
        # check can blame the `for` statement instead of the whole kernel.
        self.magic_loop_line:int = 0
        # Image parameters that an OpenCL kernel samples. Each one needs a sampler
        # constant emitted next to the kernel, and the host side builds one sampler
        # per image, so this set drives that generation.
        self.required_image_samplers:Set[str] = set()
        # Image parameters a CUDA kernel stores into: those need a surface handle,
        # which is a separate device object from the texture.
        self.required_image_surfaces:Set[str] = set()
        self.atomic_targets:Set[str] = set()
        self._tmp_counter:int = 0
        self.local_buffers:List[Tuple[str, str, int]] = []
        self.return_type:Optional[object] = None
        # A device helper's `return a, b` (see _visit_return): the element types, in
        # order. The call site appends one `T*` output parameter per element and passes
        # the destinations it unpacks into; the helper stores through them.
        self.tuple_return_types:Optional[List[object]] = None
        # The element types read for the tuple call currently being unpacked (see
        # _tuple_call_element_count, which stashes them for _emit_tuple_call_unpack).
        self._pending_tuple_types:Optional[List[object]] = None
        # (line index, name, initial type) for every local declaration `_visit_assign`
        # emitted, so the renderer can rewrite one whose variable a later assignment
        # widened (float -> double; see _define).
        self._declaration_sites:List[Tuple[int, str, str]] = []
        # A local bound to an array extent above the magic loop (`rows: int =
        # out.shape[0]`): the name resolves to (array, dim), which is what an
        # `np.ndindex(rows, ...)` bound needs to become the launch's extent parameter.
        self._shape_aliases:Dict[str, Tuple[str, int]] = {}
        # An ndenumerate index held as one name (`for index, v in ...`): the name stands
        # for the coordinate tuple, and a constant-ordinal subscript reads the
        # decomposition the head emitted (python's order, slowest first).
        self._ndenumerate_index_names:Dict[str, List[str]] = {}
        # Kernel return values. A kernel runs once per work item, so it cannot have a C
        # return value; every returned position becomes a device parameter instead (a
        # one-element buffer for a value, the array parameter's own name for an array) and
        # the first work item to reach a `return` claims the whole result.
        #
        # return_layout: one entry per value slot ({"ctype", "type"}), shared by every
        #   return site whose value has that C type.
        # return_sites: per `return` statement, its positions as the translator saw them.
        # _site_slot_map: per site, (position ordinal, slot index) pairs it writes.
        self.return_layout:List[ReturnLayoutEntry] = []
        self.return_sites:List[List[ReturnPosition]] = []
        self._site_slot_map:Dict[int, List[Tuple[int, int]]] = {}
        # True when statements currently being visited run once per work item after the
        # magic loop (see _visit_region and _visit_kernel_return).
        self.after_magic_loop:bool = False
        # True when statements currently being visited are inside (or after) the magic
        # loop, where building data would happen once per work item (see _visit_assign).
        self.creation_below_loop:bool = False
        # The conditions of the enclosing `if` blocks, innermost last. A store to a
        # launch-wide location is safe when a guard decides the same way for every work
        # item (`if i == 0:`); see `_store_is_guarded`.
        self._guard_conditions:List[ast.expr] = []
        # True while the statements being visited are the magic loop's own body -- one
        # execution per work item, as opposed to `after_magic_loop`, which means after the
        # dispatch. A shared store is racy in the first case and refused (see
        # `_named_outer_target`), and the two flags are deliberately separate because a
        # statement inside the loop is not "after" it.
        self.in_work_item_body:bool = False
        # True when any site returns a value; then the kernel needs the claim and site
        # parameters even if some other site is a bare `return`.
        self.has_returns:bool = False
        # True when a `return` actually reaches the generated device code. A body whose only
        # return sits after the magic loop hands its value back through the host tail (see
        # PyKernel._run_host_region), so it never claims, and the claim/site words it would
        # otherwise declare are two wasted uploads and two never-read parameters.
        self._claims_on_device:bool = False
        # How many real C loops enclose the statement being visited. Only a nested serial
        # `for`/`while` that is emitted as a loop counts: the magic loop itself does not,
        # because it becomes the work-item dispatch rather than a loop in the generated
        # source (see _visit_work_item_body). `break` and `continue` are meaningful only
        # while this is non-zero, which is what decides how they are translated.
        self.loop_depth:int = 0
        # Outer-scope variables: a name the body uses that is neither a parameter nor a
        # local becomes a device parameter instead (see _bind_outer_variables). Each entry
        # records how the host reaches the name ("closure" through a cell, "global" through
        # func.__globals__), its device type, and whether the body declares it writable
        # (`nonlocal`/`global`) -- which decides between `T* name` + `*name` and a plain `T`.
        self.outer_vars:Dict[str, OuterVarEntry] = {}
        # Values the host creates for the kernel: an assignment above the magic loop whose
        # right-hand side only builds data (np.zeros, np.full, ...) is executed by the
        # interpreter, not translated, and the object travels as a parameter (see
        # _host_created_assignment).
        self.host_values:Dict[str, HostValueEntry] = {}
        self.return_type:Optional[object] = None
        self.lines:List[str] = []
        self._indent:int = 1
        self.scopes:List[Dict[str, DeviceType]] = []

        args:List[ast.arg] = self.tree.args.args
        if self.tree.args.vararg or self.tree.args.kwarg or self.tree.args.kwonlyargs:
            raise SyntaxError("*args/**kwargs/keyword-only args are not supported", self.tree.args) from None
        # Default values are accepted: a kernel argument is always supplied by the
        # caller, so a default is a convenience for reading the signature rather than
        # something the device code has to represent.
        receiver_type:object = None
        if self_param and args and args[0].arg == self_param:
            if self.receiver_as_argument:
                # The receiver travels as this function's own first argument: take its
                # type off the front and name it before the remaining parameters.
                receiver_type = param_types[0] if param_types else None
                param_types = list(param_types[1:])
            args = args[1:]
        # `param_types` holds the positional arguments, then the receiver struct, then one
        # type per ndarray attribute (see PyKernel._invoke_inner and _emit_device_call).
        expected:int = len(args) + len(self.self_attr_types) + (
            1 if self.self_struct_members else 0)
        if expected != len(param_types):
            raise SyntaxError(f"{func.__name__}() takes {expected} device arguments but {len(param_types)} types were given") from None

        # The receiver is a method's first parameter, the way python spells one: `self` is
        # the receiver, so the device signature reads `Class* self,` before the caller's
        # own arguments. `self.params` *is* the signature order, so the receiver is
        # appended before the arguments rather than after them. The slot it occupies in
        # `param_types` stays where the host packs it (after the positional arguments), so
        # the type is taken from there and the two lists keep their own orders.
        scope:Dict[str, DeviceType] = {}
        if self.self_struct_members:
            struct_type:object = (param_types[len(args)] if len(param_types) > len(args)
                               else self.self_struct_type)
            if self.self_struct_type is None:
                self.self_struct_type = struct_type
            self.params.append((self.self_param, struct_type))
        if receiver_type is not None:
            # The receiver is not a `_self_` field set: it is one structure pointer,
            # spelled by the function's own first parameter name.
            scope[self.self_param] = receiver_type
            self.params.append((self.self_param, receiver_type))
        for arg, t in zip(args, param_types):
            # Annotations are accepted but ignored: the device type comes from the
            # actual argument at call time (see infer_ctype), because one kernel is
            # specialized per argument-type signature. Keeping them lets the same
            # function be read by editors and type checkers.
            scope[arg.arg] = t
            self.params.append((arg.arg, t))
        for attr, t in self.self_attr_types.items():
            if attr in self.self_struct_members:
                continue                  # carried inside the receiver struct
            pname:str = f"_self_{attr}"
            scope[pname] = t
            self.params.append((pname, t))
        self.scopes.append(scope)

    # ---------- basic utilities ----------

    @property
    def printf_backend(self)->str:
        """The key into the printf tables: hip compiles CUDA-style code, so it joins cuda."""
        return "cuda" if self.backend in ("cuda", "hip") else "opencl"

    def _error(self, message:str, node:Optional[ast.AST]=None)->NoReturn:
        """Raise a translation error. Declared NoReturn so callers can rely on it not
        falling through (the checker then stops treating each call site as nullable).

        ``from None`` suppresses the chain: the translator's own frames are an implementation
        detail, and what matters is the position drawn from the user's file.
        """
        raise _syntax_error(message, node, self.filename, self.source_text,
                            self.first_line) from None

    def _emit(self, line:str)->None:
        self.lines.append("    " * self._indent + line)

    def _render_body(self, lines:Optional[List[str]]=None)->str:
        """The emitted lines with separators placed where a reader expects them.

        The walkers record a blank line after every closed block and between statement
        runs. This pass keeps one where a *new statement at the same scope* follows a run
        or a block, and drops one where the scope simply ends (the next line dedents back
        to the parent), because there is nothing more in this scope to separate. Doubles
        collapse, and no blank may lead the body or survive at its end.

        Before the separators are placed, every local declaration whose variable a later
        assignment *widened* (`scale = 1.0` here, a double inside the loop) has its type
        rewritten -- the declaration was spelled with the type the first assignment had,
        and the C compiler rejects a store of the wider value into it.
        """
        source:List[str] = (self.lines if lines is None else lines)
        if self._declaration_sites:
            source = list(source)
            for line_index, name, initial_type in self._declaration_sites:
                final_t:DeviceType = self._lookup(name)
                final_type:str = (self._device_type(final_t)
                                  if isinstance(final_t, str) and not is_ptr(final_t)
                                  else "")
                if final_type and final_type != initial_type:
                    line:str = source[line_index]
                    marker:str = f"{initial_type} {device_name(name)} = "
                    stripped:str = line.lstrip()
                    if stripped.startswith(marker):
                        indent:str = line[:len(line) - len(stripped)]
                        source[line_index] = (f"{indent}{final_type} "
                                              f"{device_name(name)} = "
                                              f"{stripped[len(marker):]}")
        rendered:List[str] = []
        for line in source:
            if line == "" and (not rendered or rendered[-1] == ""):
                continue
            rendered.append(line)
        # A block that ends just before the brace closing its parent (`while { ... }` as
        # the last statement of a kernel) leaves two closing braces on consecutive lines.
        # The inner one still ends a block, so the separator every closing brace gets is
        # inserted between them.
        # The trailing blank the caller may have appended is trimmed first: the last real
        # line is what decides whether a separator is needed before it.
        trimmed:List[str] = list(rendered)
        while trimmed and trimmed[-1] == "":
            trimmed.pop()
        kept:List[str] = list(trimmed)
        # A blank line directly before a closing brace separated a block that ends there.
        # When the very next line closes the *parent*, that separator would sit between two
        # braces instead of between two statements, so it is dropped. Every other closing
        # brace keeps its own separator (`if`/`else` already has none by construction).
        spaced:List[str] = []
        for index, line in enumerate(kept):
            if line == "" and index + 1 < len(kept) and kept[index + 1].strip() == "}":
                continue
            spaced.append(line)
        kept = spaced
        while kept and kept[0] == "":
            kept.pop(0)
        while kept and kept[-1] == "":
            kept.pop()
        return "\n".join(kept)

    def _emit_blank(self)->None:
        """A separator line, recorded so the renderer can collapse or trim it."""
        if self.lines and self.lines[-1] != "":
            self.lines.append("")

    def _lookup(self, name:str)->DeviceType:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        # A host-created object and an outer variable are device parameters too, so a
        # type query has to see them; the magic-loop scan asks which arrays exist before
        # any statement of the body has been visited.
        host:Optional[HostValueEntry] = self.host_values.get(name)
        if host is not None:
            return host["type"]
        host_scalar:Optional[DeviceType] = self.host_scalars.get(name)
        if host_scalar is not None:
            return host_scalar
        for scope in reversed(self.scopes):
            if name in scope:
                declared:object = scope[name]
                if isinstance(declared, tuple) and len(declared) == 2 and declared[0] == "carried":
                    return declared
                break
        outer:Optional[OuterVarEntry] = self.outer_vars.get(name)
        if outer is not None:
            # A declared outer scalar is a pointer on the device (see _visit_name).
            if outer["writable"] and not is_ptr(outer["type"]):
                return (PTR, outer["type"])
            return outer["type"]
        return None

    def _emit_array_slice(self, name:str, node:ast.Assign)->bool:
        """Emit `local = array[lo:hi]` as a pointer into the buffer, and record its length.

        Python's slice is a *view*: it shares the buffer and adds an offset and a length.
        The device has no view type, so the same thing is spelled with a pointer it already
        has -- `T* part = a + lo;` -- and the length is kept in `_array_slice_extents`, which
        is what `part.shape[0]` reads. `a[n // 2:]` works the same way: the bound is an
        expression, and the emitted pointer just carries it.

        Two facts make the pointer arithmetic compose. The offset is added to the **base's
        own translated code**, so a slice of a slice (`inner = part[1:3]`) starts from
        `part`, not from the array the chain began with. And an omitted upper bound is the
        **immediate** base's length, so `part[1:]` is `len(part) - 1` rather than the whole
        array's extent minus one.

        A step other than 1 needs a real view (a work item cannot walk a buffer backwards),
        and a *rank* above one is refused at the call, where the shape is known: on a 2-D
        array `grid[0:2]` is two rows, which one offset cannot express.
        """
        subscript:ast.Subscript = cast(ast.Subscript, node.value)
        slice_node:ast.Slice = cast(ast.Slice, subscript.slice)
        base:ast.expr = subscript.value
        immediate:str = self._array_name(base)
        root:str = self._array_pointer_root(immediate)
        base_code, base_t = self._visit_expression(base)
        if not is_ptr(base_t):
            return False
        step:Optional[ast.expr] = slice_node.step
        if step is not None and _constant_int_of(step) != 1:
            if _constant_int_of(step) == -1:
                self._error(
                    "a reversed slice (`a[::-1]`) is not a device pointer: a work item "
                    "cannot walk backwards through a buffer with one, so read the elements "
                    "with an explicit index instead (`a[n - 1 - i]`)", step)
            self._error("a slice step must be the constant 1", step)
        lower:Optional[ast.expr] = slice_node.lower
        upper:Optional[ast.expr] = slice_node.upper
        start:str = "0" if lower is None else self._visit_expression(lower)[0]
        # An omitted upper bound is the length of what is being sliced -- the array's extent
        # for a parameter, the parent slice's own length for a slice of a slice.
        stop:str = (self._shape_var(immediate, 0) if upper is None
                    else self._visit_expression(upper)[0])
        self._define(name, base_t, node)
        self._emit(f"{self._device_type(base_t[1])}* {device_name(name)} = "
                   f"{base_code} + ({start});")
        # Two different questions are answered by two tables here, which is why both are
        # written. (1) The *length* is the slice's own, so `part.shape[0]` reads it -- that
        # is the loop's bounds guard. (2) The *launch size* has to come from a parameter,
        # and a slice is not one, so the sizing resolves to the array the chain began with:
        # the grid then covers more work items than the slice has, and the guard above skips
        # the excess. That is exactly how a rounded-up automatic work size already works.
        self._array_slice_extents[name] = f"(({stop}) - ({start}))"
        self._array_pointer_sources[name] = root
        if root not in self.sliced_arrays:
            self.sliced_arrays.append(root)
        return True

    def _loop_domain_name(self, attr_node:ast.expr)->Optional[str]:
        """The name `X` in `X.shape[0]`, exactly as the body wrote it.

        No alias or slice resolution happens here: the caller needs to know *which local*
        the bound was read from, because a slice's extent is its own length while its
        launch size is the array's.
        """
        if not (isinstance(attr_node, ast.Attribute) and attr_node.attr == "shape"):
            return None
        base:ast.expr = attr_node.value
        if isinstance(base, ast.Name):
            return base.id
        if (self.self_param and isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name) and base.value.id == self.self_param):
            return f"_self_{base.attr}"
        return None

    def _pointer_source_name(self, value:ast.expr)->Optional[str]:
        """The array a pointer assignment copies from, when the right-hand side is one.

        Only a bare name (`src = a`, resolved through the alias chain) or a receiver member
        (`buf = self.data`) counts. Every other attribute is a *value* -- `lane =
        threadIdx.x` is the usual one -- and asking `_array_name` about it would refuse a
        statement that has nothing to do with arrays.
        """
        if isinstance(value, ast.Name):
            return self._array_pointer_root(value.id)
        if (self.self_param and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name) and value.value.id == self.self_param):
            return f"_self_{value.attr}"
        return None

    def _record_element_write(self, base:ast.expr)->None:
        """A store into an element makes the *array it came from* writable.

        The read/write analysis reads the generated source, where a local pointer
        (`double* part = a + 1;`) is not the parameter -- so a store through the local would
        leave `a` marked read-only and the host would drop the write silently. The name is
        followed to the array the local points at, and that name is the force-writable
        channel the member-store path already uses (see `atomic_targets`).
        """
        try:
            name:str = self._array_name(base)
        except SyntaxError:
            return
        root:str = self._array_pointer_root(name)
        if is_ptr(self._lookup(root)):
            self.atomic_targets.add(root)

    def _define(self, name:str, t:DeviceType, node:Optional[ast.AST])->None:
        """Record a variable's type; node is optional because some declarations have no
        source node to blame in an error message."""
        existing:DeviceType = self._lookup(name)
        if existing is None:
            self.scopes[-1][name] = t
        elif existing != t:
            if is_int(existing) and is_float(t):
                self.scopes[-1][name] = t
            elif is_float(existing) and is_float(t):
                # Widening only ever goes up (float -> double), the same rule the int ->
                # float case above follows: a python variable is untyped, so `scale = 1.0`
                # above the loop followed by `scale = x * 2.0` (a double under the
                # mapping) refused with "changes type from float to double". The scope
                # entry is what every later read and the declaration itself see, so
                # re-pointing it widens the variable in place.
                self.scopes[-1][name] = t
            else:
                self._error(f"variable '{name}' changes type from {existing} to {t}", node)

    def _is_const_struct(self, name:str)->bool:
        """Whether a name denotes a structure parameter the body never writes.

        Its signature carries `const`, so an alias of it cannot be mutable.
        """
        binding:DeviceType = self._lookup(name)
        if not (isinstance(binding, tuple) and is_ptr(binding)
                and isinstance(binding[1], str)):
            return False
        from . import structs as _structs_module  # noqa: PLC0415

        if _structs_module.dtype_for(binding[1]) is None:
            return False
        return name not in self.member_writes

    def _record_member_write(self, member_code:str)->None:
        """Remember that this structure parameter's members are written.

        The generated expression is `(base)->field` or `(base[0]).field`; the base is the
        parameter name, and that parameter must not be declared `const`.
        """
        import re as _re

        # The base is the parameter name, optionally wrapped and optionally indexed
        # (`(g[0]).field` is a record-array element of the parameter `g`).
        match = _re.search(r"(?:\*?&&?)?\(?([A-Za-z_]\w*)", member_code)
        base:str = match.group(1) if match else ""
        if base:
            self.member_writes.add(base)
            # The alias case (`q = p; q.a = ...`) writes the same buffer.
            alias:Optional[DeviceType] = (self.scopes[-1].get(base)
                                          if self.scopes else None)
            if isinstance(alias, tuple) and len(alias) == 2 and alias[0] == "alias":
                self.member_writes.add(alias[1])

    def _shape_var(self, array_name:str, dim:int)->str:
        # A local pointer (`src = a`) is a *device variable*, but it has no extent parameter
        # of its own: the extent lives on the array it was assigned from, because that is
        # what the launch passes. Resolving here is what makes `src.shape[0]` compile -- and
        # it is the single place that decides it, because every extent parameter is built by
        # this call.
        # A slice (`part = a[lo:hi]`) has a length of its own, recorded when it was bound.
        if dim == 0 and array_name in self._array_slice_extents:
            return f"({self._array_slice_extents[array_name]})"
        array_name = self._array_pointer_root(array_name)
        # A private array created in the work-item body has a constant shape, so its extents
        # are compile-time numbers rather than launch parameters.
        binding:DeviceType = self._lookup(array_name)
        if isinstance(binding, tuple) and len(binding) == 4 and binding[0] == "bodyarr":
            shape:Tuple[int, ...] = binding[2]
            if 0 <= dim < len(shape):
                return str(shape[dim])
        key = (array_name, dim)
        if key not in self.shapes_used:
            self.shapes_used.append(key)
        # The extent parameter is named after the array it bounds, and an array called
        # `register` or `new` would put the reserved word inside the identifier. Mapping the
        # array name here is the one place that decides the spelling, because every extent
        # parameter is built by this call.
        return f"_{device_name(array_name)}_shape{dim}"

    def _array_pointer_root(self, name:str)->str:
        """The array a local pointer variable was assigned from, or the name itself.

        An array parameter reaches the device as `T*`, and `src = a` declares one more
        pointer (`T* src = a;`) rather than copying anything. The pointer itself has no
        extent, though: extents are launch parameters named after the *array*, so a shape
        read through the local has to resolve to the array it was assigned from. A chain
        (`x = src`) follows to the first name, bounded so a malformed pair cannot loop.

        The table is filled ahead of the walk (see `_collect_array_pointers`), because the
        *loop detection* asks about `enumerate(src)` before any statement is translated.
        """
        seen:Set[str] = set()
        while name in self._array_pointer_sources and name not in seen:
            seen.add(name)
            name = self._array_pointer_sources[name]
        return name

    def _collect_array_pointers(self, body:List[ast.stmt])->None:
        """Record every `local = <array>` in the body before it is walked.

        A pointer local is a fact about the body, and the magic-loop scan needs it before
        the statement that creates it is reached: `enumerate(src)` is only the dispatch when
        `src` is known to be an array. The sources are recorded with their own name here and
        resolved through `_array_pointer_root`, so a chain (`two = one`) works too.
        """
        for statement in body:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                continue
            target_name:str = statement.targets[0].id
            source:ast.expr = statement.value
            if isinstance(source, ast.Name):
                self._array_pointer_sources.setdefault(target_name, source.id)
            elif isinstance(source, ast.Attribute) and isinstance(source.value, ast.Name) \
                    and self.self_param and source.value.id == self.self_param:
                self._array_pointer_sources.setdefault(target_name, f"_self_{source.attr}")

    def _trim_source(self)->str:
        return get_dedented_source(self.func)

    # ---------- translation entry ----------

    @property
    def nests_a_launch(self)->bool:
        """True when this kernel launches another kernel, i.e. needs linked device code."""
        return bool(self.nested_kernels)

    def translate(self)->None:
        body:List[ast.stmt] = self.tree.body
        # A local pointer (`src = a`) is known before the body is walked: the magic-loop
        # scan asks about `enumerate(src)` before that statement is reached (see
        # _collect_array_pointers).
        self._collect_array_pointers(body)
        # `weights = [1.0, 2.0]` feeding a loop is a compile-time table, recorded before the
        # walk so the binding emits no device code and the loop can read it.
        self._collect_constant_lists(body)
        # Outer-scope names and host-created objects must be known before the body is
        # visited: every use of them compiles against the parameter that will carry them,
        # and the magic-loop scan asks which arrays exist before any statement is reached
        # (a created array can be the loop's iteration domain).
        self._bind_outer_variables()
        if self.is_kernel:
            self._bind_host_values(body)
            # Whether any return site carries a value decides the device signature, and
            # that is known only after the whole body has been seen; every site's code
            # depends on it, so the body is scanned once first.
            self.has_returns = _returns_a_value(body)
            # Whether any of those returns is device code, which is a different question:
            # a tail return runs on the host and claims nothing (see render_return_params).
            self._claims_on_device = _has_device_return(body)
            # A local initialized above the loop and written inside it is one location for
            # the whole launch, not one per work item -- see _carried_scalars.
            self._bind_carried_scalars(body)
            for offender in self._non_atomic_writes(body, self.carried_scalars):
                self._error(
                    "this variable is one location shared by every work item (it is "
                    "accumulated inside the loop and read after it), so a plain store here "
                    "races: whichever work item finishes last decides the value. Use `+=` "
                    "(which becomes a device atomic), or keep the per-work-item value in a "
                    "different variable and reduce it afterwards", offender)
            # A body with several magic loops is several *passes* over the work items. Each
            # one is translated on its own into a kernel of its own, because a second loop
            # inside one kernel would be run by every work item instead of once. They are
            # launched back to back on the same stream, so the arrays stay on the device.
            self.phases = self.magic_loop_phases(body)
            if len(self.phases) == 1:
                self._translate_magic_loop(body)
            else:
                self._translate_phase_kernels()
        else:
            for stmt in body:
                self._visit_statement(stmt)
            if self.return_type is None:
                self.return_type = "void"

    def _translate_phase_kernels(self)->None:
        """Translate one kernel per magic loop, each owning its own statements.

        A phase is a separate kernel, so a name it reads has to be one of its own
        parameters. The names a *different* phase computes therefore have to cross the
        boundary, and how depends on what the name holds:

        * a **created array** (`mid = np.zeros(4)`) is built once on the host and travels as
          a trailing parameter, exactly as it does in a single-phase body -- whether it sits
          above the first loop or between two of them;
        * a **plain scalar** (`scale = 10`) is copied into the body of every later phase and
          recomputed there. It is a value each work item can derive for itself, so shipping
          it as a parameter would add an argument nothing needs, and it is what keeps a
          body's `k = 2 * factor` readable across the split.
        """
        # The statements every phase shares: those above the first loop, plus the gaps
        # between the loops (a creation there has to exist before the pass that reads it).
        header:List[ast.stmt] = self.phases[0][:_magic_loop_index(self.phases[0])]
        carried:List[ast.stmt] = []
        for index in range(len(self.phases) - 1):
            gap:List[ast.stmt] = self.phases[index][_magic_loop_end(self.phases[index]):]
            for stmt in gap:
                # Every gap statement runs on the host: python executes it once, between the
                # two passes. A creation is bound by _bind_host_values, which already runs on
                # the host; anything else is a value the next pass receives as a parameter.
                #
                # A statement the interpreter *already ran* (its line is in
                # host_statement_lines) is not copied into the header: copying it made
                # every phase emit device code for it under the same function name, and
                # the two definitions then collided. Its value travels as a host scalar
                # instead (see _bind_host_statement), which is what the next pass needs.
                if (int(getattr(stmt, "lineno", 0) or 0) in self.host_statement_lines
                        and _assigned_name_of(stmt) is not None):
                    carried.append(stmt)
                    continue
                header.append(stmt)
                if not self._is_device_creation(stmt) and _assigned_name_of(stmt) is not None:
                    carried.append(stmt)

        # The names a gap computed on the host: each is a parameter of the pass that reads
        # it, so it is declared (and marked like a carried slot) before the body is visited.
        gap_values:List[str] = [
            name for name in (_assigned_name_of(stmt) for stmt in carried)
            if name is not None]
        self.gap_names = list(gap_values)
        # The statements after each phase's loop, in phase order: the host runs region i
        # between launch i and launch i + 1 (see PyKernel._run_host_region).
        self.phase_gaps:List[List[ast.stmt]] = [
            list(self.phases[index][_magic_loop_end(self.phases[index]):])
            for index in range(len(self.phases) - 1)]

        for index, phase_body in enumerate(self.phases):
            # The gap that follows this phase belongs to the *next* one (it is either a
            # creation the host builds or a value the next pass recomputes), so it is cut
            # from this phase's body: emitted here it would be device code, and a creation
            # has no device form.
            body:List[ast.stmt] = (list(phase_body[:_magic_loop_index(phase_body) + 1]
                                        if index < len(self.phases) - 1 else phase_body))

            phase:FunctionTranslator = FunctionTranslator(
                self.func, self.param_types, self.backend, self.resolve_global,
                self.device_cache, is_kernel=True, self_param=self.self_param,
                self_attr_types=self.self_attr_types, self_owner=self.self_owner,
                launch_backend=self.launch_backend,
                host_creation_values=self.host_creation_values,
                host_executed_lines=sorted(self.host_statement_lines))
            # The tree is the same function, so the phase is visited by transplanting its
            # statements onto the phase translator's own tree.
            phase.tree.body = body
            # The gap this phase precedes is host code, but the phase still has to see the
            # names it feeds: a scalar the loop accumulates and the gap reads keeps its
            # shared slot in the writing phase (see _carried_scalars).
            phase.phase_gap = list(self.phases[index][_magic_loop_end(self.phases[index]):])
            phase._bind_outer_variables()
            phase.phase_carried_hint = list(self.carried_scalars)
            # A phase skips the main translate() walk, so the shared-slot binding has to run
            # here: without it, a scalar the loop accumulates compiles as per-item scratch
            # and the reduction silently loses every contribution but one work item's.
            phase._bind_carried_scalars(body)
            for offender in phase._non_atomic_writes(body, phase.carried_scalars):
                phase._error(
                    "this variable is one location shared by every work item (it is "
                    "accumulated inside the loop and read after it), so a plain store here "
                    "races: whichever work item finishes last decides the value. Use `+=` "
                    "(which becomes a device atomic), or keep the per-work-item value in a "
                    "different variable and reduce it afterwards", offender)
            # A later phase does not contain the shared creations, so it binds them here;
            # the object is the same host object for every phase.
            phase._bind_host_values(header + body if index else body)
            used:Set[str] = {node.id for node in ast.walk(ast.Module(body=body, type_ignores=[]))
                             if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                             and node.id in gap_values}
            # A gap value is one of the pass's own parameters, but it must be declared
            # *after* the body is visited: the shape extents are appended while translating,
            # and inserting a parameter ahead of them shifts every later argument.
            for name in gap_values:
                # A value the interpreter bound between the passes (a host-region statement)
                # travels as a host scalar of its *own* type (see _bind_host_statement):
                # declaring it here as a carried `int*` slot as well made the launch pass
                # the string it was computed from into an int parameter.
                if name in phase.host_scalars:
                    continue
                if name in used and name not in phase.scopes[-1]:
                    phase.scopes[-1][name] = ("carried", name)
            phase.has_returns = _returns_a_value(body)
            # Every phase is its own kernel, so whether one of them claims is a fact about
            # its own slice: a pass whose only return sits in its tail hands it back on the
            # host and needs no claim word (see render_return_params).
            phase._claims_on_device = _has_device_return(body)
            phase._translate_magic_loop(body)
            # The statements after this phase's own loop. For the last phase that is the
            # body's tail, which the host runs after the final launch; for an earlier one it
            # is the gap that feeds the next pass, and the launch already runs it as a region
            # (see _run_host_region). Recording it on the phase keeps the tail honest for a
            # multi-loop body: only the main translator used to keep one, so a body split
            # into passes lost its tail entirely and a `return out` after the last loop
            # answered None.
            phase.body_tail = list(body[_magic_loop_end(body):])
            for name in gap_values:
                if name in phase.host_scalars:
                    continue
                if name in used and name not in dict(phase.params):
                    phase.params.append((name, (PTR, "int")))
                    phase.carried_scalars.append(name)
            self.phase_translators.append(phase)
            # The host builds one argument list for the whole call, so every phase's
            # parameters have to be known here: a value only a later phase needs (an array
            # created between the loops) is still a parameter of this translation.
            for name, value_type in phase.params:
                if name not in dict(self.params):
                    self.params.append((name, value_type))
            # The same for outer-scope values: the host builds one launch, so every name any
            # phase reads has to be known here.
            for name, outer in phase.outer_vars.items():
                if name not in self.outer_vars:
                    self.outer_vars[name] = outer

    def _is_device_creation(self, stmt:ast.stmt)->bool:
        """True when ``stmt`` builds data, as opposed to computing a value."""
        value:Optional[ast.expr] = _creation_target_of(stmt)
        return value is not None and self._is_creation_expression(value)

    # ---------- outer-scope variables ----------

    def _bind_outer_variables(self)->None:
        """Turn every outer name the body uses into a device parameter.

        Python resolves such a name through the enclosing function's closure or the
        module globals; a device kernel has neither, so the value travels as an argument
        the host supplies at launch (see PyKernel._compile / _invoke_inner). Which names
        qualify is decided from the decorated function itself:

        * a `nonlocal`/`global` declaration makes the name writable: it becomes `T*` and
          the body reads and writes it through the pointer, so the host can copy the final
          value back into the scope;
        * any other name the function references but does not define is a free variable
          (`co_freevars`, i.e. a closure cell) or a module global (`co_names`): it becomes
          a plain `T` parameter and is read-only, because python itself would not let the
          body assign to it without a declaration.

        A cell's content can change between calls, so a read-only name is still a
        parameter -- inlining the current value would make the kernel stale, which is the
        whole reason this exists.
        """
        code:object = getattr(self.func, "__code__", None)
        if code is None:
            return
        cell_names:Set[str] = set(code.co_freevars)
        declared:Set[str] = _declared_scope_names(self.tree)
        # Names the body defines locally never reach the scope; parameters are already bound.
        bound:Set[str] = self._local_names() | set(self.self_attr_types)
        bound |= {name for name, _ in self.params}
        if self.self_param:
            bound.add(self.self_param)
        # `co_names` also holds attribute names (`x.shape` contributes "shape") and the
        # names of called builtins/helpers, so only names the body actually reads as a
        # plain variable can become parameters. A name the body *assigns* without a
        # declaration is python's local, not a scope variable, so it is not a candidate --
        # but `co_names` still lists it for the global case, hence the `referenced` filter.
        referenced:Set[str] = self._referenced_names()

        candidates:List[str] = []
        for name in sorted(declared | cell_names | referenced | set(code.co_names)):
            if name in bound or name in self.outer_vars:
                continue
            if name in _BUILTIN_NAMES:
                continue
            candidates.append(name)
        for name in candidates:
            value:object = self.resolve_global(name)
            if value is _UNRESOLVED:
                continue
            entry:Optional[OuterVarEntry] = self._outer_var_entry(name, value, code, declared)
            if entry is None:
                continue
            # Which module's namespace the name came from. A helper imported from its own
            # module may read a global of *that* module, and the host has to look there --
            # reading the kernel's globals would find nothing and pass None.
            entry["scope"] = getattr(self.resolve_global, "source_globals", None)
            self.outer_vars[name] = entry
            if entry["writable"]:
                self.params.append((name, (PTR, entry["type"])))
            else:
                self.params.append((name, entry["type"]))

    def _outer_var_entry(self, name:str, value:object, code:object,
                         declared:Set[str])->Optional[OuterVarEntry]:
        """The device parameter for one outer name, or None when it is not a value.

        A module, a class, a builtin or a helper function is not a value a parameter can
        carry: those names are resolved by the translator's own dispatch (numpy is
        `np`, `gpu`, `math`, helper calls, ...), so they are left alone. An outer name
        that *is* a value has to be a supported argument type, and a declared write has to
        be a scalar, because the scope only ever holds that variable's value.
        """
        if isinstance(value, (type, types.ModuleType)) or callable(value):
            # A module, class, builtin or helper function is not a value a parameter can
            # carry: those names are resolved by the translator's own dispatch (numpy is
            # `np`, `gpu`, `math`, helper calls, ...), so they are left alone.
            return None
        writable:bool = name in declared
        try:
            device_type:DeviceType = infer_ctype(value, self.type_map)
        except (SyntaxError, ValueError) as error:
            if not writable:
                # Not a name this feature handles; the normal "unknown name" path will
                # explain it with the right position if the body really reads it.
                return None
            raise type(error)(f"outer variable '{name}' cannot be written back: {error}") from None
        if writable and is_ptr(device_type):
            raise SyntaxError(
                f"cannot rebind the outer array '{name}': the array object itself is "
                f"shared with the caller, so a device kernel can only write its elements") from None
        return {"type": device_type, "writable": writable,
                "kind": ("closure" if name in code.co_freevars else "global")}

    def _local_names(self)->Set[str]:
        """Names this function defines as locals (assigns, unpacks, loop targets).

        A name declared `global`/`nonlocal` is *not* local even though the body assigns
        to it: python binds it in the scope, which is exactly what this feature reproduces.
        An undeclared name that is both read and assigned is python's local variable (it
        shadows any outer name of the same name), so it stays local here too.
        """
        declared:Set[str] = _declared_scope_names(self.tree)
        names:Set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in declared:
                    names.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        return names

    def _referenced_names(self)->Set[str]:
        """Names the body loads as a plain variable, not as an attribute or call target."""
        names:Set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
        return names

    def _shape_holder_param(self, attr_node:ast.expr)->Optional[str]:
        """When X in `X.shape` is an array parameter, return its parameter name.

        A local alias of an array (`src = a`) resolves to the array it names: the extent is
        the same one, and the loop over `src.shape[0]` is the same dispatch.
        """
        if not (isinstance(attr_node, ast.Attribute) and attr_node.attr == "shape"):
            return None
        base:ast.expr = attr_node.value
        if isinstance(base, ast.Name):
            return self._array_pointer_root(base.id)
        if (self.self_param and isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                and base.value.id == self.self_param):
            return f"_self_{base.attr}"
        return None

    def _shape0_range_param(self, iter_node:ast.expr)->Optional[str]:
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "range" and len(iter_node.args) == 1 and not iter_node.keywords):
            return None
        arg:ast.expr = iter_node.args[0]
        if not (isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant) and arg.slice.value == 0):
            return None
        return self._shape_holder_param(arg.value)

    def _is_shape0_range(self, iter_node:ast.expr)->bool:
        return (self._shape0_range_param(iter_node) is not None)

    def _enumerate_magic_target(self, iter_node:ast.expr)->Optional[Translated]:
        """Resolve the array of `for i, x in enumerate(arr)`, returning (name, pointer type).

        The domain may also be a **literal list** (`enumerate([1.0, 2.0, 3.0])`): numpy
        accepts it, its length is a compile-time number, and the elements are constants the
        work item can read directly. That form has no array parameter, so it is reported
        with the length the list states and no pointer type (`None`), which is what the
        caller uses to emit a constant table.
        """
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "enumerate" and len(iter_node.args) == 1 and not iter_node.keywords):
            return None
        arg:ast.expr = iter_node.args[0]
        if isinstance(arg, (ast.List, ast.Tuple)) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                and not isinstance(e.value, bool) for e in arg.elts):
            self._enumerate_constants = [e.value for e in arg.elts]
            return "", None
        if isinstance(arg, ast.Name) and arg.id in self._constant_lists:
            # `weights = [1.0, 2.0]`, then `enumerate(weights)`: the list is the table.
            self._enumerate_constants = list(self._constant_lists[arg.id])
            return "", None
        self._enumerate_constants = []
        param:str = ""
        if isinstance(arg, ast.Name):
            # A local alias of an array (`src = a`, then `ndenumerate(src)`) names the same
            # buffer, so its extents are the aliased array's own.
            param = self._array_pointer_root(arg.id)
        elif (self.self_param and isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id == self.self_param):
            param = f"_self_{arg.attr}"
        else:
            return None
        t:DeviceType = self._lookup(param)
        if not is_ptr(t):
            return None
        return param, t

    def _collect_constant_lists(self, body:List[ast.stmt])->None:
        """Record `weights = [1.0, 2.0, ...]` when a loop enumerates it.

        A literal list is a compile-time table: its length is the dispatch bound and each
        element is a constant, so the loop reads them from the source itself (see
        `_emit_constant_enumerate_head`) and the binding emits nothing. Only a list an
        `enumerate` names is treated that way -- any other list stays an ordinary value.
        """
        enumerated:Set[str] = set()
        for statement in body:
            for node in ast.walk(statement):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "enumerate" and len(node.args) == 1
                        and isinstance(node.args[0], ast.Name)):
                    enumerated.add(node.args[0].id)
        for statement in body:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                continue
            if statement.targets[0].id not in enumerated:
                continue
            value:ast.expr = statement.value
            if isinstance(value, (ast.List, ast.Tuple)) and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                    and not isinstance(e.value, bool) for e in value.elts):
                self._constant_list_names.add(statement.targets[0].id)
                self._constant_lists[statement.targets[0].id] = [
                    e.value for e in value.elts]

    def _emit_constant_enumerate_head(self, index_name:str, value_name:str,
                                      loop:ast.For)->None:
        """The work-item head for `for i, x in enumerate([1.0, 2.0, ...])`.

        The list is a compile-time table: its length is the dispatch's bound and each
        element is a constant, so the value is a `switch` over the index rather than a
        buffer the host would have to build and pass. The bound is a literal, so the guard
        is too.
        """
        values:List[object] = self._enumerate_constants
        extent:int = len(values)
        self.work_item_indices = [index_name]
        self._define(index_name, "int", None)
        if self._opencl_syntax:
            self._emit(f"const int {index_name} = get_global_id(0);")
        else:
            self._emit(f"const int {index_name} = "
                       f"(int)(blockIdx.x * blockDim.x + threadIdx.x);")
        self._emit(f"if ({index_name} >= {extent})")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")
        self._emit_blank()
        # The element's type is the one its constants promote to (int beside a float is a
        # `double` under the mapping, python's own arithmetic).
        element:str = ("double" if any(isinstance(v, float) for v in values) else "int")
        self._define(value_name, element, loop.target)
        self._emit(f"{self._device_type(element)} {value_name} = 0;")
        self._emit(f"switch ({index_name})")
        self._emit("{")
        for position, value in enumerate(values):
            literal:str = (self._real_literal(float(value), element)
                           if element == "double" else str(int(value)))
            self._emit(f"    case {position}: {value_name} = {literal}; break;")
        self._emit("}")
        self._emit_blank()

    def _emit_work_item_head(self, index_name:str, param:str)->None:
        """Emit the gid definition and the out-of-range guard."""
        self.work_item_indices = [index_name]
        self._define(index_name, "int", None)
        gid_bound:str = self._shape_var(param, 0)
        if self._opencl_syntax:
            self._emit(f"const int {index_name} = get_global_id(0);")
        else:
            self._emit(f"const int {index_name} = (int)(blockIdx.x * blockDim.x + threadIdx.x);")
        self._emit(f"if ({index_name} >= {gid_bound})")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")
        self._emit_blank()

    def _nd_magic_target(self, iter_node:ast.expr)->Optional[Tuple[str, str, str]]:
        """Parse the np.ndindex(...) / np.ndenumerate(...) magic loops.

        Returns (form, array parameter name, module name); ndenumerate's element type is
        looked up in scope by the caller.

        numpy accepts three equivalent spellings for the same iteration domain, and all of
        them have to be recognised -- otherwise the loop is not seen as a magic loop at
        all, every statement above it is translated as device code, and a host-created
        array among them is never lifted out:

            for i, j in np.ndindex(arr.shape):
            for i, j in np.ndindex(arr.shape[0], arr.shape[1]):
            for i, j in np.ndindex(2, 3):

        The number of dimensions comes from the names the loop unpacks; this method only
        records where each extent comes from (``self._nd_loop_extents``), and None means
        "the single array's own shape, one dimension per unpacked name".
        """
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Attribute)
                and isinstance(iter_node.func.value, ast.Name) and not iter_node.keywords):
            return None
        module:object = self.resolve_global(iter_node.func.value.id)
        if module is not np:
            return None
        form:str = iter_node.func.attr
        if form not in ("ndindex", "ndenumerate"):
            return None

        if form == "ndindex":
            explicit:Optional[List[Tuple[str, int]]] = self._ndindex_extents(iter_node)
            if explicit is None:
                return None
            self._nd_loop_extents = explicit or None
            # A literal-only domain (np.ndindex(2, 3)) names no array, which is valid: the
            # extents are stated outright and no shape parameter is involved.
            holder:str = (explicit[0][0] if explicit else self._nd_shape_holder)
            return form, holder, iter_node.func.value.id

        if len(iter_node.args) != 1:
            return None
        arg:ast.expr = iter_node.args[0]
        param:str = ""
        if isinstance(arg, ast.Name):
            # A local alias of an array (`src = a`, then `ndenumerate(src)`) names the same
            # buffer, so its extents are the aliased array's own.
            param = self._array_pointer_root(arg.id)
        elif (self.self_param and isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id == self.self_param):
            param = f"_self_{arg.attr}"
        if not param:
            return None
        holder_type:DeviceType = self._lookup(param)
        # An array contributes its extents as pointer parameters; an image carries
        # its extents in its own descriptor, which the backend knows at launch. Both
        # can define the iteration domain of a magic loop.
        if not (is_ptr(holder_type) or is_image(holder_type)):
            return None
        self._nd_loop_extents = None
        return form, param, iter_node.func.value.id

    def _ndindex_extents(self, iter_node:ast.Call)->Optional[List[Tuple[str, int]]]:
        """Where each extent of one np.ndindex(...) call comes from.

        An empty list means the call named a whole array shape (``np.ndindex(arr.shape)``
        or ``np.ndindex((a, b))``), so its dimensions are the loop's own unpacked names.
        Otherwise one entry per extent: ``(array, dimension)`` for an array dimension, or
        ``("", value)`` for a literal count.
        """
        args:List[ast.expr] = list(iter_node.args)
        if len(args) == 1:
            arg:ast.expr = args[0]
            if isinstance(arg, ast.Attribute) and arg.attr == "shape":
                holder:Optional[str] = self._shape_holder_param(arg)
                if holder is None:
                    return None
                holder_type:DeviceType = self._lookup(holder)
                if not (is_ptr(holder_type) or is_image(holder_type)):
                    return None
                self._nd_shape_holder = holder
                return []
            if isinstance(arg, ast.Tuple):
                args = list(arg.elts)
            elif isinstance(arg, ast.Subscript) and self._extent_source(arg) is not None:
                # A single axis spelled out: `np.ndindex(arr.shape[0])`. numpy reads it as
                # a one-dimensional domain of that length, exactly like the tuple and
                # multi-argument forms, so the loop is the work-item dispatch just the same.
                # This used to fall through to "not a magic loop", which pushed the creation
                # above the loop into device code and reported it as a *second* magic loop --
                # a long message about a shape the body never wrote.
                args = [arg]
            else:
                return None

        extents:List[Tuple[str, int]] = []
        for arg in args:
            pair:Optional[Tuple[str, int]] = self._extent_source(arg)
            if pair is None:
                return None
            extents.append(pair)
        return extents

    def _extent_source(self, node:ast.expr)->Optional[Tuple[str, int]]:
        """One extent of np.ndindex: ``<array>.shape[d]``, ``self.<attr>``, or a literal count."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            # np.ndindex(2, 3): the iteration domain is stated outright.
            return "", int(node.value)
        if isinstance(node, ast.Name):
            # `rows: int = out.shape[0]` above the loop, then `np.ndindex(rows, columns)`.
            # The name's *value* is the extent expression the assignment bound it to, so
            # the bound resolves through the alias table instead of the variable itself
            # (which travels as a plain C int, not as the array's extent parameter).
            binding:Optional[Tuple[str, int]] = self._shape_aliases.get(node.id)
            if binding is not None:
                holder_type:DeviceType = self._lookup(binding[0])
                if is_ptr(holder_type) or is_image(holder_type):
                    return binding
            return None
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and self.self_param and node.value.id == self.self_param):
            # A host-side integer attribute is a constant for this launch, like a literal:
            # `np.ndindex(self.image_height, self.image_width)`. The extent is read from the
            # argument carrying the attribute (see the -2 marker in PyKernel._invoke_inner).
            attribute_type:object = self.self_attr_types.get(node.attr)
            if attribute_type is not None and is_int(attribute_type):
                return f"_self_{node.attr}", -2
            if node.attr in self.self_struct_members:
                # The attribute travels inside the receiver struct: the extent reads the
                # member through the pointer (the -3 marker in _nd_bound_code, and the
                # member name in work_size_members for the host).
                member:str = self.self_struct_members[node.attr]
                from . import structs as _structs_module  # noqa: PLC0415

                struct_dtype:Optional[np.dtype] = _structs_module.dtype_for(
                    self.self_struct_type[1])
                member_type:object = _structs_module.member_types(
                    struct_dtype, self.backend).get(member)
                if member_type is not None and is_int(member_type):
                    self.work_size_members.append(node.attr)
                    return f"(self)->{member}", -3
            return None
        if not isinstance(node, ast.Subscript):
            return None
        if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
            return None
        holder:Optional[str] = self._shape_holder_param(node.value)
        if holder is None:
            return None
        holder_type:DeviceType = self._lookup(holder)
        if not (is_ptr(holder_type) or is_image(holder_type)):
            return None
        return holder, node.slice.value

    def _nd_bound_code(self, source:Tuple[str, int])->str:
        """The C expression bounding one dimension (a shape variable, a scalar, or a literal)."""
        name, dim = source
        if not name:
            return str(dim)
        if dim == -2:
            # `np.ndindex(self.image_height, ...)`: the extent is the value of that argument
            # itself, which travels as a plain scalar parameter.
            return name
        if dim == -3:
            # The extent is a receiver-struct member: read it through the pointer.
            return name
        return self._shape_var(name, dim)

    def _launch_id_code(self, axis:int)->str:
        """The work-item id along one launch axis."""
        # Recorded so a device helper can tell whether it needs Metal's prologue
        # parameters passed down to it (see `_needs_prologue_parameters`).
        self._uses_index_builtins = True
        if self._opencl_syntax:
            # Metal shares this spelling: its prelude defines `get_global_id(dim)` as a
            # macro over the attribute-decorated kernel parameters (see metal_codegen).
            return f"get_global_id({axis})"
        component:str = (".x" if axis == 0 else ".y" if axis == 1 else ".z")
        return f"(int)(blockIdx{component} * blockDim{component} + threadIdx{component})"

    def _emit_nd_head(self, names:List[str], param:str)->None:
        """Multi-dimensional work-item head: names[0] maps to dimension 0 (slowest).

        The body compiles once per signature and the generated source is cached by it, so the
        head cannot depend on how many axes the *launch* happens to have -- a bracket is not
        part of the signature, and a source compiled for a 1-D launch would be reused by a
        2-D one. It is therefore always the same statement: fold the launch's ids into one
        flat work-item position, then decompose that position over the loop's own domain.

        Every axis folds the same way, which is what makes the launch rank irrelevant.
        Normalising each id by its grid extent (the C remainder semantics, so a one-item axis
        contributes its only value) and then by the stride to the next axis gives a position
        where axis 0 is fastest -- and the decomposition, which has the *last* loop name
        fastest, walks the domain from the last stride back. A launch of exactly the loop's
        rank and one of a single flat axis are both instances of the same fold, so neither
        needs a different source.
        """
        ndim:int = len(names)
        self.work_item_indices = list(names)
        sources:List[Tuple[str, int]] = self._nd_loop_extents or [(param, d) for d in range(ndim)]
        for d, name in enumerate(names):
            self._define(name, "int", None)
            self._nd_bound_code(sources[d])
        # The launch supplies at most three ids (get_global_id has three dimensions). The
        # name is addressed by the loop's position, not by its domain: `param` may itself be
        # an expression (`(self)->image_height`, for a receiver member), which is no C
        # identifier.
        gid:str = "_work_position"
        self._emit(f"const int {gid} = {self._flattened_launch_id(param)};")
        checks:List[str] = []
        for d, name in enumerate(names):
            # The stride to the next-faster loop axis is the size of the axes after this one,
            # so the slowest axis divides by every later extent and the fastest takes the
            # remainder of its own. That makes the decomposition the exact inverse of the
            # row-major flattening `_nd_flat_index` emits for a subscript.
            divisor:str = gid
            for inner in range(d + 1, ndim):
                divisor = f"{divisor} / {self._nd_bound_code(sources[inner])}"
            if d:
                divisor = f"{divisor} % {self._nd_bound_code(sources[d])}"
            self._emit(f"const int {name} = {divisor};")
            checks.append(f"{name} >= {self._nd_bound_code(sources[d])}")
        self._emit(f"if ({' || '.join(checks)})")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")
        self._emit_blank()

    def _flattened_launch_id(self, param:str)->str:
        """One flat work-item position, folded from whichever launch axes are present.

        `get_global_id(d)` reports a number in `[0, size(d))`, and a grid axis of one item
        reports 0, so folding every axis by its own grid extent and the strides of the axes
        between them yields the same position for a 1-D grid and for an N-D one of the same
        total. The grid extents are launch-time values the backend already exposes, so the
        generated source does not have to know the rank.

        Axis 0 is the fastest one, which is the order `_nd_flat_index` flattens the loop's own
        names in reverse -- so the loop's last name lands on gid 0.
        """
        if self._opencl_syntax:
            sizes:List[str] = [f"get_global_size({d})" for d in range(3)]
            ids:List[str] = [f"get_global_id({d})" for d in range(3)]
        else:
            sizes = ["(int)(gridDim.x * blockDim.x)", "(int)(gridDim.y * blockDim.y)",
                     "(int)(gridDim.z * blockDim.z)"]
            ids = [self._launch_id_code(axis) for axis in range(3)]
        # ((z % nz) * ny + y % ny) * nx + x % nx: one expression, so a rank-1 grid folds to
        # its own gid. Every group is parenthesised explicitly -- `%` and `*` share a
        # precedence level in C and associate left, so `a % b * c` is `(a % b) * c`, but
        # leaving the operands bare would also silently regroup the outer sum.
        return (f"((({ids[2]} % {sizes[2]}) * {sizes[1]} + ({ids[1]} % {sizes[1]}))"
                f" * {sizes[0]} + ({ids[0]} % {sizes[0]}))")

    def _nd_flat_index(self, names:List[str], param:str)->str:
        """Row-major flat index expression: n0*s1*s2 + n1*s2 + n2."""
        terms:List[str] = []
        ndim:int = len(names)
        for d, name in enumerate(names):
            term:str = name
            for inner in range(d + 1, ndim):
                term = f"{term} * {self._shape_var(param, inner)}"
            terms.append(term)
        return " + ".join(terms)

    def magic_loop_phases(self, body:List[ast.stmt])->List[List[ast.stmt]]:
        """Split a kernel body into one statement list per magic loop.

        A body with a single magic loop is one phase and the split is trivial. Two or more
        are separate *passes* over the work items, and they are translated as consecutive
        launches rather than one kernel: a second loop inside one kernel would be executed
        by every work item instead of once (see the refusal in _visit_statement).

        The statements before the first loop stay with it, since they belong to the first
        pass, and everything from the last loop to the end of the body is the last phase.
        Each returned list has exactly one magic loop, which is what the single-phase
        translation below expects.
        """
        # Only loops *shaped* like a magic loop count, and only from the second one on: the
        # first is the work-item dispatch, so the phases start at it and each later one adds
        # a boundary.
        loops:List[int] = [
            index for index, stmt in enumerate(body)
            if isinstance(stmt, ast.For) and self._is_secondary_magic_loop(stmt)]
        if len(loops) < 2:
            return [body]
        boundaries:List[int] = loops[1:]
        phases:List[List[ast.stmt]] = []
        start:int = 0
        for boundary in boundaries:
            phases.append(body[start:boundary])
            start = boundary
        phases.append(body[start:])
        return phases

    def _bind_carried_scalars(self, body:List[ast.stmt])->None:
        """Give every `carried` local a one-element buffer parameter instead of a device `T`.

        The parameter is appended in the same trailing position an outer value uses, and it
        is writable, so the body reads and writes it through the pointer and the host reads
        the final value back after the launch.
        """
        names:List[str] = list(self._carried_scalars(body))
        # Names the whole body carried that this phase initializes or writes: the read that
        # made them shared lives in a later pass, which this phase's slice no longer shows.
        hint:Set[str] = set(getattr(self, "phase_carried_hint", []) or [])
        magic_index:int = _magic_loop_index(body)
        written:Set[str] = set()
        for stmt in body[magic_index if magic_index < len(body) else len(body):]:
            for node in ast.walk(stmt):
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    written.add(node.target.id)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            written.add(target.id)
        for name in sorted(hint):
            if name not in names and name in written:
                names.append(name)
        for name in names:
            declared:DeviceType = self._lookup(name)
            if declared is not None:
                continue
            # The slot is the carried value's own width: the initializer above the loop
            # (`total = 0.0` is a float, `total = 0` an int) says what python answers,
            # and the atomic below is typed the same way. The slot used to be `int` for
            # every carrier, which made a float sum's `atomicAdd` an int atomic -- the
            # contributions were truncated (`0.5 + 0.25` answered 0), and the
            # interpreter, which runs python's own arithmetic, answered 0.75.
            element_t:object = "int"
            init_stmt:Optional[ast.stmt] = next(
                (stmt for stmt in body[:_magic_loop_index(body)]
                 if _assigned_name_of(stmt) == name), None)
            if init_stmt is not None:
                init_value:Optional[ast.expr] = _creation_target_of(init_stmt)
                if isinstance(init_value, ast.Constant):
                    if isinstance(init_value.value, float):
                        element_t = self.type_map.get("float", "float")
                    elif isinstance(init_value.value, int) \
                            and not isinstance(init_value.value, bool):
                        element_t = self.type_map.get("int", "int")
            self.carried_scalars.append(name)
            self.params.append((name, (PTR, element_t)))
            self._define(name, ("carried", name), body[0])

    def _translate_magic_loop(self, body:List[ast.stmt])->None:
        # The statements after the loop are python: they run once on the host, after the
        # launch, reading whatever the work items wrote (see PyKernel._run_host_region).
        # Recorded here so every dispatch shape treats the tail the same way.
        #
        # "After the loop" has to mean after the *dispatch* loop: a serial `for` above it
        # (`for j in range(lane * 4, lane * 4 + 4)`) is per-work-item code, and taking the
        # last `ast.For` of any kind as the boundary pushed everything between them -- a
        # `zeros_per_group` declaration included -- into the host tail, where a device-only
        # builtin has no work item and fails.
        self.body_tail = list(body[_magic_loop_end(body):])
        # `rows: int = out.shape[0]` above the loop: the local stands for the extent, and
        # an `np.ndindex(rows, ...)` bound resolves through it. Scanned up front, because
        # the loop *detection* below needs the table -- a bound held in a local is still
        # the work-item domain, and missing it made the loop read as a "second magic
        # loop" instead.
        for stmt in body[:_magic_loop_index(body)]:
            value:Optional[ast.expr] = _creation_target_of(stmt)
            name:Optional[str] = _assigned_name_of(stmt)
            if value is None or name is None:
                continue
            if isinstance(value, ast.Subscript) \
                    and isinstance(value.slice, ast.Constant) \
                    and isinstance(value.slice.value, int):
                holder:Optional[str] = self._shape_holder_param(value.value)
                if holder is not None:
                    holder_t:DeviceType = self._lookup(holder)
                    if is_ptr(holder_t) or is_image(holder_t):
                        self._shape_aliases[name] = (holder, int(value.slice.value))
        magic_index:Optional[int] = None
        enumerate_form:Optional[Translated] = None
        nd_form:Optional[Tuple[str, str, str]] = None
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, ast.For):
                continue
            if self._is_shape0_range(stmt.iter):
                magic_index = idx
                break
            enum_info:Optional[Translated] = self._enumerate_magic_target(stmt.iter)
            if enum_info is not None:
                magic_index = idx
                enumerate_form = enum_info
                break
            nd_info:Optional[Tuple[str, str, str]] = self._nd_magic_target(stmt.iter)
            if nd_info is not None:
                magic_index = idx
                nd_form = nd_info
                break

        if magic_index is None:
            # No magic loop: the user indexes via gpu.get_global_id and friends. The iteration
            # domain then has to come from somewhere: the decorator's `domain=`, or the bracket
            # at call time. Without either, the launch size cannot be derived (see
            # PyKernel._invoke).
            for stmt in body:
                self._visit_statement(stmt)
            return

        # Statements before the magic loop are translated as usual (e.g. local variable initialization).
        # The magic loop already fixes the iteration domain, so a `domain=` declaration is
        # redundant: PyKernel rejects a kernel that carries both before it reaches here.
        self.magic_loop_present = True
        self.magic_loop_line = int(getattr(body[magic_index], "lineno", 0) or 0)
        for stmt in _outside_loops(body):
            # A creation the device cannot perform is executed by the interpreter instead
            # and becomes a parameter; that is only possible here, above the loop, because
            # below it the statement would run once per work item.
            if self._host_created_assignment(stmt):
                continue
            self._visit_host_statement(stmt)

        if enumerate_form is not None:
            loop:ast.For = body[magic_index]
            param, arr_t = enumerate_form
            target = loop.target
            if not (isinstance(target, ast.Tuple) and len(target.elts) == 2
                    and all(isinstance(e, ast.Name) for e in target.elts)):
                self._error(
                    "the enumerate magic loop must unpack exactly two names: "
                    "`for i, x in enumerate(array)`", target)
            index_name:str = target.elts[0].id
            value_name:str = target.elts[1].id

            if arr_t is None:
                # A literal list: the domain is `len(list)` and the element is the constant
                # at that position. No array parameter is involved, so the dispatch reads the
                # length the list states.
                self.gid_param = ""
                self._emit_constant_enumerate_head(index_name, value_name, loop)
                self._visit_work_item_body(loop.body)
                # The domain is the list's own length: an empty array name marks a literal
                # extent (see the `name.startswith` mapping in PyKernel._compile).
                self.work_size_dims = [("", len(self._enumerate_constants))]
                return

            self.gid_param = param
            self._emit_work_item_head(index_name, param)
            # x is an element-value snapshot (matches python semantics: rebinding x does not write back to the array)
            elem_t:object = arr_t[1]
            self._define(value_name, elem_t, target.elts[1])
            self._emit(f"{elem_t} {value_name} = {param}[{index_name}];")

            self._visit_work_item_body(loop.body)
            self.work_size_dims = [(param, 0)]
            return

        if nd_form is not None:
            form, param, module_name = nd_form
            loop:ast.For = body[magic_index]
            target = loop.target

            # Target unpacking: ndindex -> (y, x); ndenumerate -> ((y, x), v)
            value_name:str = ""
            names_node:ast.expr = target
            if form == "ndenumerate":
                if not (isinstance(target, ast.Tuple) and len(target.elts) == 2
                        and isinstance(target.elts[1], ast.Name)):
                    self._error(
                        f"{module_name}.ndenumerate must unpack like "
                        f"`for (y, x), v in {module_name}.ndenumerate(array)`", target)
                names_node = target.elts[0]
                value_name = target.elts[1].id
            if not (isinstance(names_node, ast.Name)
                    or (isinstance(names_node, ast.Tuple)
                        and all(isinstance(e, ast.Name) for e in names_node.elts))):
                self._error(
                    f"{module_name}.{form} indices must be unpacked into plain names, "
                    f"e.g. `for y, x in {module_name}.ndindex(array.shape)`", target)
            # One name is the 1-D spelling (`for i in np.ndindex(arr.shape[0])`), which numpy
            # accepts; a tuple is the multi-dimensional one.
            names:List[str] = ([names_node.id] if isinstance(names_node, ast.Name)
                               else [e.id for e in names_node.elts])
            ndim:int = len(names)
            # A one-name domain (`for i in np.ndindex(arr.shape[0])`) is legal numpy and is
            # legal here: the head below handles any rank, and refusing this rank pushed a
            # plain 1-D body into the "not a magic loop" path, where the array created above
            # it became device code and the error named the wrong thing entirely.

            # One extent per unpacked name. A mismatch between the two is a *count*
            # mismatch, and python reports it as an unpacking ValueError raised by the
            # `for` statement itself -- so the device says the same thing, in the same
            # words, and the check below raises a ValueError rather than a translation
            # SyntaxError.
            # The whole-shape spelling takes its extents from the array, one per name; that
            # rank is known only at launch, because a pointer carries none.
            explicit:Optional[List[Tuple[str, int]]] = self._nd_loop_extents
            sources:List[Tuple[str, int]] = (explicit
                                             if explicit is not None
                                             else [(param, d) for d in range(ndim)])
            if explicit is not None and len(explicit) != ndim:
                # Positioned at the loop, so the message carries the file and line the
                # interpreter would blame (see _body_frame, which turns them into a frame).
                raise _unpacking_error(_unpack_count_message(ndim, len(explicit)),
                                       target, self.filename, self.source_text,
                                       self.first_line)
            if explicit is None:
                # `np.ndindex(arr.shape)` asks for the array's whole shape, but a pointer
                # carries no rank, so "whole" can only be checked at launch. Remember which
                # array it was: the launch compares its rank with `ndim` and refuses a loop
                # that names fewer dimensions than the array has.
                self.nd_whole_shape_param = param
            self._nd_loop_extents = sources

            self.gid_param = param
            self._emit_nd_head(names, param)
            if form == "ndenumerate":
                arr_t:DeviceType = self._lookup(param)
                elem_t:object = arr_t[1]
                self._define(value_name, elem_t, None)
                self._emit(f"{elem_t} {value_name} = {param}[{self._nd_flat_index(names, param)}];")
            if form == "ndenumerate" and isinstance(names_node, ast.Name):
                # `for index, v in ndenumerate(arr)` holds the index as a *tuple* in
                # python (`index == (i,)`, or `(i, j)` in 2-D), and a body may subscript
                # it (`index[0]`) before unpacking it -- numpy's own protocol. The name
                # below stands for that tuple: a constant ordinal reads the
                # decomposition the head already emitted, in python's order (slowest
                # first), which is the reverse of the head's fastest-last names.
                self._ndenumerate_index_names[names_node.id] = list(reversed(names))

            self._visit_work_item_body(loop.body)
            # The launch sizes dimensions fastest-first, which is the reverse of the order
            # the loop unpacks them: gid 0 is the last index. Getting this backwards used
            # to launch a 2x3 domain as 3x2, so the last column or row was never written.
            self.work_size_dims = list(reversed(sources))
            # The launch sizes dimensions fastest-first, which is the reverse of the loop's
            # unpack order -- the member names read by the host follow the same reversal, or
            # `image_height` would be read for the fastest axis.
            self.work_size_members = list(reversed(self.work_size_members))
            return

        loop:ast.For = body[magic_index]
        range_call:ast.Call = loop.iter
        bound:ast.expr = range_call.args[0]

        # Two names, because the two questions they answer have different answers for a
        # slice. `guard_name` is what the per-item bound is read from, which is the local
        # itself (`part.shape[0]` is the slice's length); `size_name` is the parameter the
        # *launch* is sized from, which is the array the local points at, because extents
        # are launch parameters named after arrays. Launching the array's full extent and
        # skipping the excess with the guard is the same arrangement the automatic work
        # size already uses when it rounds a grid up.
        raw_name:Optional[str] = self._loop_domain_name(bound.value)
        if raw_name is None:
            self._error(
                "a magic loop's iteration domain must be `<array>.shape[0]`", bound)
        guard_name:str = (raw_name if raw_name in self._array_slice_extents
                          else self._array_pointer_root(raw_name))
        size_name:str = self._array_pointer_root(raw_name)
        outer:Optional[OuterVarEntry] = self.outer_vars.get(size_name)
        host:Optional[HostValueEntry] = self.host_values.get(size_name)
        is_outer_array:bool = ((outer is not None and is_ptr(outer["type"]))
                               or (host is not None and is_ptr(host["type"])))
        if not is_outer_array and not is_ptr(self._lookup(guard_name)):
            self._error(f"'{raw_name}' is not an array parameter of this kernel", bound)

        self.gid_param = guard_name

        gid_bound:str = self._shape_var(guard_name, 0)
        target:str = loop.target.id
        # This branch emits its own work-item head (the 1-D form), so it has to record the
        # index variable that makes a `arr[i] += v` provably race-free just as the
        # multi-dimensional head does.
        self.work_item_indices = [target]
        self._define(target, "int", loop.target)
        if self._opencl_syntax:
            self._emit(f"const int {target} = get_global_id(0);")
        else:
            self._emit(f"const int {target} = (int)(blockIdx.x * blockDim.x + threadIdx.x);")
        self._emit(f"if ({target} >= {gid_bound})")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")
        self._emit_blank()

        # The loop body is the work-item body; nested range loops translate as real serial loops
        self._visit_work_item_body(loop.body)
        self.work_size_dims = [(size_name, 0)]

    # ---------- statements ----------

    # ---------- host-created values ----------
    #
    # A statement above the magic loop that only *builds* data -- np.zeros(...),
    # np.full(...), np.arange(...), np.float32(2.5), ... -- has no device meaning at all.
    # The interpreter runs it as written, the object it produced is passed to the kernel as
    # a parameter (or inlined, when it is a scalar constant the body only reads), and the
    # device code just uses it. This keeps the python semantics exactly: the array exists,
    # has the shape and dtype numpy gave it, and can be returned to the caller as the very
    # same object.
    #
    # The statement must be above the magic loop: below it, it would run once per work
    # item, creating N arrays where python creates one.

    def _bind_host_values(self, body:List[ast.stmt])->None:
        """Record every host-created object above the magic loop before the body is visited.

        The magic-loop scan needs to know which arrays exist -- `for i in
        np.ndindex(created.shape)` defines the iteration domain from a created array -- and
        every later use needs the object's type, so the declarations are collected once,
        up front, instead of being discovered while walking the body.
        """
        for stmt in body[:_magic_loop_index(body)]:
            value:Optional[ast.expr] = _creation_target_of(stmt)
            name:Optional[str] = _assigned_name_of(stmt)
            if value is None or name is None:
                continue
            if not self._is_creation_expression(value):
                continue
            produced:object = self.host_creation_values.get(name, _UNRESOLVED)
            if produced is _UNRESOLVED:
                self._error(
                    f"'{name}' cannot be created here: the expression only builds "
                    f"data, but it could not be evaluated on the host; build it from values "
                    f"that exist at call time (a shape, a constant, another array)",
                    value)
            device_type:DeviceType = infer_ctype(produced, self.type_map)
            self.host_values[name] = {"expr": value, "type": device_type}
            # The object travels as a parameter, appended after the outer-scope values
            # (see render_kernel and PyKernel._invoke_inner for the launch order).
            self.params.append((name, device_type))

    def _host_created_assignment(self, node:ast.stmt)->bool:
        """Skip one creation statement: it was already bound by _bind_host_values.

        Returns True when the statement was handled here (the device needs no code for it).
        Only a plain `name = <creation>` counts: a creation buried in an expression the
        device must evaluate is not a creation this can lift out.
        """
        name:Optional[str] = _assigned_name_of(node)
        if name is None:
            return False
        return name in self.host_values

    def _visit_host_statement(self, node:ast.stmt)->None:
        """Bind one host-region statement, or translate it when the host could not run it.

        A statement above the first magic loop (or between two of them) runs **once**, on
        the host, exactly as python reads it -- never once per work item. So when the
        interpreter already ran it (its line is in `host_statement_lines`, see
        `_host_region_values`), the device gets no code for it at all: only the values it
        bound travel to the kernel, as parameters.

        A statement the host could not run stays device code, and that is what keeps
        `zeros_per_group(...)` above a loop working: a shared buffer is declared by the
        kernel, so there is nothing for the host to execute.
        """
        if int(getattr(node, "lineno", 0) or 0) in self.host_statement_lines:
            self._bind_host_statement(node)
            return
        self._visit_statement(node)

    def _loop_written_names(self)->Set[str]:
        """Names the magic loops' bodies assign as a plain python local.

        Such a name is a plain local the work item updates; only one that is *also*
        initialized above the loop and read after it is carried through a device slot (see
        `_carried_scalars`). The distinction matters: the names here are what keeps a host
        scalar from being bound twice, so this is deliberately narrower than "any name the
        loop assigns" -- a `self.<attr> += ...` or an element store is not a local at all.
        """
        body:List[ast.stmt] = list(self.tree.body)
        start:int = _magic_loop_index(body)
        names:Set[str] = set()
        for statement in body[start:]:
            if not isinstance(statement, ast.For):
                continue
            for child in ast.walk(statement):
                if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                    names.add(child.target.id)
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
        return names

    def _bind_host_statement(self, node:ast.stmt)->None:
        """Give the device the values a host statement bound.

        The statement ran on the host (see `_host_region_values`), so its value exists at
        launch time; what is recorded here is which of its names the device has to receive.
        A name a device argument can carry becomes a scalar or array parameter. A name it
        cannot carry -- a string, a dict, an arbitrary-precision number -- stays on the
        host: nothing is bound for the device, and no statement that reads it can be device
        code (that was decided when the host ran it). A statement that binds no value
        (`import json`) contributes neither.
        """
        # `lo, hi = bounds(1)` above the loop: the host ran the helper, so both values are
        # known, and each name becomes its own scalar parameter. The device never calls the
        # helper -- the pair is data by the time the launch happens.
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Tuple):
            for element in node.targets[0].elts:
                if isinstance(element, ast.Name):
                    self._bind_host_value(element.id, element)
            return
        name:Optional[str] = _assigned_name_of(node)
        if name is None or name in self.host_values:
            return
        # An outer-scope name (`nonlocal held` in an enclosing `def`, or `global`) is *not*
        # a host value: it travels through its own slot (see _bind_outer_variables), and
        # every use of it already compiles against that. Treating it as a host scalar
        # declared it twice and gave the device a pointer where the outer slot is a value.
        if name in self.outer_vars:
            return
        # A name the *loop* updates (`total = 0.0` above it, `total = total + x[i]` inside)
        # is one location for the whole launch, carried through a device slot (see
        # _bind_carried_scalars). Its initializer is host work in the sense that python ran
        # it, but the slot is what carries the value, and the kernel is what finalizes it.
        # A `return total` *inside* the loop does not qualify as a carried scalar (the
        # helper requires the name to be read after the loop), so the loop body is
        # consulted directly.
        if name in self.carried_scalars or name in self._loop_written_names():
            # Not a host value: the loop updates it, so the ordinary walk owns the
            # initializer (a device local, or a carried slot). Translating it here is what
            # keeps the work item's own copy declared -- and it is what the body meant.
            self._visit_statement(node)
            return
        # `part = a[0:4]` is a pointer into the array, not a host value (see
        # _emit_array_slice): the device binds the offset itself.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript) \
                and isinstance(node.value.slice, ast.Slice):
            return
        # `weights = [1.0, 2.0, 3.0]` feeding `enumerate(weights)` is a compile-time table:
        # the loop reads the elements from the list's own literal (see
        # _emit_constant_enumerate_head), so the binding emits no device code and no
        # parameter carries it.
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            assigned:Optional[str] = _assigned_name_of(node)
            if assigned is not None and assigned in self._constant_list_names:
                return
        # `src = a` is a pointer assignment, not a host value: no parameter carries the
        # local, and letting it become a host scalar made the launch look for an extent the
        # argument list does not have (an `IndexError` from the shape check).
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            source_name:Optional[str] = self._pointer_source_name(node.value)
            if source_name is not None and is_ptr(self._lookup(source_name)):
                self._visit_statement(node)
                return
        produced:object = self.host_region_values.get(name, _UNRESOLVED)
        if produced is _UNRESOLVED:
            # The host ran the statement but bound no value under this name, so there is
            # nothing for the device to receive; the name simply stays host-side.
            return
        self._bind_host_value(name, node)

    def _bind_host_value(self, name:str, node:Optional[ast.AST])->None:
        """Give the device one host-bound name, as a scalar (or array) parameter.

        The value came from a statement the interpreter ran (see `_host_region_values`), so
        it exists at launch time. A type a device argument can carry becomes a parameter; a
        string, a dict or an arbitrary-precision number cannot, and stays on the host.
        """
        if name in self.host_values or name in self.host_scalars:
            return
        produced:object = self.host_region_values.get(name, _UNRESOLVED)
        if produced is _UNRESOLVED:
            return
        try:
            device_type:DeviceType = infer_ctype(produced, self.type_map)
        except SyntaxError:
            device_type = None
        if device_type is None:
            # Not something a device argument can carry: the value stays on the host.
            return
        # A *host value* is its own entry list, not an entry of host_values: host_values is
        # what the body *created* (see _bind_host_values), and the launch treats those
        # differently -- a creation is passed as an object, this one as a scalar.
        self.host_scalars[name] = device_type
        self.params.append((name, device_type))

    def _is_creation_expression(self, node:ast.expr)->bool:
        """True when this expression *builds data* the device cannot build itself.

        A plain constant is ordinary python that the device handles directly, so it is not
        a creation: `total = 0.0` stays a local variable.
        """
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.Name):
            return False
        if isinstance(node, ast.Attribute):
            # np.float32 and friends: a dtype/type object, not a value.
            if isinstance(node.value, ast.Name) and self.resolve_global(node.value.id) is np:
                return True
            return False
        if isinstance(node, ast.Call):
            func:ast.expr = node.func
            if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and self.resolve_global(func.value.id) is np):
                return False
            if self._is_numpy_dtype_name(func.attr):
                # `np.float32(2.5)` builds a constant the host can hold, while
                # `np.float32(x[i] * 2.0)` converts a device value in place (see
                # _emit_numpy_cast). Only the constant form is a creation, so a cast inside
                # the loop keeps working instead of being refused as a per-work-item one.
                return all(isinstance(arg, ast.Constant) for arg in node.args)
            return self._is_creation_function(func.attr)
        return False

    def _is_numpy_dtype_name(self, name:str)->bool:
        """True for a numpy scalar type (`np.float32`, `np.int32`, ...)."""
        return name in _NUMPY_DEVICE_TYPES or name in self.RETURN_SLOT_TYPES

    def _is_numpy_constant_cast(self, node:ast.expr)->bool:
        """True for `np.<dtype>(<constants>)`: one number, known at translation time.

        The creation guard treats a constant dtype call as a creation (see
        `_is_creation_expression`), and above the magic loop that is right -- the host
        builds the value and it travels as a parameter. Below the loop the value is one
        per work item, and a scalar is exactly that: the device cast computes the same
        number python would, so folding it to a constant keeps the statement a plain
        assignment instead of refusing a cast as if it allocated.
        """
        if not isinstance(node, ast.Call):
            return False
        func:ast.expr = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and self.resolve_global(func.value.id) is np):
            return False
        if not self._is_numpy_dtype_name(func.attr):
            return False
        return all(isinstance(arg, ast.Constant) for arg in node.args)

    def _is_creation_function(self, name:str)->bool:
        """A numpy function that only allocates or reshapes host data."""
        if name in _HOST_CREATION_FUNCTIONS:
            return True
        return name.endswith("_like")

    def _visit_work_item_body(self, statements:List[ast.stmt])->None:
        """Visit the magic loop's body: these statements run once per work item.

        A `return` here is exactly what a kernel return is (one work item leaving early),
        so this is *not* the "after the loop" region -- but a creation here would build one
        object per work item, which is why creations are refused with `after_loop_only`.

        `in_work_item_body` is set for the duration: a store to a launch-wide location here
        is one location shared by every work item, which is a race unless it is atomic or
        guarded (see `_named_outer_target`). It is deliberately **not** `after_magic_loop`:
        that flag means "runs after the dispatch", which a statement inside the loop does
        not, and setting it there refused a perfectly good `return` from inside the loop.
        """
        previous:bool = self.after_magic_loop
        previous_creation:bool = self.creation_below_loop
        previous_body:bool = self.in_work_item_body
        self.after_magic_loop = previous
        self.creation_below_loop = True
        self.in_work_item_body = True
        try:
            for stmt in statements:
                self._visit_statement(stmt)
        finally:
            self.after_magic_loop = previous
            self.creation_below_loop = previous_creation
            self.in_work_item_body = previous_body

    def _store_is_guarded(self, node:ast.AST)->bool:
        """Whether a store is inside a branch every work item takes the same way.

        A template for one work item's slice is a real pattern (`if i == 0: total = x[0]`),
        and its store is not racy because exactly one work item enters the branch. The
        condition has to be a test on a *launch-wide* value -- the work-item index, a scalar
        argument, an outer scalar -- so a data-driven condition (`if x[i] > 0`) stays refused:
        there every work item decides for itself, which is the race.
        """
        for guard in self._guard_conditions:
            names:List[str] = [n.id for n in ast.walk(guard)
                               if isinstance(n, ast.Name)]
            if not names:
                continue
            if all(self._is_launch_wide_name(name) for name in names):
                return True
        return False

    def _is_launch_wide_name(self, name:str)->bool:
        """Whether a name's value is the same in every work item.

        The magic loop's own indices, the kernel's scalar parameters and its outer-scope
        values all are; an array element, a per-item local and a value computed inside the
        loop are not.
        """
        if name in self.work_item_indices:
            return True
        binding:DeviceType = self._lookup(name)
        if binding is None:
            # An unresolved name is a module global or a builtin -- the same for everyone.
            return True
        if isinstance(binding, tuple):
            return False
        if is_ptr(binding) or is_image(binding):
            # A pointer says nothing by itself: `x[i]` is per-item even though `x` is not.
            return False
        return True

    def _visit_loop_exit(self, node:ast.stmt, spelling:str)->None:
        """Translate `break`/`continue` for the loop they actually sit in.

        Three cases, and the difference between them is not cosmetic:

        * **Inside a real loop** (a nested serial `for`/`while`, or the body of an element
          loop): the generated source has a C loop there, so the statement is emitted
          verbatim and keeps python's meaning.
        * **Directly in a work item** (the magic loop's body, which becomes the dispatch
          rather than a loop): `continue` means "this work item stops here and the next one
          -- which is another thread -- does its own work", so it becomes a plain `return;`.
          That is the *same* control flow, because there is no loop left to continue. It must
          NOT go through _visit_kernel_return: that one claims the kernel result, so a work
          item merely skipping its turn would race the value-returning sites for the answer
          and could hand back None.
        * **`break` in a work item**: there is no meaning to give it. Stopping the magic
          loop means stopping every other work item too, and they are already running in
          parallel, so the only honest translation is a refusal.
        """
        if self.loop_depth:
            self._emit(f"{spelling};")
            return
        if not self.is_kernel or not self.creation_below_loop:
            # Outside a loop and outside a work item: nothing to exit and nothing to
            # enforce. `break`/`continue` outside a loop is a SyntaxError in python too,
            # and the body's own statements above the loop never reach here as device code.
            self._emit(f"{spelling};")
            return
        if spelling == "continue":
            self._emit("return;")
            return
        self._error(
            "`break` cannot stop a magic loop: its iterations are work items running in "
            "parallel, so stopping the loop means stopping every other work item too, which "
            "the device cannot express. Guard the rest of this work item's body with `if`, "
            "or put the sequential loop you want to leave inside the body with "
            "`for j in range(...)` and `break` that one instead", node)

    def _carried_scalars(self, body:List[ast.stmt])->List[str]:
        """The locals assigned *above* the magic loop and written *inside* it.

        Such a name is a device `T` today, so every work item accumulates into its own copy
        and the result is one work item's partial sum rather than the whole one -- silently,
        because nothing about the shape of the program looks wrong. Python has one `total`:
        it is initialized once, then updated by every iteration in turn.

        Carrying it as a one-element array is the same shape the framework already uses for a
        writable outer scalar (`T*`, read and written through the pointer), so the device
        shares one location across the whole launch and the host can read the final value.
        The statements above the loop still run on the host -- that is where the initial
        `total = 0` comes from -- and the kernel receives the slot, not the value.
        """
        magic_index:int = _magic_loop_index(body)
        initialized:Dict[str, ast.stmt] = {}
        for stmt in body[:magic_index]:
            name:Optional[str] = _assigned_name_of(stmt)
            if name is None:
                continue
            initialized[name] = stmt
        written:set = set()
        for stmt in body[magic_index:]:
            for node in ast.walk(stmt):
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    written.add(node.target.id)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            written.add(target.id)
        # Only a name the loop's *result* carries needs the shared location: one whose value
        # is still read after the loop. A name used only inside the loop is each work item's
        # own scratch, and making it shared would turn a per-item value into a race -- the
        # behaviour is deterministic today and would stop being so.
        # Everything after the loop *itself*: the loop's body runs once per work item, so a
        # name read there is per-item scratch, while one read after the loop is the value the
        # launch is supposed to have produced.
        after:List[ast.stmt] = body[magic_index + 1:]
        used_after:set = {
            node.id for stmt in after for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        carried:List[str] = []
        for name in initialized:
            if name in written and name in used_after:
                carried.append(name)
        return carried

    def _non_atomic_writes(self, body:List[ast.stmt], names:List[str]) -> List[ast.stmt]:
        """The statements inside the loop that *store* one of ``names`` without `+=`.

        A carried slot is one location for the whole launch, so a plain store there is a
        race: every work item overwrites it, and the value that survives is whichever
        finished last. `total += x[i]` is fine (it becomes an atomic), and so is a
        conditional update built on it; `best = x[i]` inside an `if` is not, and it has to be
        refused because it compiles and then answers one work item's number.
        """
        if not names:
            return []
        magic_index:int = _magic_loop_index(body)
        if magic_index >= len(body) or not isinstance(body[magic_index], ast.For):
            # A gid-style body has no magic loop, so there is no per-item region to check.
            return []
        loop:ast.For = body[magic_index]
        offenders:List[ast.stmt] = []
        for stmt in loop.body:
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        offenders.append(node)
        return offenders

    def _is_secondary_magic_loop(self, node:ast.For)->bool:
        """True for a `for` shaped like the work-item dispatch, at the top level.

        The shape is the one the first loop has -- `range(arr.shape[0])`, `enumerate(arr)`,
        `np.ndindex(...)` -- but its position is what matters: only the first one is the
        launch. A `range` over something that is not an array's shape (`range(8)`) does not
        name a work-item domain, so it is an ordinary serial loop and stays allowed, exactly
        as it was before the several-pass split existed.

        An argument that is a local alias of an array (`src = a`, then `enumerate(src)`) is
        the same domain: resolving the alias here is what keeps a body with one dispatch
        loop from reading as a body with two.
        """
        iterator:ast.expr = node.iter
        if isinstance(iterator, ast.Call):
            func:ast.expr = iterator.func
            if isinstance(func, ast.Name) and func.id == "enumerate":
                return True
            if isinstance(func, ast.Name) and func.id == "range":
                # Only a data-shaped range is a dispatch; a literal count is a plain loop.
                return self._shape0_range_param(iterator) is not None
            if (isinstance(func, ast.Attribute) and func.attr in ("ndindex", "ndenumerate")
                    and isinstance(func.value, ast.Name)):
                return True
        return False

    def _visit_region(self, statements:List[ast.stmt], after_magic_loop:bool)->None:
        """Visit a run of statements, marking whether they run once per work item.

        Everything after the magic loop is emitted into every work item, so a value
        `return` there hands back one work item's own local -- not the single value the
        python function would compute after its loop. `_visit_kernel_return` refuses such
        a return; an array return stays allowed, since the object is the same whichever
        work item names it.
        """
        previous:bool = self.after_magic_loop
        self.after_magic_loop = previous or after_magic_loop
        try:
            for stmt in statements:
                self._visit_statement(stmt)
        finally:
            self.after_magic_loop = previous

    def _visit_statement(self, node:ast.stmt)->None:
        # A creation statement was executed by the interpreter and travels as a parameter,
        # so the device emits nothing for it. Checked here as well as in the magic-loop
        # header: a gid-style body has no loop above which to lift them, yet its top-level
        # statements are just as much "once per launch".
        if self.is_kernel and self._host_created_assignment(node):
            return
        if isinstance(node, ast.Assign):
            # An initializer for a carried scalar already ran on the host, and the slot that
            # travels is the buffer it filled: re-emitting 	otal = 0 would overwrite the
            # parameter's type and reset the shared location for every work item.
            target:object = node.targets[0] if len(node.targets) == 1 else None
            if (isinstance(target, ast.Name) and target.id in self.carried_scalars
                    and not self.creation_below_loop):
                return
            self._visit_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._visit_augassign(node)
        elif isinstance(node, ast.If):
            # `if KERNEL_TRANSLATING:` / `if not KERNEL_TRANSLATING:` is a *compile-time*
            # branch, the translator's counterpart of `typing.TYPE_CHECKING`: the constant is
            # False in python (so the interpreter backend runs the python branch, which it
            # does by executing the body as written) and True here, so exactly one branch is
            # translated and the other is dropped -- a macro, not a runtime test. Any other
            # condition is an ordinary device `if`.
            taken:Optional[List[ast.stmt]] = _translation_branch(node.test, node.body,
                                                                 node.orelse)
            if taken is not None:
                self._emit_block(taken)
                return
            cond:str = self._condition_code(node.test)
            # `_condition_code` already wraps the condition in the one pair C's syntax needs
            # -- it has to, because the boolean join and the ternary below splice it into a
            # larger expression. Emitting `if ({cond})` therefore produced `if ((n == k))`,
            # and the OpenCL compiler reported the inner pair as an extraneous one around a
            # comparison (with a `use '=' ...` hint that would change the meaning).
            self._emit(f"if {cond}")
            self._emit("{")
            self._guard_conditions.append(node.test)
            try:
                self._push_scope_block(node.body)
            finally:
                self._guard_conditions.pop()
            if node.orelse:
                self._emit("}")
                self._emit("else")
                self._emit("{")
                self._push_scope_block(node.orelse)
            self._emit("}")
            self._emit_blank()
        elif isinstance(node, ast.While):
            cond:str = self._condition_code(node.test)
            # Same as the `if` above: the condition arrives with its own pair.
            self._emit(f"while {cond}")
            self._emit("{")
            self._push_scope_block(node.body, is_loop=True)
            self._emit("}")
            self._emit_blank()
        elif isinstance(node, ast.For):
            # A second *top-level* magic loop would silently become an ordinary serial loop
            # inside every work item -- each item running the whole phase, all of them racing
            # on the same array. That is a different computation, so it is refused here
            # rather than emitted; splitting the body across several @kernel calls is what
            # this means. A `for` inside the work-item body (`creation_below_loop`) is an
            # ordinary loop and stays allowed.
            if (self.is_kernel and not self.creation_below_loop
                    and self._is_secondary_magic_loop(node)):
                self._error(
                    "this kernel body has more than one magic loop, and only the first "
                    "becomes the work-item dispatch: a second one would run inside every "
                    "work item instead of once, so several work items would repeat the same "
                    "phase and race on the same array. Give each phase its own @kernel and "
                    "call them in order, which keeps the data on the device", node)
            self._visit_for(node)
        elif isinstance(node, ast.Return):
            self._visit_return(node)
        elif isinstance(node, ast.Expr):
            # Bare string literals (e.g. the function docstring) are ignored, not emitted into the kernel
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                pass
            elif self._is_print_call(node.value):
                # print(...) maps to device printf calls that print the same text
                self._emit_print(node.value)
            else:
                # Other expression statements (e.g. gpu.atomic_add(...)/gpu.barrier()) must be emitted, otherwise side effects are lost
                code, _ = self._visit_expression(node.value)
                if code:
                    self._emit(f"{code};")
        elif isinstance(node, ast.AnnAssign):
            # `x: float = 0.0` is ordinary Python. The annotation has no device meaning --
            # the type comes from the assigned value, as it does for every other name -- so
            # it is dropped and the statement becomes a plain assignment. A bare `x: float`
            # declares nothing, which a kernel has no use for, so it is ignored the way a
            # docstring is.
            if node.value is not None:
                self._visit_assign(ast.Assign(targets=[node.target], value=node.value,
                                              lineno=node.lineno,
                                              col_offset=node.col_offset))
        elif isinstance(node, ast.Break):
            self._visit_loop_exit(node, "break")
        elif isinstance(node, ast.Continue):
            self._visit_loop_exit(node, "continue")
        elif isinstance(node, (ast.Nonlocal, ast.Global)):
            # The scope declarations themselves emit nothing: they were already read while
            # binding the outer variables (see _bind_outer_variables), so the name travels
            # as a device parameter and every use goes through it.
            pass
        elif isinstance(node, ast.Pass):
            pass
        else:
            self._error(f"unsupported statement: {node.__class__.__name__}", node)

    def _emit_block(self, statements:List[ast.stmt])->None:
        """Translate statements as a plain block, with a scope of their own.

        Used by the compile-time branch (`if KERNEL_TRANSLATING`), which contributes its
        statements *inline*: it is a macro, so no `if` and no braces survive into the
        generated source.
        """
        self.scopes.append({})
        for statement in statements:
            self._visit_statement(statement)
        self.scopes.pop()

    def _push_scope_block(self, statements:List[ast.stmt], is_loop:bool=False)->None:
        """Translate a braced block, optionally marking it as a real C loop body.

        `is_loop` is what makes `break`/`continue` mean their C meaning inside this block:
        the generated source emits them verbatim, and only a real loop can contain them
        (see _visit_loop_exit). A plain block -- the `if` half of an `if`/`else` -- leaves
        the depth alone, so an exit statement inside it belongs to whichever enclosing
        loop there is.
        """
        self.scopes.append({})
        self._indent += 1
        if is_loop:
            self.loop_depth += 1
        try:
            for stmt in statements:
                self._visit_statement(stmt)
        finally:
            if is_loop:
                self.loop_depth -= 1
            self._indent -= 1
            self.scopes.pop()

    def _visit_for(self, node:ast.For)->None:
        if isinstance(node.iter, (ast.Name, ast.Attribute)):
            # `for x in arr`: an element-wise walk becomes an indexed one, with `x` bound to
            # the current element by value -- the same read-only snapshot `enumerate` gives.
            self._emit_element_loop(node)
            return
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "enumerate"):
            self._error(
                "enumerate() is only supported as the outermost magic loop "
                "(`for i, x in enumerate(array)`), not as a real inner loop", node.iter)
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr in ("ndindex", "ndenumerate")
                and isinstance(node.iter.func.value, ast.Name)
                and self.resolve_global(node.iter.func.value.id) is np):
            self._error(
                f"np.{node.iter.func.attr} is only supported as the outermost magic loop, "
                f"not as a real inner loop", node.iter)
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"):
            self._error("only `for ... in range(...)` loops are supported", node.iter)
        args:List[ast.expr] = node.iter.args
        if len(args) not in (1, 2, 3) or node.iter.keywords:
            self._error("range() takes 1-3 positional arguments", node.iter)
        if not isinstance(node.target, ast.Name):
            self._error("loop target must be a plain variable", node.target)

        values:List[Translated] = [self._visit_expression(a) for a in args]
        for a, (_, t) in zip(args, values):
            if not is_int(t):
                self._error("range() arguments must be integers", a)
        target:str = node.target.id
        self._define(target, "int", node.target)

        if len(values) == 1:
            start, stop, step = "0", values[0][0], None
        elif len(values) == 2:
            start, stop, step = values[0][0], values[1][0], None
        else:
            start, stop, step = values[0][0], values[1][0], values[2][0]

        if step is None:
            self._emit(f"for (int {target} = {start}; {target} < {stop}; {target}++)")
            self._emit("{")
        elif re.fullmatch(r"-?\d+", step):
            cond_op:str = (">" if step.startswith("-") else "<")
            self._emit(f"for (int {target} = {start}; {target} {cond_op} {stop}; {target} += {step})")
            self._emit("{")
        else:
            self._error("range() step must be a constant integer", node.iter.args[-1])
        self._push_scope_block(node.body, is_loop=True)
        self._emit("}")
        self._emit_blank()

    def _emit_element_loop(self, node:ast.For)->None:
        """`for x in arr:` as an indexed loop with `x` a copy of the current element.

        A work item cannot write through an iterator, so the element is read-only -- which is
        what python's iteration gives for a record too. The index variable is hidden, so a
        body that also declares `i` does not collide with it.
        """
        if not isinstance(node.target, ast.Name):
            self._error("iterating an array needs a plain variable: `for x in arr`", node.target)
        code, t = self._visit_expression(node.iter)
        if not is_ptr(t):
            self._error(
                "iterating a value directly needs an array; this is a scalar. Use an array "
                "argument, or a range() loop", node.iter)
        element_t:object = t[1]
        holder:ast.expr = node.iter
        # The extent: a parameter's own shape variable, or a structure member's fixed length.
        if isinstance(holder, ast.Name):
            stop:str = self._shape_var(holder.id, 0)
        elif isinstance(holder, ast.Attribute) and isinstance(holder.value, ast.Name):
            base_t:DeviceType = self._lookup(holder.value.id)
            base_element:object = base_t[1] if is_ptr(base_t) else base_t
            dtype:Optional[np.dtype] = (structs_dtype_for(base_element)
                                        if isinstance(base_element, str) else None)
            if dtype is None or holder.attr not in (dtype.names or ()):
                self._error(
                    f"cannot iterate '{ast.unparse(holder)}': it is not an array member",
                    holder)
            field_dtype:np.dtype = np.dtype(dtype.fields[holder.attr][0])
            if field_dtype.subdtype is None:
                self._error(
                    f"cannot iterate '{ast.unparse(holder)}': a scalar member is not an array",
                    holder)
            stop = str(field_dtype.subdtype[1][0])
        else:
            self._error(
                f"cannot iterate '{ast.unparse(holder)}': use a named array parameter or an "
                f"array member of one", node.iter)

        name:str = node.target.id
        index:str = f"_{name}_idx"
        self._emit(f"for (int {index} = 0; {index} < {stop}; {index}++)")
        self._emit("{")
        # The declaration belongs to the loop body, so it is emitted after the indent step:
        # before it, the line landed at the `for` line's own level and every element loop in
        # a generated source read as if the copy lived outside its block.
        self._define(name, element_t, node.target)
        self._indent += 1
        self.loop_depth += 1
        try:
            self._emit(f"{self._device_type(element_t)} {name} = {code}[{index}];")
            for statement in node.body:
                self._visit_statement(statement)
        finally:
            self.loop_depth -= 1
            self._indent -= 1
        self._emit("}")
        self._emit_blank()

    # ---------- kernel return values ----------
    #
    # A kernel is one work item, so it has no C return value to hand back: a launch of N
    # work items returns N values, and only one of them can be "the" result. The kernel
    # therefore returns through parameters: every returned position gets a slot of its own
    # -- a one-element buffer for a value, the array parameter's own name for an array --
    # plus one shared claim flag. The first work item to reach a `return` claims the
    # result with an atomic compare-and-swap and writes its own return site's values.
    #
    # Python lets every `return` statement have its own shape, so the sites are NOT
    # required to agree: site 7 may return a float and site 9 a (float, int) pair. What
    # the device signature needs is the union of the slots every site can write, which is
    # collected here; the winning site is recorded so the host can rebuild exactly the
    # value that site produced (and None for a bare `return`).
    #
    # Nesting a tuple inside a returned tuple is refused: it has no slot layout, and the
    # host could not tell it from a flat tuple of the same length.

    RETURN_CLAIM:str = "_ret_claim"
    RETURN_SITE:str = "_ret_site"

    # The C element type a value slot is declared with, and the numpy dtype the host uses
    # for the same slot. A slot is internal storage, so its C type only has to hold the
    # value: bool travels as a byte (OpenCL rejects a _Bool kernel parameter and the CUDA
    # binding table has no bool entry), and char also travels as an unsigned byte because
    # OpenCL's `char` binds to ctypes.c_char (|S1), which is not an integer array at all.
    # The host converts when it reads the slot back.
    RETURN_SLOT_TYPES:Dict[str, Translated] = {
        "bool": ("uchar", np.uint8),
        "char": ("uchar", np.uint8),
        "uchar": ("uchar", np.uint8),
        "short": ("short", np.int16),
        "ushort": ("ushort", np.uint16),
        "int": ("int", np.int32),
        "uint": ("uint", np.uint32),
        "long": ("long", np.int64),
        "ulong": ("ulong", np.uint64),
        "float": ("float", np.float32),
        "double": ("double", np.float64),
        # A complex value's slot is one complex element for the same reason: the structure
        # the device declares is the complex layout, so a complex array *is* the buffer.
        "c64": ("c64", np.complex64),
        "c128": ("c128", np.complex128),
    }

    def _return_position(self, node:ast.expr)->ReturnPosition:
        """Classify one returned expression as an array position or a value position."""
        if isinstance(node, ast.Subscript):
            # A slice/element of an array is not an array: it would be a single element,
            # which a kernel cannot hand back as a view (the device sees a flat pointer).
            if isinstance(node.slice, ast.Slice):
                self._error(
                    "a kernel cannot return an array slice: return the whole array "
                    "parameter by name, or return the element as a value", node)
        code, t = self._visit_expression(node)
        if is_image(t):
            self._error("cannot return an image: image parameters are opaque handles", node)
        if is_ptr(t):
            # An array travels as its own parameter: the host already holds the object, so
            # there is nothing to copy and nothing to allocate. Only a name identifies a
            # parameter, so an expression is refused rather than silently copied. A
            # `self.<attr>` array counts as a name: it is a parameter too (`_self_<attr>`).
            if not isinstance(node, (ast.Name, ast.Attribute)):
                self._error(
                    "a kernel can return an array parameter, but it must name it "
                    "(e.g. `return out`), not an expression", node)
            name:str = self._array_name(node)
            if name not in {param_name for param_name, _ in self.params}:
                # A host-created array is a parameter too, just not a python argument.
                if name not in self.host_values:
                    self._error(f"cannot return the array '{name}': it is not a parameter", node)
            # A carried slot is a pointer to one value, so returning it hands back that value
            # -- not an array. The parameter already holds it, so no slot is allocated.
            if name in self.carried_scalars:
                return {"kind": "value", "type": t[1], "code": f"{name}[0]",
                        "slot_parameter": name}
            if name in self.promoted_parameters:
                # Promoted for the return's sake: it is a *value*, and the parameter it now
                # travels as is the slot it lands in -- no extra argument.
                self.promoted_returns.add(name)
                return {"kind": "value", "type": t[1], "code": f"{name}[0]",
                        "slot_parameter": name}
            return {"kind": "array", "name": name, "type": t}
        if isinstance(node, ast.Name) and (
                node.id in self.carried_scalars or node.id in self.promoted_parameters):
            # Both are one location the kernel already carries: the parameter *is* the slot.
            return {"kind": "value", "type": t, "code": code,
                    "slot_parameter": node.id}
        return {"kind": "value", "type": t, "code": code}

    def _return_site(self, node:ast.Return)->List[ReturnPosition]:
        """The shape of one `return` statement, regardless of what other sites return."""
        if node.value is None:
            return []
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            # `return None` is the same as a bare `return` in python.
            return []
        if isinstance(node.value, ast.Tuple):
            for item in node.value.elts:
                if isinstance(item, ast.Tuple):
                    self._error(
                        "a returned tuple position must be a single value, not a nested tuple", item)
            return [self._return_position(item) for item in node.value.elts]
        return [self._return_position(node.value)]

    def _promote_returned_parameters(self, node:ast.Return)->None:
        """Turn a by-value parameter the body returns into a pointer parameter.

        A scalar argument travels by value, so the device only holds a copy and there is
        nothing for a return slot to point at -- the kernel would allocate a second
        parameter just to carry a number the caller already passed. Passing such an
        argument by pointer instead gives the return slot the parameter itself: one
        location, no extra argument, and the same value on both sides.
        """
        if node.value is None:
            return
        values:List[ast.expr] = (list(node.value.elts) if isinstance(node.value, ast.Tuple)
                                else [node.value])
        for value in values:
            if not isinstance(value, ast.Name):
                continue
            for index, (name, t) in enumerate(self.params):
                if name != value.id or is_ptr(t) or is_image(t):
                    continue
                # A writable carried slot is already a pointer; this only reaches an ordinary
                # scalar argument (`scale: int`).
                self.params[index] = (name, (PTR, t))
                # The scope has to follow, or a later lookup still reports the by-value type
                # and the layout allocates a slot for a value that already has one.
                for scope in self.scopes:
                    if name in scope:
                        scope[name] = (PTR, t)
                self.promoted_parameters.add(name)

    def _return_is_shared(self, node:ast.Return)->bool:
        """True when every name a post-loop return reads is a launch-wide slot.

        A carried scalar (written inside the loop, read after it) is one location for the
        whole launch, so a work item's copy of the expression is the call's result -- there
        is nothing per-item left for the claim to arbitrate. A kernel argument is shared for
        the same reason. Anything else (a local assigned below the loop) is one work item's
        value, which is what the refusal is about.
        """
        if node.value is None:
            return True
        values:List[ast.expr] = (list(node.value.elts) if isinstance(node.value, ast.Tuple)
                                else [node.value])
        for value in values:
            for name in ast.walk(value):
                if not isinstance(name, ast.Name) or not isinstance(name.ctx, ast.Load):
                    continue
                if name.id in self.carried_scalars:
                    continue
                # A kernel argument is launch-wide too: every work item reads the same value,
                # so returning one answers the same thing everywhere. A pointer argument (an
                # array) was already allowed by the array-return rule.
                if name.id in dict(self.params) or is_ptr(self._lookup(name.id)):
                    continue
                return False
        return True

    def _visit_kernel_return(self, node:ast.Return)->None:
        """Translate a `return` in a kernel body into a claim plus slot stores."""
        # Promotion comes first: it decides the parameter's type, and the layout below reads
        # that type to tell a reused parameter from a value needing a slot of its own.
        if self.is_kernel:
            self._promote_returned_parameters(node)
        positions:List[ReturnPosition] = self._return_site(node)
        if self.after_magic_loop and any(position["kind"] == "value" for position in positions):
            # The statements after the loop run once per work item, so a value accumulated
            # there is that work item's -- unless it is read from a *carried* slot, which is
            # one location for the whole launch. `total = 0; for ...: total += x[i];
            # return total` therefore answers the same thing in every work item, and the
            # claim word hands back the one value the call computed.
            if not self._return_is_shared(node):
                self._error(
                    "a value-returning return cannot appear after the magic loop: the "
                    "statements after the loop run once per work item, so a value computed "
                    "there belongs to one work item, not to the whole call. Accumulate it in "
                    "a scalar above the loop with `+=` (one shared slot), return from inside "
                    "the loop, or reduce into an array argument and return that array", node)
        site:int = len(self.return_sites)
        self.return_sites.append(positions)
        self._record_return_slots(site, positions, node)

        if not self.has_returns:
            # No return site of this kernel returns a value at all (see
            # _returns_a_value): a plain return, and the result is None.
            self._emit("return;")
            return

        # Every site claims, including a bare `return`: in python the first work item to
        # leave the function decides the result, so a bare return must be able to win
        # with None just as a value site wins with its tuple.
        claim:str = f"{self.RETURN_CLAIM}[0]"
        if self._metal_syntax:
            # MSL's compare-and-swap takes the expected value by pointer and answers a
            # bool rather than the old value. So the claim is tried with a zeroed
            # expected local and then read back: the word is 1 only for the work item
            # whose exchange actually happened, which is what `claimed == 1` tests.
            self._emit("int _claim_expected = 0;")
            self._emit(f"atomic_compare_exchange_weak_explicit("
                       f"(device atomic_int*)(&{claim}), &_claim_expected, 1, "
                       f"memory_order_relaxed, memory_order_relaxed);")
            self._emit(f"const bool _claimed = ({claim} == 1);")
            guard = "_claimed"
        elif self._opencl_syntax:
            guard = f"atomic_cmpxchg((volatile __global int*)(&{claim}), 0, 1) == 0"
        else:
            guard = f"atomicCAS(&{claim}, 0, 1) == 0"
        self._emit(f"if ({guard})")
        self._emit("{")
        # The winning site records what it returned, so the host rebuilds that site's
        # shape rather than a shape inferred from every site at once.
        self._emit(f"    {self.RETURN_SITE}[0] = {site + 1};")
        for slot, position in self._site_slots(site):
            if position["kind"] != "value":
                continue
            if slot < 0:
                # The value already lives in the parameter it travels in: there is no slot to
                # store it into, and the host reads it from the argument it passed.
                continue
            stored:str = self._complex_store(self.return_layout[slot]["type"],
                                             position["type"], position["code"])
            self._emit(f"    {self._return_slot(slot)}[0] = {stored};")
        self._emit("}")
        self._emit("return;")

    def _return_slot(self, index:int)->str:
        return f"_ret{index}"

    def _record_return_slots(self, site:int, positions:List[ReturnPosition],
                             node:ast.Return)->None:
        """Give each value position of this site a slot, extending the shared layout.

        A slot is keyed by its C element type, so two sites returning a float share one
        buffer while a site returning an int gets a buffer of its own. A returned array
        occupies no slot: the host hands back the very object it passed in.
        """
        for ordinal, position in enumerate(positions):
            if position["kind"] != "value":
                continue
            if position.get("slot_parameter"):
                # The value already has a location of its own -- the parameter it travels
                # in -- so no slot is allocated for it: there is nothing to copy and the
                # host reads it from the argument it passed.
                self._site_slot_map.setdefault(site, []).append((ordinal, -1))
                continue
            ctype:str = self._return_param_type(position["type"])
            slot:Optional[int] = self._slot_for_type(ctype)
            if slot is None:
                slot = len(self.return_layout)
                self.return_layout.append({"ctype": ctype, "type": position["type"]})
            self._site_slot_map.setdefault(site, []).append((ordinal, slot))

    def _slot_for_type(self, ctype:str)->Optional[int]:
        for index, entry in enumerate(self.return_layout):
            if entry["ctype"] == ctype:
                return index
        return None

    def _site_slots(self, site:int)->List[Tuple[int, ReturnPosition]]:
        """(slot, position) pairs this site stores into."""
        positions:List[ReturnPosition] = self.return_sites[site]
        pairs:List[Tuple[int, ReturnPosition]] = []
        for ordinal, slot in self._site_slot_map.get(site, []):
            pairs.append((slot, positions[ordinal]))
        return pairs

    def _return_param_type(self, t:object)->str:
        """The C element type of a value slot."""
        from . import structs

        if is_vec(t):
            return t
        if isinstance(t, str) and structs.dtype_for(t) is not None:
            # A structure value travels as a one-element buffer of that structure, so the
            # slot element is the structure type itself.
            return t
        slot:Optional[Translated] = self.RETURN_SLOT_TYPES.get(t)
        if slot is None:
            self._error(f"cannot return a value of type {t}")
        return self._device_type(slot[0])

    def return_slot_dtype(self, index:int)->np.dtype:
        """The numpy dtype the host uses for one value slot.

        A vector slot uses the *host* layout (a 3-component vector has three components);
        `pygpgpu.vec_types.layout.pack` re-lays it out for the backend that needs padding,
        the same way it does for vector arguments.
        """
        from ..vec_types.layout import _host_classes
        from . import structs

        entry:ReturnLayoutEntry = (self.return_layout or [])[index]
        if is_vec(entry["type"]):
            return _host_classes()[entry["type"]].numpy_dtype()
        structure:object = structs.dtype_for(entry["type"]) if isinstance(entry["type"], str) else None
        if structure is not None:
            return structure
        return self.RETURN_SLOT_TYPES[entry["type"]][1]

    def render_return_params(self)->List[str]:
        """The extra device parameters a kernel with returns needs.

        A kernel that returns only an array still needs the claim and site words, so this
        is gated on `has_returns` rather than on the value slots alone.

        The claim and site words are withheld when this kernel contains **no device-side
        return at all**: a body whose only `return` sits after the magic loop hands its value
        back through the host tail (see PyKernel._run_host_region), so no work item ever
        reaches the claim, the word stays zero, and the two buffers would become a pair of
        wasted uploads. A phase of a multi-loop body is considered the same way and on its
        own: every phase is a separate kernel, and whether one of them claims is a fact
        about the statements it contains.
        """
        if not self.has_returns:
            return []
        qualifier:str = self._buffer_qualifier()
        params:List[str] = [f"{qualifier}{entry['ctype']}* {self._return_slot(index)}"
                            for index, entry in enumerate(self.return_layout)]
        # A movable claim/site pair, decided by whether anything on the device claims: a
        # tail-only return never reaches the translator, so it needs neither word.
        if not self._claims_on_device:
            return params
        params.append(f"{qualifier}int* {self.RETURN_CLAIM}")
        params.append(f"{qualifier}int* {self.RETURN_SITE}")
        return params

    def _buffer_qualifier(self)->str:
        """The address-space qualifier a buffer parameter carries, with its trailing space.

        One place decides it, because the three questions it answers are different:
        Metal spells the space `device`, OpenCL spells it `__global`, and CUDA/HIP
        spell nothing at all (a plain pointer is already in the global space).
        """
        if self._metal_syntax:
            return "device "
        if self._opencl_syntax:
            return "__global "
        return ""

    def return_site_layout(self)->List[ReturnSiteLayout]:
        """What the host needs to rebuild the result of each return site.

        Per site: the positions in return order, each "value" with its slot index or
        "array" with the parameter name to hand back, plus whether the site returned a
        tuple. A site with no positions is a bare `return`, i.e. None.
        """
        layout:List[ReturnSiteLayout] = []
        for site, positions in enumerate(self.return_sites):
            entries:List[ReturnSitePosition] = [
                {"kind": "value", "ordinal": ordinal, "slot": slot,
                 # A negative slot means the value already lives in the parameter it
                 # travelled in, so the host reads it from the launch arguments.
                 **({"reused": positions[ordinal]["slot_parameter"]}
                    if slot < 0 else {})}
                for ordinal, slot in self._site_slot_map.get(site, [])]
            for ordinal, position in enumerate(positions):
                if position["kind"] == "array":
                    entries.append({"kind": "array", "ordinal": ordinal, "name": position["name"]})
            entries.sort(key=lambda entry: entry["ordinal"])
            layout.append({"positions": entries,
                           "is_tuple": len(positions) > 1})
        return layout

    def _visit_return(self, node:ast.Return)->None:
        if self.is_kernel:
            self._visit_kernel_return(node)
            return

        if node.value is None:
            self.return_type = self.return_type or "void"
            self._emit("return;")
            return

        # `return a, b` from a device helper: C has no tuple, so the elements become
        # stores into the `T*` output parameters appended after the helper's own
        # signature (see `tuple_return_types` and `_emit_device_call`, which appends
        # them and passes the call site's destinations). `@kernel` supports a tuple
        # return through its own claim machinery; this is the helper-scale form of the
        # same feature, and the unpacking at the call site (`x, y = pair(...)`) is what
        # python writes for it.
        if isinstance(node.value, ast.Tuple):
            elements:List[Translated] = [self._visit_expression(e) for e in node.value.elts]
            for _code, t in elements:
                if is_ptr(t):
                    self._error("cannot return an array from a function", node)
            self.tuple_return_types = [t for _code, t in elements]
            for ordinal, (code, _t) in enumerate(elements):
                self._emit(f"*_tuple_out{ordinal} = {code};")
            self.return_type = "void"
            self._emit("return;")
            return

        code, t = self._visit_expression(node.value)
        if is_ptr(t):
            self._error("cannot return an array from a function", node)

        if self.return_type is None:
            self.return_type = t
        elif self.return_type != t:
            unified:object = promote(self.return_type, t)
            if unified != self.return_type:
                # The cast has to be spelled in this backend's own type names: CUDA C has
                # no `uchar`/`ushort`/`uint`/`ulong`, so a bare translator name here made
                # every device helper returning one of those fail to compile under NVRTC.
                code = f"(({self._device_type(self.return_type)})({code}))"
            self.return_type = unified
        self._emit(f"return {code};")

    def _eval_constant(self, node:Optional[ast.expr])->object:
        """Evaluate a compile-time expression: a shape, a dtype, a fill value.

        Only the forms a creation statement uses are supported (constants, names resolved in
        the kernel's namespace, attribute access, and calls), and an unsupported form raises
        so the caller can refuse the statement with its own message.
        """
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.resolve_global(node.id)
        if isinstance(node, ast.Attribute):
            return getattr(self._eval_constant(node.value), node.attr)
        if isinstance(node, ast.Call):
            function:object = self._eval_constant(node.func)
            arguments:List[object] = [self._eval_constant(argument) for argument in node.args]
            keywords:Dict[str, object] = {keyword.arg: self._eval_constant(keyword.value)
                                       for keyword in node.keywords if keyword.arg}
            return function(*arguments, **keywords)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._eval_constant(element) for element in node.elts)
        raise SyntaxError("not a compile-time constant expression")

    def _private_element_type(self, dtype_value:object)->Optional[Translated]:
        """(C element type, translator type) for a private array's element, or None."""
        from . import structs

        if dtype_value is None:
            dtype_value = np.float64
        try:
            dtype:np.dtype = np.dtype(dtype_value)
        except TypeError:
            return None
        from ..vec_types.layout import _vector_dtype_name

        vector:Optional[str] = _vector_dtype_name(dtype)
        if vector is not None:
            # Device-side storage: the padded width is what the device type means here.
            return self._device_type(vector), vector, False
        if structs.is_record(dtype):
            name:str = structs.register(dtype)
            if dtype not in self.private_structs:
                self.private_structs.append(dtype)
            return name, name, False
        scalar:Optional[str] = DTYPE_MAP.get(dtype.name)
        if scalar is None:
            return None
        if scalar == "bool":
            scalar = "uchar"
        return self._device_type(scalar), scalar, True

    def _emit_private_array(self, name:str, node:ast.Assign)->bool:
        """Emit a per-work-item C array for `name = np.zeros(<constant shape>, ...)`.

        Returns False when the statement is not something a private array can represent, so
        the caller reports the reason.
        """
        call:ast.Call = node.value  # type: ignore[assignment]
        if isinstance(call.func, ast.Attribute):
            constructor:str = call.func.attr
        else:
            return False
        if constructor not in ("zeros", "ones", "full", "empty"):
            return False

        # np.zeros(4, ...) / np.zeros((2, 3), ...) -- one positional shape argument.
        positional:List[ast.expr] = list(call.args)
        try:
            shape_value:object = self._eval_constant(positional[0]) if positional else None
            dtype_node:Optional[ast.expr] = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "dtype"), None)
            dtype_value:object = self._eval_constant(dtype_node)
            fill_value:object = None
            if constructor == "full" and len(positional) > 1:
                fill_value = self._eval_constant(positional[1])
        except SyntaxError:
            return False

        if shape_value is None:
            shape:Tuple[int, ...] = ()
        elif isinstance(shape_value, int):
            shape = (shape_value,)
        elif isinstance(shape_value, (tuple, list)) and all(
                isinstance(extent, int) for extent in shape_value):
            shape = tuple(int(extent) for extent in shape_value)
        else:
            return False
        if not shape or any(extent <= 0 for extent in shape):
            return False

        element:Optional[Tuple[str, Any, bool]] = self._private_element_type(dtype_value)
        if element is None:
            return False
        ctype, element_t, scalar_element = element
        total:int = 1
        for extent in shape:
            total *= extent

        initialiser:Optional[str] = self._private_initialiser(constructor, total, fill_value,
                                                             scalar_element)
        if initialiser is None:
            return False
        self._emit(f"{ctype} {device_name(name)}[{total}]{initialiser};")
        self._define(name, ("bodyarr", total, shape, element_t), node)
        return True

    def _private_initialiser(self, constructor:str, total:int, fill_value:object,
                             scalar_element:bool)->Optional[str]:
        """The C initialiser for a private array, matching the numpy constructor.

        CUDA/HIP compile as C++, where a vector or structure aggregate is value-initialised
        with `{}`; OpenCL C needs `{0}`. A non-scalar element cannot take a `full`/`ones`
        value without a conversion the initialiser cannot express, so that is refused (the
        caller reports it) rather than emitting something that does not compile.
        """
        empty:str = " = {}" if self.backend != "opencl" else " = {0}"
        if constructor == "empty":
            return ""
        if constructor == "zeros":
            return empty
        value:object = 1 if constructor == "ones" else fill_value
        if value is None:
            return empty
        if not scalar_element:
            return None
        if total > self.PRIVATE_ARRAY_MAX_UNROLL:
            return None
        return " = {" + ", ".join(f"({value})" for _ in range(total)) + "}"

    def _visit_assign(self, node:ast.Assign)->None:
        if len(node.targets) != 1:
            self._error("chained assignment is not supported", node)
        target:ast.expr = node.targets[0]

        # `weights = [1.0, 2.0, 3.0]` feeding `enumerate(weights)` is a compile-time table:
        # the loop reads the elements from the literal itself (see
        # _emit_constant_enumerate_head), so the binding emits nothing and no parameter
        # carries it. Handled here rather than only where a host region binds it, because
        # both paths reach this method.
        if isinstance(target, ast.Name) and target.id in self._constant_list_names:
            return

        # `src = a` is a plain pointer assignment: an array parameter *is* a `T*` on the
        # device, so the work item binds a second name to the same buffer and reads or
        # writes through either. The statement itself is emitted by the ordinary assignment
        # path below; what is recorded here is which array it was assigned from, because the
        # *extents* are launch parameters named after the array -- a shape read through the
        # local (`src.shape[0]`) has to resolve to them, or the launch looks for an argument
        # that does not exist (an `IndexError` from the shape check). Only a bare name or a
        # receiver member is a candidate: `threadIdx.x` is an attribute too, and it is a
        # value, not an array.
        if isinstance(target, ast.Name) and self._lookup(target.id) is None:
            source_name:Optional[str] = self._pointer_source_name(node.value)
            if source_name is not None and is_ptr(self._lookup(source_name)):
                self._array_pointer_sources[target.id] = source_name

        # A *slice* of an array (`part = a[lo:hi]`) is a pointer into the same buffer: the
        # device has no view type, so the slice becomes `T* part = a + lo;` and its length
        # is the one the slice wrote down. A constant slice is a fixed extent, which is what
        # makes `part.shape[0]` a compile-time number (see _shape_var).
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Subscript) \
                and isinstance(node.value.slice, ast.Slice) \
                and self._lookup(target.id) is None:
            if self._emit_array_slice(target.id, node):
                return

        # `box.total = v` is a call to the property's setter. It is handled here rather
        # than in _visit_target because a setter takes the value, which only the
        # assignment knows -- and because the result is a statement, not an expression.
        if isinstance(target, ast.Attribute):
            base_code, base_t = self._visit_expression(target.value)
            if isinstance(base_t, tuple) and is_ptr(base_t):
                from . import structs as _structs_module  # noqa: PLC0415

                dtype:Optional[np.dtype] = _structs_module.dtype_for(base_t[1])
                if dtype is not None and self._emit_property_write(
                        _structs_module.declared_classes(dtype), target.attr,
                        (base_code, base_t), node.value, node):
                    return

        # A creation below the magic loop runs once per work item, so its object cannot be a
        # host parameter. When the shape is a compile-time constant it is exactly a C local
        # array -- private storage per work item, which is what python's fresh array means --
        # and it is emitted as one. A runtime extent has no fixed-size declaration, so it is
        # refused instead of being silently shared between work items.
        if (self.creation_below_loop and isinstance(target, ast.Name)
                and self._is_creation_expression(node.value)):
            # A *scalar* dtype cast with constant arguments (`two = np.float64(2.0)`) is
            # not an allocation: python answers one number here, exactly as it does above
            # the loop, where the same statement is lifted to the host and travels as a
            # parameter. Emitting it as a device cast instead answers the same value with
            # no launch machinery -- the cast spelling is a cast wherever it appears. The
            # refusal used to trip on the dtype names sitting on the creation list, which
            # made one spelling answer three ways depending on where it appeared.
            if self._is_numpy_constant_cast(node.value):
                # The device spelling is the cast itself (`(double)(2.0)`), which answers
                # the same number python computes. Rebuilding the assignment would meet
                # the same creation guard again, so the store is emitted directly.
                func_attr:str = cast("str", node.value.func.attr)  # type: ignore[union-attr]
                element_code:str = self._emit_numpy_cast(
                    func_attr,
                    [(self._visit_expression(a)[0], "int") for a in node.value.args],
                    node.value)[0]
                element_t:object = DTYPE_MAP[func_attr]
                self._define(target.id, element_t, node)
                self._emit(f"{self._device_type(element_t)} {device_name(target.id)}"
                           f" = {element_code};")
                return
            if self._emit_private_array(target.id, node):
                return
            self._error(
                "a value cannot be created below the magic loop unless its shape is a "
                "compile-time constant: the statements there run once per work item, so a "
                "constant shape becomes a private array and anything else would create one "
                "object per work item. Create it above the loop, or give it a constant "
                "shape (e.g. np.zeros(4, dtype=np.float32))", node)

        # self.xxx = ... : a scalar member of the receiver structure is a field of the
        # pointer, so the store lands in the caller's structure -- the shape
        # `HitRecord.set_face_normal` needs. A host attribute (an ndarray, or the receiver's
        # own state) is still refused: device code has nowhere to write it back from.
        if (self.self_param and isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == self.self_param):
            from . import structs as _structs_module  # noqa: PLC0415

            member:str = self.self_struct_members.get(target.attr, target.attr)
            receiver_t:DeviceType = self._lookup(self.self_param)
            receiver_dtype:Optional[np.dtype] = (
                _structs_module.dtype_for(receiver_t[1])
                if isinstance(receiver_t, tuple) and is_ptr(receiver_t) else None)
            is_member:bool = (receiver_dtype is not None
                              and member in _structs_module.member_types(
                                  receiver_dtype, self.backend))
            if is_member:
                value_code, value_t = self._visit_expression(node.value)
                if np.dtype(receiver_dtype.fields[member][0]).name == "bool":
                    # A bool member is a byte on the device: a float assigned straight into
                    # it truncates (`0.99` became 0), and the branch reading it went the
                    # wrong way -- the sphere lost its blue channel and came out grey.
                    value_code = f"(({value_code}) != 0)"
                self.member_writes.add(self.self_param)
                self._emit(f"({self.self_param})->{member} = {value_code};")
                return
            self._error(
                f"cannot assign to self.{target.attr}: host attributes cannot be written by device code; "
                f"write into an array element (self.{target.attr}[i] = ...) or use a local variable", node)

        # shared memory declaration: buf = zeros_per_group(64, np.float32)
        if (isinstance(node.value, ast.Call)
                and self._builtin_call_name(node.value) in ("zeros_per_group",
                                                        "zeros_per_block")):
            self._handle_local_declaration(node)
            return

        # Tuple unpacking/swap: a, b = b, a -- also when an item is an array element,
        # as in the idiomatic swap a[i], b[i] = b[i], a[i].
        if isinstance(target, ast.Tuple):
            # `x, y = pair(...)`, where the helper answers a tuple: python unpacks the
            # pair, and the device does the same by passing the destinations as the
            # helper's output parameters (see _visit_return's tuple branch, which
            # stores through them). Any other non-tuple right-hand side stays refused.
            if not isinstance(node.value, ast.Tuple):
                if isinstance(node.value, ast.Call) \
                        and self._tuple_call_element_count(node.value) is not None:
                    count:int = cast(int, self._tuple_call_element_count(node.value))
                    if len(target.elts) != count:
                        self._error(
                            f"the call answers {count} values, and the tuple assignment "
                            f"unpacks {len(target.elts)} names -- the counts have to "
                            f"match, as they do in python", node)
                    self._emit_tuple_call_unpack(target.elts, node.value)
                    return
                self._error("tuple assignment requires matching tuples on both sides", node)
            if not isinstance(node.value, ast.Tuple) or len(target.elts) != len(node.value.elts):
                self._error("tuple assignment requires matching tuples on both sides", node)
            if not all(isinstance(t, (ast.Name, ast.Subscript)) for t in target.elts):
                self._error(
                    "tuple assignment targets must be plain variables or array elements",
                    node)

            values:List[Translated] = [self._visit_expression(v) for v in node.value.elts]
            temps:List[Translated] = []
            for code, t in values:
                if is_ptr(t):
                    self._error("cannot swap array parameters", node)
                self._tmp_counter += 1
                tmp:str = f"_swap{self._tmp_counter}"
                self._define(tmp, t, node)
                self._emit(f"{self._device_type(t)} {tmp} = {code};")
                temps.append((tmp, t))

            for (tmp, t), target_item in zip(temps, target.elts):
                if isinstance(target_item, ast.Subscript):
                    # Evaluate the destination after every right-hand side has been
                    # captured, so a swap really reads both values first.
                    code, _ = self._visit_array_target(target_item.value)
                    index_code, _ = self._visit_subscript_index(target_item)
                    self._emit(f"{code}[{index_code}] = {tmp};")
                    continue
                name:str = target_item.id
                existing:DeviceType = self._lookup(name)
                if existing is not None and is_ptr(existing):
                    self._error(f"cannot assign to array parameter '{name}'", node)
                is_new:bool = (existing is None)
                self._define(name, t, node)
                decl:str = (f"{self._device_type(t)} " if is_new else "")
                self._emit(f"{decl}{name} = {tmp};")
            return
        code, t = self._visit_expression(node.value)
        from . import structs as _structs_module

        if is_ptr(t) and not (isinstance(t, tuple) and _structs_module.dtype_for(t[1]) is not None):
            # A pointer to a *structure* is an aliasable handle (`q = item; q.a` counts as a
            # read of `item.a`), so it copies like a scalar. An array pointer is the same
            # thing on the device -- an array parameter *is* a `T*` -- so `src = a` binds a
            # second name to the same buffer and the work item reads/writes it through
            # either. Rebinding an *outer* array stays refused (the object is shared with the
            # caller, so only its elements may be written), and that is the one case left.
            if isinstance(target, ast.Name) and target.id in self.outer_vars:
                self._error(
                    f"cannot rebind the outer array '{target.id}': the array object itself "
                    f"is shared with the caller, so a device kernel can only write its elements", node)
            if isinstance(target, ast.Name) and self._lookup(target.id) is None:
                # One pointer assigned to a fresh local: a real C declaration, so every later
                # use of the name is an ordinary variable read.
                self._emit(f"{self._device_type(t[1])}* {device_name(target.id)} = {code};")
                self._define(target.id, t, node)
                return
            self._error("cannot assign an array to a variable", node)

        if isinstance(target, ast.Name):
            outer:Optional[Dict[str, object]] = self._named_outer_target(target, node)
            if outer is not None:
                self._emit(f"{device_name(target.id)}[0] = {code};")
                return
            # `rows = out.shape[0]`: the name stands for the array's extent, which is
            # what an `np.ndindex(rows, ...)` bound resolves through (see
            # _extent_source). Recorded before the visit of the loop, whose head reads
            # the table.
            if isinstance(node.value, ast.Subscript) \
                    and isinstance(node.value.slice, ast.Constant) \
                    and isinstance(node.value.slice.value, int):
                holder:Optional[str] = self._shape_holder_param(node.value.value)
                if holder is not None:
                    holder_t:DeviceType = self._lookup(holder)
                    if is_ptr(holder_t) or is_image(holder_t):
                        self._shape_aliases[target.id] = (holder,
                                                          int(node.value.slice.value))
            existing:DeviceType = self._lookup(target.id)
            if isinstance(existing, tuple) and len(existing) == 2 and existing[0] == "carried":
                # One location for the whole launch: store through the pointer, exactly as
                # the outer-scalar path does.
                self._emit(f"{device_name(target.id)}[0] = {code};")
                return
            if existing is not None and is_ptr(existing):
                self._error(f"cannot assign to array parameter '{target.id}'", node)
            is_new:bool = (existing is None)
            # A local holding the block id stays marked as one, so `buf[blk]` later still
            # reads as a group selection rather than an ordinary index (see
            # _is_block_id_expression).
            stored:object = t
            if self._is_block_id_expression(node.value):
                stored = "blockid"
            self._define(target.id, stored, node)
            decl:str = ""
            if is_new:
                if isinstance(t, tuple) and is_ptr(t):
                    # A structure pointer alias: declared with the *same* const qualification
                    # the source expression has, because a `const Struct*` parameter cannot
                    # initialise a mutable `Struct*`.
                    const:str = ("const " if isinstance(node.value, ast.Name)
                                 and self._is_const_struct(node.value.id) else "")
                    decl = f"{const}{self._kernel_scalar_type(t[1])}* "
                else:
                    decl = self._device_type(t) + " "
                # A later widening (`scale = 1.0` here, a double inside the loop) has to
                # rewrite this declaration's type -- see _render_body's fixup pass.
                self._declaration_sites.append((len(self.lines), target.id,
                                                self._device_type(t)))
            self._emit(f"{decl}{device_name(target.id)} = {code};")
        elif isinstance(target, ast.Subscript):
            # A string subscript on a structure value writes a member: `s["i"] = 1` becomes
            # `s.i = 1`. Recognized before the array path, which rejects a structure base
            # because a structure is a value, not a pointer.
            member:Optional[Translated] = self._struct_member_access(target)
            if member is not None:
                # A member store makes the structure parameter writable: it must not be
                # declared `const` (see render_kernel / render_device).
                self._record_member_write(member[0])
                # The generated field assignment goes through a parenthesised member
                # expression, which the static read/write analysis does not attribute to the
                # array; record the target so the host still reads the array back.
                if isinstance(target.value, ast.Subscript):
                    try:
                        self.atomic_targets.add(self._array_name(target.value.value))
                    except SyntaxError:
                        pass
                self._emit(f"{member[0]} = {code};")
                return
            # A store into a zeros_per_group() slice flattens like the read side, and has
            # to be recognized before the array path (a slice is not a pointer parameter).
            slice_code, slice_t = self._visit_expression(target.value)
            if isinstance(slice_t, tuple) and slice_t and slice_t[0] == SHARED_SLICE:
                store_code, store_t = self._visit_shared_index(slice_code, slice_t, target)
                store_code, _ = self._scalar_position(store_code, store_t, target)
                self._emit(f"{store_code} = {code};")
                return
            base_code, base_t = self._visit_array_target(target.value)
            if is_image(base_t):
                self._emit_image_write(base_code, base_t, target, code)
                return
            index_code, _ = self._visit_subscript_index(target)
            code = self._complex_store(base_t, t, code)
            self._emit(f"{base_code}[{index_code}] = {code};")
        elif isinstance(target, ast.Attribute):
            # `p.x = ...`: a C member store on a structure value, the write twin of the
            # attribute read. The read/write analysis cannot attribute the field store to
            # the caller's argument (it sees a member expression, not a subscript), so this
            # path is only reached for synthesized object arguments, whose host side copies
            # the final member values home after the launch.
            target_code, target_t = self._visit_target(target)
            # A member store on a structure parameter makes it writable, exactly as the
            # string-subscript store does; without this the parameter was declared `const`
            # and the generated store failed to compile.
            self._record_member_write(target_code)
            if isinstance(target.value, ast.Subscript):
                # A record-array element store (`rows[i].i = ...`) writes the array the
                # parameter names, so the parameter is not read-only either.
                try:
                    self.member_writes.add(self._array_name(target.value.value))
                except SyntaxError:
                    pass
            if self._target_is_bool_member(target, target_t, t):
                # A bool member is a byte on the device, so a float would truncate
                # (`front_face = 0.99` -> 0). The store keeps boolean semantics instead.
                code = f"(({code}) != 0)"
            self._emit(f"{target_code} = {code};")
        else:
            self._error(f"unsupported assignment target: {target.__class__.__name__}", node)

    def _named_outer_target(self, target:ast.Name, node:ast.AST)->Optional[Dict[str, object]]:
        """Resolve a plain-name assignment to an outer variable, refusing what cannot work.

        A `nonlocal`/`global`-declared variable is written through its pointer, so
        `total = total + x` writes the scope value and the host copies it back after the
        launch. An undeclared outer name is refused: python would create a local there,
        which would silently shadow the scope variable instead of touching it.

        **A plain store from inside the work-item body is a race, so it is refused.** The
        variable is one location for the whole launch, so every work item overwrites it and
        the value that survives is whichever finished last -- the same hazard the *carried*
        scalars are checked for (see `_non_atomic_writes`). `total += x[i]` is the spelling
        that works, because it becomes a device atomic; a store guarded by something every
        work item agrees on (`if i == 0`) is a template for one work item's slice, and is
        also fine. What is not fine is an unguarded `counter = counter + x[i]`, which
        compiles and then answers one work item's partial value.
        """
        outer:Optional[Dict[str, object]] = self.outer_vars.get(target.id)
        if outer is None:
            return None
        if not outer["writable"]:
            self._error(
                f"cannot assign to the outer variable '{target.id}': without a "
                f"'nonlocal {target.id}' or 'global {target.id}' declaration this would "
                f"create a local variable, shadowing the outer one", node)
        if self.in_work_item_body and not self._store_is_guarded(node):
            self._error(
                f"'{target.id}' is one location shared by every work item (it is an outer "
                f"variable of the enclosing scope), so a plain store here races: whichever "
                f"work item finishes last decides the value. This is the same rule an "
                f"accumulated local follows -- write `{target.id} += value` (which becomes a "
                f"device atomic), or guard the store with the iteration index "
                f"(`if i == 0: {target.id} = ...`), or keep the work item's own value in a "
                f"local and reduce it afterwards", node)
        if is_ptr(outer["type"]):
            self._error(
                f"cannot rebind the outer array '{target.id}': the array object itself is "
                f"shared with the caller, so a device kernel can only write its elements", node)
        return outer

    def _handle_local_declaration(self, node:ast.Assign)->None:
        """`buf = zeros_per_group(64, np.float32)` -> a shared array, one slice per work group.

        A work item can only touch memory inside its own group, so the value the name
        stands for is a **dictionary keyed by block id**: `buf[block_id(0)][j]` is this
        group's slice. Indexing it with anything else is refused, because there is no
        meaning for "the buffer of another group" inside a work item.

        opencl: one `__local` parameter per buffer, sized for a single group, and the
        kernel reads it through the group's own offset in that parameter;
        cuda: a `__shared__` array, which is already per-group.
        """
        call:ast.Call = node.value
        if len(call.args) != 2 or call.keywords:
            self._error("zeros_per_group(shape, dtype) takes exactly 2 arguments", call)

        # `np.zeros(shape, dtype)`, argument for argument: the shape comes first and may be
        # an int (`256`) or a tuple (`(64, 64)`), and the dtype second.
        shape_node:ast.expr = call.args[0]
        extents:Optional[List[int]] = self._constant_shape(shape_node, "zeros_per_group()")
        if extents is None:
            self._error(
                "zeros_per_group() shape must be a constant positive integer or a tuple of "
                "them: the buffer is a fixed-size __shared__/__local array, so a launch-time "
                "shape cannot size it", shape_node)

        dtype_node:ast.expr = call.args[1]
        dtype_name:str = ""
        if isinstance(dtype_node, ast.Attribute) and isinstance(dtype_node.value, ast.Name):
            module:object = self.resolve_global(dtype_node.value.id)
            if module is np:
                dtype_name = dtype_node.attr
            else:
                self._error("zeros_per_group() dtype must be a numpy dtype like np.float32", dtype_node)
        elif isinstance(dtype_node, ast.Constant) and isinstance(dtype_node.value, str):
            dtype_name = dtype_node.value
        else:
            self._error("zeros_per_group() dtype must be a numpy dtype like np.float32", dtype_node)

        if dtype_name not in DTYPE_MAP:
            self._error(f"zeros_per_group() unsupported dtype: {dtype_name}", dtype_node)

        ctype:str = DTYPE_MAP[dtype_name]
        if ctype == "half":
            self._error("zeros_per_group() does not support half (fp16)", dtype_node)

        total:int = 1
        for extent in extents:
            total *= extent
        c_name:str = f"_local{len(self.local_buffers)}"
        self.local_buffers.append((c_name, ctype, total))

        target:ast.expr = node.targets[0]
        if not isinstance(target, ast.Name):
            self._error("zeros_per_group() result must be assigned to a plain variable", node)
        self._define(target.id, ("shared", ctype, c_name, total, tuple(extents)), node)

    def _constant_shape(self, node:ast.expr, what:str)->Optional[List[int]]:
        """The extents of a constant shape: an int, or a tuple of positive ints."""
        def one(candidate:ast.expr)->Optional[int]:
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, int) \
                    and candidate.value > 0:
                return candidate.value
            return None

        single:Optional[int] = one(node)
        if single is not None:
            return [single]
        if isinstance(node, ast.Tuple) and node.elts:
            extents:List[int] = []
            for element in node.elts:
                extent:Optional[int] = one(element)
                if extent is None:
                    return None
                extents.append(extent)
            return extents
        return None

    def _is_local_buffer(self, node:ast.expr)->bool:
        """True when ``node`` reaches `zeros_per_group(...)` memory, i.e. work-group shared storage.

        Both spellings count: the buffer itself (`bins`) and the group's slice of it
        (`bins[blockIdx]`). The slice form is what an atomic call passes -- `atomic_add(
        bins[blockIdx], i, 1)` -- and answering False there made the OpenCL cast say
        `__global`, which the compiler rejects as a change of address space.
        """
        if isinstance(node, ast.Subscript):
            node = node.value
        if not isinstance(node, ast.Name):
            return False
        t:DeviceType = self._lookup(node.id)
        return isinstance(t, tuple) and len(t) == 5 and t[0] == "shared"

    def _builtin_call_name(self, node:ast.expr)->Optional[str]:
        """The builtin this call invokes, or None when it invokes something else.

        Identity, not spelling: the callee is resolved the way python would resolve it and
        the resulting object is looked up in the registry, so a bare name (``dot(...)``), an
        attribute of the package (``pygpgpu.dot(...)``) and a name imported from the module
        that defines it all agree, while a user function that happens to share a name is
        recognized as theirs.
        """
        if not isinstance(node, ast.Call):
            return None
        callee:ast.expr = node.func
        if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
            owner:object = self.resolve_global(callee.value.id)
            if owner is None:
                return None
            return builtin_name_of(getattr(owner, callee.attr, None))
        if isinstance(callee, ast.Name):
            return builtin_name_of(self.resolve_global(callee.id))
        return None

    def _library_call_name(self, node:ast.expr)->Optional[str]:
        """The registered device function this call invokes, or None.

        The same identity rule as the builtins: the callee is resolved the way python
        would resolve it and the resulting object is looked up in the registry, so a
        user's own function that happens to share a name is not mistaken for the
        registered one.
        """
        if not isinstance(node, ast.Call):
            return None
        callee:ast.expr = node.func
        if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
            owner:object = self.resolve_global(callee.value.id)
            if owner is None:
                return None
            return library_function_name(getattr(owner, callee.attr, None))
        if isinstance(callee, ast.Name):
            return library_function_name(self.resolve_global(callee.id))
        return None

    def _emit_library_function(self, name:str, args:List[Translated],
                               node:ast.Call)->Translated:
        """A call to a registered device-side library function.

        The device gets the symbol the registry names for this backend; the headers the
        registry asks for are recorded, so the compilation includes them.
        """
        entry:object = library_functions().get(name)
        symbol:Optional[str] = (entry.symbols.get(self.backend) if entry is not None else None)
        if symbol is None:
            available:str = (", ".join(sorted(entry.symbols)) if entry is not None else "")
            self._error(
                f"gpu.{name}() is a registered device function that the {self.backend} "
                f"backend has no implementation for"
                + (f" (it has: {available})" if available else ""), node)
        # The headers the function needs: they go into the generated source, so the symbol
        # the registry names is actually declared.
        for header in entry.headers.get(self.backend, ()):
            if header not in self.device_library_headers:
                self.device_library_headers.append(header)
        arg_codes:List[str] = [code for code, _ in args]
        for code, t in args:
            if is_vec(t):
                self._error(
                    f"gpu.{name}() takes scalars: a registered device function is called "
                    f"as it is declared, and CUDA has no vector overload for it", node)
        # The result type: the first argument's type, unless the function is registered
        # with a different one. A function that takes nothing answers nothing.
        result_t:object = args[0][1] if args else "float"
        if not arg_codes:
            result_t = "void"
        return f"{symbol}({', '.join(arg_codes)})", result_t

    # ---------- print() -> device printf ----------

    def _is_print_call(self, node:ast.expr)->bool:
        """True for a statement-level call of the builtin ``print``.

        Only the builtin counts: a module that defines its own print() means its own
        function, which is then translated (or rejected) like any other call. The builtin
        is not a module global, so it is looked up in the function's own namespace.
        """
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print" and self._print_is_builtin())

    def _print_is_builtin(self)->bool:
        """True while translating a function whose global scope does not shadow print."""
        namespace:object = getattr(self.func, "__globals__", None)
        if namespace is None:
            return True
        return namespace.get("print", print) is print

    def _emit_print(self, call:ast.Call)->None:
        """Translate ``print(...)`` into a device printf that prints the same text.

        Python's print writes to the host process's stdout, which a kernel has no access
        to; the device counterpart is printf. The rewrite mirrors python's text: arguments
        separated by one space, a trailing newline, and every value rendered the way
        ``str()`` renders it (True/False, decimal integers, python's shortest round-trip
        floats, and ``float4(1.0, 2.0, 3.0, 4.0)`` for vectors).

        One line becomes one printf call site per branch leaf, selected by a computed
        index. That keeps each line whole: a device printf call site has its own output
        buffer and the buffers are drained call site by call site, so a line assembled
        from several printf calls would come out split up and interleaved with other work
        items. Each leaf repeats the whole format string, so a line is never broken.

        The HIP backend is excluded on purpose. Its driver drops the device ("unspecified
        launch failure", followed by a Windows display-driver reset on the tested ROCm 7.2
        machine) once a kernel with this many printf call sites is launched, and debug
        output is not worth crashing the machine over. Use OpenCL or CUDA for printing.
        """
        # launch_backend, not backend: a hip kernel goes through cuda code generation, so
        # self.backend is "cuda" there and only the runtime tells the two apart.
        if self.launch_backend == "hip":
            self._error(
                "print() is not supported on the hip backend: launching a kernel with the "
                "generated printf call sites makes the driver drop the device on this "
                "runtime. Run the kernel with the opencl or cuda backend to see its output, "
                "or write results into an array and inspect them on the host", call)
        if self._metal_syntax:
            # The emitted call sites are OpenCL's: `printf` writes into a per-call-site
            # output buffer that the host drains after the launch. MSL has no such
            # buffer -- `printf` in Metal goes to a shared log that the command buffer
            # owns and no host code reads -- so the generated code would compile and
            # silently lose the output. Refuse rather than pretend, the same way the
            # HIP path above does.
            self._error(
                "print() is not supported on the metal backend: the generated call sites "
                "write into a per-call-site output buffer that only the OpenCL runtime "
                "drains, while Metal's printf goes to a log no host code here reads. Run "
                "the kernel with the opencl or cuda backend to see its output, or write "
                "results into an array and inspect them on the host", call)
        # sep/end are fixed text, so both can be folded into the format string. file and
        # flush are not translated: there is only one device output stream.
        separator, terminator = self._print_sep_end(call)

        pieces:List[Tuple[str, str, Any]] = []
        for index, argument in enumerate(call.args):
            if index:
                pieces.append(("text", separator, None))
            pieces.extend(self._print_argument(argument))

        # The digit-count code for a real argument has to run before the switch that picks
        # the leaf, so it is emitted as the pieces are expanded.
        leaves:List[Tuple[str, List[str]]] = [("", [])]
        selectors:List[Tuple[str, int]] = []
        self._expand_print_pieces(pieces, leaves, selectors)
        self._emit_print_leaves(leaves, selectors, call, terminator)

    def _print_fstring(self, node:ast.JoinedStr)->List[Tuple[str, str, Any]]:
        """An f-string as print pieces: its literal runs are text, its fields are values.

        `f"slot {i}"` is the same output as `print("slot", i, sep="")`, and that is how it is
        emitted -- python's f-string is a `JoinStr` of `Constant` and `FormattedValue` parts,
        and the second sort is exactly one value. A conversion (`!r`/`!s`/`!a`) or a format
        spec (`:>4`, `:.2f`) asks python's formatting machinery for text the device has no
        formatter for, so each is refused with the reason rather than silently ignored.
        """
        pieces:List[Tuple[str, str, Any]] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pieces.append(("text", part.value, None))
                continue
            if not isinstance(part, ast.FormattedValue):
                self._error(
                    "an f-string part has no device form: use literal text and plain "
                    "`{expression}` fields", part)
            if part.conversion not in (-1, None):
                self._error(
                    "an f-string conversion (`!r`, `!s`, `!a`) formats through python's "
                    "repr/str, which the device has no formatter for: print the value in a "
                    "plain `{expression}` field instead", part)
            if part.format_spec is not None:
                self._error(
                    "an f-string format spec (`{value:>4}`, `{value:.2f}`) needs python's "
                    "formatting machinery, which the device does not have: print the value "
                    "in a plain `{expression}` field instead", part)
            pieces.extend(self._print_argument(part.value))
        return pieces

    def _print_sep_end(self, call:ast.Call)->Tuple[str, str]:
        """The separator and terminator a print() call asks for.

        Both become part of the format string, which has to be a compile-time constant on
        the device (OpenCL rejects a computed format), so they must be string literals.
        ``file`` and ``flush`` are refused: a kernel has one output stream and nothing to
        flush.
        """
        separator:str = " "
        terminator:str = "\n"
        for keyword in call.keywords:
            if keyword.arg is None:
                self._error(
                    "print() does not accept **kwargs: the printed text must be fixed in "
                    "the device code", call)
            if keyword.arg == "file":
                self._error(
                    "print(file=...) is not supported in a kernel: device printf writes to "
                    "the backend's own output stream", call)
            if keyword.arg == "flush":
                self._error(
                    "print(flush=...) has no meaning in a kernel: device output is "
                    "drained when the launch completes", call)
            if keyword.arg not in ("sep", "end"):
                self._error(
                    f"print() got an unexpected keyword argument '{keyword.arg}'; only sep "
                    f"and end are supported", call)
            if not (isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)):
                self._error(
                    f"print({keyword.arg}=...) must be a string literal: the format string "
                    f"is fixed in the device code, so it cannot be computed", call)
            if keyword.arg == "sep":
                separator = keyword.value.value
            else:
                terminator = keyword.value.value
        return separator, terminator

    def _print_argument(self, node:ast.expr)->List[Tuple[str, str, Any]]:
        """Split one print() argument into pieces.

        A piece is one of:
          ("text", literal, None)   fixed text, copied into the format string
          ("int", code, type)       an integer, printed by a printf conversion
          ("bool", code, "bool")    a bool, printed as True/False
          ("real", code, type)      a float or double, printed like python's repr
          ("vec", code, type)       a vector, printed like its python repr

        An f-string is exactly a list of those pieces already: its literal runs are fixed
        text and each `{expression}` is one value, so it is expanded here rather than
        refused. A conversion or a format spec (`{x!r}`, `{x:>4}`) is not, because python's
        formatting machinery has no device counterpart -- the value is printed its own way.
        """
        if isinstance(node, ast.JoinedStr):
            return self._print_fstring(node)
        if isinstance(node, ast.Constant) and node.value is None:
            return [("text", "None", None)]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A literal string is fixed text, and the only way a string can reach print():
            # the translator has no string type.
            return [("text", node.value, None)]

        code, t = self._visit_expression(node)
        if is_vec(t):
            return [("vec", code, t)]
        if is_ptr(t):
            self._error(
                "print() cannot print an array: python prints the whole array's repr, which "
                "has no device counterpart. Print an element (print(a[i])) or a value instead",
                node)
        if t == "bool":
            return [("bool", code, "bool")]
        if t in ("float", "double"):
            return [("real", code, t)]
        if is_int(t):
            return [("int", code, t)]
        self._error(f"print() does not support a value of type {t}", node)

    def _expand_print_pieces(self, pieces:List[Tuple[str, str, Any]],
                             leaves:List[Tuple[str, List[str]]],
                             selectors:List[Tuple[str, int]])->None:
        """Fold pieces into leaves (one printf each), recording the selectors in order.

        Leaves are ordered so that the leaf index is the mixed-radix number formed by the
        selectors taken in order, which is what _emit_print_leaves relies on: each bool
        contributes a radix-2 digit, each real a digit of its own budget.
        """
        for kind, code, extra in pieces:
            if kind == "text":
                leaves[:] = [(fmt + self._c_string_text(code), args) for fmt, args in leaves]
            elif kind == "int":
                conversion:str = "%" + self._int_format(extra)
                argument:str = self._int_argument(code, extra)
                leaves[:] = [(fmt + conversion, args + [argument]) for fmt, args in leaves]
            elif kind == "bool":
                # Python prints a bool as True/False, and no printf conversion produces
                # that text from a computed value, so the line is split into the two
                # spellings. True is the first leaf, hence selector 0 for a true value.
                expanded:List[Tuple[str, List[str]]] = []
                for fmt, args in leaves:
                    expanded.append((fmt + "True", list(args)))
                    expanded.append((fmt + "False", list(args)))
                leaves[:] = expanded
                selectors.append((f"(({code}) ? 0 : 1)", 2))
            elif kind == "real":
                selector, budget = self._print_real_selector(code, extra)
                expanded = []
                for fmt, args in leaves:
                    for choice in range(budget):
                        text:str = self._print_real_text(choice)
                        conversion:str = self._print_real_conversion(choice, extra)
                        if text:
                            # A special value spells itself and takes no argument.
                            expanded.append((fmt + text, list(args)))
                        else:
                            expanded.append((fmt + conversion, list(args) + [f"((double)({code}))"]))
                leaves[:] = expanded
                selectors.append((selector, budget))
            elif kind == "component":
                n:str = self._next_temp()
                self._emit(f"double _pa{n} = fabs((double)({code}));")
                expanded = []
                for fmt, args in leaves:
                    for choice in (0, 1):
                        conversion:str = "%.1f" if choice == 0 else "%.17g"
                        expanded.append((fmt + conversion, list(args) + [f"((double)({code}))"]))
                leaves[:] = expanded
                selectors.append((f"((_pa{n} == floor(_pa{n}) && _pa{n} < 1e16) ? 0 : 1)", 2))
            else:
                sub_pieces:List[Tuple[str, str, Any]] = self._print_vec_pieces(code, extra)
                sub_leaves:List[Tuple[str, List[str]]] = [("", [])]
                sub_selectors:List[Tuple[str, int]] = []
                self._expand_print_pieces(sub_pieces, sub_leaves, sub_selectors)
                expanded = []
                for fmt, args in leaves:
                    for sub_fmt, sub_args in sub_leaves:
                        expanded.append((fmt + sub_fmt, args + sub_args))
                leaves[:] = expanded
                selectors.extend(sub_selectors)

    def _print_vec_pieces(self, code:str, vec:str)->List[Tuple[str, str, Any]]:
        """The pieces of a vector's python repr: "float4(" + components + ")".

        A vector's repr renders each component as a Python float, i.e. the exact double the
        component widens to. Every component therefore gets the same two shapes: a whole
        value keeps the ".0" python shows, and everything else is printed with 17
        significant digits, which is what the widened float32 needs to read back exactly
        (the low bits of such a double are zero, so 17 digits are its shortest form too).
        Keeping the layout that small is what lets a whole vector stay one printf.
        """
        pieces:List[Tuple[str, str, Any]] = [("text", f"{vec}(", None)]
        for position, (field_name, _field_type) in enumerate(_vec_components(vec)):
            if position:
                pieces.append(("text", ", ", None))
            component_t:str = _C_FIELD_TYPES[(vec, field_name)]
            component:str = f"({code}).{field_name}"
            if component_t in ("float", "double"):
                pieces.append(("component", component, "double"))
            elif component_t == "bool":
                pieces.append(("bool", component, "bool"))
            else:
                pieces.append(("int", component, component_t))
        pieces.append(("text", ")", None))
        return pieces

    def _print_real_conversion(self, choice:int, t:str)->str:
        """The printf conversion for one leaf of a real argument, or its fixed text.

        The leaves are a fixed layout, so a single integer selects one:
          0..4        nan, inf, -inf, 0.0 and -0.0: python's exact spelling, so the leaf
                      carries the text and no argument
          5           a whole number below the notation threshold: python prints 2.0
                      where %g would print 2
          6..6+max-1  the "%g" significant-digit budgets, 1..max: inside python's
                      fixed-notation band %g switches notation exactly as python does
          ..          the "%.Ne" budgets, 0..max-1 decimals, for the band where python
                      writes the exponent form but a "%g" of that budget would not
        """
        if choice < PRINT_REAL_SPECIAL_COUNT:
            return ""
        if choice == PRINT_REAL_SPECIAL_COUNT:
            return "%.1f"
        max_digits:int = PRINT_REAL_DIGITS[t == "float"]
        offset:int = choice - PRINT_REAL_SPECIAL_COUNT - 1
        if offset < max_digits:
            return "%." + str(offset + 1) + "g"
        # An exponent leaf prints the digits "%g" of that budget would print, only with the
        # exponent form made mandatory: p significant digits are p-1 decimals. A value at
        # the very top of the precision's range needs no decimals, so this starts at 0.
        decimals:int = offset - max_digits
        return "%." + str(max(decimals, 0)) + "e"

    def _print_real_text(self, choice:int)->str:
        """The fixed text of a special leaf (nan, inf, -inf, 0.0, -0.0); otherwise empty."""
        specials:List[str] = ["nan", "inf", "-inf", "0.0", "-0.0"]
        return specials[choice] if 0 <= choice < len(specials) else ""

    def _print_real_selector(self, code:str, t:str)->Tuple[str, int]:
        """Emit the leaf selection for a real argument and return (selector, leaf count).

        Python prints a real with the fewest significant digits that read back as the same
        value, so the digit count depends on the value: it is found here, at run time, by
        taking the first budget whose rounded value lands within half the gap to the
        neighbouring value the argument's own precision can hold.

        The gap is taken from the argument's binary exponent (ldexp, because OpenCL C has no
        nextafter), and the rounding is done on the value itself, scaled by 10^(budget-1-e).
        Both powers are built from two factors so the intermediate stays in range even for
        the smallest and largest values.
        """
        is_float_value:bool = (t == "float")
        max_digits:int = PRINT_REAL_DIGITS[is_float_value]
        plain_max:str = PRINT_REAL_PLAIN_MAX[is_float_value]
        sci_hi:int = PRINT_REAL_SCI_HI[is_float_value]
        big:str = PRINT_REAL_BIG[is_float_value]
        whole_leaf:int = PRINT_REAL_SPECIAL_COUNT
        general_base:int = whole_leaf + 1
        exponent_base:int = general_base + max_digits
        # The gap to the next representable value: one float mantissa step, or one double.
        gap:str = ("ldexp(1.0, ilogb((float)_pa%s) - 23)" if is_float_value
                   else "ldexp(1.0, ilogb(_pa%s) - 52)")
        smallest_normal:str = ("1.17549435e-38" if is_float_value else "2.2250738585072014e-308")
        smallest_step:str = ("1.40129846e-45" if is_float_value else "4.9406564584124654e-324")
        n:str = self._next_temp()
        # These temporaries stay in the current block: the switch that follows the
        # expansion is emitted at the same level and reads the selector out of them.
        self._emit(f"double _pv{n} = (double)({code});")
        self._emit(f"double _pa{n} = fabs(_pv{n});")
        self._emit(f"int _pd{n} = {general_base};")
        # The five values python spells out own the first leaves.
        self._emit(f"if (!(_pa{n} <= {big}))"
                   f" _pd{n} = (_pa{n} != _pa{n}) ? 0 : (_pv{n} < 0.0 ? 2 : 1);")
        self._emit(f"else if (_pa{n} == 0.0) _pd{n} = signbit(_pv{n}) ? 4 : 3;")
        self._emit(f"else if (_pa{n} == floor(_pa{n}) && _pa{n} < {plain_max})"
                   f" _pd{n} = {whole_leaf};")
        self._emit("else")
        self._emit("{")
        self._indent += 1
        self._emit(f"double _pgap{n} = {gap % n};")
        # A power of two has only half that gap below it.
        self._emit(f"if (_pa{n} == ldexp(1.0, ilogb(_pa{n}))) _pgap{n} *= 0.5;")
        # A subnormal value has a fixed smallest step, so ldexp would underflow it.
        self._emit(f"if (_pa{n} < {smallest_normal}) _pgap{n} = {smallest_step};")
        self._emit(f"int _pe{n} = (int)floor(log10(_pa{n}));")
        self._emit(f"double _ps{n} = pow(10.0, (double)_pe{n});")
        self._emit(f"if (_ps{n} > _pa{n})")
        self._emit("{")
        self._emit(f"    _pe{n} -= 1; _ps{n} = pow(10.0, (double)_pe{n});")
        self._emit("}")
        self._emit(f"else if (_ps{n} * 10.0 <= _pa{n})")
        self._emit("{")
        self._emit(f"    _pe{n} += 1; _ps{n} = pow(10.0, (double)_pe{n});")
        self._emit("}")
        # Half the gap either side, then the shortest budget whose rounding fits inside it.
        self._emit(f"double _plo{n} = _pa{n} - 0.5 * _pgap{n};")
        self._emit(f"double _phi{n} = _pa{n} + 0.5 * _pgap{n};")
        self._emit(f"int _pbest{n} = {max_digits};")
        self._emit(f"for (int _pk{n} = 0; _pk{n} < {max_digits}; _pk{n}++)")
        self._emit("{")
        self._indent += 1
        # Round to _pk+1 significant digits: divide by 10^(e-_pk), round, multiply back.
        # The exponent is split in two so no factor leaves the double range.
        self._emit(f"int _pstep{n} = _pe{n} - _pk{n};")
        self._emit(f"int _phalf{n} = _pstep{n} / 2;")
        self._emit(f"double _pdown{n} = pow(10.0, (double)_phalf{n})"
                   f" * pow(10.0, (double)(_pstep{n} - _phalf{n}));")
        self._emit(f"double _pq{n} = round(_pa{n} / _pdown{n}) * _pdown{n};")
        self._emit(f"if (_pq{n} >= _plo{n} && _pq{n} <= _phi{n})")
        self._emit("{")
        self._emit(f"    _pbest{n} = _pk{n} + 1; break;")
        self._emit("}")
        self._indent -= 1
        self._emit("}")
        self._emit_blank()
        # Python's notation bands: outside them the text is in the exponent form, inside
        # them "%g" keeps the fixed notation python uses. The exponent leaf with budget p
        # prints exactly the digits of "%.pg", so the same budget selects both.
        sci_lo:int = PRINT_REAL_SCI_LO[is_float_value]
        self._emit(f"if (_pe{n} <= {sci_lo} || _pe{n} >= {sci_hi})"
                   f" _pd{n} = {exponent_base} + _pbest{n} - 1;")
        self._emit(f"else _pd{n} = {general_base} + _pbest{n} - 1;")
        self._indent -= 1
        self._emit("}")
        # The last leaf is index exponent_base + max_digits - 1, so the line needs one more.
        return f"_pd{n}", exponent_base + max_digits

    def _emit_print_leaves(self, leaves:List[Tuple[str, List[str]]],
                           selectors:List[Tuple[str, int]], node:ast.Call,
                           terminator:str="\n")->None:
        """Emit the leaves as printfs selected by the mixed-radix leaf index."""
        if len(leaves) > PRINT_MAX_LINES:
            self._error(
                f"print() would need {len(leaves)} printf calls for this one line, above the "
                f"{PRINT_MAX_LINES} a kernel allows: too many booleans or floats in one "
                f"line. Split the text over several print() calls", node)

        tail:str = self._c_string_text(terminator)
        if len(leaves) == 1 and not selectors:
            format_text:str = leaves[0][0] + tail
            self._emit('printf("%s"%s);' % (format_text, (", " + ", ".join(leaves[0][1])) if leaves[0][1] else ""))
            return

        # Mixed-radix index over the selectors, in the order the leaves were expanded.
        index:str = "0"
        for selector, radix in selectors:
            index = f"(({index}) * {radix} + ({selector}))"
        cases:List[str] = []
        for position, (format_text, args) in enumerate(leaves):
            arguments:str = (", " + ", ".join(args)) if args else ""
            cases.append('case %d: printf("%s%s"%s); break;'
                         % (position, format_text, tail, arguments))
        cases.append("default: break;")
        self._emit(f"switch ((int){index})")
        self._emit("{")
        self._emit("\n".join(f"    {case}" for case in cases))
        self._emit("}")

    def _int_format(self, t:str)->str:
        """The printf conversion for an integer type on this backend."""
        return PRINT_INT_FORMATS[self.printf_backend].get(t, "d")

    def _int_argument(self, code:str, t:str)->str:
        """The C expression the printf conversion for an integer type expects."""
        cast:str = PRINT_INT_CASTS[self.printf_backend].get(t, "int")
        return f"({cast})({code})"

    def _c_string_text(self, text:str)->str:
        """Escape text for a C string literal, keeping % a literal percent."""
        escaped:str = text.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        return escaped.replace("%", "%%")

    def _next_temp(self)->str:
        """A fresh name suffix for the temporaries a translation step introduces."""
        self._tmp_counter += 1
        return str(self._tmp_counter)

    def _augassign_is_race_free(self, target:ast.Subscript)->bool:
        """True when each work item provably touches a different element.

        The only case the translator can prove without knowing the launch geometry is the
        one where the subscript is built from the magic loop's own index variables: every
        index is one of them, each appears once, and all of them appear. Distinct work
        items then have distinct index tuples, so `out[i, j] += v` cannot overlap with
        another work item and a plain read-modify-write is exact.

        Anything else -- a subscript driven by a data value (`bins[image[i]] += 1`), a
        partial tuple (`out[i] += v` in a 2-D loop), or an arithmetic index -- may collide
        and is treated as racy.
        """
        indices:ast.expr = target.slice
        elements:List[ast.expr] = list(indices.elts) if isinstance(indices, ast.Tuple) else [indices]
        if not all(isinstance(element, ast.Name) for element in elements):
            return False
        names:List[str] = [element.id for element in elements]  # type: ignore[union-attr]
        work_items:List[str] = self.work_item_indices
        if not work_items:
            # A gid-style kernel states its work item through gpu.get_global_id, which this
            # check cannot relate to a subscript, so nothing is provable there.
            return False
        return sorted(names) == sorted(work_items)

    # The augmented operators that have a device atomic counterpart, and its gpu name.
    ATOMIC_ASSIGN_OPS:Dict[str, str] = {
        "Add": "atomic_add", "Sub": "atomic_sub",
        "BitAnd": "atomic_and", "BitOr": "atomic_or", "BitXor": "atomic_xor",
    }

    def _visit_atomic_augassign(self, node:ast.AugAssign, op_name:str,
                                atomic_name:str,
                                pointer_code:Optional[str]=None)->None:
        """Emit a racy `arr[i] op= value` as the matching device atomic operation.

        ``pointer_code`` is set when the target is an outer scalar: the device parameter is
        already a pointer to one element, so there is no subscript to compute.
        """
        if pointer_code is not None:
            name:str = node.target.id  # type: ignore[union-attr]
            outer:Dict[str, object] = self.outer_vars[name]
            self._emit_atomic_update(node, op_name, atomic_name, outer["type"],
                                     array_code=name, index_code="0",
                                     pointer_code=pointer_code)
            return

        target:ast.Subscript = node.target  # type: ignore[assignment]
        # `bins[blockIdx][value] += 1` on shared memory: the slice is the array, and any
        # offset already consumed folds into the index.
        slice_code, slice_t = self._visit_expression(target.value)
        if isinstance(slice_t, tuple) and slice_t and slice_t[0] == SHARED_SLICE:
            pending:str = slice_t[3] if len(slice_t) > 3 else ""
            index_code, _ = self._visit_shared_index("", slice_t, target)
            base_code:str = slice_code
            element_t:object = slice_t[1]
            flat:int = index_code.find("[")
            inner:str = index_code[flat + 1:-1] if flat >= 0 else "0"
            if pending:
                inner = f"({pending}) + ({inner})"
            self._emit_atomic_update(node, op_name, atomic_name, element_t,
                                     array_code=base_code, index_code=inner,
                                     is_local=True)
            return
        # _visit_array_target already narrows a pointer to its element type.
        base_code, element_t = self._visit_array_target(target.value)
        self._record_element_write(target.value)
        if is_image(element_t):
            self._error(
                "'op=' on an image element is not supported; write `img[x, y] = value` "
                "or use gpu.atomic_* on an array buffer", node)
        index_code, _ = self._visit_subscript_index(target)
        self._emit_atomic_update(node, op_name, atomic_name, element_t,
                                 array_code=base_code, index_code=index_code,
                                 is_local=self._is_local_buffer(target.value))

    def _emit_atomic_update(self, node:ast.AugAssign, op_name:str, atomic_name:str,
                            element_t:object, array_code:str, index_code:str,
                            pointer_code:Optional[str]=None,
                            is_local:bool=False)->None:
        """The shared tail of an atomic `op=` on an array element or an outer scalar."""
        # A float slot is addable through the portable CAS helper (see
        # float_atomic_add_source); every other non-integer element has no atomic at all.
        float_add:bool = (element_t == "float" and op_name == "Add")
        if not is_int(element_t) and not float_add:
            self._error(
                f"'{self._augassign_symbol(op_name)}=' on this location may be written by "
                f"several work items at once, and a {element_t} value has no device atomic "
                f"operation. Python would add every contribution, so the plain "
                f"read-modify-write would lose some. Use an integer counter, or accumulate "
                f"per work item and reduce afterwards (see examples/09_reduction.py)", node)
        value_code, value_t = self._visit_expression(node.value)
        if not is_int(value_t) and not is_float(value_t):
            self._error(f"'{op_name}' needs a numeric value", node)
        if pointer_code is None:
            self.atomic_targets.add(self._array_name(node.target.value))  # type: ignore[union-attr]
        else:
            self.atomic_targets.add(node.target.id)  # type: ignore[union-attr]
        if float_add and float_atomic_add_source(self.backend) \
                not in self.device_helpers:
            self.device_helpers.append(float_atomic_add_source(self.backend))
        call:str = emit_atomic(atomic_name, array_code, index_code, value_code,
                               element_t, self.backend, pointer_code=pointer_code,
                               is_local=is_local)
        self._emit(f"{call};")

    def _augassign_symbol(self, op_name:str)->str:
        return {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//",
                "Mod": "%", "Pow": "**", "LShift": "<<", "RShift": ">>",
                "BitAnd": "&", "BitOr": "|", "BitXor": "^"}.get(op_name, op_name)

    def _tuple_call_element_count(self, call:ast.Call)->Optional[int]:
        """How many values a call answers, when it answers a tuple -- or None.

        The answer is known only for a *helper* call whose body returns a tuple; the
        interpreter backend runs the same call and python unpacks the pair there, so this
        is the device side of one spelling. The count comes from the function's return
        statements -- `return a, b` has a fixed arity, whatever the types turn out to be
        at the call.
        """
        count:Optional[int] = self._tuple_call_return_info(call)[0]
        if count is not None:
            # The element types, read from the returns as well: each one is the device
            # type the callee's own translation produces (a np.float64 cast is a double,
            # a python float result is the mapping's width). They decide the
            # destinations' declarations, so a plain name unpacks like an assignment.
            types:List[object] = self._tuple_call_element_types(call)
            if types is not None:
                self._pending_tuple_types = types
        return count

    def _tuple_call_return_info(self, call:ast.Call)->Tuple[Optional[int],
                                                             List[object]]:
        """A tuple-returning helper's arity and element types (empty when unknown).

        The arity comes from the function's return statements. The element types are the
        ones a *first pass* of the call would produce: the cheapest reliable source is a
        dry translation of the helper at this call's argument types, which is exactly
        what `_emit_device_call` does -- so the caller unpacks after asking here, and the
        types come back through the cache entry.
        """
        if not isinstance(call.func, ast.Name):
            return None, []
        target:object = self.resolve_global(call.func.id)
        if not isinstance(target, types.FunctionType):
            return None, []
        counts:Set[int] = set()
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
        except (OSError, TypeError):
            return None, []
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
                counts.add(len(sub.value.elts))
            elif isinstance(sub, ast.Return) and sub.value is not None:
                return None, []
        if len(counts) != 1:
            return None, []
        return counts.pop(), []

    def _tuple_call_element_types(self, call:ast.Call)->Optional[List[object]]:
        """The device type of each element a tuple-returning helper answers, or None.

        Read from the helper's own returns: a `np.<dtype>(...)` cast names its width, a
        python float result names the mapping's width, a python int result is an int, and
        arithmetic over its own parameters (`return n, n + 2`) takes the type of the
        parameter it starts from -- the same rule the kernel's own expressions follow. That
        last case is what makes an ordinary helper usable: `lo, hi = bounds(i)` is how
        python spells "give me two numbers", and a helper written that way has no cast to
        read. Anything whose type is not statically known is refused rather than guessed.
        """
        if not isinstance(call.func, ast.Name):
            return None
        target:object = self.resolve_global(call.func.id)
        if not isinstance(target, types.FunctionType):
            return None
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
        except (OSError, TypeError):
            return None
        python_float_kind:str = self.type_map.get("float", "float")
        # The helper's own parameters, at the types its call site passes: `n` in
        # `def bounds(n)` is whatever the caller handed over, which is what the arithmetic
        # below has to start from.
        parameter_types:Dict[str, object] = {}
        try:
            names:List[str] = list(target.__code__.co_varnames[
                :target.__code__.co_argcount])
        except AttributeError:
            names = []
        for position, name in enumerate(names):
            if position < len(call.args):
                parameter_types[name] = self._visit_expression(call.args[position])[1]
        sites:List[List[object]] = []
        for sub in ast.walk(tree):
            if not (isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple)):
                continue
            element_types:List[object] = []
            for element in sub.value.elts:
                inferred:Optional[object] = self._tuple_element_type(
                    element, python_float_kind, parameter_types)
                if inferred is None:
                    return None
                element_types.append(inferred)
            sites.append(element_types)
        if len({tuple(site) for site in sites}) != 1 or not sites:
            return None
        return sites[0]

    def _tuple_element_type(self, element:ast.expr, python_float_kind:str,
                            parameter_types:Dict[str, object])->Optional[object]:
        """One element of a helper's tuple return, at its device type, or None.

        The shapes an ordinary two-value helper uses: a cast, a float/int constant, one of
        the helper's own parameters, or arithmetic over those. Everything else -- an
        argument's member, another call -- is not statically known here.
        """
        if isinstance(element, ast.Call) \
                and isinstance(element.func, ast.Attribute) \
                and isinstance(element.func.value, ast.Name) \
                and self.resolve_global(element.func.value.id) is np:
            return DTYPE_MAP.get(element.func.attr)
        if isinstance(element, ast.Constant):
            if isinstance(element.value, bool):
                return "bool"
            if isinstance(element.value, float):
                return python_float_kind
            if isinstance(element.value, int):
                return "int"
            return None
        if isinstance(element, ast.Name):
            return parameter_types.get(element.id)
        if isinstance(element, ast.UnaryOp):
            return self._tuple_element_type(element.operand, python_float_kind,
                                            parameter_types)
        if isinstance(element, ast.BinOp) and isinstance(
                element.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            left:Optional[object] = self._tuple_element_type(
                element.left, python_float_kind, parameter_types)
            right:Optional[object] = self._tuple_element_type(
                element.right, python_float_kind, parameter_types)
            if isinstance(element.op, ast.Div):
                # True division of two ints answers a float in python, so the result is the
                # mapping's float width rather than an int.
                return python_float_kind if left is not None and right is not None else None
            if left is None or right is None:
                return None
            if is_float(left) or is_float(right):
                return "double" if "double" in (left, right) else python_float_kind
            return "int"
        return None

    def _emit_tuple_call_unpack(self, targets:List[ast.expr], call:ast.Call)->None:
        """`x, y = pair(...)`: translate the call, passing the destinations as outputs.

        The helper's signature ends in one `T*` output per element (see `_visit_return`'s
        tuple branch and `render_device`), so the destinations travel by address and the
        helper's stores land in them -- the device shape of python's unpacking. A plain
        name is declared at its element type first, so the call site reads like the
        assignment it is; the types come from the helper's own returns
        (`_tuple_call_element_types`).
        """
        element_types:List[object] = (self._pending_tuple_types
                                      or self._tuple_call_element_types(call) or [])
        # The destinations are evaluated first, exactly as a tuple swap evaluates every
        # right-hand side before storing: an element destination captures its own address.
        address_codes:List[str] = []
        for ordinal, target_item in enumerate(targets):
            if isinstance(target_item, ast.Subscript):
                base_code, _base_t = self._visit_array_target(target_item.value)
                index_code, _ = self._visit_subscript_index(target_item)
                if isinstance(target_item.value, ast.Name):
                    self.atomic_targets.add(target_item.value.id)
                address_codes.append(f"&({base_code}[{index_code}])")
                continue
            if not isinstance(target_item, ast.Name):
                self._error(
                    "a tuple-returning call unpacks into plain variables or array "
                    "elements", target_item)
            name:str = target_item.id
            existing:DeviceType = self._lookup(name)
            if existing is not None and is_ptr(existing):
                self._error(f"cannot assign to array parameter '{name}'", target_item)
            if existing is None:
                if ordinal >= len(element_types):
                    self._error(
                        "cannot tell the type of the value this helper answers for "
                        f"'{name}': its returns are not all `np.<dtype>(...)` casts, "
                        f"float results, or the same shape -- declare the variable "
                        f"yourself (`x: np.float64 = 0.0`) and unpack into it", target_item)
                self._define(name, element_types[ordinal], target_item)
                self._emit(f"{self._device_type(element_types[ordinal])} {name};")
            address_codes.append(f"&({device_name(name)})")
        self._pending_tuple_types = None
        call_code, call_t = self._emit_device_call(
            call.func.id, self.resolve_global(call.func.id),
            [self._visit_expression(a) for a in call.args], call,
            tuple_targets=address_codes)
        if call_t != "void":
            self._error("the call does not answer a tuple; unpack a tuple-returning "
                        "helper", call)
        # The call is a statement here: the helper's own body stores through the
        # destinations, and the (void) answer carries nothing to assign.
        self._emit(f"{call_code};")

    def _visit_augassign(self, node:ast.AugAssign)->None:
        # A racy `arr[i] op= v` must not compile into a plain read-modify-write: every work
        # item reads the old value and writes it back, so contributions are lost with no
        # error at all. Python accumulates every one of them, so the device code has to use
        # an atomic operation whenever the translator cannot prove the elements are
        # distinct. `out[i] += x[i]` over the loop's own index stays a plain store.
        op_name:str = node.op.__class__.__name__
        # An outer scalar named `nonlocal`/`global` is one device location shared by every
        # work item, so an in-place update there is racy in exactly the same way a
        # data-driven array index is -- python accumulates every contribution.
        outer:Optional[Dict[str, object]] = (
            self.outer_vars.get(node.target.id)
            if isinstance(node.target, ast.Name) else None)
        carried:bool = (isinstance(node.target, ast.Name)
                        and self._lookup(node.target.id) is not None
                        and isinstance(self._lookup(node.target.id), tuple)
                        and self._lookup(node.target.id)[0] == "carried")
        racy:bool = ((isinstance(node.target, ast.Subscript)
                      and not self._augassign_is_race_free(node.target))
                     or (outer is not None and bool(outer["writable"]))
                     or carried)
        if racy:
            atomic_name:Optional[str] = self.ATOMIC_ASSIGN_OPS.get(op_name)
            if atomic_name is None:
                where:str = ("this shared outer variable"
                             if outer is not None else "this array")
                self._error(
                    f"'{self._augassign_symbol(op_name)}=' on {where} may be updated by "
                    f"several work items at once, and there is no device atomic operation "
                    f"for it. Write the update explicitly, e.g. use atomic_add for a sum, "
                    f"or accumulate per work item and reduce afterwards "
                    f"(see examples/09_reduction.py)", node)
            if carried:
                # One location shared by every work item, updated through its own buffer.
                # The atomic is the slot's own width (see _bind_carried_scalars, which
                # types the slot from the initializer): an int atomic handed a float
                # value truncated every contribution. The carried marker's second entry
                # is the *name*; the slot's type is the parameter's.
                carried_t:object = "int"
                for parameter_name, parameter_t in self.params:
                    if parameter_name == node.target.id and isinstance(parameter_t, tuple) \
                            and is_ptr(parameter_t):
                        carried_t = parameter_t[1]
                        break
                self._emit_atomic_update(node, op_name, atomic_name, carried_t,
                                         array_code=node.target.id, index_code="0",
                                         pointer_code=node.target.id)
                return
            pointer:Optional[str] = node.target.id if outer is not None else None
            self._visit_atomic_augassign(node, op_name, atomic_name, pointer_code=pointer)
            return

        target_code, target_t = self._visit_target(node.target)
        value_code, value_t = self._visit_expression(node.value)

        # A complex target has no C in-place operator either, but its arithmetic *does* have
        # a device meaning: the header's function, exactly as `_emit_complex_binop` builds
        # it for `z = z + c`. Python reads `z += c` as `z = z + c` when the class defines no
        # `__iadd__` -- the header's record defines none -- so the store is the same call.
        # Without this branch the plain C `+=` reached the compiler and was refused for a
        # struct (`no operator "+=" matches these operands (c128 += c128)`).
        if self._complex_dtype_of(target_t) is not None \
                and op_name in ("Add", "Sub", "Mult", "Div", "Pow"):
            operator_node:ast.BinOp = ast.BinOp(
                left=node.target, op=node.op, right=node.value,
                lineno=node.lineno, col_offset=node.col_offset)
            stored:Translated = self._emit_complex_binop(
                op_name, target_code, target_t, value_code, value_t, operator_node)
            self._emit(f"{target_code} = {stored[0]};")
            return

        # A structure target has no C in-place operator: `v += w` on a class is that
        # class's `__iadd__`, or its `__add__` when it defines no in-place form. The call
        # returns a new value, which is stored back -- exactly what python does.
        if self._is_structure_operand(target_t):
            inplace:Optional[str] = MAGIC_INPLACE.get(op_name)
            owner:Optional[Tuple[type, Any]] = (
                self._magic_owner(target_t, inplace) if inplace is not None else None)
            called:str = inplace or ""
            if owner is None:
                # No in-place method: python answers `v = v + w`, so the plain operator's
                # method (or the refusal that explains there is none) decides.
                magic:Optional[str] = MAGIC_METHODS.get(op_name)
                owner = (self._magic_owner(target_t, magic)
                         if magic is not None else None)
                called = magic or ""
            if owner is not None:
                _cls, method = owner
                call:Translated = self._emit_device_call(
                    called, method, [(target_code, target_t), (value_code, value_t)],
                    node, self_param=_receiver_param_of(method), receiver_as_argument=True)
                self._emit(f"{target_code} = {call[0]};")
                return
            self._error(self._structure_operator_message(op_name, target_t, value_t, node), node)

        # A vector has no C in-place operator either (CUDA rejects `v += w`), so the store
        # is written out as the plain vector operation it means. The right operand goes into
        # a temporary first: the per-component form names it once per component, and a call
        # with side effects (`pixel_color += scene.ray_color(...)`) must be evaluated once.
        vector_store:Optional[str] = self._emit_vector_inplace(
            target_code, op_name, target_code, target_t, value_code, value_t)
        if vector_store is not None:
            if value_code.strip().endswith(")") and "(" in value_code:
                temporary:str = f"_inplace{self._next_temp()}"
                self._emit(f"{value_t} {temporary} = {value_code};")
                vector_store = self._emit_vector_inplace(
                    target_code, op_name, target_code, target_t, temporary, value_t)
            self._emit(vector_store)
            return

        name:str = op_name
        if name == "Div":
            if is_int(target_t) and is_int(value_t):
                self._error("integer /= integer changes type (use // or float operands)", node)
            self._emit(f"{target_code} /= {value_code};")
        elif name == "FloorDiv":
            if is_int(target_t) and is_int(value_t):
                # `a //= b` must floor like python, not truncate like C (see _emit_floored).
                code, _ = self._emit_floored(("floor_div", "floor_mod"), "floor_div",
                                             target_code, target_t, value_code,
                                             value_t, node)
                self._emit(f"{target_code} = {code};")
            else:
                self._emit(f"{target_code} = floor(({target_t})({target_code}) / ({target_t})({value_code}));")
        elif name == "Pow":
            # There is no **= in C; the power is a function call, so the store is
            # written explicitly. An integer target would change type under python's
            # rules, which a C variable cannot do, so only floats are accepted.
            if is_int(target_t) and is_int(value_t):
                self._error("integer **= changes type (use float operands or pow())", node)
            self._emit(f"{target_code} = ({target_t})pow(({target_t})({target_code}), "
                       f"({target_t})({value_code}));")
        elif name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor", "Add", "Sub", "Mult"):
            symbol:str = {"LShift": "<<", "RShift": ">>", "BitOr": "|", "BitAnd": "&", "BitXor": "^",
                          "Add": "+", "Sub": "-", "Mult": "*"}[name]
            self._emit(f"{target_code} {symbol}= {value_code};")
        elif name == "Mod":
            if is_int(target_t) and is_int(value_t):
                # `a %= b` must carry the divisor's sign like python's `%` (see _emit_floored).
                code, _ = self._emit_floored(("floor_div", "floor_mod"), "floor_mod",
                                             target_code, target_t, value_code,
                                             value_t, node)
                self._emit(f"{target_code} = {code};")
            else:
                symbol = "%"
                self._emit(f"{target_code} {symbol}= {value_code};")
        else:
            self._error(f"unsupported operator: {name}", node)

    def _lookup_type_of_index(self, node:ast.AugAssign)->str:
        return "int"

    def _visit_target(self, target:ast.expr)->Translated:
        if isinstance(target, ast.Name):
            outer:Optional[OuterVarEntry] = self.outer_vars.get(target.id)
            if outer is not None:
                if not outer["writable"]:
                    self._error(
                        f"cannot assign to the outer variable '{target.id}': without a "
                        f"'nonlocal {target.id}' or 'global {target.id}' declaration this "
                        f"would create a local variable, shadowing the outer one", target)
                # A store through the parameter as `name[0]` rather than `(*name)`: both are
                # the same location, but only the subscript form is recognized as a write
                # by the static read/write analysis, which decides whether the value is
                # read back to the host.
                return f"{target.id}[0]", outer["type"]
            t:DeviceType = self._lookup(target.id)
            if t is None:
                self._error(f"variable '{target.id}' used before assignment", target)
            if is_ptr(t):
                self._error(f"cannot assign to array parameter '{target.id}'", target)
            return target.id, t
        if isinstance(target, ast.Subscript):
            # A string subscript on a structure value writes a member: `s["i"] = 1` becomes
            # `s.i = 1`. It has to be recognized before the array path, which would reject a
            # structure base (it is a value, not a pointer).
            member:Optional[Translated] = self._struct_member_access(target)
            if member is not None:
                return member
            # A store into a zeros_per_group() slice: the same flattening as the read side.
            slice_code, slice_t = self._visit_expression(target.value)
            if isinstance(slice_t, tuple) and slice_t and slice_t[0] == SHARED_SLICE:
                element_code, element_t = self._visit_shared_index(slice_code, slice_t, target)
                return self._scalar_position(element_code, element_t, target)
            base_code, base_t = self._visit_array_target(target.value)
            index_code, _ = self._visit_subscript_index(target)
            self._record_element_write(target.value)
            return f"{base_code}[{index_code}]", base_t
        if isinstance(target, ast.Attribute):
            # `p.x = ...` on a structure value: a C member store. The read side already
            # handles attribute access on structures; this is its write twin. A written
            # object argument travels as a one-element record array, so the store goes
            # through the element (`p[0].x = ...`), which is what makes it reach the host.
            code:str
            t:object
            code, t = self._visit_expression(target.value)
            if isinstance(t, tuple) and is_ptr(t):
                dtype:object = structs_dtype_for(t[1])
                if dtype is None:
                    self._error(
                        f"cannot assign to '{ast.unparse(target)}': {t[1]} is not a "
                        f"structure the translator knows", target)
                # `box.total = v` is a call to the property's setter, next to the getter
                # the read side already answers.
                member_types:Dict[str, object] = structs_member_types(dtype, self.backend)
                if target.attr not in member_types:
                    self._error(
                        f"structure value has no member '{target.attr}' "
                        f"(available: {', '.join(member_types)})", target)
                # The static analysis sees a member expression, not a subscript, so it never
                # marks the array as written; without this flag the host would not read the
                # stores back. `atomic_targets` is exactly the force-writable channel.
                self.atomic_targets.add(target.value.id)
                if code.startswith("&"):
                    # A method's own receiver, by address: a plain pointer store.
                    return f"({code})->{target.attr}", member_types[target.attr]
                return f"({code}[0]).{target.attr}", member_types[target.attr]
            if is_vec(t) or not isinstance(t, str) or structs_dtype_for(t) is None:
                self._error(
                    f"cannot assign to '{ast.unparse(target)}': member stores need a "
                    f"structure value, and {t} is not one", target)
            member_types:Dict[str, object] = structs_member_types(structs_dtype_for(t), self.backend)
            if target.attr not in member_types:
                self._error(
                    f"structure value has no member '{target.attr}' "
                    f"(available: {', '.join(member_types)})", target)
            return f"({code}).{target.attr}", member_types[target.attr]
        self._error(f"unsupported assignment target: {target.__class__.__name__}", target)

    def _target_is_bool_member(self, target:ast.expr, target_t:object, value_t:object)->bool:
        """Whether a member store lands in a bool-typed field while holding a wider type."""
        from . import structs  # noqa: PLC0415

        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
            return False
        if value_t == "bool":
            return False
        base_t:DeviceType = self._lookup(target.value.id)
        struct_t:object = base_t[1] if is_ptr(base_t) else base_t
        dtype:object = structs.dtype_for(struct_t) if isinstance(struct_t, str) else None
        if dtype is None or target.attr not in (dtype.names or ()):
            return False
        return np.dtype(dtype.fields[target.attr][0]).name == "bool"

    def _visit_array_target(self, node:ast.expr)->Translated:
        code, t = self._visit_expression(node)
        if is_image(t):
            # An image keeps its own marker: indexing it means sampling, not
            # pointer arithmetic, so the caller decides from the marker.
            return code, t
        if not is_ptr(t):
            self._error("only array (numpy) and image arguments can be indexed", node)
        return code, t[1]

    def _record_subscript_indices(self, node:ast.Subscript)->None:
        """Remember how many indices each array is subscripted with.

        A pointer parameter carries no rank, so the translator cannot tell whether
        ``out[i, j]`` addresses one element or selects a slice; the value passed at call
        time can. Recording the count here lets the launch compare it with the array's
        real shape instead of silently writing one element of a 3-D array (which is what
        ``out[i, j] = v`` used to compile into).
        """
        base:ast.expr = node.value
        name:str = ""
        if isinstance(base, ast.Name):
            name = base.id
        elif (self.self_param and isinstance(base, ast.Attribute)
              and isinstance(base.value, ast.Name) and base.value.id == self.self_param):
            name = f"_self_{base.attr}"
        else:
            return
        if not is_ptr(self._lookup(name)):
            return
        count:int = len(node.slice.elts) if isinstance(node.slice, ast.Tuple) else 1
        previous:int = self.subscript_indices.get(name, 0)
        if count > previous:
            # The widest subscript is the one that decides; among equally wide ones the
            # FIRST is the one python would raise at, because it is the first the body
            # reaches. Its line is in the parsed copy's numbering, which _body_frame
            # converts back to the user's file, and the column is where the subscript
            # expression starts, so the caret underlines the indexing rather than the
            # whole statement.
            self.subscript_lines[name] = getattr(node, "lineno", 0) or 0
            self.subscript_columns[name] = getattr(node, "col_offset", 0) or 0
            self.subscript_texts[name] = ast.get_source_segment(self.source_text, node) or ""
        self.subscript_indices[name] = max(previous, count)

    def _visit_subscript_index(self, node:ast.Subscript)->Translated:
        self._record_subscript_indices(node)
        return self._flatten_index(node.slice, node.value, node)

    def _shared_slice_shape(self, node:ast.expr)->Optional[Tuple[int, ...]]:
        """The shape a `zeros_per_group` slice was declared with, or None if it is not one."""
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            return None
        t:DeviceType = self._lookup(node.value.id)
        if not (isinstance(t, tuple) and len(t) == 5 and t[0] == "shared"):
            return None
        if not self._is_block_id_expression(node.slice):
            return None
        shape:Tuple[int, ...] = tuple(t[4])
        return shape

    def _flatten_index(self, index:ast.expr, target:ast.expr,
                       node:ast.AST)->Translated:
        """One flat offset for an index that may name several axes.

        A tuple index (`a[i, j]`) is row-major flattening over the *declared* shape, which is
        why it needs the array's name; a scalar index is already a flat offset and is left
        alone, so the 1-D spelling keeps working on a multidimensional array.

        A **negative index** is python's from-the-end spelling, and emitting it raw was a
        silent out-of-bounds access: `a[-1]` read the word *before* the buffer and answered
        0.0 where python answers the last element, and `a[-1] = v` wrote outside the array.
        A constant negative index becomes `extent - n` at translation time; a computed one
        gets the same correction around a `%`-free select, so `i - n` cannot underflow into
        a neighbouring array. An extent of zero is python's own `IndexError` territory, so
        the expression is clamped to 0 rather than left negative (the caller's bounds guard
        already skips the work item when the array is empty).

        An atomic call passes the whole array as the target, so an atomic on a shared buffer
        must name the group's slice (`buf[blockIdx]`) rather than the buffer.
        """
        if isinstance(index, ast.Tuple):
            slice_shape:Optional[Tuple[int, ...]] = self._shared_slice_shape(target)
            if slice_shape is not None:
                # An atomic on a shared buffer names the group's slice as its array, and the
                # slice carries its own shape, so the flattening needs no `_shape_var`.
                dims:List[Tuple[str, object]] = [
                    (self._visit_expression(e), e) for e in index.elts]
                terms:List[str] = []
                for d, ((code, t), element) in enumerate(dims):
                    if not is_int(t):
                        self._error("array indices must be integers", node)
                    code = self._python_index(
                        code, element,
                        literal_extent=(slice_shape[d] if d < len(slice_shape) else None))
                    term:str = code
                    for inner in range(d + 1, len(dims)):
                        if inner >= len(slice_shape):
                            self._error(
                                f"this shared buffer has {len(slice_shape)} dimension(s), so "
                                f"it cannot be indexed with {len(dims)}", node)
                        term = f"{term} * {slice_shape[inner]}"
                    terms.append(term)
                return " + ".join(terms), "int"
            if self._lookup(self._array_name(target)) is not None \
                    and isinstance(self._lookup(self._array_name(target)), tuple) \
                    and self._lookup(self._array_name(target))[0] == "shared":
                self._error(
                    "a shared buffer is indexed by its group first, so pass the tuple after "
                    "the group: `buf[blockIdx][i, j]`", node)
            array_name:str = self._array_name(target)
            dims = [(self._visit_expression(e), e) for e in index.elts]
            terms = []
            for d, ((code, t), element) in enumerate(dims):
                if not is_int(t):
                    self._error("array indices must be integers", node)
                code = self._python_index(code, element, target, dim=d)
                term = code
                for inner in range(d + 1, len(dims)):
                    term = f"{term} * {self._shape_var(array_name, inner)}"
                terms.append(term)
            return " + ".join(terms), "int"
        code, t = self._visit_expression(index)
        if not is_int(t):
            self._error("array index must be an integer", node)
        return self._python_index(code, index, target), t

    def _python_index(self, code:str, node:ast.expr,
                      target:Optional[ast.expr]=None, dim:int=0,
                      literal_extent:Optional[int]=None)->str:
        """A python index as the device reads it, with a constant negative index folded.

        `a[-1]` is python's last element, and the device has no such addressing: the index
        is a plain offset, so a raw `-1` reaches memory *before* the buffer. A constant
        negative index is therefore rewritten as `extent - n`, which is the element python
        names -- `a[-1]` and `a[n - 1]` are the same access, so the fold costs nothing.

        The extent is looked up **only** when the fold actually happens, and that matters
        beyond tidiness: `_shape_var` is what *registers* an extent parameter, so asking
        for one eagerly made every kernel subscript declare a shape it never read. A
        gid-style Metal body then grew an `_out_shape0` parameter it is defined not to
        have, and a 2-D one grew axis 0 as well as axis 1.

        A *computed* index is passed through unchanged, deliberately. Wrapping it in a
        sign test would evaluate the expression twice (wrong for a call or a subscript) and
        would put a select around every array access in every kernel to pay for a case that
        is an out-of-range index by python's own rules -- `a[k]` with a negative `k` is the
        same kind of mistake as `a[100]` on a short array, which the device has never
        bounds-checked either.
        """
        constant:Optional[int] = _constant_int_of(node)
        if constant is None or constant >= 0:
            return code
        extent:Optional[str] = None
        if literal_extent is not None:
            extent = str(literal_extent)
        elif target is not None:
            try:
                extent = self._shape_var(self._array_name(target), dim)
            except SyntaxError:
                extent = None
        if extent is None:
            self._error(
                "a negative index needs the array's extent to count from the end, and "
                "the array this index addresses is not known here", node)
        # `extent - n` is python's answer. If the array is shorter than the index python
        # raises IndexError, and no work item can raise it, so the offset is clamped.
        needed:int = -constant
        return f"((({extent}) - {needed} >= 0) ? (({extent}) - {needed}) : 0)"
        if constant is not None:
            if constant >= 0:
                return code
            if extent is None:
                self._error(
                    "a negative index needs the array's extent to count from the end, and "
                    "the array this index addresses is not known here", node)
            # `extent - n` is python's answer. If the array is shorter than the index python
            # raises IndexError, and no work item can raise it, so the offset is clamped.
            needed:int = -constant
            return f"((({extent}) - {needed} >= 0) ? (({extent}) - {needed}) : 0)"
        if extent is None:
            return f"((({code}) >= 0) ? ({code}) : 0)"
        return f"((({code}) >= 0) ? ({code}) : (({code}) + ({extent})))"

    def _array_name(self, node:ast.expr)->str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return self._array_name(node.value)
        if (self.self_param and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == self.self_param):
            return f"_self_{node.attr}"
        self._error("unsupported array expression", node)

    # ---------- expressions ----------

    def _visit_expression(self, node:ast.expr)->Translated:
        if isinstance(node, ast.Constant):
            return self._visit_constant(node)
        if isinstance(node, ast.Name):
            return self._visit_name(node)
        if isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._visit_unaryop(node)
        if isinstance(node, ast.Compare):
            return self._visit_compare(node)
        if isinstance(node, ast.BoolOp):
            # Each operand is a *condition* in python (`and`/`or` read their operands as
            # bools), so a complex operand reaches the truthiness rule instead of arriving
            # at the compiler as a struct the operator cannot take.
            parts:List[str] = [self._condition_code(v) for v in node.values]
            symbol:str = ("&&" if isinstance(node.op, ast.And) else "||")
            joined:str = f" {symbol} ".join(parts)
            return f"({joined})", "bool"
        if isinstance(node, ast.IfExp):
            cond:str = self._condition_code(node.test)
            then_code, tt = self._visit_expression(node.body)
            else_code, et = self._visit_expression(node.orelse)
            if tt != et:
                unified:object = promote(tt, et)
                return f"({cond} ? ({unified})({then_code}) : ({unified})({else_code}))", unified
            return f"({cond} ? {then_code} : {else_code})", tt
        if isinstance(node, ast.Call):
            return self._visit_call(node)
        if isinstance(node, ast.Subscript):
            return self._visit_subscript(node)
        if isinstance(node, ast.Attribute):
            return self._visit_attribute(node)
        self._error(f"unsupported expression: {node.__class__.__name__}", node)

    def _visit_enum_constant(self, entry:Tuple[Any, Any], node:ast.AST)->Translated:
        """Emit an enum member as the constant it stands for.

        An `IntEnum` member's value is an int and an `Enum`'s may be anything; only the
        numeric ones have a device meaning, so a member whose value is not a number is
        refused with the value's type named -- there is nothing to emit for it.
        """
        payload:object = entry[0]
        member:object = entry[1]
        name:str = f"{type(member).__name__}.{getattr(member, 'name', '?')}"
        if isinstance(payload, enum.Enum):
            payload = payload.value
        if isinstance(payload, bool):
            return ("true" if payload else "false"), "bool"
        if isinstance(payload, int):
            if payload == 0:
                return "0", "int"
            # An enumerator is spelled in the C source, so a negative one keeps its sign.
            return str(payload), "int"
        if isinstance(payload, float):
            return self._visit_constant(ast.Constant(value=payload))
        self._error(
            f"{name} is an enum member whose value is a {type(payload).__name__}, which has "
            f"no device equivalent: give the member a numeric value (an int or a float), or "
            f"read the value on the host and pass it as a scalar argument", node)

    def _visit_constant(self, node:ast.Constant)->Translated:
        value:object = node.value
        if isinstance(value, bool):
            return ("true" if value else "false"), "bool"
        if isinstance(value, int):
            return str(value), "int"
        if isinstance(value, float):
            # A python float literal has no width of its own, so it takes the one the
            # `float` mapping chooses -- the same rule a python complex literal follows
            # (`1 + 2j` is a c128 under `float=np.float64`), and the same rule a python
            # float *argument* follows. Spelling every literal with the `f` suffix cut a
            # 17-digit constant down to float32 inside a double kernel.
            if self.type_map.get("float", "float") == "double":
                return self._real_literal(value, "double"), "double"
            text:str = repr(value)
            if not ("." in text or "e" in text or "E" in text or "inf" in text or "nan" in text):
                text += ".0"
            if not text.endswith(("f", "F", "y")) and "inf" not in text:
                text += "f"
            return text, "float"
        if isinstance(value, complex):
            # A complex literal is written with its two parts, because C has no complex
            # constant: the header builds the value from the reals. A python complex has no
            # width of its own, so it takes the one the `float` mapping chooses -- the same
            # rule a python float follows -- which is why `1 + 2j` meets a complex64 operand
            # as a complex64 while `@kernel(float=np.float64)` makes it a complex128.
            kind:str = self._python_complex_kind()
            return self._emit_complex_literal(value.imag, value.real, kind), kind
        self._error(f"unsupported constant: {value!r}", node)

    def _complex_store(self, target_t:object, value_t:object, code:str)->str:
        """`code` as a value the target type can hold, for a complex assignment.

        C has no complex conversion operator, so a store between two different widths --
        `out64[i] = z` where `z` is a complex128, or a return slot of the other width -- would
        emit `c64 = c128` and fail to compile. The header's widening and narrowing functions
        are the conversion, and they are the *same* ones a cast uses, so a store and an
        explicit `np.complex64(z)` mean the same thing.

        A real source assigned into a complex target is lifted to the complex value with a
        zero imaginary part, which is numpy's rule (`out64[i] = 2.0` is `2 + 0i`).
        """
        target_complex:Optional[np.dtype] = self._complex_dtype_of(target_t)
        value_complex:Optional[np.dtype] = self._complex_dtype_of(value_t)
        if target_complex is None:
            if value_complex is not None:
                self._error(
                    f"cannot assign a complex value ({value_t}) into a {target_t}: a complex"
                    f" number has no real counterpart, so write `.real` or `.imag` if that"
                    f" is what you meant")
            return code
        kind:str = self._complex_side(target_t, target_complex)
        self._require_complex(target_complex)
        if value_complex is None:
            if is_ptr(value_t) or is_vec(value_t) or is_image(value_t):
                return code
            element:str = "double" if kind == "c128" else "float"
            return f"{kind}_from_real(({element})({code}))"
        value_kind:str = self._complex_side(value_t, value_complex)
        return self._convert_complex(code, value_kind, kind)

    def _convert_complex(self, code:str, from_kind:str, to_kind:str)->str:
        """`code` as a value of the other complex width, through the header's conversion.

        C has no complex conversion operator, so the header's widening and narrowing functions
        are the conversion. They are the *same* ones a cast uses, so a store, an explicit
        `np.complex64(z)` and a python complex reaching the mapped width all mean one thing.
        """
        if from_kind == to_kind:
            return code
        self.complex_needs.update(complex_support.mixed_needs())
        converter:str = (f"{from_kind}_widen_{to_kind}" if to_kind == "c128"
                         else f"{from_kind}_narrow_{to_kind}")
        return f"{converter}({code})"

    def _python_complex_kind(self)->str:
        """The header's spelling of the width a python `complex` travels as in this kernel.

        The one decision point on this side of the translation: the mapping's `float` entry
        chooses it through `python_complex_type`, so a body literal and `complex(re, im)`
        cannot disagree with the `c64`/`c128` a python complex *argument* is declared as.
        """
        return python_complex_type(self.type_map.get("float", "float"))

    def _emit_complex_literal(self, imaginary:float, real:float, kind:str)->str:
        """One complex constant, as the header's constructor over two real literals."""
        element:str = "double" if kind == "c128" else "float"
        self._require_complex(np.dtype(np.complex128 if kind == "c128" else np.complex64))
        return (f"{kind}_make({self._real_literal(real, element)},"
                f" {self._real_literal(imaginary, element)})")

    def _real_literal(self, value:float, width:str)->str:
        """A real literal of the given width, spelled so the C compiler reads it as one."""
        text:str = repr(float(value))
        if not ("." in text or "e" in text or "E" in text or "inf" in text or "nan" in text):
            text += ".0"
        if width != "double" and not text.endswith(("f", "F", "y")) and "inf" not in text:
            text += "f"
        return text

    def _visit_name(self, node:ast.Name)->Translated:
        host:Optional[HostValueEntry] = self.host_values.get(node.id)
        if host is not None:
            # The host created this object; on the device it is just the parameter that
            # carries it (see _host_created_assignment).
            return device_name(node.id), host["type"]
        host_scalar:Optional[DeviceType] = self.host_scalars.get(node.id)
        if host_scalar is not None:
            # A host-region statement bound this value; the device reads the parameter
            # that carries it (see _bind_host_statement).
            return device_name(node.id), host_scalar
        outer:Optional[Dict[str, object]] = self.outer_vars.get(node.id)
        if outer is not None:
            # A writable outer variable travels as `T*`, so every use is a dereference; a
            # read-only one travels as `T` and is simply the parameter itself.
            if outer["writable"]:
                return f"(*{device_name(node.id)})", outer["type"]
            return device_name(node.id), outer["type"]
        t:DeviceType = self._lookup(node.id)
        if isinstance(t, tuple) and len(t) == 2 and t[0] == "carried":
            # One location for the whole launch. A subscript rather than (*name) for the
            # same reason an outer scalar uses one: the read/write analysis only recognizes
            # that form as a write, so the host reads the final value back.
            return f"{device_name(t[1])}[0]", "int"
        if t is not None:
            if isinstance(t, tuple) and len(t) == 5 and t[0] == "shared":
                # A shared buffer is not itself a pointer: it is one slice per group, and
                # only uf[block_id(d)] picks the caller's out (see _visit_shared_slice).
                self._error(
                    f"'{node.id}' is a zeros_per_group() buffer, which is one slice per work group: "
                    f"index it with the group first, as '{node.id}[blockIdx]'", node)
            if isinstance(t, tuple) and len(t) == 4 and t[0] == "bodyarr":
                # A private array created in the body is indexed like any array; its extents
                # are constants (see _shape_var).
                return device_name(node.id), (PTR, t[3])
            return device_name(node.id), t
        # CUDA's index vectors are values, not calls (`buf[blockIdx][j]`), so the *name*
        # resolves to the vector the backend builds from its own launch terms.
        if node.id in INDEX_VECTOR_BUILTINS:
            table:Dict[str, Tuple[str, str]] = (
                INDEX_VECTOR_OPENCL if self._opencl_syntax else INDEX_VECTOR_BUILTINS)
            vector_type, code = table[node.id]
            return code, vector_type
        value:object = self.resolve_global(node.id)
        enum_value:Optional[Tuple[Any, Any]] = _enum_member_value(value)
        if enum_value is not None:
            # An enum member is a named constant, so the device reads its value. Python has
            # no device enum type to name here -- the number is what every operator and
            # every function that accepts it already wants.
            return self._visit_enum_constant(enum_value, node)
        if isinstance(value, bool):
            return self._visit_constant(ast.Constant(value=value))
        if isinstance(value, (int, float)) and not isinstance(value, np.ndarray):
            return self._visit_constant(ast.Constant(value=value))
        self._error(f"unknown name '{node.id}' (only parameters, local variables and int/float/bool global constants are supported)", node)

    def _emit_vector_inplace(self, target_code:str, op_name:str,
                             left:str, lt:object, right:str, rt:object)->Optional[str]:
        """`v += w` on a vector as `v = v + w`, or None when that form does not apply.

        A vector has no C in-place operator: CUDA rejects `v += w` outright. The ordinary
        `+` over vectors already produces the per-component form the vector dispatch knows
        (see `_visit_binop`), so the store is written out instead.
        """
        symbol:str = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/"}.get(op_name, "")
        if not symbol or not (is_vec(lt) or is_vec(rt)):
            return None
        if is_vec(lt) and is_vec(rt):
            if lt != rt:
                return None      # the caller reports the mismatched types
            if self._opencl_syntax:
                return f"{target_code} = ({left} {symbol} {right});"
            # The right operand can be a call with side effects (`pixel_color +=
            # scene.ray_color(...)`), and the per-component form names it once per
            # component: evaluating it three times would be wrong and wasteful, so it goes
            # into a temporary first, which the caller emits to get a name back.
            return f"{target_code} = {self._vec_componentwise(symbol, left, lt, right=right)};"
        # A vector with a scalar broadcast: the vector side decides which is which.
        vector_type:object = lt if is_vec(lt) else rt
        vector_operand:str = left if is_vec(lt) else right
        scalar_operand:str = right if is_vec(lt) else left
        if self._opencl_syntax:
            return f"{target_code} = ({left} {symbol} {right});"
        expanded:str = self._vec_componentwise(symbol, vector_operand, vector_type,
                                               scalar=scalar_operand)
        return f"{target_code} = {expanded};"

    def _vec_componentwise(self, op:str, left:str, vec:str, right:str="", scalar:str="")->str:
        """CUDA has no vector arithmetic/math overloads; expand into per-component form.

        When op is an infix symbol (+ - * /) generate `l.c op r.c`; when it is a
        function name (sqrt etc.) generate `op(l.c, r.c)`.
        right is a vector operand, scalar is a broadcast scalar operand.
        """
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        infix:bool = (op in ("+", "-", "*", "/"))
        terms:List[str] = []
        for c in components:
            if scalar:
                terms.append(f"({left}.{c} {op} {scalar})" if infix else f"{op}({left}.{c}, {scalar})")
            elif right:
                terms.append(f"({left}.{c} {op} {right}.{c})" if infix else f"{op}({left}.{c}, {right}.{c})")
            else:
                terms.append(f"({op}{left}.{c})" if infix else f"{op}({left}.{c})")
        return self._vec_maker(vec, ", ".join(terms))

    def _emit_quaternion_binop(self, operator:str, left:str, lt:object, right:str, rt:object,
                               node:ast.BinOp)->Optional[Translated]:
        """A quaternion operation, as a call to one of the header's functions.

        `pygf` gives a quaternion three meanings: `q * q` composes two rotations, `q * v`
        rotates a vector, and a scalar scales it. None is a C operator on a structure, so each
        becomes a function from the fixed header and the expression becomes a call.

        A float and a double operand mix freely, and pygf's rule is that the result is the
        **double** form -- `quatf * quatd` is a `quatd`, `quatf * double3` is a `double3`,
        `quatd * float3` is a `double3`. Widening only ever goes that way, so nothing is
        rounded down; each mixed combination has its own function in the header.
        """
        from . import quaternion  # noqa: PLC0415

        # A structure operand travels as a pointer, so a quaternion variable is `quatf*`:
        # unwrap it for the type, and dereference it in the emitted expression.
        if isinstance(lt, tuple) and is_ptr(lt) and structs_dtype_for(lt[1]) is not None:
            lt, left = lt[1], f"(*{left})"
        if isinstance(rt, tuple) and is_ptr(rt) and structs_dtype_for(rt[1]) is not None:
            rt, right = rt[1], f"(*{right})"
        left_quat:Optional[np.dtype] = quaternion.is_quaternion(lt)
        right_quat:Optional[np.dtype] = quaternion.is_quaternion(rt)
        if left_quat is None and right_quat is None:
            return None
        if operator not in ("Add", "Sub", "Mult", "Div"):
            return None

        def call(name:str, dtype:np.dtype, args:str, result:object,
                 mixed:bool = False) -> Translated:
            # The types and their arithmetic both come from the fixed header; all this records
            # is which widths are in use, so the compile step can enable their blocks. A mixed
            # operation touches both, because its body widens one width to the other.
            if mixed:
                self.quaternion_needs.update(quaternion.mixed_needs())
            else:
                self._require_quaternion_header(dtype)
            return f"{name}({args})", result

        def widen(left_dtype:np.dtype, right_dtype:np.dtype, operation:str) -> str:
            """The header function for an operation on these two operand types.

            The same width is `<type>_<operation>`; a mixed pair is
            `<left type>_<operation>_<right type>`, which is how the header spells every mixed
            combination. Deriving the name keeps the two in step without a table.
            """
            left_name:str = structs_name(left_dtype)
            if left_name == structs_name(right_dtype):
                return f"{left_name}_{operation}"
            return f"{left_name}_{operation}_{structs_name(right_dtype)}"

        if operator == "Mult":
            if left_quat is not None and right_quat is not None:
                name:str = widen(left_quat, right_quat, "mul")
                # A mixed pair answers the double form, which is a real structure the header
                # declares, so it is registered under its fixed name like any other.
                result:object = (lt if structs_name(left_quat) == structs_name(right_quat)
                              else structs_name(quaternion.double_dtype()))
                return call(name, quaternion.wider(left_quat, right_quat),
                            f"{left}, {right}", result,
                            mixed=structs_name(left_quat) != structs_name(right_quat))
            if left_quat is not None and is_vec(rt):
                # A float quaternion rotating a double vector answers a double vector, and a
                # double one rotating a float vector likewise; only the matching pair is the
                # plain `_rotate`.
                same:bool = quaternion.matches_vector(left_quat, rt)
                name = (f"{structs_name(left_quat)}_rotate" if same
                        else f"{structs_name(left_quat)}_rotate_{rt}")
                result_vec:object = rt if same else (
                    "double3" if rt.startswith("float") else rt)
                # A widened rotation needs the double block too, since it answers a double3.
                return call(name, left_quat, f"{left}, {right}", result_vec, mixed=not same)
            if left_quat is not None and is_float(rt):
                return call(f"{structs_name(left_quat)}_scale", left_quat,
                            f"{left}, {right}", lt)
            if right_quat is not None and is_float(lt):
                # `2.0 * q` multiplies the same way, and the scalar comes first in C.
                return call(f"{structs_name(right_quat)}_scale", right_quat,
                            f"{right}, {left}", rt)
            return None

        if operator == "Div" and left_quat is not None and is_float(rt):
            return call(f"{structs_name(left_quat)}_scale", left_quat,
                        f"{left}, (1.0f / ({right}))", lt)

        if operator in ("Add", "Sub"):
            # Component-wise, which is what the base class would do.
            if left_quat is None or right_quat is None:
                return None
            suffix:str = "add" if operator == "Add" else "sub"
            name = widen(left_quat, right_quat, suffix)
            result = (lt if structs_name(left_quat) == structs_name(right_quat)
                      else structs_name(quaternion.double_dtype()))
            return call(name, quaternion.wider(left_quat, right_quat), f"{left}, {right}",
                        result, mixed=structs_name(left_quat) != structs_name(right_quat))

        return None

    # ---------- complex values ----------

    def _require_complex(self, dtype:object)->None:
        """Record that this source uses a complex width, and enable its block."""
        need:Optional[str] = complex_support.needs(dtype)
        self.uses_complex = True
        if need is not None:
            self.complex_needs.add(need)

    def _complex_dtype_of(self, t:object)->Optional[np.dtype]:
        """The complex dtype behind a translated type name, or None."""
        base:object = t[1] if isinstance(t, tuple) and is_ptr(t) else t
        if not isinstance(base, str):
            return None
        return complex_support.dtype_for_type_name(base)

    def _complex_operand(self, code:str, t:object)->Tuple[str, Optional[np.dtype]]:
        """A complex operand as a value: a pointer is dereferenced first."""
        dtype:Optional[np.dtype] = self._complex_dtype_of(t)
        if dtype is not None and isinstance(t, tuple) and is_ptr(t):
            return f"(*{code})", dtype
        return code, dtype

    def _condition_code(self, node:ast.expr)->str:
        """A condition as C reads it: `(code)` for what is already a scalar test.

        A complex value is truthy in python when either part is non-zero, so a condition
        holding one (`if z:`, `while z and n < max:`) is rewritten as the comparison the
        record itself cannot express -- the plain `if (z)` reached the compiler and was
        refused with "expression must have bool type". A zero complex value means
        `z.re == 0 && z.im == 0`, which is the negation of the header's own inequality
        against the zero the constructor builds.
        """
        code, t = self._visit_expression(node)
        dtype:Optional[np.dtype] = self._complex_dtype_of(t)
        if dtype is None:
            return f"({code})"
        self._require_complex(dtype)
        kind:str = self._complex_side(t, dtype)
        return (f"({kind}_equal({code}, {kind}_from_real(0)) == 0)")

    def _complex_side(self, t:object, dtype:Optional[np.dtype])->str:
        """One operand's width, in the header's spelling: ``c64``/``c128``/``float``/``double``."""
        if dtype is not None:
            return complex_support.complex_type_name(dtype) or "c64"
        base:object = t[1] if isinstance(t, tuple) and is_ptr(t) else t
        return "double" if base == "double" else "float"

    def _emit_complex_binop(self, operator:str, left:str, lt:object, right:str, rt:object,
                            node:ast.BinOp)->Optional[Translated]:
        """A complex operation, as a call to one of the header's functions.

        A complex value is a structure on the device, so none of `+ - * / **` is a C operator:
        each becomes a function from the fixed header, named after the operand widths. The
        arithmetic itself is numpy's, which is also what the interpreter computes.

        A float and a double operand mix freely and the result is the **double** form, the same
        rule the quaternion arithmetic follows: `c64 + c128` is a `c128`, `c64 * 2.0` (a python
        float, which is a double) is a `c128`, while `c64 * np.float32(2)` stays a `c64`.
        Widening only ever goes that way, so nothing is rounded down.
        """
        left_code, left_dtype = self._complex_operand(left, lt)
        right_code, right_dtype = self._complex_operand(right, rt)
        if left_dtype is None and right_dtype is None:
            return None
        if operator not in ("Add", "Sub", "Mult", "Div", "Pow"):
            return None

        left_side:str = self._complex_side(lt, left_dtype)
        right_side:str = self._complex_side(rt, right_dtype)
        widened:bool = "c128" in (left_side, right_side) or "double" in (left_side, right_side)
        kind:str = "c128" if widened else "c64"
        suffix:str = {"Add": "add", "Sub": "sub", "Mult": "mul", "Div": "div",
                      "Pow": "pow"}[operator]

        # A mixed operation's body widens one width to the other, so its source needs both
        # blocks; a same-width one needs only its own.
        if left_dtype is not None and right_dtype is not None \
                and left_side != right_side:
            self.complex_needs.update(complex_support.mixed_needs())
        elif (left_dtype is not None and right_dtype is not None) and widened:
            self._require_complex(np.dtype(np.complex128))
        elif not widened and (left_dtype is not None or right_dtype is not None):
            self._require_complex(np.dtype(np.complex64))
        elif widened:
            self._require_complex(np.dtype(np.complex128))
            if (left_side == "double" or right_side == "double") \
                    and (left_side == "c64" or right_side == "c64"):
                self.complex_needs.update(complex_support.mixed_needs())

        if left_dtype is not None and right_dtype is not None:
            if left_side == right_side:
                return f"{kind}_{suffix}({left_code}, {right_code})", kind
            return f"{left_side}_{suffix}_{right_side}({left_code}, {right_code})", kind

        # One complex operand and one real scalar. The scalar's own width decides whether the
        # result widens, and the header names the mixed conversion after the widths it joins.
        complex_is_left:bool = left_dtype is not None
        complex_side:str = left_side if complex_is_left else right_side
        scalar_side:str = right_side if complex_is_left else left_side
        complex_code:str = left_code if complex_is_left else right_code
        scalar_code:str = right_code if complex_is_left else left_code
        mixed_scalar:bool = scalar_side == "double"
        # The real operand is cast to the width the function declares: a literal `1` in
        # `1 + 2j` is an int in C, and the header's `float`/`double` parameter would then be
        # ambiguous (CUDA rejects it with "no suitable constructor exists").
        element:str = "double" if mixed_scalar or kind == "c128" else "float"
        typed_scalar:str = f"(({element})({scalar_code}))"
        # `kind` is the *result* width, so a c64 operand meeting a double scalar still has to
        # be widened to reach the c128 function the header declares.
        typed_complex:str = complex_code
        if mixed_scalar and complex_side == "c64":
            typed_complex = f"c64_widen_c128({complex_code})"
            self.complex_needs.update(complex_support.mixed_needs())

        def scalar_call(operation:str)->Translated:
            """`z op s`, as the header's function: the scalar is the second argument.

            Never `f"{kind}_{operation}_c128"`: the header's `_c128` suffix marks an operand
            of that width (`c64_scale_c128(c64, double)`), and because `kind` here is already
            the widened result, appending it named helpers that do not exist -- every
            complex-op-float spelling was refused with "c128_scale_c128 is undefined".
            """
            return f"{kind}_{operation}({typed_complex}, {typed_scalar})", kind

        def reversed_call(operation:str)->Translated:
            """`s op z`, whose scalar is the *first* argument the header declares."""
            return f"{kind}_{operation}({typed_scalar}, {typed_complex})", kind

        if operator == "Mult":
            # `s * z` and `z * s` are the same product: the scalar is the second argument.
            return scalar_call("scale")
        if operator == "Div":
            if complex_is_left:
                return scalar_call("scale_div")
            return reversed_call("rdiv")
        if operator in ("Add", "Sub"):
            # `z + s` and `z - s` add or subtract the real part; `s + z` is the *same* sum as
            # `z + s` (addition commutes, so it reuses the same function), while `s - z` has to
            # be its own operation -- the earlier spelling used `rsub` for both, which turned
            # the commonest constant in a body (`3 + 4j`, an `Add` over a real 3) into `3 - 4j`.
            if operator == "Add":
                if not complex_is_left:
                    scalar_code, complex_code = complex_code, scalar_code
                return scalar_call("add_real")
            if complex_is_left:
                return scalar_call("sub_real")
            return reversed_call("rsub")
        if operator == "Pow":
            if complex_is_left:
                # `z ** 2` (the Mandelbrot iteration shape) is the direct product: the
                # general power goes through exp(2 * log(z)), whose round trip loses the
                # real part entirely at c64 (`(4-4j) ** 2` answered -1.4e-06 - 32j where
                # numpy answers -32j). The real path special-cases a constant 2 into
                # `x * x` (see _visit_binop); the complex path owes the same case. An
                # integer exponent of 2 only -- python's `z ** 2.0` is a float power.
                if isinstance(node.right, ast.Constant) \
                        and isinstance(node.right.value, int) \
                        and node.right.value == 2 and not isinstance(node.right.value, bool):
                    return f"{kind}_mul({complex_code}, {complex_code})", kind
                return scalar_call("pow_real")
            return reversed_call("rpow")
        return None

    def _emit_complex_constructor(self,
                                  node:ast.Call)->Translated:
        """`complex(re, im)` -- python's own constructor for a complex value.

        Python offers three spellings and they all mean the same value, so all three are
        accepted: `complex(re, im)`, `complex(z)` (a copy, or a real widened), and
        `complex("1+2j")` is refused because a string has no device value -- the text would
        have to be parsed on the host, and a caller who knows the value can write it as a
        literal.

        The result carries the width the `float` mapping chooses for a python `complex`, which
        is the same rule `1 + 2j` follows: `c64` under the 32-bit default, `c128` when the
        kernel asks for 64-bit reals. A caller who wants the other width for one value writes
        `np.complex64(complex(re, im))` (or `np.complex128(...)`), and that cast converts it.
        """
        argv:List[ast.expr] = list(node.args)
        if len(argv) not in (1, 2):
            self._error(
                f"complex() takes 1 or 2 arguments (complex(re, im) or complex(z)), got"
                f" {len(argv)}", node)
        for argument in argv:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                # `complex("3+4j")` parses the text on the host. A device has no parser, so the
                # text would have to be translated into the value it spells -- and a caller who
                # knows the value can write it as a literal or as `complex(re, im)`.
                self._error(
                    f'complex("{argument.value}") is a string parse, which device code cannot'
                    f" do: write the value it spells instead (e.g. 3 + 4j), or pass the real"
                    f" and imaginary parts as scalars", argument)
        args:List[Translated] = [self._visit_expression(argument) for argument in argv]
        for code, t in args:
            if is_ptr(t) or is_image(t) or is_vec(t):
                self._error(
                    f"complex() components must be scalars, got {t}", node)
        kind:str = self._python_complex_kind()
        element:str = "double" if kind == "c128" else "float"
        dtype:np.dtype = np.dtype(np.complex128 if kind == "c128" else np.complex64)
        if len(args) == 1:
            code, t = args[0]
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is not None:
                # `complex(z)` is z at the width a python complex travels as here: python
                # answers its own (double) type, and the mapping stands in for that type, so a
                # complex64 operand beside `float=` on a double is widened and a complex128 one
                # under the 32-bit default is narrowed -- exactly as a python float argument is.
                self._require_complex(dtype)
                value_kind:str = self._complex_side(t, complex_dtype)
                return self._convert_complex(code, value_kind, kind), kind
            if not is_float(t) and not is_int(t):
                self._error(
                    f"complex({t}) has no device meaning: pass a real or a complex value,"
                    f" or use complex(re, im)", node)
            # A real value, widened: the imaginary part is zero.
            self._require_complex(dtype)
            return f"{kind}_from_real(({element})({code}))", kind

        (real_code, real_t), (imag_code, imag_t) = args
        for code, t in args:
            if not is_float(t) and not is_int(t):
                self._error(
                    f"complex(re, im) needs real components, got {t}", node)
        # A numpy scalar part keeps its own width, as it does everywhere else
        # (`np.float64` arguments are narrowed deliberately, but a value that *travels*
        # as a double carries its width into the result): numpy's own
        # `complex(np.float64(1.0), np.float64(2.0))` answers a complex128. A *python*
        # float part has no width of its own and follows the mapping, which is the rule
        # `1 + 2j` already uses -- the mapping is what chose `kind` above.
        parts_are_double:bool = real_t == "double" or imag_t == "double"
        if parts_are_double:
            kind = "c128"
        element = "double" if kind == "c128" else "float"
        dtype = np.dtype(np.complex128 if kind == "c128" else np.complex64)
        self._require_complex(dtype)
        return (f"{kind}_make(({element})({real_code}), ({element})({imag_code}))", kind)

    def _emit_complex_compare(self, operator:str, left:str, lt:object, right:str,
                              rt:object) -> Optional[Translated]:
        """`z == w` / `z != w` as the header's equality test.

        numpy compares the two parts, so an equality is a real comparison of both. A real
        operand is widened (`c64 == np.float32(2)` is the comparison with `2 + 0i`), which is
        the same widening the arithmetic uses; anything else has no device meaning.
        """
        left_code, left_dtype = self._complex_operand(left, lt)
        right_code, right_dtype = self._complex_operand(right, rt)
        if left_dtype is None and right_dtype is None:
            return None
        left_side:str = self._complex_side(lt, left_dtype)
        right_side:str = self._complex_side(rt, right_dtype)
        widened:bool = "c128" in (left_side, right_side) or "double" in (left_side, right_side)
        kind:str = "c128" if widened else "c64"
        if left_dtype is not None and right_dtype is not None:
            if left_side == right_side:
                self._require_complex(left_dtype)
                comparison:str = f"{kind}_equal({left_code}, {right_code})"
            else:
                self.complex_needs.update(complex_support.mixed_needs())
                comparison = (f"{left_side}_equal_{right_side}"
                              f"({left_code}, {right_code})")
        else:
            complex_is_left:bool = left_dtype is not None
            scalar_side:str = right_side if complex_is_left else left_side
            complex_code:str = left_code if complex_is_left else right_code
            scalar_code:str = right_code if complex_is_left else left_code
            if scalar_side == "double" and not widened:
                self.complex_needs.update(complex_support.mixed_needs())
                comparison:str = f"{kind}_equal_c128({complex_code}, {scalar_code})"
            else:
                # The scalar is widened by building the complex value it means, which the
                # header spells as its own conversion.
                self._require_complex(np.dtype(np.complex128 if widened else np.complex64))
                widener:str = "c128_from_real" if widened else "c64_from_real"
                scalar_element:str = "double" if widened else "float"
                comparison = (f"{kind}_equal({complex_code},"
                              f" {widener}(({scalar_element})({scalar_code})))")
        if operator == "NotEq":
            return f"(!{comparison})", "bool"
        return f"({comparison})", "bool"

    def _emit_complex_method(self, attr:str, base_code:str, base_t:object,
                             dtype:np.dtype, node:ast.Call)->Optional[Translated]:
        """One of a complex value's own methods, as the header's function.

        numpy's complex scalars define `conjugate()` (and its alias `conj()`), plus
        `astype()` for a width change. Nothing else is a method -- the rest of the complex
        vocabulary is the module functions -- so a miss returns None and the caller names
        the methods that exist.
        """
        kind:str = self._complex_side(base_t, dtype)
        if not node.args and attr in ("conjugate", "conj"):
            self._require_complex(dtype)
            return f"{kind}_conjugate({base_code})", kind
        return None

    def _magic_receiver_type(self, t:DeviceType)->DeviceType:
        """The type a magic method's receiver parameter takes.

        A structure argument travels as a pointer, so a class method's receiver is a
        pointer too; a vector (the package's own types define no magic method) and a scalar
        keep their value type.
        """
        return t

    def _property_descriptor(self, classes:object, name:str,
                             code:str)->Optional[object]:
        """The property descriptor for `name`, from the class that actually owns it.

        The launch knows the exact class of a structure argument (see `_class_of`), and
        that answer wins: two classes with one layout share a dtype, so asking the dtype
        answered with the class that registered it first -- and `b.total` then compiled
        `a`'s getter and returned its value.
        """
        exact:Optional[type] = self._class_of(code, None)
        if exact is not None:
            descriptor:Optional[object] = _property_of(exact, name)
            if descriptor is not None:
                return descriptor
        return _property_carrier(classes, name)

    def _emit_property_write(self, cls:Optional[type], name:str, base:Translated,
                             value:ast.expr, node:ast.AST)->bool:
        """`box.total = v` as a device call to the property's setter, or False.

        A setter is an ordinary method taking the new value, so the assignment becomes a
        call statement of its own. False when the class defines no such property, which
        leaves the field store to answer.
        """
        descriptor:Optional[object] = self._property_descriptor(cls, name, base[0])
        if descriptor is None:
            return False
        setter:Optional[Callable] = _property_accessor(descriptor, writer=True)
        if setter is None:
            self._error(
                f"property '{name}' has no setter: give the class a '@{name}.setter'",
                node)
        value_code, value_t = self._visit_expression(value)
        call:Translated = self._emit_device_call(
            name, setter, [base, (value_code, value_t)], node,
            self_param=_receiver_param_of(setter), receiver_as_argument=True)
        self._emit(f"{call[0]};")
        # The store happens inside the setter, which the static analysis cannot see: it
        # reads `h.doubled = v` but the generated statement is a call, so the parameter
        # would stay read-only and the host would drop the write. `atomic_targets` is the
        # force-writable channel a member store already uses.
        if isinstance(value, ast.expr) and isinstance(base[0], str):
            written_name:str = base[0].replace("&", "").replace("(", "").replace(")", "")
            if written_name.isidentifier():
                self.atomic_targets.add(written_name)
                self.member_writes.add(written_name)
        return True

    def _emit_property_read(self, cls:Optional[type], name:str, base:Translated,
                            node:ast.AST)->Optional[Translated]:
        """`box.total` as a device call to the property's getter, or None.

        A property is a member the class computes, so the read is a method call with the
        receiver as its argument -- the same shape a plain method call takes. None when the
        class defines no such property, which leaves the field lookup to answer.
        """
        descriptor:Optional[object] = self._property_descriptor(cls, name, base[0])
        if descriptor is None:
            return None
        getter:Optional[Callable] = _property_accessor(descriptor, writer=False)
        if getter is None:
            self._error(f"property '{name}' has no getter", node)
        return self._emit_device_call(
            name, getter, [base], node,
            self_param=_receiver_param_of(getter), receiver_as_argument=True)

    def _emit_magic_conversion(self, magic:str, code:str, t:object,
                               node:ast.AST)->Optional[Translated]:
        """`int(box)` / `abs(box)` and friends as a device call to the class's method.

        A class that defines the conversion gives the builtin a meaning the device keeps:
        casting an aggregate is not what the caller wrote. None when the operand carries
        no such method, which leaves the ordinary scalar path to answer.
        """
        owner:Optional[Tuple[type, Any]] = self._magic_owner(t, magic)
        if owner is None:
            return None
        _cls, method = owner
        return self._emit_device_call(
            magic, method, [(code, t)], node,
            self_param=_receiver_param_of(method), receiver_as_argument=True)

    def _emit_magic_binop(self, magic:str, left:str, lt:object, right:str, rt:object,
                          node:ast.BinOp)->Optional[Translated]:
        """`a op b` as a device call to the class's magic method, or None.

        Python resolves `a + b` as `a.__add__(b)`, and then -- if the left class does not
        define it -- as `b.__radd__(a)`. Both spellings are kept: the reflected form is how a
        scalar on the left reaches a class that only knows how to multiply itself by one
        (`2.0 * v` calls `v.__rmul__(2.0)`), which is a shape a kernel body does write.
        """
        owner:Optional[Tuple[type, Any]] = self._magic_owner(lt, magic)
        if owner is not None:
            _cls, method = owner
            return self._emit_device_call(
                magic, method, [(left, lt), (right, rt)], node,
                self_param=_receiver_param_of(method), receiver_as_argument=True)

        reflected:Optional[str] = MAGIC_REFLECTED.get(_operator_name(magic))
        if reflected is not None:
            owner = self._magic_owner(rt, reflected)
            if owner is not None:
                _cls, method = owner
                # The reflected method takes the right operand's receiver first, so the
                # operands are handed over in the order the call has them.
                return self._emit_device_call(
                    reflected, method, [(right, rt), (left, lt)], node,
                    self_param=_receiver_param_of(method), receiver_as_argument=True)
        return None

    def _emit_magic_compare(self, magic:str, left:str, lt:object, right:str, rt:object,
                            op:ast.cmpop)->Optional[Translated]:
        """`a == b` as a device call to the class's comparison, or None."""
        # A containment test asks the RIGHT operand: `x in box` is `box.__contains__(x)`.
        # Its result is a truth value, so `not in` negates the same call. A chained form
        # (`x in a in b`) is refused above, where the operator is read.
        if magic == "__contains__":
            owner:Optional[Tuple[type, Any]] = self._magic_owner(rt, magic)
            if owner is not None:
                _cls, method = owner
                call:Translated = self._emit_device_call(
                    magic, method, [(right, rt), (left, lt)], op,
                    self_param=_receiver_param_of(method), receiver_as_argument=True)
                if op.__class__.__name__ == "NotIn":
                    return f"(!{call[0]})", "bool"
                return call
            return None

        owner:Optional[Tuple[type, Any]] = self._magic_owner(lt, magic)
        if owner is not None:
            _cls, method = owner
            return self._emit_device_call(
                magic, method, [(left, lt), (right, rt)], op,
                self_param=_receiver_param_of(method), receiver_as_argument=True)

        # Python reads `a < b` as `b.__gt__(a)` when the left class has no `__lt__`.
        reflected:Optional[str] = MAGIC_COMPARE_REFLECTED.get(_operator_name(magic))
        if reflected is not None:
            owner = self._magic_owner(rt, reflected)
            if owner is not None:
                _cls, method = owner
                return self._emit_device_call(
                    reflected, method, [(right, rt), (left, lt)], op,
                    self_param=_receiver_param_of(method), receiver_as_argument=True)
        return None

    def _class_of(self, code:str, t:object)->Optional[type]:
        """The class a value was built from, when this launch knows it exactly.

        One dtype can stand for several classes -- two classes with one layout share it --
        so a dtype alone cannot say whose method `p.total` is. The call does know: the
        caller handed over an instance, and its type travels in `struct_arg_classes` under
        the parameter's name. That answer wins over the dtype, which only has the class
        that registered the layout first.
        """
        name:str = code.strip().lstrip("&").strip("()").strip()
        exact:Optional[type] = self.struct_arg_classes.get(name)
        return exact

    def _magic_owner(self, t:object, magic:str,
                     code:str = "")->Optional[Tuple[type, Any]]:
        """The class whose magic method an operand's operator should call, and that method.

        A python class that defines `__add__` gives `a + b` a meaning of its own, and a
        device translation has to keep it: the operator becomes a device call to that
        method, with the operands as its arguments, exactly as `ray.at(t)` becomes a call to
        `Ray_at`. Returns None when the operand carries no such method, which leaves the
        ordinary operator path (a scalar or a vector operation) to answer.
        """
        from . import structs as _structs_module  # noqa: PLC0415

        base:str = t[1] if isinstance(t, tuple) and is_ptr(t) else t
        if not isinstance(base, str):
            return None
        dtype:object = _structs_module.dtype_for(base)
        if dtype is None:
            return None
        # The launch may know the exact class of this value (see `_class_of`); that answer
        # is the one that counts, because two classes sharing a layout would otherwise be
        # answered by whichever registered first.
        exact:Optional[type] = self._class_of(code, t) if code else None
        if exact is not None:
            method:object = _magic_method_of(exact, magic)
            if method is not None:
                return exact, method
        # Otherwise ask every class the layout carries -- a shared layout means the first
        # registered one is not necessarily the class whose method this is.
        for cls in _structs_module.declared_classes(dtype):
            method:object = _magic_method_of(cls, magic)
            if method is not None:
                return cls, method
        return None

    def _is_structure_operand(self, t:object)->bool:
        """Whether a translated type is a structure value (a quaternion is one too).

        A structure argument travels as a pointer, so the pointer type names one as well.
        """
        if isinstance(t, tuple) and is_ptr(t):
            return structs_dtype_for(t[1]) is not None
        if not isinstance(t, str) or is_vec(t):
            return False
        return structs_dtype_for(t) is not None

    def _structure_operator_message(self, operator:str, lt:object, rt:object,
                                    node:ast.AST)->str:
        """Why a structure cannot take part in an operator, and what to write instead."""
        from . import structs  # noqa: PLC0415

        operand:object = lt if self._is_structure_operand(lt) else rt
        dtype:object = structs.dtype_for(operand)
        members:str = ", ".join(dtype.names or ()) if dtype is not None else "?"
        symbol:str = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
                      "Pow": "**", "FloorDiv": "//", "Lt": "<", "LtE": "<=", "Gt": ">",
                      "GtE": ">=", "Eq": "==", "NotEq": "!="}.get(operator, operator)
        return (
            f"a structure cannot be used with the '{symbol}' operator: device code has no "
            f"arithmetic for a structure value, and the operands here are {lt} and {rt}. "
            f"Operate on the members instead (`s.{members.split(', ')[0]} {symbol} ...`), or "
            f"move the computation into a host method before the kernel -- which is how a "
            f"quaternion rotation belongs, unless a dedicated device function for it exists")

    def _visit_binop(self, node:ast.BinOp)->Translated:
        left, lt = self._visit_expression(node.left)
        right, rt = self._visit_expression(node.right)
        name:str = node.op.__class__.__name__

        # A quaternion is a structure too, but its arithmetic *does* have a device meaning
        # (pygf defines it), so it is handled by generated device functions before the
        # general structure refusal below.
        quat_result:Optional[Translated] = self._emit_quaternion_binop(
            name, left, lt, right, rt, node)
        if quat_result is not None:
            return quat_result

        # A complex value is a structure too, and its arithmetic likewise has a device
        # meaning -- numpy's own, which the header implements -- so it is answered here,
        # before the general structure refusal below.
        complex_result:Optional[Translated] = self._emit_complex_binop(
            name, left, lt, right, rt, node)
        if complex_result is not None:
            return complex_result

        # A class that defines the operator's magic method gives it a meaning, and the
        # translation keeps that meaning: `a + b` becomes `Class___add__(&a, &b)`, the same
        # shape `ray.at(t)` takes. Only a method written in the class body counts, so an
        # inherited identity comparison cannot silently answer for a structure.
        magic:Optional[str] = MAGIC_METHODS.get(name)
        if magic is not None:
            call:Optional[Translated] = self._emit_magic_binop(
                magic, left, lt, right, rt, node)
            if call is not None:
                return call

        # A structure value has no arithmetic: none of `+ - * / % ** < <= > >= == !=` has a
        # device meaning for an aggregate, and emitting one produced either a C++ operator
        # error or -- worse for `Mult` -- a per-component scalar multiply that silently
        # computed something the language never defined. Every other structure is refused
        # here, before the vector dispatch, so the message can say what to do instead.
        if self._is_structure_operand(lt) or self._is_structure_operand(rt):
            self._error(self._structure_operator_message(name, lt, rt, node), node)

        if is_vec(lt) or is_vec(rt):
            symbol:str = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
                          "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
                          "Eq": "==", "NotEq": "!="}.get(name)
            if symbol is None:
                self._error(f"operator {name} is not supported on vector types; operate per-component", node)

            if is_vec(lt) and is_vec(rt):
                if lt != rt:
                    self._error(f"vector operands must have the same type, got {lt} and {rt}", node)
                if self._opencl_syntax:
                    result_t:object = (lt if name in ("Add", "Sub", "Mult", "Div") else self._get_int_vec_type(lt))
                    return f"({left} {symbol} {right})", result_t
                if name in ("Add", "Sub", "Mult", "Div"):
                    return self._vec_componentwise(symbol, left, lt, right=right), lt
                return self._vec_componentwise_compare(symbol, left, lt, right), self._get_int_vec_type(lt)

            # vector with scalar: broadcast per component
            vec, vec_side_left = (lt, True) if is_vec(lt) else (rt, False)
            if self._opencl_syntax:
                code:str = f"({left} {symbol} {right})" if vec_side_left else f"({right} {symbol} {left})"
                return code, vec
            if vec_side_left:
                return self._vec_componentwise(symbol, left, vec, scalar=right), vec
            return self._vec_componentwise(symbol, right, vec, scalar=left), vec

        if name == "Div":
            t:object = promote(lt, rt) if not (is_int(lt) and is_int(rt)) else "float"
            if is_int(lt) and is_int(rt):
                return f"((({t})({left})) / (({t})({right})))", t
            return f"({left} / {right})", t

        if name == "FloorDiv":
            if is_int(lt) and is_int(rt):
                # C's `/` truncates and python's `//` floors, so the two disagree for a
                # negative pair. The helper applies the correction (see
                # device_builtins.PYTHON_FLOOR_SOURCES); for an unsigned width it is
                # empty and the plain operator already agrees.
                return self._emit_floored(("floor_div", "floor_mod"), "floor_div",
                                          left, promote(lt, rt), right, rt, node)
            t:object = promote(lt, rt)
            return f"(({t})floor(({t})({left}) / ({t})({right})))", t

        if name == "Pow":
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and node.right.value == 2:
                return f"({left} * {left})", lt
            t:object = promote(lt, rt)
            if is_int(t):
                t = "float"
            return f"(({t})pow(({t})({left}), ({t})({right})))", t

        if name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor"):
            if not (is_int(lt) and is_int(rt)):
                self._error(f"operator {name} requires integer operands", node)
            symbol:str = {"LShift": "<<", "RShift": ">>", "BitOr": "|", "BitAnd": "&", "BitXor": "^"}[name]
            return f"({left} {symbol} {right})", promote(lt, rt)

        if name == "Mod":
            t:object = promote(lt, rt)
            if is_int(t):
                # Same discrepancy as `//`: C's `%` takes the dividend's sign and python's
                # takes the divisor's.
                return self._emit_floored(("floor_div", "floor_mod"), "floor_mod",
                                          left, t, right, rt, node)
            return f"fmod(({t})({left}), ({t})({right}))", t

        symbol:str = {"Add": "+", "Sub": "-", "Mult": "*"}.get(name)
        if symbol is None:
            self._error(f"unsupported operator: {name}", node)
        result_t:object = promote(lt, rt)
        if is_int(lt) and is_int(rt):
            # Python's int is arbitrary precision and the device's is the mapped width, so a
            # product of two literal ints used to wrap silently: `5000 * 1000000` answered
            # 705032704 with a compiler warning nobody piped up. When both sides are literal
            # arithmetic the host can compute the exact value and run the same width check a
            # scalar *argument* gets (`_check_integer_fits`); a literal that does not fit is
            # refused with the numpy spelling that carries it, not wrapped.
            folded:Optional[int] = self._folded_int_value(node)
            if folded is not None:
                _check_integer_fits(folded, result_t if isinstance(result_t, str) else "int")
        return f"({left} {symbol} {right})", result_t

    def _folded_int_value(self, node:ast.BinOp)->Optional[int]:
        """The exact value of a literal int BinOp, or None when it is not foldable.

        A "literal" here is an expression built only from int constants, the arithmetic
        operators python folds with (`+ - * / // % **`), and unary signs --
        `5000000 * 1000000`, `2 ** 40`, `-7 // 2`. A name (a parameter or a local) is *not*
        folded: its value is a launch-time fact, and the mapping `int=np.int64` is the
        escape hatch for a runtime product that needs the wider device type.
        """
        operations:Dict[type, object] = {
            ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b,
        }

        def evaluate(expr:ast.expr)->Optional[int]:
            if isinstance(expr, ast.Constant) and isinstance(expr.value, int) \
                    and not isinstance(expr.value, bool):
                return int(expr.value)
            if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
                inner:Optional[int] = evaluate(expr.operand)
                if inner is None:
                    return None
                return -inner if isinstance(expr.op, ast.USub) else inner
            if isinstance(expr, ast.BinOp) and type(expr.op) in operations:
                left:Optional[int] = evaluate(expr.left)
                right:Optional[int] = evaluate(expr.right)
                if left is None or right is None:
                    return None
                try:
                    return int(operations[type(expr.op)](left, right))
                except (ZeroDivisionError, ValueError, OverflowError):
                    return None
            return None

        return evaluate(node)

    def _visit_unaryop(self, node:ast.UnaryOp)->Translated:
        code, t = self._visit_expression(node.operand)
        name:str = node.op.__class__.__name__
        # A class that defines `__neg__` / `__pos__` / `__invert__` gives the unary operator
        # a meaning; without this, `-a` on a structure emitted `(-a)` and the compiler
        # rejected it as arithmetic on an aggregate.
        magic:Optional[str] = MAGIC_METHODS.get(name)
        if magic is not None:
            owner:Optional[Tuple[type, Any]] = self._magic_owner(t, magic)
            if owner is not None:
                cls, method = owner
                return self._emit_device_call(
                    magic, method, [(code, t)], node,
                    self_param=(method.__code__.co_varnames[0]
                                if getattr(method, "__code__", None) else "self"),
                    receiver_as_argument=True)
        if name == "USub":
            if is_vec(t) and self.backend == "cuda":
                # CUDA has no unary minus on a vector type: the negation is per component.
                return self._vec_componentwise("-", code, t), t
            if self._complex_dtype_of(t) is not None:
                # A complex value is a structure, so `-z` is the header's own negation.
                self._require_complex(self._complex_dtype_of(t))
                kind:str = self._complex_side(t, self._complex_dtype_of(t))
                return f"{kind}_neg({code})", kind
            return f"(-{code})", t
        if name == "UAdd":
            if self._complex_dtype_of(t) is not None:
                # `+z` is z: python answers the value unchanged, and so does a C++ copy.
                return code, self._complex_side(t, self._complex_dtype_of(t))
            return f"(+{code})", t
        if name == "Invert":
            if not is_int(t):
                self._error("~ requires an integer operand", node)
            return f"(~{code})", t
        if name == "Not":
            # The operand is parenthesised, or `not a == b` reaches C as `!a == b` -- a
            # comparison against the negated left operand, which is not what python means
            # (`not (a == b)`), and the OpenCL compiler says so with "logical not is only
            # applied to the left hand side of this comparison".
            return f"(!({code}))", "bool"
        self._error(f"unsupported unary operator: {name}", node)

    def _visit_compare(self, node:ast.Compare)->Translated:
        symbols:Dict[str, str] = {"Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
                                  "Eq": "==", "NotEq": "!=", "In": "in", "NotIn": "not in"}

        left, lt = self._visit_expression(node.left)
        if is_vec(lt):
            # vector comparison (opencl natively returns an int vector; cuda generates per-component), same semantics as genVec._compare_op
            if len(node.ops) != 1:
                self._error("chained comparison is not supported on vector types", node)
            name:str = node.ops[0].__class__.__name__
            if name not in symbols:
                self._error(f"unsupported comparison: {name}", node)
            right, rt = self._visit_expression(node.comparators[0])
            if is_vec(rt):
                if rt != lt:
                    self._error(f"vector operands must have the same type, got {lt} and {rt}", node)
                return self._vec_componentwise_compare(symbols[name], left, lt, right), self._get_int_vec_type(lt)
            self._error("vector comparison requires vector operands on both sides", node)

        parts:List[str] = []
        for op, comparator in zip(node.ops, node.comparators):
            name:str = op.__class__.__name__
            if name not in symbols:
                self._error(f"unsupported comparison: {name}", op)
            right, rt = self._visit_expression(comparator)
            # A class that defines `__eq__` / `__lt__` and its siblings gives the comparison
            # a meaning, and the translation keeps it as a device call.
            magic:Optional[str] = MAGIC_METHODS.get(name)
            if magic is not None:
                call:Optional[Translated] = self._emit_magic_compare(
                    magic, left, lt, right, rt, op)
                if call is not None:
                    # A chained comparison cannot be a single call: `a < b < c` means
                    # `a < b and b < c`, which the loop already joins with `&&`.
                    if len(node.ops) != 1:
                        self._error(
                            "chained comparison is not supported on a class that defines "
                            "its own comparison; write the two comparisons separately", op)
                    return call
            # A complex value compares with `==` and `!=` (numpy compares the two parts), and
            # nothing else: python has no ordering for a complex number, so numpy and the
            # device raise the same TypeError rather than inventing one.
            if self._complex_dtype_of(lt) is not None or self._complex_dtype_of(rt) is not None:
                if name not in ("Eq", "NotEq"):
                    self._error(
                        f"complex numbers have no '{symbols[name]}' comparison: python orders"
                        f" no complex value, so only == and != are defined. Compare the"
                        f" magnitude (abs) or a component (.real / .imag) instead", op)
                call:Optional[Translated] = self._emit_complex_compare(name, left, lt, right, rt)
                if call is None:
                    self._error(
                        f"cannot compare {lt} with {rt}: a complex value compares with a"
                        f" complex number, or with a real one it is widened to", op)
                parts.append(call[0])
                left = right
                continue
            # A structure value has no comparison either: C rejects `s == s` outright, so it
            # is refused here with the same explanation the arithmetic gets.
            if self._is_structure_operand(lt) or self._is_structure_operand(rt):
                self._error(self._structure_operator_message(name, lt, rt, node), op)
            parts.append(f"{left} {symbols[name]} {right}")
            left = right
        # A comparison is not wrapped here: an `if` adds the one pair C's syntax needs, and a
        # second pair made it emit `if (((a == b)))` -- the middle one, which the OpenCL
        # compiler reports as "equality comparison with extraneous parentheses" (and points
        # at `=` as a "fix", which would silently change the meaning). An operand that needs
        # grouping of its own -- a nested comparison, an arithmetic expression -- already
        # carries parentheses from its own visitor. A chained comparison joins its parts with
        # `&&`, and the `&&` expression gets the pair that groups it.
        if len(parts) > 1:
            return f"({' && '.join(parts)})", "bool"
        return parts[0], "bool"

    def _visit_subscript(self, node:ast.Subscript)->Translated:
        value:ast.expr = node.value
        # `index[0]` on an ndenumerate index held as one name: python's index is a tuple
        # of coordinates, and a constant ordinal reads one of them. The head already
        # emitted the decomposition (see the ndenumerate branch of
        # _translate_magic_loop), so the subscript is that coordinate's own name.
        if isinstance(value, ast.Name) \
                and value.id in self._ndenumerate_index_names \
                and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, int):
            coordinates:List[str] = self._ndenumerate_index_names[value.id]
            ordinal:int = int(node.slice.value)
            if 0 <= ordinal < len(coordinates):
                return coordinates[ordinal], "int"
            self._error(
                f"the ndenumerate index has {len(coordinates)} coordinate(s); "
                f"[{ordinal}] is outside it", node)
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            array_name:str = self._array_name(value.value)
            t:DeviceType = self._lookup(array_name)
            # A private array created in the body has a constant shape, so `.shape[d]` is a
            # compile-time number just like for a parameter (see _shape_var).
            private:bool = (isinstance(t, tuple) and len(t) == 4 and t[0] == "bodyarr")
            if not is_ptr(t) and not private:
                self._error("'.shape' is only available on array parameters", node)
            index:ast.expr = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                return self._shape_var(array_name, index.value), "int"
            self._error("only `<array>.shape[<constant>]` is supported", node)

        # A string subscript on a structure value reads a member: numpy's spelling for a
        # record field, emitted as C member access. It covers nested structures
        # (`s["inner"]["a"]`) because the member itself is a structure-typed value.
        member:Optional[Translated] = self._struct_member_access(node)
        if member is not None:
            return member

        # `box[i]` on a class that defines `__getitem__` is that method, exactly as
        # `box.attr` is `box.__getattribute__`. The member read above already covered the
        # string spelling (`s["field"]`), so only a computed index reaches here.
        if (not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str))
                and isinstance(node.value, ast.Name)):
            # Only a plain name is a class instance this class's `__getitem__` can answer
            # for. A shared buffer (`bins[blockIdx]`) and an array are indexed by the paths
            # below, and visiting their base here would raise the shared-buffer refusal
            # before that path could handle the outer subscript.
            binding:DeviceType = self._lookup(node.value.id)
            # A structure travels as a pointer too, so `is_ptr` alone would call it an
            # array and skip the `__getitem__` its class may define.
            is_struct_ptr:bool = (isinstance(binding, tuple) and is_ptr(binding)
                                  and structs_dtype_for(binding[1]) is not None)
            is_array:bool = (is_ptr(binding) and not is_struct_ptr) or (
                isinstance(binding, tuple) and len(binding) == 4 and binding[0] == "bodyarr")
            is_shared:bool = (isinstance(binding, tuple) and len(binding) == 5
                              and binding[0] == "shared")
            if not is_array and not is_shared:
                base_code, base_t = self._visit_expression(node.value)
                owner:Optional[Tuple[type, Any]] = self._magic_owner(base_t, "__getitem__")
                if owner is not None:
                    _cls, method = owner
                    index_code, index_t = self._visit_expression(node.slice)
                    return self._emit_device_call(
                        "__getitem__", method,
                        [(base_code, base_t), (index_code, index_t)],
                        node, self_param=_receiver_param_of(method),
                        receiver_as_argument=True)

        # `buf[k]` on a `zeros_per_group(...)` value: the index selects a work group, and only
        # caller's own group is reachable from inside a work item, so `k` must be the block
        # id. The result is that group's slice, which is then indexed normally.
        shared:Optional[Translated] = self._visit_shared_slice(node)
        if shared is not None:
            return shared

        # A slice used directly as an index (`a[:][i]`, `a[2:][j]`) is the same pointer at
        # an offset: emit the offset arithmetically so the expression needs no local. This
        # has to come *before* the generic visit below, which would try to evaluate the
        # slice as an expression of its own (and there is no Slice value on the device).
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
            return self._visit_sliced_subscript(node)

        # A zeros_per_group() slice remembers the shape it was declared with, so a
        # multidimensional subscript can be flattened exactly like an array parameter's.
        readonly_code, readonly_t = self._visit_expression(value)
        if isinstance(readonly_t, tuple) and readonly_t and readonly_t[0] == SHARED_SLICE:
            element_code, element_t = self._visit_shared_index(readonly_code, readonly_t, node)
            return self._scalar_position(element_code, element_t, node)

        base_code, base_t = self._visit_array_target(value)
        if is_image(base_t):
            return self._emit_image_read(base_code, base_t, node)
        index_code, _ = self._visit_subscript_index(node)
        return f"{base_code}[{index_code}]", base_t

    def _visit_sliced_subscript(self, node:ast.Subscript)->Translated:
        """`a[lo:hi][i]`: the element of the slice, as `array[lo + i]` on the buffer.

        A slice is a pointer at an offset, and indexing it is indexing that offset. Folding
        the two into one subscript is what lets the spelling appear in an expression without
        a local to hold the view. A slice with a step is refused by `_emit_array_slice`'s
        rules, and a nested slice (`a[1:][1:]`) is refused here rather than guessed at.
        """
        sliced:ast.Subscript = cast(ast.Subscript, node.value)
        base:ast.expr = sliced.value
        slice_node:ast.Slice = cast(ast.Slice, sliced.slice)
        if isinstance(base, ast.Subscript) and isinstance(base.slice, ast.Slice):
            self._error("a slice of a slice is not supported: take one slice into a local "
                        "and index it", node)
        base_code, base_t = self._visit_expression(base)
        if not is_ptr(base_t):
            self._error("only an array parameter can be sliced", node)
        step:Optional[ast.expr] = slice_node.step
        if step is not None and _constant_int_of(step) != 1:
            if _constant_int_of(step) == -1:
                self._error(
                    "a reversed slice (`a[::-1]`) is not a device pointer: a work item "
                    "cannot walk backwards through a buffer with one, so read the elements "
                    "with an explicit index instead (`a[n - 1 - i]`)", step)
            self._error("a slice step must be the constant 1", step)
        start:str = "0" if slice_node.lower is None else self._visit_expression(slice_node.lower)[0]
        index_code, index_t = self._visit_expression(node.slice)
        if not is_int(index_t):
            self._error("array index must be an integer", node)
        # `_visit_expression` answered the *pointer* type (a subscript on a name is a
        # pointer), so the element type is what the caller expects back.
        return f"{base_code}[({start}) + ({index_code})]", (cast(tuple, base_t))[1]

    def _visit_shared_index(self, code:str, t:object, node:ast.Subscript)->Translated:
        """Index a shared slice: `buf[blockIdx][i]`, `[i, j]`, and `[i][j]` all flatten.

        A tuple subscript consumes several axes at once; a single subscript consumes one and
        hands back a slice with the remaining axes, so a nested spelling reaches the same
        element (`buf[blockIdx][row][col]` is `buf[row * 4 + col]`).
        """
        ctype:object = t[1]
        shape:Tuple[int, ...] = tuple(t[2])
        offset_base:int = t[3] if len(t) > 3 else 0
        index:ast.expr = node.slice
        if isinstance(index, ast.Tuple):
            elements:List[ast.expr] = list(index.elts)
        else:
            elements = [index]
        if len(elements) > len(shape):
            self._error(
                f"this zeros_per_group() buffer has {len(shape)} dimension(s) "
                f"({', '.join(str(e) for e in shape)}), so it cannot be indexed with "
                f"{len(elements)}", index)
        terms:List[str] = []
        for position, element in enumerate(elements):
            element_code, element_t = self._visit_expression(element)
            if not is_int(element_t):
                self._error("an index into a zeros_per_group() buffer must be an integer",
                            element)
            # A negative index counts from that axis's extent, exactly as it does on an
            # array parameter. Left raw it addressed the shared array's own header.
            term:str = self._python_index(element_code, element,
                                          literal_extent=shape[position])
            for extent in shape[position + 1:len(shape)]:
                term = f"({term}) * {extent}"
            terms.append(term)
        flat:str = " + ".join(terms) if terms else "0"
        remaining:Tuple[int, ...] = shape[len(elements):]
        if not remaining:
            if offset_base:
                return f"{code}[({offset_base}) + ({flat})]", ctype
            return f"{code}[{flat}]", ctype
        # Partly indexed: the caller's code keeps the running offset and the axes left. Such a
        # value is a *slice*, not an element, so it can only be indexed further -- writing to
        # it would target the whole buffer. `_scalar_position` refuses that where a scalar is
        # required.
        combined:str = f"({offset_base}) + ({flat})" if offset_base else flat
        return f"{code}", (SHARED_SLICE, ctype, remaining, combined)

    def _scalar_position(self, code:str, t:object, node:ast.AST)->Translated:
        """A shared slice's element, refusing a partly indexed one.

        `buf[blockIdx]` with `(2, 2)` axes and one index is a slice of a row; it has no
        device value of its own, so using it where a scalar is needed is an error rather than
        a buffer with no subscript.
        """
        if isinstance(t, tuple) and t and t[0] == SHARED_SLICE:
            self._error(
                f"this shared buffer has {len(t[2])} dimension(s) left, so it is a slice rather "
                f"than an element: index every axis, as `buf[blockIdx][i, j]`", node)
        return code, t

    def _visit_shared_slice(self, node:ast.Subscript)->Optional[Translated]:
        """`buf[blockIdx]` on a shared buffer: **this** group's storage, and nothing more.

        The subscript picks *which* group's slice is meant, which is a real choice on the
        host (where the value is a mapping) but not on a device: the hardware already
        gives every group its own `__shared__`/`__local` storage, so the answer is always
        the caller's own slice and **no index arithmetic is emitted at all**. `buf[blockIdx]`
        folds to the buffer itself.

        The subscript is therefore validated rather than computed: it must be a block id
        (`blockIdx`, or `get_group_id(d)`), because any other expression would name a group
        a work item cannot reach. Emitting that as an offset would be worse than refusing --
        it would read the next group's copy, which is precisely the memory a work item must
        never touch.
        """
        if not isinstance(node.value, ast.Name):
            # The inner `buf[block_id(0)]` is already a pointer by the time the outer
            # `[j]` is visited, so that subscript indexes an array and is not a group
            # selection.
            return None
        t:DeviceType = self._lookup(node.value.id)
        if not (isinstance(t, tuple) and len(t) == 5 and t[0] == "shared"):
            return None
        ctype, c_name, shape = t[1], t[2], t[4]
        index:ast.expr = node.slice
        if not self._is_block_id_expression(index):
            self._error(
                "a zeros_per_group() buffer is one slice per work group, and a work item can only "
                "reach its own: index it with the group first, as `buf[blockIdx][j]` "
                "(or `buf[get_group_id(0)][j]`)", index)
        # Both backends give each work group its own copy of the buffer -- OpenCL allocates
        # one `__local` parameter per group, CUDA/HIP a `__shared__` array per block -- so
        # the slice a work item reaches is the whole buffer, and the block id only names
        # which group is asking. No offset is added: doing so would read into the next
        # group's copy, which is exactly the memory a work item must not touch.
        return c_name, (SHARED_SLICE, ctype, shape)

    def _is_block_id_expression(self, node:ast.expr)->bool:
        """Whether an expression names a work group, i.e. evaluates to the block id.

        Both spellings qualify -- `block_id(d)` and `get_group_id(d)` -- and so does a local
        the body assigned from one (`gid = block_id(0)`, then `buf[gid]`), because it is the
        same number by another name.
        """
        if isinstance(node, ast.Name):
            # `blockIdx` is the group vector itself; a local holding one is the same axis
            # under another name.
            return node.id == "blockIdx" or self._lookup(node.id) == "blockid"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            # `blockIdx.x`: one axis of the group vector.
            return node.value.id == "blockIdx" and node.attr in ("x", "y", "z")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return False
        return self._builtin_call_name(node) == "get_group_id"

    def _struct_member_access(self, node:ast.Subscript)->Optional[Translated]:
        """`s["field"]` on a structure value, or `v["x"]` on a vector, or None.

        A string subscript is numpy's spelling for a record field, and it works for a vector
        too (`v["x"]`). Both are member access in C, so both are emitted the same way.
        """
        from . import structs

        index:ast.expr = node.slice
        if not (isinstance(index, ast.Constant) and isinstance(index.value, str)):
            return None
        base_code, base_t = self._visit_expression(node.value)
        field:str = index.value

        if is_vec(base_t):
            component:str = _vec_component_name(field)
            if component not in _vec_field_names(base_t):
                self._error(
                    f"vector type {base_t} has no component '{field}' "
                    f"(available: {''.join(_vec_field_names(base_t))})", node)
            return f"({base_code}).{component}", _C_FIELD_TYPES[(base_t, component)]

        dtype:Optional[object] = structs.dtype_for(base_t) if isinstance(base_t, str) else None
        if dtype is None and (isinstance(base_t, tuple) and is_ptr(base_t)):
            dtype = structs.dtype_for(base_t[1])
        if dtype is None:
            return None
        members:Dict[str, object] = structs.member_types(dtype, self.backend)
        if field not in members:
            self._error(
                f"structure value has no member '{field}' "
                f"(available: {', '.join(members)})", node)
        # The parenthesised member expression is what C expects; a pointer (a structure
        # argument always travels as one) goes through '->', a value through '.'. A record
        # array's element (`g[0]`) is a value, so its own store form is `.`, while the
        # parameter itself is the pointer.
        if isinstance(base_t, tuple) and is_ptr(base_t) and not base_code.startswith("&"):
            if re.search(r"\]\s*$", base_code):
                return f"({base_code}).{field}", members[field]
            return f"({base_code})->{field}", members[field]
        return f"({base_code}).{field}", members[field]

    def _reject_metal_image(self, node:ast.expr, writing:bool)->None:
        """Refuse an image on the Metal backend with the reason, not a compile error.

        MSL *does* have texture objects (`texture2d<float, access::sample>` and
        `access::write`), but the whole image path is Metal's own and is not
        implemented: the parameter type, the sampling call with its sampler state,
        the store call, and the host-side `MTLTexture` binding all differ from the
        OpenCL formulation this backend reuses. Without them the generated source
        carries `__read_only image2d_t`, which MSL has no notion of, so the compiler
        would fail on an undeclared type. The translator is the right place to say
        so, because the message can name the backends that do work.
        """
        if not self._metal_syntax:
            return
        action = "write" if writing else "read"
        self._error(
            "the metal backend does not support images (%s) yet: the MSL texture "
            "formulation (texture2d<float, access::sample|write>, the sampling call "
            "with its sampler state, and the host-side MTLTexture binding) is not "
            "implemented, so the generated source would name an image2d_t that MSL "
            "does not have. Use the cuda, opencl or python backend for image kernels."
            % action, node)

    def _reject_hip_image(self, node:ast.expr, writing:bool)->None:
        """Refuse an image on the HIP backend with the reason, not a compile error.

        HIP reuses CUDA code generation, so an image parameter is emitted as a
        texture/surface pair and the device would fail to compile. Beyond that, the
        device may have no texture hardware at all: on the tested ROCm 7.2 Windows
        build with an integrated Radeon, hipTexObjectCreate reports "Texture not
        supported on the device". Either way a clear message here is better than a
        compiler diagnostic about an undeclared type.
        """
        if self.launch_backend != "hip":
            return
        action = "write" if writing else "read"
        self._error(
            "the hip backend does not support images (%s): HIP reuses CUDA code "
            "generation, which has no texture object type, and hipTexObjectCreate is "
            "rejected with \"operation not supported\" on hardware without texture "
            "support. Use the cuda, opencl or python backend for image kernels."
            % action, node)

    def _emit_image_read(self, image_code:str, image_t:object, node:ast.Subscript)->Translated:
        """Translate ``img[x, y]`` into the sampling call of the current backend.

        Coordinates are texel space, as :mod:`pygpgpu.image` documents: an integer
        coordinate names a texel and samples its centre, while a fractional one
        addresses a position between texels. CUDA expresses both with a float
        coordinate, OpenCL needs int2 for the former and float2 for the latter, so
        the coordinate type decides which form is emitted.
        """
        self._reject_metal_image(node, writing=False)
        self._reject_hip_image(node, writing=False)
        channels:int = image_channels(image_t)
        scalar:str = image_scalar(image_t)

        indices = self._image_indices(node)
        if channels not in (1, 4):
            self._error(
                "sampling an image of %d channels is not supported: use 1 or 4"
                % channels, node)

        if self._opencl_syntax:
            # OpenCL takes a float coordinate for interpolation, and an integer
            # coordinate means the texel boundary rather than its centre -- which a
            # linear sampler blends across, unlike CUDA where an integer subscript
            # lands on the centre. Both integer and fractional subscripts are
            # therefore emitted as float coordinates, with integers moved to +0.5 so
            # every backend samples the same position.
            parts = []
            for code, t in indices:
                if is_int(t):
                    parts.append("((float)(%s) + 0.5f)" % code)
                else:
                    parts.append("((float)(%s))" % code)
            coord_code = "((float%d)(%s))" % (len(parts), ", ".join(parts))
            sampler = self._image_sampler(image_code)
            call = "read_image%s(%s, %s, %s)" % (
                {"float": "f", "int": "i", "uint": "ui"}[scalar],
                image_code, sampler, coord_code)
        else:
            # CUDA: a float coordinate; integer subscripts land on the texel centre.
            parts = []
            for code, t in indices:
                if is_int(t):
                    parts.append("((float)(%s) + 0.5f)" % code)
                else:
                    parts.append("((float)(%s))" % code)
            if scalar == "float":
                call = "tex%dD<float>(%s, %s)" % (
                    len(parts), image_code, ", ".join(parts))
            else:
                self._error(
                    "sampling a %s image is not supported on cuda: only float images "
                    "can be sampled" % scalar, node)

        if channels == 4:
            if self._opencl_syntax:
                return "(%s).xyzw" % call, self._image_vec_type(scalar, 4)
            return call, self._image_vec_type(scalar, 4)
        if self._opencl_syntax:
            # read_imagef returns a vector; single-channel takes .x
            return "(%s).x" % call, scalar
        return call, scalar

    def _image_indices(self, node:ast.Subscript)->List[Translated]:
        """Compile an image subscript into (x, y, z) order.

        A kernel writes ``img[y, x]`` because that mirrors the host array's row-major
        shape, while every sampling call takes (x, y) -- so the compiled coordinates
        are reversed here, exactly like an array subscript's flattening reverses them.
        """
        index:ast.expr = node.slice
        items = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        compiled = [self._visit_expression(item) for item in items]
        return list(reversed(compiled))

    def _image_vec_type(self, scalar:str, channels:int)->str:
        return "%s%d" % (scalar, channels)

    def _image_param_type(self, name:str, image_t:object)->str:
        """The declaration type of an image parameter on the current backend.

        CUDA reads through a texture object and writes through a surface object, so
        the parameter's type follows how the kernel uses it. OpenCL uses one image
        type for both directions, with a qualifier saying which.
        """
        if self._opencl_syntax:
            written:bool = name in self.required_image_surfaces
            qualifier:str = "__write_only" if written else "__read_only"
            return f"{qualifier} image2d_t"

        if name in self.required_image_surfaces:
            return "cudaSurfaceObject_t"
        return "cudaTextureObject_t"

    def _emit_image_write(self, image_code:str, image_t:object,
                          target:ast.Subscript, value_code:str)->None:
        """Translate ``img[x, y] = v`` into the store call of the current backend.

        A write names one texel, so the coordinate is always integral here: a
        fractional subscript is rejected rather than silently rounded, because there
        is no filtering rule for a store.
        """
        self._reject_metal_image(target, writing=True)
        self._reject_hip_image(target, writing=True)
        channels:int = image_channels(image_t)
        scalar:str = image_scalar(image_t)

        # Record the parameter as a write target: the parameter's declared type
        # depends on it, and the host side needs a surface rather than a texture.
        self.required_image_surfaces.add(image_code)

        indices = self._image_indices(target)
        for _, t in indices:
            if not is_int(t):
                self._error(
                    "an image store needs integer coordinates: a fractional "
                    "coordinate has no texel to write", target)

        if self._opencl_syntax:
            coord_code = "((int%d)(%s))" % (len(indices),
                                            ", ".join(c for c, _ in indices))
            if channels == 1:
                packed = "((float4)(%s))" % value_code
            else:
                packed = "((float4)(%s))" % value_code
            self._emit("write_image%s(%s, %s, %s);" % (
                {"float": "f", "int": "i", "uint": "ui"}[scalar],
                image_code, coord_code, packed))
            return

        # CUDA: a surface store. Its first axis is addressed in bytes, so the x
        # coordinate is scaled by the element size.
        coords = [c for c, _ in indices]
        element_bytes = 4
        x_code = "((%s) * %d)" % (coords[0], element_bytes)
        rest = coords[1:]
        if channels == 1:
            self._emit("surf%dDwrite<%s>(%s, %s, %s);" % (
                len(coords), scalar, value_code, image_code,
                ", ".join([x_code] + rest)))
        else:
            self._emit("surf%dDwrite<vec%d>(%s, %s, %s);" % (
                len(coords), channels, value_code, image_code,
                ", ".join([x_code] + rest)))

    def _surface_name(self, image_code:str)->str:
        return "__surface_for_%s" % image_code

    def _image_sampler(self, image_code:str)->str:
        """The sampler argument an OpenCL image read needs.

        The sampler is fixed per image, so it is emitted as a constant at translation
        time rather than passed as another kernel parameter -- that keeps the kernel
        signature identical to the CUDA one.
        """
        self.required_image_samplers.add(image_code)
        return self._sampler_name(image_code)

    def _sampler_name(self, image_code:str)->str:
        return "__sampler_for_%s" % image_code

    def _index_vector_axis(self, node:ast.Attribute, vector_type:str)->Optional[str]:
        """The scalar expression for `blockIdx.x`-style axis, or None when it is not one.

        The index vectors are *values* in the source (`buf[blockIdx][j]`), so their name
        resolves to a built vector. Taking a component off that rebuilt all three terms and
        indexed one: `int i = make_int3((int)threadIdx.x, ...).x`. The axis itself is a single
        launch term, and this answers it directly -- the same number the `get_*` spelling
        gives, which is what the axis meant in the first place.
        """
        if len(node.attr) != 1:
            return None
        base:ast.expr = node.value
        if not isinstance(base, ast.Name) or base.id not in INDEX_VECTOR_BUILTINS:
            return None
        if INDEX_VECTOR_BUILTINS[base.id][0] != vector_type:
            return None
        component:str = _vec_component_name(node.attr)
        axis:int = "xyz".find(component)
        if axis < 0:
            return None
        table:Dict[str, Tuple[str, str, str]] = (
            INDEX_VECTOR_AXES_OPENCL if self._opencl_syntax else INDEX_VECTOR_AXES_CUDA)
        return table[base.id][axis]

    def _visit_attribute(self, node:ast.Attribute)->Translated:
        # self.xxx -> the receiver struct's member, or an extra parameter _self_xxx for an
        # ndarray attribute (the read/write analysis and the shape plumbing need it apart).
        if (self.self_param and isinstance(node.value, ast.Name) and node.value.id == self.self_param):
            if node.attr not in self.self_struct_members and node.attr not in \
                    self.self_attr_types:
                # The receiver may be a structure pointer instead of a field set
                # (`def hit(self, ...)` called with a Scene): its scalar members are
                # fields of that pointer.
                from . import structs as _structs_module  # noqa: PLC0415

                pointer_t:DeviceType = self._lookup(self.self_param)
                if isinstance(pointer_t, tuple) and is_ptr(pointer_t):
                    pointer_dtype:Optional[np.dtype] = _structs_module.dtype_for(
                        pointer_t[1])
                    if pointer_dtype is not None:
                        pointer_members:Dict[str, object] = _structs_module.member_types(
                            pointer_dtype, self.backend)
                        if node.attr in pointer_members:
                            return (f"({self.self_param})->{node.attr}",
                                    pointer_members[node.attr])
            if node.attr in self.self_struct_members:
                from . import structs as _structs_module  # noqa: PLC0415

                member:str = self.self_struct_members[node.attr]
                struct_dtype:Optional[np.dtype] = _structs_module.dtype_for(
                    self.self_struct_type[1])
                members:Dict[str, object] = _structs_module.member_types(
                    struct_dtype, self.backend)
                if node.attr not in members:
                    self._error(
                        f"structure {self.self_struct_type[1]} has no member '.{node.attr}' "
                        f"(available: {', '.join(members)})", node)
                # The receiver struct travels as a pointer, so the member read goes
                # through '->'.
                return (f"({self.self_param})->{member}", members[member])
            if node.attr not in self.self_attr_types:
                # A receiver that arrived as this function's own first argument
                # (`def at(self, t)` called as `ray.at(t)`) or as one structure pointer
                # (`def hit(self, ...)` called with a Scene) is a structure pointer with no
                # field set: read its members through the pointer, like any other
                # structure parameter.
                receiver_t:DeviceType = self._lookup(self.self_param)
                if isinstance(receiver_t, tuple) and is_ptr(receiver_t):
                    from . import structs as _structs_module  # noqa: PLC0415

                    dtype:Optional[np.dtype] = _structs_module.dtype_for(receiver_t[1])
                    if dtype is not None:
                        members:Dict[str, object] = _structs_module.member_types(
                            dtype, self.backend)
                        if node.attr in members:
                            return (f"({self.self_param})->{node.attr}" ,
                                    members[node.attr])
                kind:str
                _target:object
                kind, _target = _self_member_call_kind(self.self_owner, node.attr)
                if kind in ("method", "staticmethod"):
                    # A helper of the kernel's own class: call it instead of reading it,
                    # which is the one supported form (a function value has no device type).
                    self._error(
                        f"self.{node.attr} is a function; call it as self.{node.attr}(...) "
                        f"instead of reading it", node)
                self._error(
                    f"self.{node.attr} was not discovered on this kernel "
                    f"(discovered attributes: {', '.join(self.self_attr_types) or 'none'})", node)
            return f"_self_{node.attr}", self.self_attr_types[node.attr]

        if isinstance(node.value, ast.Name):
            module:object = self.resolve_global(node.value.id)
            # `Kind.B`: reading a member off an enum *class* is that member's value, exactly
            # as reading the bare name `B` would be. The class itself has no device
            # representation, so this has to be answered before it reaches the generic
            # attribute path.
            #
            # The base has to be a real enum *class* before the attribute is read: `blockIdx`
            # is an `_IndexVector` whose attributes compute the launch's numbers, so an
            # unconditional `getattr` here asked for a work item that does not exist yet and
            # made every `blockIdx.x` fail with "get_group_id() is only available inside a
            # @kernel function body".
            if isinstance(module, enum.EnumMeta):
                enum_member:Optional[Tuple[Any, Any]] = _enum_member_value(
                    getattr(module, node.attr, None))
                if enum_member is not None:
                    return self._visit_enum_constant(enum_member, node)
            if module is math or module is np:
                if node.attr in _MATH_CONSTS:
                    return _MATH_CONSTS[node.attr], "float"
                if module is np and node.attr in _NUMPY_DEVICE_TYPES:
                    # np.float32 / np.int32 / ... name a type, not a value. Reading it as a
                    # value has no device meaning; a cast is `x.astype(...)` or `np.float32(x)`,
                    # and a dtype keyword is consumed by the creation this feature lifts out.
                    self._error(
                        f"np.{node.attr} is a type: use it as a keyword (dtype=np.{node.attr}) "
                        f"or as a constructor (np.{node.attr}(value)), not as a value", node)
                self._error(f"{node.value.id}.{node.attr} is not supported", node)

        base_code, base_t = self._visit_expression(node.value)
        # A complex value's members are its real and imaginary parts, which the header
        # spells as its own accessors: the structure's members are `re`/`im` on the device
        # and `.real`/`.imag` in numpy, and a reader should never have to know which.
        complex_dtype:Optional[np.dtype] = self._complex_dtype_of(base_t)
        if complex_dtype is not None:
            kind:str = self._complex_side(base_t, complex_dtype)
            member:str = {"real": "real", "imag": "imag", "re": "real", "im": "imag"}.get(
                node.attr, "")
            if member:
                self._require_complex(complex_dtype)
                element:str = "double" if kind == "c128" else "float"
                return f"{kind}_{member}({base_code})", element
            self._error(
                f"a complex value has no member '.{node.attr}': use '.real', '.imag', or"
                f" call '.conjugate()'", node)
        if is_vec(base_t):
            component:str = _vec_component_name(node.attr)
            if len(component) > 1:
                return self._emit_vec_swizzle(base_code, base_t, component, node)
            field_key:(str, str) = (base_t, component)
            if field_key not in _C_FIELD_TYPES:
                self._error(
                    f"vector type {base_t} has no component '.{node.attr}'", node)
            # `threadIdx.x` is one launch term, not a component of a built vector: the *name*
            # resolves to a whole `int3` (`make_int3(...)` on CUDA, a compound literal on
            # OpenCL), so reading one component off it rebuilt all three and then indexed
            # them. The scalar spelling is what the axis means, and it is what the OpenCL
            # path already answers for `get_local_id(0)`.
            axis:Optional[str] = self._index_vector_axis(node, base_t)
            if axis is not None:
                return axis, _C_FIELD_TYPES[field_key]
            return f"({base_code}.{component})", _C_FIELD_TYPES[field_key]

        # A structure member, read with python's attribute spelling. It cannot be confused
        # with a vector component: the branch above has already claimed every base that is a
        # vector, so a name reaching here is only a member if the base is a structure.
        from . import structs

        if isinstance(base_t, tuple) and is_ptr(base_t):
            # `p.x` on a structure pointer: a structure argument always travels as one, so
            # the member read goes through '->'. A record array (a written object argument)
            # reads its element first, and a receiver passed by address (`&record`) is
            # addressed as `(&record)->x`.
            dtype:Optional[np.dtype] = structs.dtype_for(base_t[1])
            if dtype is not None:
                # A `@property` is a member the class computes: `box.total` is a call to
                # its getter, not a field read. It is tried before the field lookup, and
                # only for a name that is not a real member -- a property name that
                # shadows a field would be a field read in python too, because the class
                # body's descriptor replaces the field of the same name.
                property_read:Optional[Translated] = self._emit_property_read(
                    structs.declared_classes(dtype), node.attr, (base_code, base_t), node)
                if property_read is not None:
                    return property_read
                members:Dict[str, object] = structs.member_types(dtype, self.backend)
                if node.attr not in members:
                    self._error(
                        f"structure {base_t[1]} has no member '.{node.attr}' "
                        f"(available: {', '.join(members)})", node)
                if base_code.startswith("&"):
                    # A receiver passed by address: no record array around it.
                    return f"({base_code})->{node.attr}", members[node.attr]
                if structs.is_declared_structure(np.dtype(dtype)) and \
                        not base_code.endswith(']'):
                    # A structure pointer parameter: addressed directly.
                    return f"({base_code})->{node.attr}", members[node.attr]
                return f"({base_code}[0]).{node.attr}", members[node.attr]

        if isinstance(base_t, str):
            dtype:Optional[np.dtype] = structs.dtype_for(base_t)
            if dtype is not None:
                members:Dict[str, object] = structs.member_types(dtype, self.backend)
                if node.attr not in members:
                    self._error(
                        f"structure {base_t} has no member '.{node.attr}' "
                        f"(available: {', '.join(members)})", node)
                return f"({base_code}).{node.attr}", members[node.attr]

        self._error(f"unsupported attribute access: .{node.attr}", node)

    # ---------- calls ----------

    def _require_quaternion_header(self, dtype:np.dtype)->None:
        """Record that this source uses a quaternion of this width.

        One header serves both widths; what the compile step needs to know is which blocks to
        enable, because a `double` block does not compile on a device without fp64.
        """
        from . import quaternion  # noqa: PLC0415

        need:Optional[str] = quaternion.needs(dtype)
        if need is not None:
            self.quaternion_needs.add(need)

    def _emit_quaternion_call(self, node:ast.Call)->Optional[Translated]:
        """`conjugate(q)` as the generated device function for the same meaning.

        `pygf.funcs.conjugate` is an ordinary python function, so without this it was
        translated as a device *helper* -- and its body builds `m.__class__(...)`, naming a
        class a kernel has no meaning for. The quaternion meaning is what the call is after,
        so it maps onto the generated `{type}_conjugate` instead.
        """
        from . import quaternion  # noqa: PLC0415

        def called_name(expr:ast.expr)->Optional[str]:
            if isinstance(expr, ast.Name):
                return expr.id
            if isinstance(expr, ast.Attribute):
                return expr.attr
            return None

        if called_name(node.func) != "conjugate" or len(node.args) != 1:
            return None
        if isinstance(node.func, ast.Name):
            callee:object = self.resolve_global(node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            owner:object = self.resolve_global(node.func.value.id)
            callee = getattr(owner, node.func.attr, None)
        else:
            return None
        import pygf.funcs as pygf_funcs  # noqa: PLC0415

        if callee is not pygf_funcs.conjugate:
            return None
        argument:Translated = self._visit_expression(node.args[0])
        # The argument may travel as a pointer: unwrap the type, dereference the value.
        arg_code:str = argument[0]
        arg_type:object = argument[1]
        if isinstance(arg_type, tuple) and is_ptr(arg_type):
            arg_type, arg_code = arg_type[1], f"(*{arg_code})"
        dtype:Optional[np.dtype] = quaternion.is_quaternion(arg_type)
        if dtype is None:
            # `conjugate` serves both meanings: a quaternion flips its axis, a complex value
            # its imaginary part. pygf's own `conjugate` accepts either, so the device has to
            # as well -- otherwise one kernel would mean two things per type.
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(arg_type)
            if complex_dtype is not None:
                kind:str = self._complex_side(arg_type, complex_dtype)
                self._require_complex(complex_dtype)
                return f"{kind}_conjugate({arg_code})", kind
            self._error(
                f"conjugate() needs a quaternion or a complex value, got {arg_type}", node)
        type_name:str = structs_name(dtype)
        self._require_quaternion_header(dtype)
        return f"{type_name}_conjugate({arg_code})", type_name

    def _visit_call(self, node:ast.Call)->Translated:
        func:ast.expr = node.func
        if node.keywords:
            # A structure constructor takes member names as keywords -- Ray(origin=p) is
            # how python's ctypes spelling addresses members -- so it is translated before
            # the positional reordering below, which only knows about plain functions.
            if isinstance(func, ast.Name):
                candidate:object = self.resolve_global(func.id)
                if isinstance(candidate, type) and self._is_structure_class(candidate):
                    return self._visit_structure_constructor_call(node, candidate)
                # A plain class the scan synthesized into a structure is a constructor too --
                # it has to be handled *here*, before keyword reordering, because reordering
                # fills a call's omitted arguments from `__init__`'s defaults, and a member
                # whose default is `None` would reach the device as a None constant.
                if isinstance(candidate, type):
                    implied:Optional[np.dtype] = self._structure_from_construction(candidate)
                    if implied is not None:
                        self.plain_class_dtypes.add(implied)
                        return self._visit_structure_constructor_call(node, implied)
            # A helper may be called with keywords; nothing below needs to know, so the
            # call is rewritten to positional order first, using the callee's signature.
            # Builtins take positional arguments only, so they keep refusing.
            node = self._reorder_keywords(node)
            if node.keywords:
                self._error("keyword arguments are not supported for this call", node)

        # A builtin is recognized by the *object* the callee resolves to, never by the
        # spelling of the name, so it does not matter whether it was imported, reached
        # through the package, or is a user function that merely shares the name.
        quaternion_result:Optional[Translated] = self._emit_quaternion_call(node)
        if quaternion_result is not None:
            return quaternion_result

        builtin:Optional[str] = self._builtin_call_name(node)
        if builtin is not None:
            if builtin in ("zeros_per_group", "zeros_per_block", "work_size"):
                self._error(
                    f"{builtin}() is a statement, not a value: use it on its own, as "
                    f"`name = {builtin}(...)`", node)
            # An atomic's index may name several axes (`atomic_add(a, (i, j), v)`), which is
            # the same row-major flattening a subscript uses, so it is flattened **instead of**
            # being visited as an expression: a bare tuple has no device value of its own. A
            # scalar index stays a flat offset, so the 1-D spelling keeps working on a
            # multidimensional array. The array is the first argument, which is where the
            # declared shape comes from.
            args:List[Translated] = []
            for position, argument in enumerate(node.args):
                if (builtin in ATOMIC_BUILTINS and position == 1
                        and isinstance(argument, ast.Tuple)):
                    args.append(self._flatten_index(argument, node.args[0], node))
                else:
                    args.append(self._visit_expression(argument))
            return self._emit_gpu_builtin(builtin, args, node)

        # A registered device-side library function: recognized by the object the callee
        # resolves to, like a builtin, and emitted as the symbol the registry names.
        library:Optional[str] = self._library_call_name(node)
        if library is not None:
            library_args:List[Translated] = [
                self._visit_expression(argument) for argument in node.args]
            return self._emit_library_function(library, library_args, node)

        if isinstance(func, ast.Name):
            name:str = func.id
            if name == "complex":
                # Handled before the arguments are visited: `complex("3+4j")` names a string
                # that has no device value at all, and visiting it first would report
                # "unsupported constant" instead of the reason a caller can act on.
                return self._emit_complex_constructor(node)
            args = [self._visit_expression(a) for a in node.args]
            arg_codes:List[str] = [c for c, _ in args]
            # `v(args)` on a structure value is `v.__call__(args)`: the callee is a value,
            # not a function, and a class that defines `__call__` gives it a meaning.
            holder:DeviceType = self._lookup(name)
            if holder is not None and self._is_structure_operand(holder):
                owner:Optional[Tuple[type, Any]] = self._magic_owner(holder, CALL_METHOD)
                if owner is not None:
                    _cls, method = owner
                    return self._emit_device_call(
                        CALL_METHOD, method, [(name, holder)] + list(args), node,
                        self_param=_receiver_param_of(method), receiver_as_argument=True)

            if name in ("int", "float", "bool"):
                if len(args) != 1:
                    self._error(f"{name}() takes exactly 1 argument", node)
                # `int(box)` is `box.__int__()`, the way `abs(box)` is `box.__abs__()`:
                # a class that defines the conversion gives it a meaning, and a device
                # keeps it rather than casting the aggregate.
                converted:Optional[Translated] = self._emit_magic_conversion(
                    CONVERSION_METHODS[name], arg_codes[0], args[0][1], node)
                if converted is not None:
                    return converted
                t:DeviceType = {"int": "int", "float": "float", "bool": "bool"}[name]
                return f"(({t})({arg_codes[0]}))", t

            if name in ("abs", "min", "max", "round", "pow"):
                # `abs(box)` and `round(box)` are single-argument calls on the class;
                # `min`/`max`/`pow` take two and are not method-shaped.
                if name in ("abs", "round") and len(args) == 1:
                    converted = self._emit_magic_conversion(
                        BUILTIN_METHODS[name], arg_codes[0], args[0][1], node)
                    if converted is not None:
                        return converted
                return self._emit_builtin(name, args, node)

            if name == "len":
                return self._emit_len(args, node)

            vec_name:Optional[str] = self._vec_type_name(name)
            if vec_name is not None:
                return self._emit_vec_constructor(vec_name, args, node)
            reason:Optional[str] = self._unsupported_vec_reason(self.resolve_global(name))
            if reason is not None:
                self._error(reason, node)

            target:object = self.resolve_global(name)
            if isinstance(target, type):
                # A `ctypes.Structure` subclass is not an object in the python sense: it is a
                # layout, and a device type is generated from it, so `Ray(origin, direction)`
                # is a structure value -- a compound literal keyed by member name. Every
                # other class stays refused: there is nothing on the device to instantiate.
                from . import structs

                structure:Optional[np.dtype] = structs.structure_dtype_of(target)
                if structure is not None:
                    return self._emit_structure_constructor(structs.register(structure),
                                                            structure, args, node)
                # A plain class whose value the body builds and then reads is a layout too,
                # decided by what the body does with it: the members it reads and the methods
                # it calls, typed from the arguments at this call site. Nothing here parses
                # the class's `__init__` or its annotations.
                implied:Optional[np.dtype] = self._structure_from_construction(target)
                if implied is not None:
                    self.plain_class_dtypes.add(implied)
                    return self._emit_structure_constructor(structs.register(implied),
                                                            implied, args, node)
                self._error(
                    f"'{name}' is a class: device code has no objects, so it cannot be "
                    f"constructed inside a kernel. Keep the values in separate scalar or "
                    f"vector variables, or pass a ctypes.Structure as a kernel argument",
                    node)
            if callable(target):
                return self._emit_device_call(name, target, args, node)
            self._error(f"unknown function '{name}'", node)

        if isinstance(func, ast.Attribute):
            receiver_name:Optional[str] = (func.value.id
                                           if isinstance(func.value, ast.Name) else None)
            if (receiver_name is not None and self.self_param
                    and receiver_name == self.self_param):
                args:List[Translated] = [self._visit_expression(a) for a in node.args]
                # The method belongs to the class the *receiver* is, not necessarily to the
                # kernel's own class: inside `Camera.render`, `scene.ray_color(...)` reaches
                # a Scene method, and a helper bound to a Scene receiver calls Scene's own
                # methods through `self.`. The receiver's dtype names that class.
                from . import structs as _structs_module  # noqa: PLC0415

                receiver_t:DeviceType = self._lookup(self.self_param)
                receiver_owner:Optional[type] = None
                if isinstance(receiver_t, tuple) and is_ptr(receiver_t):
                    dtype:Optional[np.dtype] = _structs_module.dtype_for(receiver_t[1])
                    if dtype is not None:
                        receiver_owner = _structs_module.declared_class(dtype)
                call_owner:Optional[type] = receiver_owner or self.self_owner
                kind:str
                target:object
                kind, target = _self_member_call_kind(call_owner, func.attr)
                if kind in ("method", "staticmethod") and receiver_owner is not None \
                        and call_owner is not self.self_owner:
                    # A method of the receiver's class: the receiver is this call's first
                    # argument, so it binds positionally, exactly as `ray.at(t)` does.
                    return self._emit_device_call(
                        func.attr, target,
                        [(self.self_param, receiver_t)] + args, node,
                        self_param=(target.__code__.co_varnames[0]
                                    if getattr(target, "__code__", None) else "self"),
                        receiver_as_argument=True)
                if kind in ("method", "staticmethod"):
                    return self._emit_self_method_call(kind, target, func.attr, args, node)
                self._error(
                    f"calling self.{func.attr}() from device code is not supported: "
                    f"self.{func.attr} is {self._why_not_callable(kind, func.attr)}", node)
            module:object = (self.resolve_global(receiver_name)
                          if receiver_name is not None else None)
            if module is math or module is np:
                if module is np and func.attr in ("clip", "where"):
                    # np.clip/np.where read their own arguments: a None bound means "no
                    # bound" and must not be evaluated as a device expression.
                    return self._emit_numpy_elementwise_nodes(func.attr, node)
                args = [self._visit_expression(a) for a in node.args]
                if module is np:
                    if func.attr in _NUMPY_DEVICE_TYPES:
                        # np.float32(x): a cast, exactly what `(float)x` means on the device.
                        return self._emit_numpy_cast(func.attr, args, node)
                    elementwise:Optional[Translated] = self._emit_numpy_elementwise(
                        func.attr, args, node)
                    if elementwise is not None:
                        return elementwise
                return self._emit_math_call(func.attr, args, node, module_name=receiver_name or "")
            if isinstance(module, type):
                # Class.static_method(...): a class name is not a device value, so the
                # only reachable form is a staticmethod, resolved to its plain function.
                kind, target = self_attr_kind(module, func.attr)
                if kind in ("staticmethod", "method"):
                    # A plain function taken off its class is just a function; the
                    # receiver of "method" stays an explicit first argument.
                    args = [self._visit_expression(a) for a in node.args]
                    return self._emit_device_call(func.attr, target, args, node)
                self._error(
                    f"cannot call {receiver_name}.{func.attr}() from device code: "
                    f"only static methods of a class are usable there, and "
                    f"{receiver_name}.{func.attr} is {self._why_not_callable(kind, func.attr)}", node)
            if inspect.ismodule(module) and callable(getattr(module, func.attr, None)):
                # module.function(...): a plain helper reached through its module.
                args = [self._visit_expression(a) for a in node.args]
                return self._emit_device_call(func.attr, getattr(module, func.attr), args, node)
            # Anything else is a receiver that is a *value*: `ray.hit_sphere(...)` on a
            # structure, or a method reached through one of its members.
            return self._emit_method_on_value(func, node)

        # `inner[(n,)](args)`: one work item launches another kernel. Left until here so a
        # structure constructor and a helper's keyword call are already resolved.
        if self._emit_nested_launch(node):
            return "", "void"

        self._error("unsupported call expression", node)

    def _emit_nested_launch(self, node:ast.Call)->bool:
        """`inner[(global, local)](args)` -> a device-side launch; True when this was one."""
        func:ast.expr = node.func
        if not isinstance(func, ast.Subscript):
            return False
        # Peel every bracket: child[(n,)] and the invalid child["cuda"][(n,)] both reach
        # a Name at the bottom, and the second has to be named as the mistake it is.
        callee:ast.expr = func.value
        while isinstance(callee, ast.Subscript):
            callee = callee.value
        if not isinstance(callee, ast.Name):
            return False
        target:object = self.resolve_global(callee.id)
        if not getattr(target, "is_pykernel", False):
            return False
        self._render_nested_launch(callee.id, target, func.slice, node)
        return True

    def _render_nested_launch(self, name:str, target:object, work:ast.expr,
                              node:ast.Call)->None:
        """Emit the device-side launch statement for one nested call."""
        if self.launch_backend != "cuda":
            # CUDA calls this dynamic parallelism; the OpenCL kernel language has no
            # nested-launch statement at all, and HIP does not offer it on this platform.
            self._error(
                f"'{name}' is a @kernel launched from device code, which {self.launch_backend} "
                f"does not support: device-side launch needs CUDA's dynamic parallelism. Run "
                f"this kernel on cuda, or call a plain device helper instead", node)
        entries:List[ast.expr] = []
        probe:ast.expr = node.func
        while isinstance(probe, ast.Subscript):
            entries.append(probe.slice)
            probe = probe.value
        for entry in entries:
            if isinstance(entry, ast.Constant) and isinstance(entry.value, str):
                self._error(
                    f"'{name}' is launched from device code, where the backend and the device "
                    f"cannot be chosen: the device is already running. Give only the work "
                    f"size, as `{name}[(global, local)](...)`", node)

        if isinstance(work, ast.Tuple):
            if len(work.elts) > 2:
                self._error(
                    f"a device-side launch takes at most (global, local): '{name}' was given "
                    f"{len(work.elts)} work sizes", work)
            parts:List[Translated] = [self._visit_expression(e) for e in work.elts]
            global_code, global_type = parts[0]
            local_code, local_type = parts[1] if len(parts) > 1 else ("", None)
        else:
            global_code, global_type = self._visit_expression(work)
            local_code, local_type = "", None
        if not is_int(global_type):
            self._error(f"the work size of '{name}' must be an integer", work)
        if local_code and not is_int(local_type):
            self._error(f"the local work size of '{name}' must be an integer", work)

        call_args:str = ", ".join(self._visit_expression(a)[0] for a in node.args)
        c_name:str = self._translate_nested_kernel(name, target, node)
        block:str = local_code if local_code else "1"
        grid:str = f"(({global_code}) / ({block}))"
        self._emit(f"{c_name}<<<{grid}, {block}>>>({call_args});")

    def _translate_nested_kernel(self, name:str, target:object, node:ast.Call)->str:
        """Render the launched kernel as another `__global__` function of this source."""
        raw_func:Callable = target.raw_func
        known:Optional[str] = self.nested_seen.get(raw_func)
        if known is not None:
            return known
        param_types:List[object] = [self._visit_expression(a)[1] for a in node.args]
        from .decorator import _positional_argument_names

        expected:int = len(_positional_argument_names(raw_func))
        if len(param_types) != expected:
            self._error(
                f"'{name}' takes {expected} argument(s) but a device-side launch gave "
                f"{len(param_types)}", node)
        digest:str = hashlib.md5(
            (raw_func.__name__ + repr(param_types)).encode("utf-8")).hexdigest()[:8]
        c_name:str = f"nested{len(self.nested_kernels)}_{digest}"
        # Registered before translating, so a kernel that launches itself is refused rather
        # than recursed into.
        self.nested_seen[raw_func] = c_name
        callee:FunctionTranslator = FunctionTranslator(
            raw_func, param_types, self.backend, _helper_globals(raw_func, self.resolve_global),
            self.device_cache, is_kernel=True, self_param="", self_attr_types={},
            self_owner=None, launch_backend=self.launch_backend, type_map=self.type_map)
        callee.translate()
        self.nested_kernels.append(callee.render_kernel(c_name))
        for deeper in callee.nested_kernels:
            if deeper not in self.nested_kernels:
                self.nested_kernels.append(deeper)
        from .decorator import _dependency_order

        for entry in _dependency_order(self.device_cache):
            code:str = entry["code"]
            if code not in self.nested_helpers:
                self.nested_helpers.append(code)
        return c_name

    def _emit_method_on_value(self, func:ast.Attribute, node:ast.Call)->Translated:
        """`value.method(args)` as a device call whose first argument is the value.

        A structure's methods are ordinary python functions defined in the class the caller
        declared, and their first parameter is the receiver -- which is exactly what `self`
        already is in a device helper. So there is nothing new to translate: the call becomes
        a device function whose first argument is the receiver expression, and the body then
        reads `self.origin` the way any structure-typed parameter would.

        The class has to be reachable, which is why only a `ctypes.Structure` subclass
        qualifies: a bare record dtype has no methods, and two classes with one layout share
        a dtype, so the dtype alone cannot say whose methods to compile.
        """
        from . import structs

        base_code, base_t = self._visit_expression(func.value)
        # A complex value's methods are numpy's, and the device header spells the same
        # operations: `.conjugate()` flips the imaginary part, and the rest are the free
        # functions a reader may reach for through the value itself.
        complex_dtype:Optional[np.dtype] = self._complex_dtype_of(base_t)
        if complex_dtype is not None:
            translated:Optional[Translated] = self._emit_complex_method(
                func.attr, base_code, base_t, complex_dtype, node)
            if translated is not None:
                return translated
            self._error(
                f"a complex value has no method '.{func.attr}()': numpy's complex scalars"
                f" define 'conjugate()' and 'conj()', and the rest are the module functions"
                f" (np.sqrt, np.abs, np.angle, ...)", node)
        receiver_is_pointer:bool = (isinstance(base_t, tuple) and is_ptr(base_t))
        base_struct:str = base_t[1] if receiver_is_pointer else base_t
        owner:Optional[type] = None
        if isinstance(base_struct, str):
            dtype:Optional[np.dtype] = structs.dtype_for(base_struct)
            if dtype is not None:
                # The class that actually defines the method: two callers can declare
                # classes with one layout, and only one of them carries this attribute.
                owner = (structs.declared_class_carrying(dtype, func.attr)
                         or structs.declared_class(dtype))
            elif base_struct in _vec_registry():
                owner = _vec_registry()[base_struct]
        if owner is None:
            self._error(
                f"cannot call '.{func.attr}()' on a value of type {base_t}: methods are "
                f"translated from the class they are defined on, and only a "
                f"ctypes.Structure subclass or a vector type carries one", node)

        kind:str
        target:object
        kind, target = self_attr_kind(owner, func.attr)
        if kind == "method":
            args = [self._visit_expression(a) for a in node.args]
            # A method that assigns to its receiver's members needs the receiver *by
            # address*: a by-value structure is a copy, so `self.normal = ...` inside the
            # method would be lost. `HitRecord.set_face_normal` is exactly that shape, and
            # with a by-value receiver every normal stayed zero -- the GPU drew a uniform
            # grey ball while the interpreter, which runs the method unchanged, was right.
            receiver_param:str = (target.__code__.co_varnames[0]
                                  if getattr(target, "__code__", None) else "self")
            if receiver_is_pointer:
                # A structure argument already travels as a pointer: hand it over as it is,
                # whether the callee writes its members or only reads them. The receiver's
                # parameter name is named so the callee binds it as the structure it is --
                # without it, `self.origin` inside the method would read the value form.
                return self._emit_device_call(
                    func.attr, target, [(base_code, base_t)] + args, node,
                    self_param=receiver_param, receiver_as_argument=True)
            # A value receiver is handed over by address: every structure parameter is a
            # pointer (so `self.<member>` reads the one form), and a local value's address
            # is available at the call site. The copy python makes for a by-value call is
            # the one thing that cannot be kept, and a method that assigns to its receiver
            # needs the address for the write to reach the caller.
            return self._emit_device_call(
                func.attr, target,
                [(f"&({base_code})", (PTR, base_struct))] + args, node,
                self_param=receiver_param, receiver_as_argument=True)
        if kind == "staticmethod":
            args = [self._visit_expression(a) for a in node.args]
            return self._emit_device_call(func.attr, target, args, node)
        self._error(
            f"cannot call '.{func.attr}()' on {owner.__name__}: it is "
            f"{self._why_not_callable(kind, func.attr)}", node)

    def _emit_builtin(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        # Python's own builtins (`abs`, `pow`, `min`, `max`, `round`) take a complex value
        # in the interpreter for some of these and refuse it for others, and the device has
        # to agree with that rather than with a table keyed by type (see
        # _emit_complex_builtin).
        if any(self._complex_dtype_of(t) is not None for _, t in args):
            return self._emit_complex_builtin(name, args, node)

        if name == "pow":
            # python's builtin, which differs from math.pow only in that integer arguments
            # stay exact; the device form is the same floating-point power. A pair of ints
            # promotes to `int` above, and spelling `pow((int)2, (int)10)` is ambiguous in
            # C++ -- NVRTC and OpenCL's clang refuse it ("more than one instance of
            # overloaded function pow"), while HIP's <cmath> happens to resolve it, so the
            # identical source answered on one backend and refused on the others. The `**`
            # operator already promotes an integer power to float (see _visit_binop); the
            # builtin follows the same rule.
            if len(args) != 2:
                self._error("pow() takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            t:object = promote(lt, rt)
            if is_int(t):
                t = "float"
            c_func:str = _math_device_name("pow", self.backend, t)
            return f"{c_func}(({t})({lcode}), ({t})({rcode}))", t

        if name == "abs":
            if len(args) != 1:
                self._error("abs() takes exactly 1 argument", node)
            code, t = args[0]
            if is_float(t):
                return f"fabs({code})", t
            if is_int(t):
                return f"abs({code})", t
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is not None:
                # numpy's `abs(complex)` is the magnitude, a *real*: the same rule the
                # module-level `np.abs`/`np.absolute` follows (see _emit_complex_math).
                kind:str = self._complex_side(t, complex_dtype)
                self._require_complex(complex_dtype)
                return f"{kind}_abs({code})", ("double" if kind == "c128" else "float")
            self._error("abs() requires a numeric argument", node)

        if name in ("min", "max"):
            if len(args) < 2:
                self._error(f"{name}() takes at least 2 arguments", node)
            # python's builtin takes any arity and folds left to right
            # (`min(a, b, c)` is `min(min(a, b), c)`), so the pairs chain. The mixed
            # complex pair reaches _emit_complex_builtin before this branch.
            pick:str = ("fmin" if name == "min" else "fmax") \
                if is_float(promote(args[0][1], args[1][1])) \
                else ("min" if name == "min" else "max")
            left, lt = args[0]
            for rcode, rt in args[1:]:
                t:object = promote(lt, rt)
                if is_float(t):
                    left = f"{pick}(({t})({left}), ({t})({rcode}))"
                else:
                    left = f"{pick}(({t})({left}), ({t})({rcode}))"
                lt = t
            return left, lt

        if name == "round":
            if len(args) not in (1, 2):
                self._error("round() takes 1 or 2 arguments", node)
            code, t = args[0]
            if len(args) == 2:
                # round(x, digits) with a constant digit count: python rounds the value,
                # not the digits it prints, so this scales, rounds and unscales.
                digits = node.args[1]
                if not (isinstance(digits, ast.Constant)
                        and isinstance(digits.value, int)):
                    self._error(
                        "round() digits must be a constant integer: the scale factor is "
                        "a literal in the generated code", node)
                scale = 10.0 ** int(digits.value)
                inner = f"((float)({code}))"
                return (f"(({t})rint({inner} * {scale}f) / {scale}f)"
                        if is_float(t) else f"({code})"), t
            if is_float(t):
                return f"((int)rint(({t})({code})))", "int"
            return f"({code})", t

        self._error(f"unsupported builtin: {name}", node)

    def _emit_len(self, args:List[Translated], node:ast.Call)->Translated:
        """``len(array)`` -> the array's extent along its only axis.

        A kernel works on one-dimensional arrays and its extents arrive as int
        parameters, so the length is already known: recovering it here keeps
        ``for i in range(len(a))`` written the way it would be in Python.
        """
        if len(args) != 1:
            self._error("len() takes exactly 1 argument", node)
        code, t = args[0]
        # A class that defines `__len__` states its own length, which the array rule below
        # knows nothing about.
        owner:Optional[Tuple[type, Any]] = self._magic_owner(t, "__len__")
        if owner is not None:
            _cls, method = owner
            return self._emit_device_call(
                "__len__", method, [(code, t)], node,
                self_param=_receiver_param_of(method), receiver_as_argument=True)
        if not is_ptr(t):
            self._error("len() takes an array argument", node)
        holder = node.args[0]
        if not isinstance(holder, ast.Name):
            # `len(bag.items)` on a member array: the argument spelling differs but the
            # length is the same compile-time shape variable, keyed on the base's name.
            if isinstance(holder, ast.Attribute) and isinstance(holder.value, ast.Name):
                base_t:DeviceType = self._lookup(holder.value.id)
                element_t:object = base_t[1] if is_ptr(base_t) else base_t
                dtype:Optional[np.dtype] = (structs_dtype_for(element_t)
                                            if isinstance(element_t, str) else None)
                if dtype is not None and holder.attr in (dtype.names or ()):
                    field_dtype:np.dtype = np.dtype(dtype.fields[holder.attr][0])
                    if field_dtype.subdtype is not None:
                        return (str(field_dtype.subdtype[1][0]), "int")
            self._error("len() takes an array parameter name", node)
        return self._shape_var(holder.id, 0), "int"

    def _reorder_keywords(self, node:ast.Call)->ast.Call:
        """Rewrite a call with keyword arguments into positional order.

        Only a plain-python helper has a signature to bind against; anything else
        (builtin, gpu.*, vector constructor) keeps its keywords so the caller can
        report them. Defaults are filled in, matching what the host does for a kernel
        call, so the device function always receives one value per parameter.
        """
        func:ast.expr = node.func
        target:object = None
        if isinstance(func, ast.Name):
            target = self.resolve_global(func.id)
            if not callable(target) or getattr(target, "is_pykernel", False):
                return node
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and not (self.self_param and func.value.id == self.self_param)):
            module:object = self.resolve_global(func.value.id)
            target = getattr(module, func.attr, None)
            if not callable(target):
                return node
        else:
            return node

        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return node

        positional:List[ast.expr] = list(node.args)
        supplied:Dict[str, ast.expr] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                return node
            supplied[keyword.arg] = keyword.value

        ordered:List[ast.expr] = []
        for index, (name, parameter) in enumerate(signature.parameters.items()):
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if index < len(positional):
                if name in supplied:
                    self._error(
                        "the argument '%s' is given both positionally and by keyword"
                        % name, node)
                ordered.append(positional[index])
            elif name in supplied:
                ordered.append(supplied[name])
            elif parameter.default is not parameter.empty:
                ordered.append(ast.Constant(value=parameter.default))
            else:
                self._error("missing required argument '%s'" % name, node)

        unknown = set(supplied) - set(signature.parameters)
        if unknown:
            self._error(
                "unknown keyword argument%s: %s"
                % ("" if len(unknown) == 1 else "s", ", ".join(sorted(unknown))), node)

        return ast.Call(func=node.func, args=ordered, keywords=[])

    def _emit_numpy_elementwise_nodes(self, attr:str,
                                      node:ast.Call)->Optional[Translated]:
        """``_emit_numpy_elementwise`` for the calls that read their own argument syntax.

        ``np.clip`` treats a literal ``None`` bound as "no bound on this side" and
        ``np.where`` takes a condition plus two values, so both have to look at the source
        nodes: evaluating ``None`` as a device expression fails, and evaluating every
        argument up front would lose which ones were literal ``None``.
        """
        if attr == "clip":
            if len(node.args) < 2 or len(node.args) > 3:
                self._error("np.clip() takes 2 or 3 arguments", node)
            value_code, value_t = self._visit_expression(node.args[0])
            if is_ptr(value_t) or is_image(value_t):
                self._error_numpy_array_form("clip", node)
            bounds:List[Optional[Translated]] = []
            for bound_node in node.args[1:]:
                if isinstance(bound_node, ast.Constant) and bound_node.value is None:
                    bounds.append(None)
                    continue
                bound:Translated = self._visit_expression(bound_node)
                if is_ptr(bound[1]) or is_image(bound[1]):
                    self._error("np.clip() bounds must be scalars, not arrays", bound_node)
                bounds.append(bound)
            lo:Optional[Translated] = bounds[0] if bounds else None
            hi:Optional[Translated] = bounds[1] if len(bounds) > 1 else None
            if self._complex_dtype_of(value_t) is not None:
                # numpy's clip on a complex value is its lexicographic min/max, and a `None`
                # bound means the operation on that side is simply absent. The bounds keep
                # their own types, so a real one is widened to the complex value it stands
                # for (`np.clip(z, 0, 1)`) rather than being handed to a complex function.
                value:Translated = (value_code, value_t)
                if lo is None and hi is None:
                    return value
                if hi is None:
                    return self._emit_complex_builtin("max", [value, lo], node)
                if lo is None:
                    return self._emit_complex_builtin("min", [value, hi], node)
                return self._emit_complex_builtin("clamp", [value, lo, hi], node)
            # np.clip keeps an integer input an integer, so the element type is not forced
            # to float: `np.clip(300, 0, 255)` is 255, not a float.
            result_t:object = value_t if is_int(value_t) else "float"
            return self._emit_clip(value_code, result_t,
                                   lo[0] if lo is not None else None,
                                   hi[0] if hi is not None else None), result_t
        if attr == "where":
            if len(node.args) == 1:
                self._error(
                    "np.where(cond) with one argument returns indices, which a kernel "
                    "cannot represent; use np.where(cond, a, b)", node)
            if len(node.args) != 3:
                self._error("np.where() takes exactly 3 arguments in a kernel", node)
            args:List[Translated] = [self._visit_expression(a) for a in node.args]
            for code, t in args:
                if is_ptr(t) or is_image(t):
                    self._error_numpy_array_form("where", node)
            return self._emit_where(args, node)
        return None

    def _error_numpy_array_form(self, attr:str, node:ast.AST)->NoReturn:
        self._error(
            f"np.{attr}() on an array is a host-side element-wise operation "
            f"returning a new array, which a per-work-item kernel cannot represent. "
            f"Index the element first, e.g. np.{attr}(arr[i], ...)", node)

    def _emit_numpy_elementwise(self, attr:str, args:List[Translated],
                                node:ast.Call)->Optional[Translated]:
        """numpy functions that make sense on one element, translated into device code.

        A kernel body works on single values, so the array form of these functions cannot
        be represented (it would allocate); the element form is exactly the scalar rule
        python applies to the same call:

        * ``np.clip(v, lo, hi)`` -- clamp/min-max per backend, with None bounds omitted;
        * ``np.where(cond, a, b)`` -- a ?: expression.

        Returns None when ``attr`` is not one of them, so the caller can fall through to
        the math table (which rejects the name with its own message).
        """
        if attr in ("clip", "where", "abs", "fabs", "power", "arctan2",
                    "minimum", "maximum", "square", "reciprocal", "round",
                    "angle", "real", "imag", "conj", "conjugate", "sign"):
            for code, t in args:
                if is_ptr(t) or is_image(t):
                    self._error_numpy_array_form(attr, node)
        if attr in ("abs", "fabs"):
            # numpy has two spellings; the device picks by type, as python's abs does.
            code, t = args[0]
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is not None:
                # `np.abs(complex)` and `np.fabs(complex)` are both the modulus, a real.
                kind:str = self._complex_side(t, complex_dtype)
                self._require_complex(complex_dtype)
                element:str = "double" if kind == "c128" else "float"
                return f"{kind}_abs({code})", element
            if is_float(t):
                return f"fabs({code})", t
            if is_int(t):
                return f"abs({code})", t
            self._error(f"np.{attr}() requires a numeric argument", node)
        if attr == "square":
            code, t = args[0]
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is not None:
                # A complex square is the full product, not `x * x` spelled with a C
                # operator -- the same reason the `*` operator goes through the header.
                kind:str = self._complex_side(t, complex_dtype)
                self._require_complex(complex_dtype)
                return f"{kind}_square({code})", kind
            return f"(({code}) * ({code}))", t
        if attr == "reciprocal":
            code, t = args[0]
            complex_dtype = self._complex_dtype_of(t)
            if complex_dtype is not None:
                kind = self._complex_side(t, complex_dtype)
                self._require_complex(complex_dtype)
                return f"{kind}_reciprocal({code})", kind
            return f"(1.0f / ((float)({code})))", "float"
        if attr in ("angle", "real", "imag", "conj", "conjugate", "sign"):
            # The complex-only members and functions, spelled as numpy spells them. Each one
            # is the header's own operation, and each answers either a real or a complex
            # value -- `np.real`/`np.imag`/`np.angle` are the first kind, the rest the second.
            code, t = args[0]
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is None:
                if attr == "sign" and (is_float(t) or is_int(t)):
                    # `np.sign` on a real is the ordinary sign, which the builtin answers.
                    return self._emit_shared_math_builtin("sign", args, node)
                self._error(
                    f"np.{attr}() is a complex operation; here the argument is {t}", node)
            kind:str = self._complex_side(t, complex_dtype)
            self._require_complex(complex_dtype)
            element:str = "double" if kind == "c128" else "float"
            if attr == "real":
                return f"{kind}_real({code})", element
            if attr == "imag":
                return f"{kind}_imag({code})", element
            if attr == "angle":
                return f"{kind}_angle({code})", element
            if attr == "sign":
                # numpy's complex `sign` is the direction on the unit circle, and zero at
                # zero: a header function, because the division alone is 0/0 there.
                return f"{kind}_sign({code})", kind
            return f"{kind}_conjugate({code})", kind
        if attr == "minimum" or attr == "maximum":
            if len(args) != 2:
                self._error(f"np.{attr}() takes exactly 2 arguments", node)
            if any(self._complex_dtype_of(t) is not None for _, t in args):
                # numpy gives a complex value a lexicographic order, so its minimum and
                # maximum are defined; the header's comparison is what answers them.
                return self._emit_complex_builtin(
                    "min" if attr == "minimum" else "max", args, node)
            (lcode, lt), (rcode, rt) = args
            t:object = promote(lt, rt)
            pick:str = "min" if attr == "minimum" else "max"
            if is_float(t):
                # `fmin`/`fmax` exist on both backends (`fminf` is the C spelling OpenCL
                # does not provide); the plain min/max are the integer forms.
                return f"{'fmin' if pick == 'min' else 'fmax'}(({t})({lcode}), ({t})({rcode}))", t
            return f"{pick}(({t})({lcode}), ({t})({rcode}))", t
        if attr in ("power", "arctan2"):
            # Same operation as math.pow / math.atan2 under numpy's spelling. A complex base
            # or exponent is `np.power`'s own complex power, which is the header's `pow` --
            # the unary table below could not express it, since it takes two operands.
            if attr == "power" and any(self._complex_dtype_of(t) is not None
                                       for _, t in args):
                return self._emit_complex_builtin("pow", args, node)
            return self._emit_math_call({"power": "pow", "arctan2": "atan2"}[attr],
                                        args, node, module_name="np")
        if attr == "round":
            # numpy rounds halfway cases to even, which is python's round too. A complex
            # value is rounded per part -- and the *builtin* `round(z)` is refused, because
            # python's own round() has no complex form; `np.round(z)` is the spelling that
            # answers one, which is why this branch came before the builtin's. The numpy
            # spelling emits the header's function directly rather than going through the
            # builtin table, where `rint` is a pygf name that rounds through python's
            # `round()` and is therefore refused like it.
            if any(self._complex_dtype_of(t) is not None for _, t in args):
                code, t = args[0]
                dtype:np.dtype = cast(np.dtype, self._complex_dtype_of(t))
                kind:str = self._complex_side(t, dtype)
                self._require_complex(dtype)
                return f"{kind}_rint({code})", kind
            return self._emit_builtin("round", args, node)
        if attr == "where":
            return self._emit_where(args, node)
        return None

    def _emit_where(self, args:List[Translated],
                    node:ast.Call)->Translated:
        """`np.where(cond, a, b)` as a conditional expression."""
        cond_code, cond_t = args[0]
        then_code, then_t = args[1]
        else_code, else_t = args[2]
        if not is_int(cond_t) and cond_t != "bool":
            self._error(
                f"np.where() needs a boolean or integer condition, got {cond_t}", node)
        unified:object = promote(then_t, else_t)
        return (f"(({cond_code}) ? ({unified})({then_code}) : "
                f"({unified})({else_code}))"), unified

    def _emit_clip(self, value_code:str, result_t:object,
                   lo:Optional[str], hi:Optional[str])->str:
        """`np.clip` for one element, in the form the current backend provides.

        OpenCL C has a real `clamp` builtin for both integer and floating types, so that
        is used directly: one call, and the compiler recognizes it. CUDA C has no `clamp`
        function at all (verified with NVRTC: it reports `identifier "clamp" is
        undefined`), so the equivalent nesting of `fmin`/`fmax` is emitted instead. Both
        forms give the same numbers; this only keeps each backend's generated code in the
        shape that backend natively understands.

        A missing bound is simply left out, matching `np.clip(a, None, hi)`.
        """
        if lo is None and hi is None:
            return f"({value_code})"
        # Every cast below must use this backend's own spelling: CUDA C has no `uchar`
        # (see _device_type), and writing the translator's internal name here made a
        # clipped uint8 element fail to compile.
        element:str = self._device_type(result_t) if not is_vec(result_t) else result_t
        if self._opencl_syntax:
            # clamp(v, lo, hi) needs both bounds; with one bound the other side is the
            # type's own extreme, which min/max expresses directly and without a literal.
            if lo is not None and hi is not None:
                return f"clamp(({element})({value_code}), ({element})({lo}), ({element})({hi}))"
            if lo is not None:
                return f"max(({element})({value_code}), ({element})({lo}))"
            return f"min(({element})({value_code}), ({element})({hi}))"

        # CUDA/HIP: no clamp builtin. An integer element stays on min/max, because fminf's
        # overloads are ambiguous for integer arguments (NVRTC reports "more than one
        # instance of overloaded function fmax matches the argument list"); a float
        # element uses fmin/fmax, whose NaN rule (return the other operand) is what
        # np.clip does with a NaN input.
        pick_min:str = "min" if is_int(result_t) else "fmin"
        pick_max:str = "max" if is_int(result_t) else "fmax"
        code:str = f"({value_code})"
        if lo is not None:
            code = f"{pick_max}(({element})({code}), ({element})({lo}))"
        if hi is not None:
            code = f"{pick_min}(({element})({code}), ({element})({hi}))"
        return f"({code})"

    def _clip_bound(self, bound:Translated,
                    node:Optional[ast.expr], call:ast.Call)->Optional[str]:
        """One bound of np.clip: its code, or None when the bound is None."""
        if node is None:
            return None
        code, t = bound
        if is_ptr(t) or is_image(t):
            self._error(
                "np.clip() bounds must be scalars, not arrays", call)
        return code

    def _emit_complex_math(self, attr:str, args:List[Translated],
                           node:ast.Call, module_name:str)->Optional[Translated]:
        """A math/numpy function applied to a complex value, as the header's function.

        numpy defines the whole elementary set on a complex scalar -- `sqrt`, `exp`, `log`,
        the trigonometric and hyperbolic families and their inverses, `square`, `absolute`,
        `angle`, `conjugate` -- and refuses the real-only ones (`floor`, `ceil`, `trunc`,
        `fmod`, `hypot`, `cbrt`, `ldexp`, `atan2`), which have no complex meaning at all. The
        device mirrors that exactly: a name numpy refuses is refused here, with the reason
        named, rather than computed into something python never would.

        Returns None when neither argument is complex, so the ordinary scalar path answers.
        """
        complex_dtypes:List[Optional[np.dtype]] = [self._complex_dtype_of(t) for _, t in args]
        if not any(dtype is not None for dtype in complex_dtypes):
            return None
        if attr in ("isnan", "isinf", "isfinite"):
            # numpy defines all three for a complex value: a NaN or an infinity in *either*
            # part answers the first two, and `isfinite` needs both parts finite. The header
            # holds the predicate, and the answer is the number python's bool stands for --
            # the shape the real path below answers with as well.
            code, t = args[0]
            kind:str = self._complex_side(t, complex_dtypes[0])
            self._require_complex(complex_dtypes[0])
            return f"({kind}_{attr}({code}) != 0)", "bool"
        if attr in COMPLEX_ABS_FUNCTIONS:
            # A real result: `abs`, `np.abs`, `np.angle`, `xFabs` on a complex is the
            # magnitude or the phase, not a complex value.
            code, t = args[0]
            dtype:np.dtype = (self._complex_dtype_of(t) or complex_dtypes[0]  # type: ignore[assignment]
                              or np.dtype(np.complex128))
            kind:str = self._complex_side(t, dtype)
            self._require_complex(dtype)
            device:str = COMPLEX_ABS_FUNCTIONS[attr]
            element:str = "double" if kind == "c128" else "float"
            if device == "":
                return f"fabs({code})", element
            return f"{kind}_{device}({code})", element
        if attr in COMPLEX_UNSUPPORTED_FUNCTIONS:
            self._error(
                f"{module_name}.{attr}() has no complex meaning: numpy refuses it for a"
                f" complex input, and so does this package. Its complex counterpart is"
                f" {COMPLEX_UNSUPPORTED_FUNCTIONS[attr]}", node)
        if attr not in COMPLEX_DEVICE_FUNCTIONS:
            self._error(
                f"{module_name}.{attr}() is not defined for a complex value: the complex"
                f" functions are {', '.join(sorted(COMPLEX_DEVICE_FUNCTIONS))}", node)
        code, t = args[0]
        dtype = self._complex_side(t, self._complex_dtype_of(t))
        self._require_complex(np.dtype(np.complex128 if dtype == "c128" else np.complex64))
        return f"{dtype}_{COMPLEX_DEVICE_FUNCTIONS[attr]}({code})", dtype

    def _emit_math_call(self, attr:str, args:List[Translated], node:ast.Call, module_name:str="math")->Translated:
        complex_result:Optional[Translated] = self._emit_complex_math(attr, args, node,
                                                                      module_name)
        if complex_result is not None:
            return complex_result
        if attr not in _MATH_FUNCS:
            self._error(f"{module_name}.{attr} is not supported", node)
        ret_override:Optional[str] = _MATH_FUNCS[attr]
        arg_codes:List[str] = []
        ret:object = (ret_override or "float")
        vec_arg:Optional[str] = None
        for i, (code, t) in enumerate(args):
            if is_ptr(t):
                # python semantics: math.* only accepts scalars; np.*(arr) is a host-side element-wise
                # operation returning a new array, which cannot be represented in a work-item kernel -> require indexing to an element
                arr_desc:str = (node.args[i].id if isinstance(node.args[i], ast.Name) else "the argument")
                self._error(
                    f"{module_name}.{attr}() cannot be applied to the array '{arr_desc}': "
                    f"in python, {module_name}.{attr}(arr) would be an element-wise host operation "
                    f"returning a new array, which a per-work-item kernel cannot represent. "
                    f"Index the element first, e.g. {module_name}.{attr}(arr[i]).", node)
            if is_vec(t):
                vec_arg = (vec_arg or t)
            if ret_override is None and t == "double":
                ret = "double"
        if ret == "float" and module_name == "np" \
                and all(is_int(t) for _, t in args) and args:
            # numpy computes every one of these in float64 for an *integer* input
            # (`np.sqrt(2)` is a numpy float64), and the interpreter runs numpy -- so the
            # device has to as well. Spelling `sqrtf((float)2)` answered 1.41421354 where
            # numpy answers 1.4142135623730951. `math.*` keeps its own rule (python's
            # math.sqrt(2) is exact at the call, and the answer rounds like a float).
            ret = "double"

        if vec_arg is not None:
            # python semantics: math/np functions do not accept vector structs (np.sqrt(float4_value) raises TypeError).
            # OpenCL natively supports per-component vector math; CUDA implements it by per-component expansion.
            if ret_override is not None:
                self._error(
                    f"{module_name}.{attr}() on vector type {vec_arg} is not supported "
                    f"(it returns {ret_override} for scalars; apply it per-component, e.g. "
                    f"{attr}(v.x))", node)
            if len(args) != 1:
                self._error(f"{module_name}.{attr}() on vectors takes exactly 1 vector argument", node)
            if self._opencl_syntax:
                return f"{attr}({args[0][0]})", vec_arg
            return self._vec_componentwise(_math_device_name(attr, "cuda"), args[0][0], vec_arg), vec_arg

        if attr in ("degrees", "radians"):
            # No CUDA intrinsic for either; the conversion factor is the definition.
            factor:str = ("57.29577951308232f" if attr == "degrees"
                          else "0.017453292519943295f")
            if self._opencl_syntax:
                return f"{attr}({args[0][0]})", ret
            return f"(({args[0][0]}) * {factor})", ret

        if attr in ("isnan", "isinf", "isfinite"):
            # The predicate takes a float; casting it to the int result type is what made the
            # CUDA call ambiguous. The device answers 0/1, which is the number python's bool
            # stands for. (A complex value answered this in _emit_complex_math, above.)
            return f"({attr}((float)({args[0][0]})) != 0)", "bool"

        if ret_override == "int" and attr in ("floor", "ceil", "trunc"):
            # python's floor/ceil/trunc answer an int, but the *argument* is a float. Casting
            # the argument instead (`floor((int)x)`) picked the integer overload -- CUDA
            # rejects it as ambiguous, OpenCL has no such overload -- and it computed the
            # wrong number wherever it did resolve: `ceil(2.1)` became `ceil(2)`, so 2.
            code:str = args[0][0]
            device:str = _math_device_name(attr, self.backend)
            return f"((int){device}((float)({code})))", "int"

        for (code, t) in args:
            arg_codes.append(code if (t == ret or (is_float(t) and is_float(ret))) else f"(({ret})({code}))")
        c_func:str = _math_device_name(attr, self.backend, ret)
        return f"{c_func}({', '.join(arg_codes)})", ret

    def _emit_numpy_cast(self, name:str, args:List[Translated],
                         node:ast.Call)->Translated:
        """np.float32(x) / np.int32(x): a plain C cast to the matching device type."""
        if len(args) != 1:
            self._error(f"np.{name}() takes exactly 1 argument", node)
        code, source_t = args[0]
        if is_ptr(source_t) or is_vec(source_t) or is_image(source_t):
            self._error(f"np.{name}() cannot convert a {source_t}", node)
        target_t:object = DTYPE_MAP.get(name)
        if target_t is None:
            self._error(f"np.{name}() has no device equivalent", node)
        # A complex cast is not a C cast: the target and the source are structures, so the
        # conversion goes through the header's own widening and narrowing functions.
        target_complex:Optional[np.dtype] = complex_support.dtype_for_type_name(target_t)
        if target_complex is not None:
            source_complex:Optional[np.dtype] = self._complex_dtype_of(source_t)
            source_kind:str = self._complex_side(source_t, source_complex)
            if source_complex is not None:
                self._require_complex(target_complex)
                if source_kind == target_t or source_complex == target_complex:
                    return code, target_t
                self.complex_needs.update(complex_support.mixed_needs())
                converter:str = (f"{source_kind}_widen_{target_t}"
                                 if target_t == "c128" else f"{source_kind}_narrow_{target_t}")
                return f"{converter}({code})", target_t
            if not is_float(source_t) and not is_int(source_t):
                self._error(
                    f"np.{name}() cannot convert {source_t} to a complex value; convert a"
                    f" real or a complex number", node)
            # A real becomes the complex value with a zero imaginary part.
            self._require_complex(target_complex)
            element:str = "double" if target_t == "c128" else "float"
            return f"{target_t}_from_real(({element})({code}))", target_t
        return f"(({self._device_type(target_t)})({code}))", target_t

    def _unsupported_vec_reason(self, target:object)->Optional[str]:
        """Why a `pygf` class has no device vector type, or None when it has one.

        The alias classes with a device counterpart resolve through `_vec_type_name`. The
        rest are fp16 vectors (which the backends do not take) and matrices (which device
        code has no type for), and they deserve a message that says so instead of the
        generic "not a function that can be compiled".
        """
        if not isinstance(target, type):
            return None
        if not str(getattr(target, "__module__", "")).startswith("pygf"):
            return None
        name:str = target.__name__
        if name.startswith(("matrix", "frame")):
            return (f"{name} is a matrix type: device code has no matrix type, so build the "
                    f"components explicitly")
        for base in target.__mro__:
            if base.__name__.startswith("half"):
                return (f"{name} holds 16-bit floats: the device backends do not support "
                        f"fp16 vectors, use a float or double vector")
        return None

    def _vec_type_name(self, name:str)->Optional[str]:
        """The device vector type a constructor name stands for, or None.

        A vector type is spelled the same way on both sides, but the host side has more
        names for one type: `pygf` gives colour/normal/point/vector aliases that all subclass
        the same vector (`color3f` -> `float3`). Walking the class's bases finds the device
        name, so `color3f(r, g, b)` compiles into the `float3` constructor the body means.
        """
        if name in _vec_registry():
            return name
        target:object = self.resolve_global(name)
        if not isinstance(target, type):
            return None
        for base in target.__mro__:
            if base.__name__ in _vec_registry():
                return base.__name__
        return None

    def _structure_from_construction(self, target:type)->Optional[np.dtype]:
        """The structure a value built by this class call should become, or None.

        The members come from the body -- what it reads off the value and which methods it
        hands it to -- and the types from the constructor's own arguments. That work happens
        once, before the translator runs, in the same scan that handles an object argument;
        this only looks the answer up.
        """
        from . import objects  # noqa: PLC0415

        dtype:Optional[np.dtype] = objects.constructed_structure(target)
        if dtype is None:
            return None
        return dtype

    def _is_structure_class(self, candidate:object)->bool:
        """Whether a global name is a `ctypes.Structure` subclass with a device layout."""
        from . import structs  # noqa: PLC0415

        return structs.structure_dtype_of(candidate) is not None

    def _visit_structure_constructor_call(self, node:ast.Call,
                                          candidate:object)->Translated:
        """A structure constructor called with keywords: `Ray(origin=p, direction=d)`.

        A member name is a keyword, exactly as python's ctypes spelling addresses one, and
        the positional arguments come first in declaration order -- the same shape
        `ctypes.Structure.__init__` accepts.

        ``candidate`` is either a declared class or the dtype a plain class was synthesized
        into. For the latter the positional arguments are matched through the class's
        constructor parameters, not the field order: the fields are sorted by name, so the
        first argument is not necessarily the first field.
        """
        from . import objects, structs  # noqa: PLC0415

        parameters:List[str] = []
        if isinstance(candidate, np.dtype):
            # A plain class the scan synthesized: the dtype IS the layout -- there is no
            # ctypes.Structure behind it to derive one from.
            dtype:Optional[np.dtype] = candidate
            owner:Optional[type] = structs.declared_class(dtype)
            if owner is not None:
                parameters = objects.constructed_parameters(owner)
        else:
            dtype = structs.structure_dtype_of(candidate)
            parameters = list(dtype.names or ()) if dtype is not None else []
        if dtype is None:
            self._error(f"'{ast.unparse(node.func)}' is not a structure type", node)
        order:List[str] = list(dtype.names or ())
        args:List[Translated] = [self._visit_expression(a) for a in node.args]
        supplied:Dict[str, Translated] = {}
        for position, value in enumerate(args):
            if position >= len(parameters):
                self._error(
                    f"{ast.unparse(node.func)}() takes at most {len(parameters)} positional "
                    f"arguments ({', '.join(parameters)}), got {position + 1}", node)
            supplied[parameters[position]] = value
        for keyword in node.keywords:
            if keyword.arg is None:
                self._error("`**kwargs` cannot construct a structure", node)
            if keyword.arg not in order:
                self._error(
                    f"{ast.unparse(node.func)} has no member '{keyword.arg}' "
                    f"(available: {', '.join(order)})", keyword.value)
            if keyword.arg in supplied:
                self._error(
                    f"member '{keyword.arg}' supplied twice in {ast.unparse(node.func)}()",
                    keyword.value)
            supplied[keyword.arg] = self._visit_expression(keyword.value)
        # Hand the constructor the values keyed by member, so a member given as a keyword
        # does not shift the positional ones; omitted members are zeroed there.
        return self._emit_structure_constructor(structs.register(dtype), dtype,
                                                args, node, supplied=supplied)

    def _emit_structure_constructor(self, name:str, dtype:np.dtype,
                                    args:List[Translated], node:ast.expr,
                                    supplied:Optional[Dict[str, Translated]] = None
                                    )->Translated:
        """`Ray(origin, direction)` as a compound literal keyed by member name.

        A `ctypes.Structure` class describes a layout, so a call to it is a structure *value*
        -- what `np.zeros(1, dtype=Ray)` builds, without the array around it. The arguments
        are positional in declaration order, which is also the order python's
        `ctypes.Structure.__init__` assigns them.
        """
        from . import structs

        if dtype not in self.private_structs:
            # The typedef has to be emitted even when the structure never appears in the
            # signature: a body that builds one locally still needs the type declared.
            self.private_structs.append(dtype)
        members:Dict[str, object] = structs.member_types(dtype, self.backend)
        order:List[str] = list(dtype.names or ())
        if supplied is None:
            supplied = {}
            # For a *declared* structure the layout is fixed, so too many arguments is the
            # clear error it always was. For a structure a body's plain class was synthesized
            # into, the members are what the body reads -- `Ray(origin, direction)` builds
            # two, and a body reading only `origin` describes a one-member structure -- so
            # the unread arguments are dropped, the way python itself leaves an attribute
            # nobody mentions unassigned.
            parameters:List[str] = order
            if dtype in self.plain_class_dtypes:
                owner:Optional[type] = structs.declared_class(dtype)
                if owner is not None:
                    from . import objects  # noqa: PLC0415

                    parameters = objects.constructed_parameters(owner) or order
                if len(args) > len(parameters):
                    args = args[:len(parameters)]
            elif len(args) > len(order):
                self._error(
                    f"{name}() takes at most {len(order)} arguments "
                    f"({', '.join(order)}), got {len(args)}", node)
            for position, (code, t) in enumerate(args):
                member:str = parameters[position]
                supplied[member] = (code, t)

        parts:List[str] = []
        positional_parts:List[str] = []
        for member in order:
            if member not in supplied:
                # A missing member is zero, which is what a C aggregate does with an
                # unmentioned field; it keeps `Ray(origin)` working. CUDA/HIP are C++ and
                # cannot skip a field in a positional initialiser, so they spell the zero
                # out; OpenCL uses designated initialisers and simply omits it.
                positional_parts.append(self._zero_value(members[member]))
                continue
            code, t = supplied[member]
            converted:str = self._structure_member_value(name, member, members[member],
                                                         code, t, node)
            parts.append(f".{member} = {converted}")
            positional_parts.append(converted)

        if self._opencl_syntax:
            # OpenCL C is C99, so a designated initialiser names the members it sets.
            return f"(({name}){{{', '.join(parts)}}})", name
        # CUDA and HIP compile as C++, where designated initialisers arrived only in C++20
        # (NVRTC is older); a braced aggregate takes its members in declaration order.
        return f"({name}{{{', '.join(positional_parts)}}})", name

    def _zero_value(self, member_type:object)->str:
        """The zero of a member's device type, for an omitted constructor argument."""
        from . import structs

        if is_ptr(member_type):
            return "0"
        if is_vec(member_type):
            return self._vec_maker(member_type,
                                   ", ".join("0" for _ in _vec_field_names(member_type)))
        if isinstance(member_type, str) and structs.dtype_for(member_type) is not None:
            dtype:object = structs.dtype_for(member_type)
            inner:str = ", ".join(self._zero_value(structs.member_types(dtype, self.backend)[m])
                                  for m in dtype.names or ())
            return f"({member_type}{{{inner}}})"
        return f"(({self._device_type(member_type)})(0))"

    def _structure_member_value(self, struct_name:str, member:str, member_type:object,
                                code:str, t:object, node:ast.expr)->str:
        """Coerce one constructor argument to the member's own device type."""
        from . import structs

        if is_ptr(member_type):
            self._error(
                f"{struct_name}(...): member '{member}' is an array; assign its elements "
                f"after construction instead", node)
        if is_vec(member_type):
            if t != member_type:
                self._error(
                    f"{struct_name}(...): member '{member}' is a {member_type}, got {t}", node)
            return code
        if isinstance(member_type, str) and structs.dtype_for(member_type) is not None:
            if t != member_type:
                self._error(
                    f"{struct_name}(...): member '{member}' needs a structure value, "
                    f"got {t}", node)
            return code
        if t == member_type:
            return code
        if member_type == "uchar" and self._member_is_bool(struct_name, member):
            # A bool member is stored as a byte, so assigning a float to it truncates --
            # `front_face = 0.99` became 0, and the branch reading it took the wrong way,
            # which showed as a missing colour channel. The conversion has to be boolean.
            return f"(({code}) != 0)"
        return f"(({self._device_type(member_type)})({code}))"

    def _member_is_bool(self, struct_name:str, member:str)->bool:
        """Whether a structure member's layout type is a bool."""
        from . import structs  # noqa: PLC0415

        dtype:object = structs.dtype_for(struct_name)
        if dtype is None or member not in (dtype.names or ()):
            return False
        return np.dtype(dtype.fields[member][0]).name == "bool"

    def _emit_vec_constructor(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        cls:object = _vec_registry()[name]
        # A 3-component vector is padded to 16 bytes for the device ABI, so its ctypes struct
        # carries a trailing `_` field that is not a component: `float3(a, b, c)` takes
        # three arguments, not four. Count the real components, not the storage fields.
        components:List[Translated] = [f for f in cls._fields_ if f[0] != "_"]
        expected:int = len(components)
        field_type:str = _C_FIELD_TYPES[(name, components[0][0])]
        # `pygf` accepts three forms, and the device call has to mean the same thing:
        # no arguments (all zeros), one argument (broadcast), or one per component.
        if len(args) == 0:
            codes:List[str] = [f"(({field_type})(0))"] * expected
        elif len(args) == 1:
            code, t = args[0]
            if is_ptr(t) or is_vec(t):
                self._error(f"{name}() components must be scalars", node)
            codes = [code if t == field_type else f"(({field_type})({code}))"] * expected
        elif len(args) == expected:
            codes = []
            for i, (code, t) in enumerate(args):
                if is_ptr(t) or is_vec(t):
                    self._error(f"{name}() components must be scalars", node)
                component_type:str = _C_FIELD_TYPES[(name, components[i][0])]
                codes.append(code if t == component_type else f"(({component_type})({code}))")
        else:
            self._error(
                f"{name}() takes no arguments, one scalar to broadcast, or {expected} "
                f"component the vector has; {len(args)} given", node)
        if self._opencl_syntax:
            return f"({name})({', '.join(codes)})", name
        return f"make_{name}({', '.join(codes)})", name

    # ---------- gpu.* device builtins ----------

    def _emit_complex_builtin(self, name:str, args:List[Translated],
                              node:ast.Call)->Translated:
        """A builtin call with a complex argument: the header's function, or a refusal.

        Both builtin dispatchers reach here first -- `_emit_gpu_builtin` for the package's own
        vocabulary and `_emit_builtin` for python's builtins -- because their tables are keyed
        by *type* and a complex value has no entry in any of them. That is what emitted
        `sqrtf(z)`, `min((int)(z), (int)(w))`, `fminf(fmaxf(z, 0.0f), 1.0f)` and a `round(z)`
        that returned the value unchanged: a call the compiler could not type, or one that
        silently did nothing.

        The split follows numpy, which is the authority the interpreter already runs: a name
        numpy defines for a complex value is answered from the complex header, and a name it
        refuses is refused here by name. Where the two sides disagree, the interpreter wins --
        which is why `round(z)` is refused (python's builtin has no complex form) while
        `np.round(z)` is answered.
        """
        for _code, t in args:
            if is_ptr(t) or is_image(t) or is_vec(t):
                self._error(
                    f"{name}() mixes a complex value with a {t}: a complex value is a scalar, "
                    f"and the device has no complex vector or array element", node)

        dtypes:List[Optional[np.dtype]] = [self._complex_dtype_of(t) for _, t in args]
        sides:List[Optional[str]] = [None if dtype is None else self._complex_side(t, dtype)
                                     for (_, t), dtype in zip(args, dtypes)]
        # A mixed pair (`min(c64, c128)`) is computed at the wider width, which is the rule
        # the arithmetic follows: nothing is ever rounded down.
        target:str = "c128" if "c128" in sides else "c64"
        element:str = "double" if target == "c128" else "float"
        codes:List[str] = []
        for (code, _t), dtype, side in zip(args, dtypes, sides):
            if dtype is None:
                # A real operand: `pow(z, 2)` and `mix(z, w, 0.5)` both have one, and it is
                # built into the complex value it stands for.
                self._require_complex(np.dtype(np.complex128 if target == "c128"
                                               else np.complex64))
                codes.append(f"{target}_from_real(({element})({code}))")
                continue
            self._require_complex(dtype)
            codes.append(code if side == target
                         else self._convert_complex(code, cast(str, side), target))

        if name in COMPLEX_BUILTIN_FUNCTIONS:
            function:str = COMPLEX_BUILTIN_FUNCTIONS[name]
            expected:int = 2 if function == "pow" else 1
            if len(args) != expected:
                self._error(f"{name}() takes exactly {expected} argument(s)", node)
            return f"{target}_{function}({', '.join(codes)})", target

        if name in COMPLEX_BUILTIN_REAL_RESULTS:
            if len(args) != 1:
                self._error(f"{name}() takes exactly 1 argument", node)
            return f"{target}_{COMPLEX_BUILTIN_REAL_RESULTS[name]}({codes[0]})", element

        if name in ("min", "max", "fmin", "fmax"):
            if len(args) != 2:
                self._error(f"{name}() takes exactly 2 arguments", node)
            # `min`/`max` propagate a NaN operand (`np.minimum`'s rule, and python's), while
            # `fmin`/`fmax` ignore it (`np.fmin`'s), so the two families are two functions in
            # the header rather than one with a flag.
            if name in ("fmin", "fmax"):
                pick:str = name
            else:
                pick = "min" if name == "min" else "max"
            return f"{target}_{pick}({codes[0]}, {codes[1]})", target

        if name in ("clamp", "saturate"):
            zero:str = f"{target}_from_real(({element})(0))"
            one:str = f"{target}_from_real(({element})(1))"
            if name == "saturate":
                bounds:List[str] = [zero, one]
            else:
                if len(args) != 3:
                    self._error("clamp(a, lo, hi) takes exactly 3 arguments", node)
                bounds = codes[1:3]
            return f"{target}_clamp({codes[0]}, {bounds[0]}, {bounds[1]})", target

        if name == "mix":
            if len(args) != 3:
                self._error("mix(a, b, t) takes exactly 3 arguments", node)
            # `x * (1 - a) + y * a`, the body pygf's `mix` computes: the blend factor is a
            # complex value here like any other operand.
            one_code:str = f"{target}_from_real(({element})(1))"
            factor:str = f"{target}_sub({one_code}, {codes[2]})"
            return (f"{target}_add({target}_mul({codes[0]}, {factor}),"
                    f" {target}_mul({codes[1]}, {codes[2]}))"), target

        if name == "mad":
            if len(args) != 3:
                self._error("mad(a, b, c) takes exactly 3 arguments", node)
            return f"{target}_add({target}_mul({codes[0]}, {codes[1]}), {codes[2]})", target

        reason:Optional[str] = COMPLEX_BUILTIN_REFUSALS.get(name)
        functions:str = ", ".join(sorted(
            set(COMPLEX_BUILTIN_FUNCTIONS) | set(COMPLEX_BUILTIN_REAL_RESULTS)
            | {"min", "max", "clamp", "mix", "mad"}))
        if reason is not None:
            self._error(f"{name}() has no complex meaning ({reason})", node)
        self._error(
            f"{name}() is not defined for a complex value: the complex operations are "
            f"{functions}", node)

    def _emit_gpu_builtin(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        # A complex argument is settled before any table is consulted: those tables are keyed
        # by type, and a complex value has no entry in them (see _emit_complex_builtin).
        if any(self._complex_dtype_of(t) is not None for _, t in args):
            return self._emit_complex_builtin(name, args, node)

        if name in INDEX_BUILTINS:
            if len(args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, int):
                self._error(f"gpu.{name}(dim) requires a constant integer dimension (0, 1 or 2)", node)
            if name in ("get_global_id", "get_global_size", "global_id"):
                self.uses_global_id = True
            return emit_index(name, node.args[0].value, self.backend), "int"

        if name == "barrier":
            if args:
                self._error("gpu.barrier() takes no arguments", node)
            if self._metal_syntax:
                from .metal_codegen import barrier
                return barrier(), "void"
            return emit_barrier(self.backend), "void"

        if name in ATOMIC_BUILTINS:
            # Every atomic is (array, index, value) except atomic_cas, which also takes the
            # value to compare against: (array, index, compare, value).
            expected:int = 4 if name == "atomic_cas" else 3
            if len(args) != expected:
                if name == "atomic_cas":
                    self._error(
                        "gpu.atomic_cas(array, index, compare, value) takes exactly 4 "
                        "arguments", node)
                self._error(f"gpu.{name}(array, index, value) takes exactly 3 arguments", node)
            (arr_code, arr_t), (idx_code, idx_t) = args[0], args[1]
            if isinstance(arr_t, tuple) and arr_t and arr_t[0] == SHARED_SLICE:
                # `atomic_add(bins[blockIdx], i, 1)`: the group's slice is the array the atomic
                # indexes into, and any offset already consumed is folded into the index.
                pending:str = arr_t[3] if len(arr_t) > 3 else ""
                if pending:
                    idx_code = f"({pending}) + ({idx_code})"
                arr_code, arr_t = arr_code, (PTR, arr_t[1])
            val_code, val_t = args[-1]
            compare_code:Optional[str] = (args[2][0] if name == "atomic_cas" else None)
            if not is_ptr(arr_t):
                shared:Optional[Dict[str, object]] = (
                    self.outer_vars.get(node.args[0].id)
                    if isinstance(node.args[0], ast.Name) else None)
                if shared is not None and shared["writable"]:
                    self._error(
                        f"{node.args[0].id} is a single shared value, not an array: update "
                        f"it with `{node.args[0].id} += value` (or -=, &=, |=, ^=), which "
                        f"the translator turns into the device atomic operation, and the "
                        f"result is written back to the enclosing scope after the call. "
                        f"{name}() itself needs an array to index", node)  # type: ignore[union-attr]
                self._error(f"{name}() first argument must be an array", node)
            if isinstance(node.args[0], ast.Name):
                # atomic ops write memory via pointer casts, which static analysis cannot recognize;
                # record the target parameter so PyKernel forcibly marks it writable after compilation
                self.atomic_targets.add(node.args[0].id)
            if arr_t[1] not in self.ATOMIC_ELEMENT_TYPES:
                self._error(
                    f"gpu.{name}() supports int32/uint32/int64/uint64 arrays, got "
                    f"{arr_t[1]}; small integer types (int8/int16) have no device atomic "
                    f"operation on either backend, so use int32 there", node)
            if not is_int(idx_t) or not is_int(val_t):
                self._error(f"gpu.{name}() index and value must be integers", node)
            if name == "atomic_cas" and not is_int(args[2][1]):
                self._error("gpu.atomic_cas() compare value must be an integer", node)
            return emit_atomic(name, arr_code, idx_code, val_code, arr_t[1], self.backend,
                               compare_code,
                               is_local=self._is_local_buffer(node.args[0])), arr_t[1]

        if (name in GENERIC_BUILTINS
                or name in SHARED_MATH_BUILTINS
                or name in SHARED_MATH_TEMPLATES
                or name in CUDA_ONLY_BUILTINS
                or name in CUDA_ONLY_CONVERSIONS):
            return self._emit_generic_builtin(name, args, node)

        self._error(f"unknown gpu builtin: gpu.{name}", node)

    def _emit_cuda_only_builtin(self, name:str, args:List[Translated],
                                node:ast.Call)->Translated:
        """A device intrinsic only the CUDA and HIP backends have.

        The fast approximations, the correctly-rounded forms and the special functions
        CUDA adds have no OpenCL counterpart, so a kernel that uses one is a CUDA or a HIP
        kernel. On OpenCL the call is refused *by name* -- "undeclared identifier" from the
        compiler would not say which of the two the user should reach for.
        """
        if self.launch_backend == "opencl":
            self._error(
                f"gpu.{name}() is a CUDA device intrinsic that the OpenCL backend has no "
                f"counterpart for: the OpenCL kernel language has no such function, so "
                f"this kernel cannot run there. Either specialize the kernel for CUDA or "
                f"HIP, or use one of the functions the shared vocabulary lists "
                f"(see pygpgpu.kernel.builtins)", node)

        # A backend whose tables carry no spelling for this name has no counterpart
        # for it -- Metal's vocabulary is its own, and it has neither the `__expf`
        # family nor the `fast::` names under these spellings. Refusing by name, and
        # naming the backends that do have it, is the whole point of the tables: the
        # device's own "undeclared identifier" would not say where to go instead.
        table:Optional[Dict[str, str]] = (CUDA_ONLY_CONVERSIONS.get(name)
                                          or CUDA_ONLY_BUILTINS.get(name))
        if table is not None and self.launch_backend not in table:
            self._error(
                f"gpu.{name}() is a CUDA device intrinsic that the "
                f"{self.launch_backend} backend has no counterpart for; run this kernel "
                f"on one of {', '.join(sorted(table))}, or compute it from the "
                f"functions all backends have (see pygpgpu.kernel.builtins)", node)

        if name in CUDA_ONLY_CONVERSIONS:
            # A conversion answers an int, so the result type is the int one.
            conversion:str = CUDA_ONLY_CONVERSIONS[name][self.launch_backend]
            expected:int = (1 if name.startswith("float_to_int")
                            else 2 if name.startswith("double_to_int") else 1)
            if len(args) != expected:
                self._error(f"gpu.{name}() takes exactly {expected} argument", node)
            code, t = args[0]
            if is_vec(t):
                self._error(
                    f"gpu.{name}() takes a scalar: CUDA has no vector overload for the "
                    f"conversion, so convert the components one at a time", node)
            # The conversion's own name says what it converts FROM, so the argument is
            # cast to that type first: `__float2int_rn` takes a float, and a double
            # argument would pick the wrong overload.
            source_type:str = ("float" if "float_to_int" in name
                               else "double" if "double_to_int" in name else "int")
            return f"({conversion}(({source_type})({code})))", "int"

        device_name:str = CUDA_ONLY_BUILTINS[name][self.launch_backend]
        # HIP spells the round-toward-zero forms with a device helper rather than an
        # intrinsic (it has no such intrinsic), so the definition goes into the source
        # before the kernel that calls it.
        helper:Optional[str] = self._round_toward_zero_helper(device_name)
        if len(args) not in (1, 2, 3):
            self._error(f"gpu.{name}() takes 1 to 3 arguments", node)
        arg_codes:List[str] = [code for code, _ in args]
        for code, t in args:
            if is_vec(t):
                self._error(
                    f"gpu.{name}() takes a scalar: CUDA spells this intrinsic for scalars "
                    f"only, so apply it per component", node)
        # The argument's type decides the overload: `__expf` takes a float, and a double
        # argument would pick the (different, less accurate) double form.
        argument_type:str = "float"
        for _, t in args:
            if t == "double":
                argument_type = "double"
        if argument_type == "double" and helper is None:
            # The double form of the fast intrinsics is spelled without the `f`.
            device_name = device_name[:-1] if device_name.endswith("f") else device_name
        cast_arguments:List[str] = [f"({argument_type})({code})" for code in arg_codes]
        return f"{device_name}({', '.join(cast_arguments)})", argument_type

    def _round_toward_zero_helper(self, device_name:str)->Optional[str]:
        """The helper a round-toward-zero call needs, or None for an ordinary symbol.

        The helper is emitted once for the whole source, the first time a call needs it.
        """
        if device_name not in HIP_ROUND_TOWARD_ZERO_HELPERS.values():
            return None
        key:str = next(key for key, value in HIP_ROUND_TOWARD_ZERO_HELPERS.items()
                       if value == device_name)
        source:str = HIP_ROUND_TOWARD_ZERO_SOURCES[key]
        if source not in self.device_helpers:
            self.device_helpers.append(source)
        return device_name

    def _emit_shared_math_builtin(self, name:str, args:List[Translated],
                                  node:ast.Call)->Translated:
        """A math function both backends have.

        OpenCL spells it under the plain name and CUDA under the `f`-suffixed one, except
        for the `native_*` family, which CUDA spells with the double-underscore
        intrinsics (that is where the "native" comes from: the device's own instruction).

        A vector argument is expanded per component, the way the ordinary math
        intrinsics are: the device has no vector overload for the scalar-only ones
        (``erf``, ``tgamma``, ...), and the ones that are component-wise (``fract``,
        ``sign``, ``fmin``) work per component as well.
        """
        if len(args) not in (1, 2, 3):
            self._error(f"gpu.{name}() takes 1 to 3 arguments", node)
        arg_codes:List[str] = [code for code, _ in args]
        vec_t:object = next((t for t, _ in args if is_vec(t)), None)
        # The functions that are component-wise: they take a vector and answer one.
        component_wise:bool = name in ("fract", "sign", "fmin", "fmax", "ldexp")
        for code, t in args:
            if is_vec(t) and not component_wise:
                self._error(
                    f"gpu.{name}() takes a scalar: the device has no vector overload for "
                    f"it, so apply it per component", node)
        result_t:object = args[0][1]
        # The functions that answer a different type than they take: the rounding
        # functions answer an int, and the conversions answer one too.
        if name in ("rint", "round", "trunc", "floor", "ceil"):
            result_t = "int"
        if vec_t is not None:
            # The component-wise ones: expand per component and rebuild the vector.
            components:List[str] = _vec_field_names(_vec_registry()[vec_t])
            terms:List[str] = []
            for component in components:
                piece_args:List[str] = [
                    (f"({self._component(code, t, component)})"
                     if is_vec(t) else code)
                    for code, t in args]
                terms.append(self._shared_math_call(name, piece_args, node))
            return self._vec_maker(vec_t, ", ".join(terms)), vec_t
        return self._shared_math_call(name, arg_codes, node), result_t

    def _shared_math_call(self, name:str, arg_codes:List[str],
                          node:ast.Call)->str:
        """One call to a shared math function, in the spelling this backend uses."""
        table:Dict[str, str] = (SHARED_MATH_TEMPLATES[name] if name in SHARED_MATH_TEMPLATES
                                else SHARED_MATH_BUILTINS[name])
        if self.launch_backend not in table:
            # The tables are keyed by backend precisely so a missing spelling cannot
            # be guessed: each one names the callee or the expression that backend
            # actually has. Say which backend and which function, instead of letting
            # a bare KeyError escape from inside the translation.
            self._error(
                f"{name}() has no spelling for the {self.launch_backend} backend; "
                f"run this kernel on one of {', '.join(sorted(table))}, "
                f"or compute it from arithmetic in the body", node)
        if name in SHARED_MATH_TEMPLATES:
            # The backends spell this one differently, so the template says which
            # spelling this backend uses.
            template:str = SHARED_MATH_TEMPLATES[name][self.launch_backend]
            return template.format(code=arg_codes[0])
        device_name:str = SHARED_MATH_BUILTINS[name][self.launch_backend]
        return f"{device_name}({', '.join(arg_codes)})"

    def _emit_generic_builtin(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        arg_codes:List[str] = [c for c, _ in args]
        arg_types:List[object] = [t for _, t in args]
        vec_t:object = next((t for t in arg_types if is_vec(t)), None)

        if name in CUDA_ONLY_BUILTINS or name in CUDA_ONLY_CONVERSIONS:
            return self._emit_cuda_only_builtin(name, args, node)

        if name in SHARED_MATH_BUILTINS or name in SHARED_MATH_TEMPLATES:
            return self._emit_shared_math_builtin(name, args, node)

        if name in ("rsqrt", "exp2"):
            if len(args) != 1:
                self._error(f"gpu.{name}() takes exactly 1 argument", node)
            code, t = args[0]
            if is_vec(t):
                if self._opencl_syntax:
                    return f"{name}({code})", t
                return self._vec_componentwise(name, code, t), t
            code = f"{GENERIC_BUILTINS[name][1]}({code})" if self.backend == "cuda" else f"{name}({code})"
            return code, t

        if name in MATH_BUILTINS:
            # A scalar math intrinsic. OpenCL has a native vector overload for every one of
            # them, so the call passes straight through; CUDA has only the `f`-suffixed
            # scalar spelling, so a vector argument is expanded per component here.
            if len(args) not in (1, 2):
                self._error(f"gpu.{name}() takes 1 or 2 arguments", node)
            cuda_name:str = MATH_BUILTINS[name][1]
            result_t:object = vec_t or args[0][1]
            if self._opencl_syntax:
                return f"{name}({', '.join(arg_codes)})", result_t
            if vec_t is not None:
                if len(args) == 1:
                    return self._vec_componentwise(cuda_name, arg_codes[0], vec_t), vec_t
                if is_vec(arg_types[0]):
                    if is_vec(arg_types[1]):
                        return self._vec_componentwise(cuda_name, arg_codes[0], vec_t,
                                                       right=arg_codes[1]), vec_t
                    return self._vec_componentwise(cuda_name, arg_codes[0], vec_t,
                                                   scalar=arg_codes[1]), vec_t
                # The vector is the second operand (`pow(2.0, v)`): expand on that side, and
                # keep the operand order, which a non-commutative pair depends on.
                components:List[str] = _vec_field_names(_vec_registry()[vec_t])
                terms:str = ", ".join(
                    f"{cuda_name}({arg_codes[0]}, {arg_codes[1]}.{c})" for c in components)
                return self._vec_maker(vec_t, terms), vec_t
            return f"{cuda_name}({', '.join(arg_codes)})", result_t

        if name == "clamp":
            if len(args) != 3:
                self._error("gpu.clamp(a, lo, hi) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self._opencl_syntax:
                    return f"clamp({', '.join(arg_codes)})", vec_t
                return self._emit_vec_clamp(arg_codes, arg_types, vec_t), vec_t
            if self._opencl_syntax:
                return f"clamp({', '.join(arg_codes)})", args[0][1]
            return f"fminf(fmaxf({arg_codes[0]}, {arg_codes[1]}), {arg_codes[2]})", args[0][1]

        if name == "mix":
            if len(args) != 3:
                self._error("gpu.mix(a, b, t) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self._opencl_syntax:
                    return f"mix({', '.join(arg_codes)})", vec_t
                return self._emit_vec_mix(arg_codes, arg_types, vec_t), vec_t
            a, b, t = arg_codes
            return f"(({a}) * (1.0f - ({t})) + ({b}) * ({t}))", args[0][1]

        if name == "mad":
            if len(args) != 3:
                self._error("gpu.mad(a, b, c) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self._opencl_syntax:
                    return f"mad({', '.join(arg_codes)})", vec_t
                return self._emit_vec_mad(arg_codes, arg_types, vec_t), vec_t
            a, b, c = arg_codes
            return f"((({a}) * ({b})) + ({c}))", args[0][1]

        if name == "saturate":
            if len(args) != 1:
                self._error("gpu.saturate(x) takes exactly 1 argument", node)
            code, t = args[0]
            if vec_t is not None:
                if self._opencl_syntax:
                    return f"clamp({code}, 0.0f, 1.0f)", vec_t
                return self._emit_vec_clamp([code], [t], vec_t), vec_t
            if self._opencl_syntax:
                return f"clamp({code}, 0.0f, 1.0f)", t
            return f"fminf(fmaxf({code}, 0.0f), 1.0f)", t

        if name == "step":
            if len(args) != 2:
                self._error("gpu.step(edge, x) takes exactly 2 arguments", node)
            (edge_code, edge_t), (x_code, x_t) = args
            if vec_t is not None and self.backend == "opencl":
                return f"step({edge_code}, {x_code})", vec_t
            if vec_t is not None:
                cls:object = _vec_registry()[vec_t]
                components:List[str] = _vec_field_names(cls)
                maker:str = f"make_{vec_t}"
                inner:str = ", ".join(
                    f"(({self._component(x_code, x_t, c)}) < ({self._component(edge_code, edge_t, c)})) ? 0.0f : 1.0f"
                    for c in components)
                return f"{maker}({inner})", vec_t
            return f"((({x_code}) < ({edge_code})) ? 0.0f : 1.0f)", "float"

        if name == "dot":
            if len(args) != 2:
                self._error("gpu.dot(a, b) takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            if not (is_vec(lt) and is_vec(rt)):
                # GLSL and OpenCL define dot for vectors only, and pygf -- the host
                # implementation of this same function -- refuses a scalar pair. Reducing it
                # to a multiply here would make the two disagree about what `dot` means, so
                # the scalar spelling is refused with the expression it stands for.
                self._error(
                    "dot() takes two vectors; for scalars write `a * b`. The device and the "
                    "host interpreter share one implementation of the builtins, so a scalar "
                    "form that only one side accepted would make a kernel mean different "
                    "things per backend", node)
            if lt != rt:
                self._error(f"gpu.dot() operands must have the same type, got {lt} and {rt}", node)
            if self._opencl_syntax:
                return f"dot({lcode}, {rcode})", "float"
            return self._emit_vec_dot(lcode, rcode, lt), "float"

        if name in ("cross", "normalize", "length", "distance"):
            return self._emit_vec_builtin(name, args, node)

        if name == "select":
            if len(args) != 3:
                self._error("gpu.select(a, b, cond) takes exactly 3 arguments", node)
            (a_code, a_t), (b_code, b_t), (c_code, c_t) = args
            if a_t != b_t:
                self._error(f"gpu.select() operands must have the same type, got {a_t} and {b_t}", node)
            if is_vec(a_t):
                if not is_vec(c_t):
                    self._error("gpu.select() on vectors requires a vector condition", node)
                # opencl's native select treats int conditions implementation-defined (NVIDIA uses the MSB):
                # (cond == 0) yields a 0/-1 mask (MSB unambiguous), select(b, a, mask) means
                # cond nonzero -> b, cond == 0 -> a
                if self._opencl_syntax:
                    int_vec:str = self._get_int_vec_type(a_t)
                    return f"select({b_code}, {a_code}, ({c_code} == ({int_vec})(0)))", a_t
                return self._emit_vec_select(a_code, b_code, c_code, a_t), a_t
            if self._opencl_syntax:
                return f"select({b_code}, {a_code}, ({c_code} == 0))", a_t
            return f"(({c_code}) ? ({b_code}) : ({a_code}))", a_t

        if name in ("isequal", "isnotequal", "isgreater", "isgreaterequal", "isless", "islessequal", "isordered", "isunordered"):
            if len(args) != 2:
                self._error(f"gpu.{name}(a, b) takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            if is_vec(lt) or is_vec(rt):
                if not (is_vec(lt) and is_vec(rt)) or lt != rt:
                    self._error(f"gpu.{name}() vector operands must have the same type", node)
                if self._opencl_syntax:
                    # opencl vector relation functions use -1 (all-1 bits) for true; & 1 normalizes it to 1
                    return f"({name}({lcode}, {rcode}) & 1)", self._get_int_vec_type(lt)
                return self._emit_vec_relation(name, lcode, rcode, lt), self._get_int_vec_type(lt)
            if self._opencl_syntax:
                return f"{name}({lcode}, {rcode})", "int"
            return self._emit_scalar_relation(name, lcode, rcode), "int"

        if name in ("any", "all"):
            if len(args) != 1 or not is_vec(args[0][1]):
                self._error(f"gpu.{name}(cond_vec) takes exactly 1 vector argument", node)
            code, t = args[0]
            if self._opencl_syntax:
                return f"{name}({code})", "int"
            cls:object = _vec_registry()[t]
            components:List[str] = _vec_field_names(cls)
            joiner:str = ("|" if name == "any" else "&")
            expr:str = f"({joiner.join(f'({code}.{c})' for c in components)})"
            return f"(({expr}) != 0)", "int"

        if name in ("abs_diff", "hadd", "rhadd", "mad24", "mul24"):
            expected:int = (3 if name == "mad24" else 2)
            if len(args) != expected:
                self._error(f"gpu.{name}() invalid argument count", node)
            if any(is_vec(t) for _, t in args):
                if not all(is_vec(t) for _, t in args) or len({t for _, t in args}) != 1:
                    self._error(f"gpu.{name}() vector operands must all be the same vector type", node)
                # generate the integer function family uniformly per-component: NVIDIA's opencl implementation
                # unsigned-izes the return type (abs_diff/hadd/rhadd/mul24/mad24), so calling it directly on int
                # vectors fails to compile or drifts the result type; per-component expansion matches the cuda path and host semantics exactly
                if name == "abs_diff":
                    return self._emit_vec_absdiff(arg_codes[0], arg_codes[1], vec_t), vec_t
                if name == "hadd":
                    return self._emit_vec_hadd(arg_codes[0], arg_codes[1], vec_t), vec_t
                if name == "rhadd":
                    return self._emit_vec_rhadd(arg_codes[0], arg_codes[1], vec_t), vec_t
                if name == "mul24":
                    return self._vec_componentwise("*", arg_codes[0], vec_t, right=arg_codes[1]), vec_t
                return self._emit_vec_mad(arg_codes, arg_types, vec_t), vec_t
            if not all(is_int(t) for _, t in args):
                self._error(f"gpu.{name}() requires integer arguments", node)
            codes:List[str] = arg_codes
            if self._opencl_syntax:
                return f"{name}({', '.join(codes)})", args[0][1]
            if name == "abs_diff":
                return f"abs(({codes[0]}) - ({codes[1]}))", args[0][1]
            if name == "hadd":
                a, b = codes
                return f"((({a}) & ({b})) + ((({a}) ^ ({b})) >> 1))", args[0][1]
            if name == "rhadd":
                a, b = codes
                return f"((({a}) | ({b})) - ((({a}) ^ ({b})) >> 1))", args[0][1]
            if name == "mul24":
                return f"(({codes[0]}) * ({codes[1]}))", args[0][1]
            a, b, c = codes
            return f"((({a}) * ({b})) + ({c}))", args[0][1]

        self._error(f"unknown gpu builtin: gpu.{name}", node)

    def _is_scalar(self, t:object)->bool:
        return isinstance(t, str) and not is_vec(t)

    def _component(self, code:str, t:object, c:str)->str:
        return f"{code}.{c}" if is_vec(t) else code

    def _emit_vec_clamp(self, codes:List[str], types:List[object], vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"fminf(fmaxf({self._component(codes[0], types[0], c)}, {self._component(codes[1], types[1], c)}), {self._component(codes[2], types[2], c)})"
            for c in components)
        return f"{maker}({inner})"

    def _emit_vec_mix(self, codes:List[str], types:List[object], vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({self._component(codes[0], types[0], c)}) * (1.0f - ({self._component(codes[2], types[2], c)})) + ({self._component(codes[1], types[1], c)}) * ({self._component(codes[2], types[2], c)}))"
            for c in components)
        return f"{maker}({inner})"

    def _emit_vec_swizzle(self, code:str, vec:str, swizzle:str, node:ast.Attribute)->Translated:
        """Read a swizzle: opencl uses native v.xy; cuda generates per-component make_float2(v.x, v.y)."""
        components:List[Translated] = _vec_components(vec)
        valid:Set[str] = {f_name for f_name, _ in components}
        chars:Set[str] = set(swizzle)
        if not chars <= valid:
            self._error(f"vector type {vec} has no components '{swizzle}' "
                        f"(available: {''.join(sorted(valid))})", node)

        # Result type = vector with the same element type and the component count (.zyx of float4 -> float3, .xy -> float2)
        result_vec:str = vec
        if len(swizzle) != len(components):
            elem_ctype:str = _C_FIELD_TYPES[(vec, components[0][0])]
            candidate:str = f"{elem_ctype}{len(swizzle)}"
            if candidate in _vec_registry():
                result_vec = candidate
            else:
                self._error(f"no vector type '{candidate}' for swizzle '.{swizzle}'", node)

        if self._opencl_syntax:
            return f"({code}.{swizzle})", result_vec
        maker:str = f"make_{result_vec}"
        inner:str = ", ".join(f"({code}.{c})" for c in swizzle)
        return f"{maker}({inner})", result_vec

    def _get_int_vec_type(self, vec:str)->str:
        """Result type of vector comparisons (same-shape integer vector, same semantics as genVec._compare_op)."""
        prefix:str = "".join(c for c in vec if not c.isdigit())
        if vec[-1].isdigit():
            return "int" + (vec[len(prefix):] if prefix else "")
        return "int" + vec

    def _vec_componentwise_compare(self, op:str, left:str, vec:str, right:str)->str:
        """Vector comparison: normalize the result to per-component 0/1 (device comparisons use -1,
        all-1 bits, for true, unlike genVec._compare_op's int(True)=1; here it is unified to 0/1)."""
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        int_vec:str = self._get_int_vec_type(vec)
        terms:List[str] = [f"(({left}.{c} {op} {right}.{c}) ? 1 : 0)" for c in components]
        inner:str = ", ".join(terms)
        if self._opencl_syntax:
            return f"({int_vec})({inner})"
        return f"make_{int_vec}({inner})"

    def _emit_vec_mad(self, codes:List[str], types:List[object], vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({self._component(codes[0], types[0], c)}) * ({self._component(codes[1], types[1], c)})) + ({self._component(codes[2], types[2], c)})"
            for c in components)
        return f"{maker}({inner})"

    def _emit_vec_select(self, a_code:str, b_code:str, c_code:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({c_code}.{c}) ? ({b_code}.{c}) : ({a_code}.{c}))" for c in components)
        return f"{maker}({inner})"

    def _emit_vec_relation(self, name:str, lcode:str, rcode:str, vec:str)->str:
        op:str = {"isequal": "==", "isnotequal": "!=", "isgreater": ">",
                  "isgreaterequal": ">=", "isless": "<", "islessequal": "<="}[name]
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        int_vec:str = self._get_int_vec_type(vec)
        maker:str = f"make_{int_vec}"
        if name == "isordered":
            inner:str = ", ".join(f"((int)((!isnan({lcode}.{c})) && (!isnan({rcode}.{c}))))" for c in components)
        elif name == "isunordered":
            inner:str = ", ".join(f"((int)(isnan({lcode}.{c}) || isnan({rcode}.{c})))" for c in components)
        else:
            inner:str = ", ".join(f"((int)(({lcode}.{c}) {op} ({rcode}.{c})))" for c in components)
        return f"{maker}({inner})"

    def _emit_scalar_relation(self, name:str, lcode:str, rcode:str)->str:
        op:str = {"isequal": "==", "isnotequal": "!=", "isgreater": ">",
                  "isgreaterequal": ">=", "isless": "<", "islessequal": "<="}[name]
        if name == "isordered":
            return f"((int)((!isnan({lcode})) && (!isnan({rcode}))))"
        if name == "isunordered":
            return f"((int)(isnan({lcode}) || isnan({rcode})))"
        return f"((int)(({lcode}) {op} ({rcode})))"

    def _vec_maker(self, vec:str, inner:str)->str:
        """Assemble the vector per backend: opencl (vec)(...) / cuda make_vec(...)."""
        if self._opencl_syntax:
            return f"({vec})({inner})"
        return f"make_{vec}({inner})"

    def _emit_vec_absdiff(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(f"abs(({lcode}.{c}) - ({rcode}.{c}))" for c in components)
        return self._vec_maker(vec, inner)

    def _emit_vec_hadd(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(
            f"((({lcode}.{c}) & ({rcode}.{c})) + ((({lcode}.{c}) ^ ({rcode}.{c})) >> 1))" for c in components)
        return self._vec_maker(vec, inner)

    def _emit_vec_rhadd(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(
            f"((({lcode}.{c}) | ({rcode}.{c})) - ((({lcode}.{c}) ^ ({rcode}.{c})) >> 1))" for c in components)
        return self._vec_maker(vec, inner)

    def _emit_vec_dot(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        terms:str = " + ".join(f"(({lcode}.{c}) * ({rcode}.{c}))" for c in components)
        return f"({terms})"

    def _emit_vec_builtin(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        opencl_name, _ = GENERIC_BUILTINS[name]
        expected_args:int = (1 if name in ("normalize", "length") else 2)
        if len(args) != expected_args:
            self._error(f"gpu.{name}() takes exactly {expected_args} arguments", node)

        vec_types:List[object] = [t for _, t in args if is_vec(t)]
        if not vec_types:
            self._error(f"gpu.{name}() requires vector arguments "
                        f"(python has no such functions; use fabs/abs for scalars)", node)
        vec:str = vec_types[0]
        if any(t != vec for t in vec_types):
            self._error(f"gpu.{name}() operands must have the same vector type", node)

        if self._opencl_syntax:
            return f"{opencl_name}({', '.join(c for c, _ in args)})", vec

        codes:List[str] = [c for c, _ in args]
        if name == "normalize":
            return self._emit_vec_normalize(codes[0], vec), vec
        if name == "length":
            return f"sqrtf({self._emit_vec_dot(codes[0], codes[0], vec)})", "float"
        if name == "distance":
            diff:str = self._vec_componentwise("-", codes[0], vec, right=codes[1])
            return f"sqrtf({self._emit_vec_dot(diff, diff, vec)})", "float"
        if name == "cross":
            return self._emit_vec_cross(codes[0], codes[1], vec), vec
        self._error(f"gpu.{name} is not supported", node)

    def _emit_vec_normalize(self, code:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        dot_expr:str = self._emit_vec_dot(code, code, vec)
        inner:str = ", ".join(f"(({code}.{c}) * rsqrtf({dot_expr}))" for c in components)
        return f"{maker}({inner})"

    def _emit_vec_cross(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        x, y, z = components[0], components[1], components[2]
        inner:str = (f"(({lcode}.{y}) * ({rcode}.{z}) - ({lcode}.{z}) * ({rcode}.{y})), "
                     f"(({lcode}.{z}) * ({rcode}.{x}) - ({lcode}.{x}) * ({rcode}.{z})), "
                     f"(({lcode}.{x}) * ({rcode}.{y}) - ({lcode}.{y}) * ({rcode}.{x}))")
        if len(components) == 4:
            inner += ", 0.0f"
        elif len(components) != 3:
            self._error(f"gpu.cross() requires a 3 or 4 component vector, got {vec}")
        return f"{maker}({inner})"

    def _emit_device_call(self, name:str, target:object, args:List[Translated], node:ast.Call,
                          self_param:str="",
                          receiver_as_argument:bool=False,
                          tuple_targets:Optional[List[str]]=None)->Translated:
        is_kernel_ref:bool = getattr(target, "is_pykernel", False)
        raw_func:Callable = (target.raw_func if is_kernel_ref else target)
        if is_kernel_ref:
            # A @kernel with a work size is a nested launch, handled in _visit_call before
            # this point; reaching here means it was called as if it were a helper.
            self._error(
                f"'{name}' is a @kernel function: from device code it is a *launch*, so give "
                f"it a work size -- `{name}[(total, local)](...)` -- rather than calling it "
                f"as a helper", node)
        if not isinstance(raw_func, types.FunctionType):
            # A builtin or a bound builtin method has no python source to translate, so it
            # cannot become a device function. Say which name it was rather than failing
            # inside inspect.getsource.
            self._error(
                f"'{name}' is not a python function this translator can compile into device "
                f"code (only functions defined in python source are supported)", node)

        # A helper that is a method of the kernel's own class keeps the same receiver:
        # its self.xxx references become extra _self_xxx parameters appended after the
        # caller's, and the call site passes them through in that order.
        #
        # A *value* method (`ray.at(t)`) is different: the receiver is already the call's
        # first argument, so it binds as the callee's own first positional parameter
        # (`def at(self, t)` is `(Ray* self, float t)`) and declares no `_self_` slot.
        self_types:Dict[str, object] = ({} if receiver_as_argument
                                     else self.self_attr_types if self_param else {})
        # A structure argument always travels as a pointer, so a local value is passed by
        # its address: the callee reads the members through `->` and a store lands in the
        # caller's own value, which is what a method's receiver needs.
        for index, (code, value_t) in enumerate(args):
            if (isinstance(value_t, str) and structs_dtype_for(value_t) is not None
                    and not is_vec(value_t)):
                # A compound literal has no address: `&((T{...}))` is not valid C, so a
                # value built straight into the call (`self.hit(ray, Interval(0, inf))`)
                # is stored in a local first. The temporary is emitted before this
                # statement, which is the same shape a named construction already takes.
                if code.startswith(f"({value_t}{{"):
                    temporary:str = f"_arg{self._next_temp()}"
                    self._emit(f"{value_t} {temporary} = {code};")
                    code = temporary
                args[index] = (f"&({code})", (PTR, value_t))
        call_codes:List[str] = [c for c, _ in args]
        if self_param:
            # The receiver is a method's first parameter, exactly as in the signature the
            # callee builds (see FunctionTranslator.__init__), so it is passed first. The
            # array attributes come after the caller's arguments, as their own parameters.
            if self.self_struct_members and not receiver_as_argument:
                call_codes.insert(0, self.self_param)
            call_codes.extend(f"_self_{attr}" for attr in self_types)
        # The caller's own arguments come first, then the receiver struct, then the array
        # attributes -- the same order the callee binds them in, and the same order the
        # call codes below follow.
        param_types:List[object] = ([t for _, t in args]
                                 + ([self.self_struct_type]
                                    if self_param and self.self_struct_members
                                    and not receiver_as_argument else [])
                                 + list(self_types.values()))

        sig_key:Tuple[Callable, Tuple[object, ...]] = (raw_func, tuple(param_types))
        entry:Optional[Dict[str, object]] = self.device_cache.get(sig_key)
        if entry is not None:
            if entry["status"] == "in_progress":
                self._error(f"recursive call detected: '{name}'", node)
            argument_codes:List[str] = (call_codes + list(entry.get("shape_codes") or [])
                                        + list(entry.get("outer_codes") or []))
            if entry.get("needs_prologue"):
                # The callee reads a work-item index, so it declared the six prologue
                # parameters; the caller passes its own (it has them, being a kernel or
                # another helper that received them the same way).
                argument_codes.extend(self._prologue_parameter_arguments())
            return (f"{entry['c_name']}({', '.join(argument_codes)})",
                    entry["return_type"])

        self.device_cache[sig_key] = {"status": "in_progress"}
        try:
            callee:FunctionTranslator = FunctionTranslator(
                raw_func, param_types, self.backend, _helper_globals(raw_func, self.resolve_global),
                self.device_cache,
                is_kernel=False, self_param=self_param, self_attr_types=self_types,
                self_owner=self.self_owner, launch_backend=self.launch_backend,
                self_struct_members=(self.self_struct_members
                                     if self_param and not receiver_as_argument
                                     else None),
                self_struct_type=(self.self_struct_type
                                  if self_param and not receiver_as_argument else None),
                receiver_as_argument=receiver_as_argument,
                # A device helper is translated *inside* the kernel's own translation unit, so
                # it has to make the same width choices the kernel does: a python complex
                # literal in a helper body follows the same `float=` mapping (see
                # `_python_complex_kind`). Without this the helper silently answered the
                # default while its caller was mapped to a double.
                type_map=self.type_map)
            callee.translate()
            c_name:str = self._device_func_name(raw_func, param_types)
            # A helper that indexes a multi-dimensional array needs that array's extents, and
            # they are parameters, not part of the pointer. They are appended to the helper's
            # signature and passed from the caller's own_ shape variables, which the kernel
            # already receives.
            shape_params:List[Tuple[str, int]] = list(callee.shapes_used)
            for array_name, dim in shape_params:
                if (array_name, dim) not in self.shapes_used:
                    self.shapes_used.append((array_name, dim))
            shape_codes:List[str] = [self._shape_var(array_name, dim)
                                     for array_name, dim in shape_params]
            # A helper reaches outer-scope names the same way the kernel does, so it gets
            # the same parameters: the call site forwards its own (see below).
            outer_codes:List[str] = []
            for outer_name, outer in callee.outer_vars.items():
                self._require_outer_parameter(outer_name, outer, node)
                outer_codes.append(self._outer_argument(outer_name, outer))
            entry = {
                "status": "done",
                "c_name": c_name,
                "return_type": callee.return_type,
                # A tuple return's element types: the call site appends one `T*`
                # destination per element and passes it after every other argument
                # (render_device appends the matching declarations).
                "tuple_return_types": list(callee.tuple_return_types or []),
                "needs_prologue": callee._needs_prologue_parameters(),
                "code": callee.render_device(c_name, shape_params),
                "shape_codes": shape_codes,
                "outer_codes": outer_codes,
                # Structure types this helper's signature or body mentions: the caller emits
                # their typedefs once, ahead of every device function.
                "structs": list(callee.private_structs),
                # The python file this function was written in: its device code goes to
                # that module's own source file (see pygpgpu.kernel.source_paths).
                "source_file": getattr(getattr(raw_func, "__code__", None),
                                        "co_filename", "") or "",
                # A helper's body may use quaternion arithmetic too; its generated functions
                # travel with the entry so the compile step can emit them once.
                "quaternion_helpers": dict(callee.quaternion_helpers),
                "quaternion_needs": set(callee.quaternion_needs),
                # The same for complex values: a helper's body may use them too.
                "complex_needs": set(callee.complex_needs),
            }
            self.device_cache[sig_key] = entry
        except Exception:
            self.device_cache.pop(sig_key, None)
            raise

        # Device code has no shared state or atomics-free write-back for images, so an
        # image a helper samples or stores into has to be declared by the kernel itself.
        if callee.required_image_samplers or callee.required_image_surfaces:
            self._error(
                f"'{name}' uses an image parameter from a device helper, which is not "
                f"supported: sample or store the image in the kernel body instead", node)

        argument_codes:List[str] = (call_codes + list(entry.get("shape_codes") or [])
                                    + list(entry.get("outer_codes") or []))
        if entry.get("needs_prologue"):
            # The callee declared Metal's six prologue parameters because its body reads
            # a work-item index; the caller has them (it is a kernel, or a helper that
            # received them the same way) and forwards them here.
            argument_codes.extend(self._prologue_parameter_arguments())
        if tuple_targets is not None:
            # A tuple return: the destinations the call site unpacks into, appended in
            # the helper's own order (see render_device, which declares them last).
            argument_codes.extend(tuple_targets)
        return (f"{entry['c_name']}({', '.join(argument_codes)})", entry["return_type"])

    def _require_outer_parameter(self, name:str, outer:Dict[str, object], node:ast.AST)->None:
        """Make sure the kernel carrying a helper's outer variable receives it too.

        The kernel's own scan only sees names its body mentions; a helper reached from it
        may reference more. The parameter is appended to the kernel's signature here, so
        the launch supplies it from the same scope (see PyKernel._compile).
        """
        if name in self.outer_vars:
            existing:Dict[str, object] = self.outer_vars[name]
            if existing["writable"] != outer["writable"] or existing["type"] != outer["type"]:
                self._error(
                    f"outer variable '{name}' is used with conflicting types by a helper: "
                    f"{existing['type']} here and {outer['type']} in the helper", node)
            return
        self.outer_vars[name] = dict(outer)
        if outer["writable"]:
            self.params.append((name, (PTR, outer["type"])))
        else:
            self.params.append((name, outer["type"]))

    def _outer_argument(self, name:str, outer:Dict[str, object])->str:
        """The call-site expression that forwards an outer variable to a helper."""
        return f"&{name}" if outer["writable"] else name

    def _emit_self_method_call(self, kind:str, target:object, attr:str,
                               args:List[Translated], node:ast.Call)->Translated:
        """Call self.<attr>(...) where the member is a method of the kernel's own class."""
        if getattr(target, "is_pykernel", False):
            self._error(
                f"self.{attr} is a @kernel method and cannot be called from device code; "
                f"move the shared logic into a plain method or function", node)
        return self._emit_device_call(attr, target, args, node,
                                      self_param=self.self_param if kind == "method" else "")

    def _why_not_callable(self, kind:str, name:str)->str:
        """Explain why a self.xxx member cannot be called, for the error message."""
        if kind == "classmethod":
            return "a classmethod (cls has no meaning on a device)"
        if kind == "kernel":
            return "a @kernel method (kernels are launched from the host, not called from device code)"
        return ("not a method or static method of the class; only those, and module-level "
                "functions, can be called from device code")

    def _device_func_name(self, func:Callable, types:List[object])->str:
        # A method of a class reads `Class_method`; a plain function keeps its own name.
        # The hash stays in the name because a helper monomorphizes per argument type
        # (`blend` over float and over float3 are two device functions), and two modules
        # can define the same helper name -- the hash covers both. Nothing is appended to
        # settle a collision: the hash is what keeps two helpers apart, so a numbered
        # suffix would only appear if the hash itself collided.
        base:str = func.__name__
        # A method is named after the class it is *defined on*, which `__qualname__` records:
        # `Ray.at` is `Ray_at`, never the receiving kernel's own class. A plain function has
        # no dot in its qualname, and a nested definition (`<locals>` in it) has no class.
        qualname:str = getattr(func, "__qualname__", "")
        if "." in qualname:
            owner_name:str = qualname.rsplit(".", 1)[0]
            if "<" not in owner_name:
                base = f"{owner_name}_{func.__name__}"
        module_file:str = getattr(getattr(func, "__code__", None), "co_filename", "")
        module_mark:str = os.path.basename(module_file) if module_file else ""
        signature:str = module_mark + '|' + type_sig(types)
        # The `int=`/`float=` mapping belongs in the hash for the same reason it is part of the
        # kernel's own C name: a python literal's width follows it, so one helper translated
        # under two mappings is two device functions -- with different bodies, and for a
        # complex value even a different return type. A source file carries every kernel that
        # was written in it, so without the mapping the two definitions would land on one name
        # and a C++ backend rejects them as "cannot overload functions distinguished by return
        # type alone".
        mapping_key:str = ",".join(f"{key}={self.type_map[key]}"
                                   for key in sorted(self.type_map))
        if mapping_key:
            signature += '|' + mapping_key
        return f"{base}_{short_hash(signature)}"

    def _device_qualifier(self)->str:
        """The storage qualifier a device function needs on this backend."""
        return "__device__ " if self.backend == "cuda" else ""

    def _emit_floored(self, operations:Tuple[str, str], operation:str,
                      left:str, ctype:str, right:str, right_t:object,
                      node:ast.AST)->Translated:
        """`left // right` or `left % right` with python's floor semantics, for integers.

        C's `/` truncates toward zero and its `%` takes the dividend's sign; python floors
        and takes the divisor's sign. They agree only when the remainder is zero or the
        operands have the same sign, so the generated code calls a small device function
        that applies the one-step correction (see `python_floor_source`).

        An **unsigned** width needs no correction -- its remainder is never negative -- and
        `(r < 0)` on it is always false anyway, which some compilers warn about, so the
        plain C operator is emitted there and the helper is not declared.

        The helper is declared once per (operation, width): a file that uses both `//` and
        `%` on `int` carries two functions, and a second kernel using `//` on `int` finds
        the definition already recorded.
        """
        device_ctype:str = self._device_type(ctype)
        helper_name:str = python_floor_name(operation, device_ctype)
        source:str = python_floor_source(operation, device_ctype, self.backend)
        if not source:
            # An unsigned width: the C operator already matches python.
            symbol:str = "/" if operation == "floor_div" else "%"
            return f"({left} {symbol} {right})", ctype
        if source not in self.device_helpers:
            self.device_helpers.append(source)
        return f"{helper_name}({left}, {right})", ctype

    def _needs_prologue_parameters(self)->bool:
        """Whether this device function must receive Metal's work-item parameters.

        On OpenCL and CUDA the index builtins are *device intrinsics* -- `get_global_id`
        and `threadIdx` are symbols the runtime provides to every function. Metal has no
        such intrinsic: the prelude defines those names as macros over attribute-decorated
        **kernel parameters**, which exist only in the entry point's own scope.

        A helper that calls one therefore expands to a reference to a parameter it does
        not have. The generated MSL compiles into `'_mtl_thread_position' was not
        declared in this scope`, which names a name the user never wrote. Passing the six
        parameters down is what makes the helper legal, and it is only needed on Metal.
        """
        # `uses_global_id` is the translator's existing record of "the body names a
        # work-item index"; it is set where the index builtins are emitted.
        return self._metal_syntax and bool(self.uses_global_id)

    def _prologue_parameter_declarations(self)->List[str]:
        """`uint3 _mtl_thread_position [[thread_position_in_grid]]`, one per axis."""
        from .metal_codegen import prologue_parameters
        return prologue_parameters()

    def _prologue_parameter_arguments(self)->List[str]:
        """The names a call site forwards for the six prologue parameters."""
        from .metal_codegen import PROLOGUE_PARAMETERS
        return [name for _t, name, _a in PROLOGUE_PARAMETERS]

    def render_device(self, c_name:str,
                      shape_params:Optional[List[Tuple[str, int]]]=None)->str:
        """Render a device function.

        ``shape_params`` are the (array, dimension) extents this helper indexes. They travel
        as extra int parameters, because a pointer carries no extents and the helper's
        multi-dimensional subscripting needs them.
        """
        params:List[str] = []
        outer_names:Set[str] = set(self.outer_vars) | set(self.host_values) | set(self.host_scalars)
        for name, t in self.params:
            if name in outer_names:
                continue
            if is_ptr(t):
                # Every structure parameter is a pointer, and one whose members the body
                # never writes is `const` -- the same rule the kernel signature follows.
                from . import structs as _structs_module  # noqa: PLC0415

                is_struct:bool = (_structs_module.dtype_for(t[1]) is not None
                                  if isinstance(t[1], str) else False)
                const:str = ("const " if is_struct and name not in self.member_writes
                             else "")
                params.append(f"{const}{self._device_type(t[1])}* {device_name(name)}")
            elif isinstance(t, str) and structs_dtype_for(t) is not None:
                # A structure value that reached a device parameter by value (a helper's
                # receiver before the pointer rule, or a caller's own spelling) travels as
                # a pointer too: one spelling for every structure.
                const = ("const " if name not in self.member_writes else "")
                params.append(f"{const}{self._device_type(t)}* {device_name(name)}")
            elif isinstance(t, tuple) and t and t[0] == SHARED_SLICE:
                # A `zeros_per_group` slice handed to a helper. It is a *pointer* into
                # the group's own storage, in the space that storage lives in -- Metal
                # spells that `threadgroup`, OpenCL `__local`, and CUDA/HIP need no
                # qualifier. Rendering the tuple itself would print the internal type
                # marker into the C source (`(('shared_slice', ...)) buf`), which no
                # compiler accepts.
                space:str = ""
                if self._metal_syntax:
                    space = "threadgroup "
                elif self._opencl_syntax:
                    space = "__local "
                element:str = self._device_type(t[1])
                params.append(f"{space}{element}* {device_name(name)}")
            elif isinstance(t, tuple) and len(t) == 3 and t[0] == IMAGE:
                # An image parameter reaching a helper. It is the backend's own image
                # type -- `__read_only image2d_t` on OpenCL, a texture/surface object on
                # CUDA -- so the same spelling the kernel signature uses applies here.
                # Falling through would print the internal marker (`('image', 4,
                # 'float')`) into the C source, which no compiler accepts.
                params.append(f"{self._image_param_type(name, t)} {device_name(name)}")
            else:
                params.append(f"{self._device_type(t)} {device_name(name)}")
        # A structure in the signature needs its typedef emitted too, and this helper's own
        # `private_structs` is what the caller collects.
        from . import structs

        for candidate in [t[1] if is_ptr(t) else t for _, t in self.params]:
            dtype:object = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in self.private_structs:
                self.private_structs.append(dtype)
        if isinstance(self.return_type, str):
            dtype = structs.dtype_for(self.return_type)
            if dtype is not None and dtype not in self.private_structs:
                self.private_structs.append(dtype)
        for array_name, dim in (shape_params or []):
            params.append(f"const int {self._shape_var(array_name, dim)}")
        # Outer-scope parameters come last, matching the call site (see _emit_device_call),
        # which forwards them after the helper's own arguments and shape extents.
        params.extend(self._outer_param_declarations(device=True))
        if self._needs_prologue_parameters():
            # Metal's work-item numbers are parameters, so a helper that reads one has to
            # receive them from its caller (see _needs_prologue_parameters).
            params.extend(self._prologue_parameter_declarations())
        # A tuple return's output parameters, after everything else (the call site
        # appends and passes them in the same order -- see _emit_device_call).
        for ordinal, element_t in enumerate(self.tuple_return_types or []):
            params.append(f"{self._device_type(element_t)}* _tuple_out{ordinal}")
        ret:str = (self.return_type or "void")
        header:str = f"{self._device_qualifier()}{self._device_type(ret)} {c_name}({', '.join(params)})"
        body:str = self._render_body()
        # No typedef block here: the caller emits every structure type once, ahead of all
        # the device functions (see PyKernel._compile). Repeating it per helper would define
        # the same struct twice, which C++ rejects.
        return f"{header}\n{{\n{body}\n}}"

    def _outer_param_declarations(self, device:bool=False)->List[str]:
        """C declarations for the parameters the host fills in at launch time.

        This covers both the outer-scope variables and the objects the kernel body
        creates: neither belongs to the python argument list, and both are appended to the
        device signature after everything the work size is derived from.
        """
        declarations:List[str] = []
        for name, outer in self.outer_vars.items():
            if is_ptr(outer["type"]):
                qualifier:str = "" if device else self._buffer_qualifier()
                declarations.append(f"{qualifier}{self._device_type(outer['type'][1])}* {name}")
            elif outer["writable"]:
                element:str = (self._return_param_type(outer["type"]) if device
                               else self._kernel_scalar_type(outer["type"]))
                qualifier = "" if device else self._buffer_qualifier()
                declarations.append(f"{qualifier}{element}* {name}")
            else:
                declarations.append(f"{self._kernel_scalar_type(outer['type'])} {name}")
        for name, entry in self.host_values.items():
            declarations.append(f"{self._host_value_declaration(name, entry, device)}")
        # The plain values a host-region statement bound: they are filled in at launch
        # time exactly like a creation, so they are declared with them. A scalar is a
        # plain parameter; an array the host built is a buffer, the same spelling a
        # creation of that shape gets.
        for name, t in self.host_scalars.items():
            if name in self.outer_vars:
                # The outer scope already declared it (an enclosing `nonlocal` slot): one
                # parameter, and it is the slot, not a second by-value copy.
                continue
            if is_ptr(t):
                qualifier = "" if device else self._buffer_qualifier()
                declarations.append(f"{qualifier}{self._device_type(t[1])}* {device_name(name)}")
            else:
                declarations.append(f"{self._kernel_scalar_type(t)} {device_name(name)}")
        return declarations

    def _host_value_declaration(self, name:str, entry:Dict[str, object], device:bool)->str:
        """The parameter declaration for one host-created object."""
        t:object = entry["type"]
        if is_ptr(t):
            qualifier:str = "" if device else self._buffer_qualifier()
            return f"{qualifier}{self._device_type(t[1])}* {device_name(name)}"
        return f"{self._kernel_scalar_type(t)} {device_name(name)}"

    def _device_type(self, t:str)->str:
        """The backend's spelling of a scalar type name.

        The translator's internal names follow OpenCL (uchar, ushort, uint, ulong). CUDA
        C does not define those four, so the CUDA path -- which HIP shares -- writes the
        full C form instead.
        """
        if self._opencl_syntax:
            return t
        return _CUDA_TYPE_NAMES.get(t, t)

    def _kernel_scalar_type(self, t:object)->str:
        """A kernel parameter's C type.

        `bool` is not portable as a kernel parameter: OpenCL C refuses a `_Bool` kernel
        argument outright, and the CUDA/HIP argument binding has no bool entry. It travels
        as a byte instead, which every backend accepts; the device reads 0/1.
        """
        if t == "bool":
            return self._device_type("uchar")
        return self._device_type(t)

    def parameter_declarations(self)->List[str]:
        """The kernel's parameter list, exactly as the generated signature spells it.

        Split out of `render_kernel` because the kernel's C name has to cover it: the name's
        hash is built from the *argument types* the caller passed, which do not include the
        extent parameters the translator derives, nor the outer-scope values, nor the shared
        buffers. A body whose derived signature changed therefore kept its old name, and the
        file then held two declarations of that name -- "more than one instance of overloaded
        function". Hashing the rendered list makes a changed signature a changed name.
        """
        # Outer-scope parameters are held in self.params so the host can index launch
        # arguments by name, but they are *declared* last: the host appends them after the
        # work-size/sampler/local-size arguments it derives (see PyKernel._invoke_inner).
        outer_names:Set[str] = set(self.outer_vars) | set(self.host_values) | set(self.host_scalars)
        params:List[str] = []
        for name, t in self.params:
            if name in outer_names:
                continue
            if is_ptr(t):
                # A structure parameter is a pointer, and one the body never writes is
                # `const`: it says the member reads are all the callee does, which is the
                # information a reader of the generated source needs.
                from . import structs as _structs_module  # noqa: PLC0415

                is_struct:bool = (_structs_module.dtype_for(t[1]) is not None
                                  if isinstance(t[1], str) else False)
                const:str = ("const " if is_struct and name not in self.member_writes
                             else "")
                qualifier = self._buffer_qualifier()
                params.append(
                    f"{qualifier}{const}{self._kernel_scalar_type(t[1])}* {device_name(name)}")
            elif is_image(t):
                params.append(f"{self._image_param_type(name, t)} {device_name(name)}")
            else:
                params.append(f"{self._kernel_scalar_type(t)} {device_name(name)}")
        for array_name, dim in self.shapes_used:
            params.append(f"const int {self._shape_var(array_name, dim)}")
        # OpenCL sampling takes the sampler as a kernel argument; one per sampled
        # image, right after that image. The host binds a matching sampler per image,
        # so the Python-visible signature stays (image) only. Metal has no sampler
        # argument at all -- a texture carries its sampler state -- so this stays
        # OpenCL's alone rather than following `_opencl_syntax`.
        if self.backend == "opencl":
            for image_name in sorted(self.required_image_samplers):
                params.append(f"const sampler_t {self._sampler_name(image_name)}")
        for lc_name, lc_type, _ in self.local_buffers:
            if self._metal_syntax:
                # MSL spells the work-group address space `threadgroup`, and a
                # `threadgroup` buffer is a kernel parameter there too.
                params.append(f"threadgroup {self._device_type(lc_type)}* {lc_name}")
            elif self._opencl_syntax:
                params.append(f"__local {self._device_type(lc_type)}* {lc_name}")

        # Outer-scope parameters after everything the launch derives the work size from,
        # matching the order the host appends them in (see PyKernel._invoke_inner).
        params.extend(self._outer_param_declarations(device=False))

        # Return slots come last, after every parameter the work size and samplers derive
        # from, so appending them at launch time cannot disturb those positions.
        params.extend(self.render_return_params())

        # Metal's work-item numbers arrive as attribute-decorated parameters, not as
        # `get_global_id()` calls (see kernel/metal_codegen.py). They are declared after
        # every parameter the host binds, because the host never binds them: the dispatch
        # fills the attributes in. The names are the ones the prelude's macros expand
        # onto, so a body keeps its OpenCL spelling.
        if self._metal_syntax:
            from .metal_codegen import prologue_parameters
            params.extend(prologue_parameters())
        return params

    def signature_digest(self)->str:
        """A digest of what this kernel's device signature declares.

        The C name appends it, so a body whose derived parameters changed is a different
        name -- and the generated file then holds one declaration instead of two that
        collide.
        """
        return hashlib.md5(";".join(self.parameter_declarations()).encode("utf-8")).hexdigest()[:8]

    def render_kernel(self, c_name:str)->str:
        params:List[str] = self.parameter_declarations()

        if self._metal_syntax:
            from .metal_codegen import kernel_prefix
            prefix:str = kernel_prefix() + " "   # "kernel void ", the return type included
        else:
            prefix = "__kernel " if self._opencl_syntax else "__global__ "
        # Metal's prefix carries the return type (`kernel void`); every other backend
        # spells the qualifier alone, so the `void` is appended only for them.
        return_type:str = "" if self._metal_syntax else "void "
        header:str = f"{prefix}{return_type}{c_name}({', '.join(params)})"
        body_lines:List[str] = []
        if self.backend == "cuda":
            for lc_name, lc_type, lc_count in self.local_buffers:
                # `_device_type` because CUDA C has no `uint`: the shared array has to be
                # declared in the backend's spelling, not the translator's internal one.
                # One indent level, like a body statement: these lines are the kernel's
                # first declarations, not part of the signature.
                body_lines.append(
                    f"    __shared__ {self._device_type(lc_type)} {lc_name}[{lc_count}];")
        body_lines.extend(self.lines)
        # A blank line between the shared-memory declarations and the body reads as a
        # separate preamble; the renderer collapses and trims whatever follows.
        body_lines.append("")
        body:str = self._render_body(body_lines)
        # The structure types are emitted once by the caller, ahead of the whole source (see
        # PyKernel._compile): a helper may need them too, and two copies of the same struct
        # is an error in C++.
        # The body already ends with the separator rule applied: two braces that close a
        # block and its parent sit together, with no blank line between them.
        return f"{header}\n{{\n{body}\n}}"

    def render_kernels(self, c_name:str)->List[Tuple[str, str]]:
        """Every kernel this translation produces, as (name, rendered source).

        A body with one magic loop renders one kernel. A body with several renders one per
        phase, each named after the phase so the host can launch them in order; they share
        the source, so a phase's helpers and typedefs are emitted once alongside the first.
        """
        if not self.phase_translators:
            return [(c_name, self.render_kernel(c_name))]
        rendered:List[Tuple[str, str]] = []
        for index, phase in enumerate(self.phase_translators):
            phase_name:str = f"{c_name}_p{index}"
            rendered.append((phase_name, phase.render_kernel(phase_name)))
        return rendered

    def structure_dtypes(self)->List[object]:
        """Every structure dtype this translation mentions, in no particular order.

        The caller emits one typedef block for the whole source, so the kernel and each
        helper contribute their own set and duplicates are dropped.
        """
        from . import structs

        dtypes:List[object] = []
        for name, t in list(self.params):
            candidate:object = t[1] if is_ptr(t) else t
            dtype:object = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in dtypes:
                dtypes.append(dtype)
        for dtype in self.private_structs:
            if dtype not in dtypes:
                dtypes.append(dtype)
        for entry in (self.return_layout or []):
            candidate = entry["type"]
            dtype = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in dtypes:
                dtypes.append(dtype)
        return dtypes

    def render_structure_typedefs(self)->str:
        """The `typedef`s for every structure this kernel's signature mentions.

        A structure is described by its numpy dtype, so the device type is generated rather
        than declared by the user. Nested structures come first (`render_structures` orders
        them), and a member that is a vector already has its device spelling.
        """
        from . import structs

        dtypes:List[object] = []
        for name, t in list(self.params):
            for candidate in (t[1] if is_ptr(t) else t,):
                dtype:object = structs.dtype_for(candidate) if isinstance(candidate, str) else None
                if dtype is not None and dtype not in dtypes:
                    dtypes.append(dtype)
        for dtype in self.private_structs:
            if dtype not in dtypes:
                dtypes.append(dtype)
        for entry in (self.return_layout or []):
            candidate:object = entry["type"]
            dtype = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in dtypes:
                dtypes.append(dtype)
        return structs.render_structures(dtypes, self.backend)
