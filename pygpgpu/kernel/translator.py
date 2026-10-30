from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import math
import os
import re
import textwrap
import types
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Tuple

import numpy as np

from .builtin_registry import builtin_name_of
from .structs import dtype_for as structs_dtype_for, member_types as structs_member_types
from .device_builtins import (
    ATOMIC_BUILTINS,
    GENERIC_BUILTINS,
    INDEX_BUILTINS,
    MATH_BUILTINS,
    emit_atomic,
    emit_barrier,
    emit_index,
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


def _get_attr(obj:Any, name:str, default:Any=None)->Any:
    """Attribute read that falls back to a default instead of raising."""
    return getattr(obj, name, default)


def _copy_ast(node:ast.AST, cls:type, fields:Dict[str, Any])->ast.AST:
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
            _value[:] = [_normalize_ast_legacy_nodes(v) for v in _value]
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


def _parse_kernel_source(source:str)->ast.Module:
    """Parse a kernel source string into an AST normalized across Python versions."""
    return _normalize_ast_legacy_nodes(ast.parse(source, mode="exec"))


def _class_attr(owner:Optional[type], name:str)->Any:
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


_MISSING:Any = object()


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
    own:Dict[str, Any] = getattr(func, "__globals__", None) or {}

    def resolve(name:str)->Any:
        if name in own:
            return own[name]
        return fallback(name)

    # The host has to read an outer value from the same place this resolver found it, or a
    # name that lives in the helper's module reads back as None at launch. Recorded on the
    # resolver rather than in a signature so `_bind_outer_variables` can pass it along.
    setattr(resolve, "source_globals", own)
    return resolve


def self_attr_kind(owner:Optional[type], name:str)->Tuple[str, Any]:
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
    raw:Any = _class_attr(owner, name)
    if isinstance(raw, staticmethod):
        func:Any = raw.__func__
        return ("kernel" if getattr(func, "is_pykernel", False) else "staticmethod"), func
    if isinstance(raw, classmethod):
        return "classmethod", raw.__func__
    if getattr(raw, "is_pykernel", False):
        return "kernel", raw
    if inspect.isfunction(raw):
        return "method", raw
    return "data", (None if raw is _MISSING else raw)


def _self_member_call_kind(owner:Optional[type], name:str)->Tuple[str, Any]:
    """Classify ``self.name(...)``: what the attribute read yields, then called.

    A staticmethod read through an instance is a plain function, so a call site has to
    reach the underlying function; a plain function the instance carries (assigned in
    __init__) is reached through the instance too, and both are receiver-less device
    calls.
    """
    kind:str
    target:Any
    kind, target = self_attr_kind(owner, name)
    if kind == "data" and owner is not None:
        attribute:Any = inspect.getattr_static(owner, name, _MISSING)
        if isinstance(attribute, staticmethod):
            func:Any = attribute.__func__
            return ("kernel" if getattr(func, "is_pykernel", False) else "staticmethod"), func
        if inspect.isfunction(attribute):
            return "method", attribute
    return kind, target


def scan_self_attrs(func:Callable, self_param:str, owner:Optional[type]=None)->List[str]:
    """Collect the self.xxx data attribute names the method (and its helpers) reference.

    A member read as a value becomes a kernel argument. With ``owner`` given, a method
    called as ``self.helper(...)`` is followed instead: the helper's own ``self.xxx``
    references then count too, recursively, so nested helpers keep working and every
    attribute they need is uploaded with the kernel. ``owner`` is None at decoration
    time (the class does not exist yet), where only the decorated body itself is read.
    """
    if not self_param:
        return []
    attrs:List[str] = []
    _collect_self_attrs(func, self_param, owner, attrs, set())
    return attrs


def _collect_self_attrs(func:Callable, self_param:str, owner:Optional[type],
                        attrs:List[str], visited:Set[Any])->None:
    """Walk one method body, recursing into the self-methods it calls."""
    if func in visited:
        return
    visited.add(func)
    tree:ast.Module = _parse_kernel_source(get_dedented_source(func))
    called:Set[str] = {node.func.attr for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and isinstance(node.func.value, ast.Name)
                       and node.func.value.id == self_param}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == self_param):
            continue
        kind:str
        target:Any
        kind, target = _self_member_call_kind(owner, node.attr)
        if node.attr in called:
            # A member that is called is not data, so it needs no kernel argument. An
            # instance method's own self.xxx reads do, so follow it; anything else
            # (a staticmethod, a @kernel method, a classmethod) is decided at the
            # call site, where a clear error can name the reason.
            if kind == "method":
                _collect_self_attrs(target, self_param, owner, attrs, visited)
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

    details:Tuple[Any, ...] = (
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


INT_TYPES:Set[str] = {"bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"}
FLOAT_TYPES:Set[str] = {"float", "double"}
SCALAR_TYPES:Set[str] = INT_TYPES | FLOAT_TYPES

DTYPE_MAP:Dict[str, str] = {
    "bool": "bool",
    "int8": "char", "int16": "short", "int32": "int", "int64": "long",
    "uint8": "uchar", "uint16": "ushort", "uint32": "uint", "uint64": "ulong",
    "float16": "half", "float32": "float", "float64": "double",
}

PTR = "*"  # type notation: ("*", base) means an array parameter, a plain string means a scalar

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

_C_FIELD_TYPES:Dict[Any, str] = {}

# Declared here (and filled on first use) because importing the driver tables eagerly would
# load a GPU runtime at `import pygpgpu` time.
_VEC_REGISTRY:Dict[str, Any]

def _vec_registry()->Dict[str, Any]:
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

def _c_scalar_name(ctypes_type:Any)->str:
    import ctypes as _ctypes
    sizes = {_ctypes.c_float: "float", _ctypes.c_double: "double",
             _ctypes.c_int8: "char", _ctypes.c_int16: "short", _ctypes.c_int32: "int",
             _ctypes.c_int64: "long", _ctypes.c_uint8: "uchar", _ctypes.c_uint16: "ushort",
             _ctypes.c_uint32: "uint", _ctypes.c_uint64: "ulong"}
    return sizes.get(ctypes_type, "float")

def is_vec(t:Any)->bool:
    return isinstance(t, str) and t in _vec_registry()

# A 3-component vector is padded to 16 bytes for the device ABI, so its ctypes struct carries
# a trailing `_` field that is not a component. Everything that names components counts them
# through this helper, so the padding never reaches a constructor, a swizzle or a print.
def _vec_components(vec:str)->List[Tuple[str, Any]]:
    """The real (field name, ctypes type) components of a vector type, in order."""
    return [field for field in _vec_registry()[vec]._fields_ if field[0] != "_"]

def _vec_field_names(cls_or_name:Any)->List[str]:
    """The component names of a vector type, given the class or its name.

    Everything that expands a vector into per-component code goes through this, so the
    padding field of a 3-component vector never becomes a member the device does not have.
    """
    fields:Any = (cls_or_name._fields_
                  if isinstance(cls_or_name, type)
                  else _vec_registry()[cls_or_name]._fields_)
    return [name for name, _ in fields if name != "_"]

# Colour-style component names, as in `color.rgb`. OpenCL C accepts them natively but CUDA's
# vector types only have x/y/z/w, so the translator normalizes them for every backend.
_VEC_COLOR_ALIASES:Dict[str, str] = {"r": "x", "g": "y", "b": "z", "a": "w"}

def _math_device_name(attr:str, backend:str)->str:
    """The device spelling of a `math.*` function.

    OpenCL's C dialect has float overloads under the plain name. CUDA has both the
    double and the `f`-suffixed float form, and asking for the plain name with a float
    argument is ambiguous whenever the argument's type is not exactly `float` -- which is
    how `floor` came to fail with "more than one instance of overloaded function".
    """
    if backend != "cuda":
        return attr
    # The three predicates are already type-generic in CUDA and take no `f` suffix.
    if attr in ("isnan", "isinf", "isfinite"):
        return attr
    return attr if attr.endswith("f") else attr + "f"


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
    "inf": "INFINITY",
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


def is_int(t:Any)->bool:
    return isinstance(t, str) and t in INT_TYPES

def is_float(t:Any)->bool:
    return isinstance(t, str) and t in FLOAT_TYPES

def is_ptr(t:Any)->bool:
    return isinstance(t, tuple) and len(t) == 2 and t[0] == PTR

def is_image(t:Any)->bool:
    """True for the marker a backend-neutral Image argument gets."""
    return isinstance(t, tuple) and len(t) == 3 and t[0] == IMAGE

def image_channels(t:Any)->int:
    return int(t[1])

def image_scalar(t:Any)->str:
    """The C scalar type of one image channel."""
    return t[2]

def promote(t1:Any, t2:Any)->Any:
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

def infer_ctype(value:Any)->Any:
    """Actual call argument -> C type."""
    from ..vec_types.layout import _vector_dtype_name
    from . import structs

    registry:Dict[str, Any] = _vec_registry()

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
        return structs.register(declared)

    if isinstance(value, np.void):
        # A structured scalar is either a vector element or a structure value: both are
        # described completely by the dtype, so both become a device type.
        if structs.is_declared_structure(value.dtype):
            # An element of a ctypes-declared array reaches here as a record, and the
            # declaration wins over the coincidental vector layout.
            return structs.register(value.dtype)
        vec_name:Optional[str] = _vector_dtype_name(value.dtype)
        if vec_name is not None:
            return vec_name
        if structs.is_record(value.dtype):
            return structs.register(value.dtype)
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
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if _is_device_handle(value):
        # A texture or surface reaches the device as an opaque 64-bit handle, which
        # the kernel reads through tex2D or writes through surf2Dwrite.
        return _handle_device_type(value)
    raise SyntaxError(f"unsupported argument type: {type(value).__name__} (only numpy arrays, int/float/bool scalars and texture handles are supported)") from None


TEXTURE_HANDLE_TYPES = ("TextureObject", "cudaTextureObject_t")
SURFACE_HANDLE_TYPES = ("SurfaceObject", "cudaSurfaceObject_t")


def _is_device_handle(value:Any)->bool:
    """True for the objects a kernel receives as an opaque texture or surface handle."""
    return type(value).__name__ in TEXTURE_HANDLE_TYPES + SURFACE_HANDLE_TYPES


def _handle_device_type(value:Any)->str:
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
def _creation_assignments(func:Callable)->List[ast.stmt]:
    """The host-executed creation statements of a kernel body, in order.

    The host runs exactly these before the launch, so the objects exist with the shape and
    dtype numpy gives them. A magic loop is what defines "above" here: an assignment below
    it would run once per work item, so it is not lifted out (the translator refuses it).
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

    globalns:Dict[str, Any] = getattr(func, "__globals__", {})

    def is_numpy(node:ast.expr)->bool:
        return (isinstance(node, ast.Name) and globalns.get(node.id) is np)

    def is_creation(node:ast.expr)->bool:
        """True when this expression only *builds* data (and cannot run on the device)."""
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
                # A scalar-type call builds a host value only when its arguments are
                # constants; `np.float32(x[i])` is a device cast (see _emit_numpy_cast).
                return all(isinstance(arg, ast.Constant) for arg in node.args)
            if name not in _HOST_CREATION_FUNCTIONS and not name.endswith("_like"):
                return False
            # The arguments may be ordinary values (a shape, a count, an input array); what
            # makes the statement host-only is the numpy constructor itself. Only a dtype
            # argument has to be a type name, which is why it is not evaluated on the device.
            return True
        return False

    del _builtins
    chosen:List[ast.stmt] = []
    for stmt in body[:magic_index]:
        value:Optional[ast.expr] = _creation_target_of(stmt)
        if value is None:
            continue
        if is_creation(value):
            chosen.append(stmt)
    return chosen


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


def _is_magic_loop_stmt(stmt:ast.stmt)->bool:
    """True for the outermost loop the translator turns into work-item dispatch."""
    if not isinstance(stmt, ast.For):
        return False
    iterator:ast.expr = stmt.iter
    if isinstance(iterator, ast.Call):
        func:ast.expr = iterator.func
        if isinstance(func, ast.Name) and func.id in ("range", "enumerate"):
            return True
        if (isinstance(func, ast.Attribute) and func.attr in ("ndindex", "ndenumerate")
                and isinstance(func.value, ast.Name)):
            return True
    return False


_UNRESOLVED:Any = object()

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


def type_sig(types:List[Any])->str:
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

    def __init__(self, func:Callable, param_types:List[Any], backend:str,
                 resolve_global:Callable[[str], Any], device_cache:Dict[Any, Any],
                 is_kernel:bool=False, self_param:str="",
                 self_attr_types:Optional[Dict[str, Any]]=None,
                 self_owner:Optional[type]=None, launch_backend:Optional[str]=None,
                 host_creation_values:Optional[Dict[str, Any]]=None):
        self.func:Callable = func
        self.filename:str = getattr(getattr(func, "__code__", None), "co_filename", "<kernel>")
        self.backend:str = backend
        # The backend the kernel is actually launched on. hip reuses cuda code generation, so
        # backend is "cuda" for a hip kernel; decisions that depend on the runtime rather than
        # on the generated syntax (for example whether device printf is safe) must use this.
        self.launch_backend:str = (launch_backend or backend)
        self.resolve_global:Callable[[str], Any] = resolve_global
        self.device_cache:Dict[Any, Any] = device_cache
        self.is_kernel:bool = is_kernel
        self.self_param:str = self_param
        self.self_attr_types:Dict[str, Any] = dict(self_attr_types or {})
        # The class the translated method belongs to, for deciding whether `self.name`
        # is a data attribute, an instance method, or a staticmethod. None for the
        # decoration-time body scan, where the class does not exist yet.
        self.self_owner:Optional[type] = self_owner
        # Objects the host already built for statements above the magic loop, keyed by the
        # name the body assigns them to (see _host_created_assignment).
        self.host_creation_values:Dict[str, Any] = dict(host_creation_values or {})

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

        self.params:List[Tuple[str, Any]] = []
        self.shapes_used:List[Tuple[str, int]] = []
        self.gid_param:Optional[str] = None
        self.work_size_dims:List[Tuple[str, int]] = []
        # One (array, dimension) source per dimension of the ndindex magic loop, in the
        # order the loop unpacks them (see _ndindex_extents). None for the other forms.
        self._nd_loop_extents:Optional[List[Tuple[str, int]]] = None
        # The array named by np.ndindex(arr.shape), whose rank is the loop's unpacking.
        self._nd_shape_holder:str = ""
        # name -> how many indices the body subscripts that array with (see
        # _record_subscript_indices); checked against the real shape at launch.
        self.subscript_indices:Dict[str, int] = {}
        # The magic loop's index variables, in unpack order. A subscript formed from
        # exactly these names addresses a different element per work item, which is what
        # makes a plain read-modify-write safe (see _augassign_is_race_free).
        self.work_item_indices:List[str] = []
        # Structure dtypes a private array created in the body mentions: their typedefs
        # have to be emitted even though they are not parameters.
        self.private_structs:List[Any] = []
        # The array whose whole shape a `np.ndindex(arr.shape)` loop iterates, when that is
        # how the domain was spelled: the launch checks its rank against the loop's names.
        self.nd_whole_shape_param:str = ""
        # Array names declared by an explicit `gpu.work_size(x)` statement; this is the
        # gid-style equivalent of a magic loop's implied iteration domain. The rank comes
        # from each array's runtime shape (a pointer type carries no rank).
        self.declared_work_arrays:List[str] = []
        # Set when the body calls gpu.get_global_id()/... , i.e. it dispatches on the launch
        # index and therefore needs an iteration domain from somewhere.
        self.uses_global_id:bool = False
        # Set once a magic loop is detected: it fixes the iteration domain, so an explicit
        # gpu.work_size(...) declaration would be redundant.
        self.magic_loop_present:bool = False
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
        self.return_type:Optional[Any] = None
        # Kernel return values. A kernel runs once per work item, so it cannot have a C
        # return value; every returned position becomes a device parameter instead (a
        # one-element buffer for a value, the array parameter's own name for an array) and
        # the first work item to reach a `return` claims the whole result.
        #
        # return_layout: one entry per value slot ({"ctype", "type"}), shared by every
        #   return site whose value has that C type.
        # return_sites: per `return` statement, its positions as the translator saw them.
        # _site_slot_map: per site, (position ordinal, slot index) pairs it writes.
        self.return_layout:List[Dict[str, Any]] = []
        self.return_sites:List[List[Dict[str, Any]]] = []
        self._site_slot_map:Dict[int, List[Tuple[int, int]]] = {}
        # True when statements currently being visited run once per work item after the
        # magic loop (see _visit_region and _visit_kernel_return).
        self.after_magic_loop:bool = False
        # True when statements currently being visited are inside (or after) the magic
        # loop, where building data would happen once per work item (see _visit_assign).
        self.creation_below_loop:bool = False
        # True when any site returns a value; then the kernel needs the claim and site
        # parameters even if some other site is a bare `return`.
        self.has_returns:bool = False
        # Outer-scope variables: a name the body uses that is neither a parameter nor a
        # local becomes a device parameter instead (see _bind_outer_variables). Each entry
        # records how the host reaches the name ("closure" through a cell, "global" through
        # func.__globals__), its device type, and whether the body declares it writable
        # (`nonlocal`/`global`) -- which decides between `T* name` + `*name` and a plain `T`.
        self.outer_vars:Dict[str, Dict[str, Any]] = {}
        # Values the host creates for the kernel: an assignment above the magic loop whose
        # right-hand side only builds data (np.zeros, np.full, ...) is executed by the
        # interpreter, not translated, and the object travels as a parameter (see
        # _host_created_assignment).
        self.host_values:Dict[str, Dict[str, Any]] = {}
        self.return_type:Optional[Any] = None
        self.lines:List[str] = []
        self._indent:int = 1
        self.scopes:List[Dict[str, Any]] = []

        args:List[ast.arg] = self.tree.args.args
        if self.tree.args.vararg or self.tree.args.kwarg or self.tree.args.kwonlyargs:
            raise SyntaxError("*args/**kwargs/keyword-only args are not supported", self.tree.args) from None
        # Default values are accepted: a kernel argument is always supplied by the
        # caller, so a default is a convenience for reading the signature rather than
        # something the device code has to represent.
        if self_param and args and args[0].arg == self_param:
            args = args[1:]
        expected:int = len(args) + len(self.self_attr_types)
        if expected != len(param_types):
            raise SyntaxError(f"{func.__name__}() takes {expected} device arguments but {len(param_types)} types were given") from None

        scope:Dict[str, Any] = {}
        for arg, t in zip(args, param_types):
            # Annotations are accepted but ignored: the device type comes from the
            # actual argument at call time (see infer_ctype), because one kernel is
            # specialized per argument-type signature. Keeping them lets the same
            # function be read by editors and type checkers.
            scope[arg.arg] = t
            self.params.append((arg.arg, t))
        # self.xxx attributes -> extra device parameters _self_xxx, filled from the instance at call time
        for attr, t in self.self_attr_types.items():
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

    def _lookup(self, name:str)->Any:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        # A host-created object and an outer variable are device parameters too, so a
        # type query has to see them; the magic-loop scan asks which arrays exist before
        # any statement of the body has been visited.
        host:Optional[Dict[str, Any]] = self.host_values.get(name)
        if host is not None:
            return host["type"]
        outer:Optional[Dict[str, Any]] = self.outer_vars.get(name)
        if outer is not None:
            # A declared outer scalar is a pointer on the device (see _visit_name).
            if outer["writable"] and not is_ptr(outer["type"]):
                return (PTR, outer["type"])
            return outer["type"]
        return None

    def _define(self, name:str, t:Any, node:Optional[ast.AST])->None:
        """Record a variable's type; node is optional because some declarations have no
        source node to blame in an error message."""
        existing:Any = self._lookup(name)
        if existing is None:
            self.scopes[-1][name] = t
        elif existing != t:
            if is_int(existing) and is_float(t):
                self.scopes[-1][name] = t
            else:
                self._error(f"variable '{name}' changes type from {existing} to {t}", node)

    def _shape_var(self, array_name:str, dim:int)->str:
        # A private array created in the work-item body has a constant shape, so its extents
        # are compile-time numbers rather than launch parameters.
        binding:Any = self._lookup(array_name)
        if isinstance(binding, tuple) and len(binding) == 4 and binding[0] == "bodyarr":
            shape:Tuple[int, ...] = binding[2]
            if 0 <= dim < len(shape):
                return str(shape[dim])
        key = (array_name, dim)
        if key not in self.shapes_used:
            self.shapes_used.append(key)
        return f"_{array_name}_shape{dim}"

    def _trim_source(self)->str:
        return get_dedented_source(self.func)

    # ---------- translation entry ----------

    def translate(self)->None:
        body:List[ast.stmt] = self.tree.body
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
            self._translate_magic_loop(body)
        else:
            for stmt in body:
                self._visit_statement(stmt)
            if self.return_type is None:
                self.return_type = "void"
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
        code:Any = getattr(self.func, "__code__", None)
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
            value:Any = self.resolve_global(name)
            if value is _UNRESOLVED:
                continue
            entry:Optional[Dict[str, Any]] = self._outer_var_entry(name, value, code, declared)
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

    def _outer_var_entry(self, name:str, value:Any, code:Any,
                         declared:Set[str])->Optional[Dict[str, Any]]:
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
            device_type:Any = infer_ctype(value)
        except SyntaxError as error:
            if not writable:
                # Not a name this feature handles; the normal "unknown name" path will
                # explain it with the right position if the body really reads it.
                return None
            raise SyntaxError(f"outer variable '{name}' cannot be written back: {error}") from None
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
        """When X in `X.shape` is an array parameter, return its parameter name (plain name or _self_ attribute parameter)."""
        if not (isinstance(attr_node, ast.Attribute) and attr_node.attr == "shape"):
            return None
        base:ast.expr = attr_node.value
        if isinstance(base, ast.Name):
            return base.id
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

    def _enumerate_magic_target(self, iter_node:ast.expr)->Optional[Tuple[str, Any]]:
        """Resolve the array parameter of `for i, x in enumerate(arr)`, returning (parameter name, pointer type)."""
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "enumerate" and len(iter_node.args) == 1 and not iter_node.keywords):
            return None
        arg:ast.expr = iter_node.args[0]
        param:str = ""
        if isinstance(arg, ast.Name):
            param = arg.id
        elif (self.self_param and isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id == self.self_param):
            param = f"_self_{arg.attr}"
        else:
            return None
        t:Any = self._lookup(param)
        if not is_ptr(t):
            return None
        return param, t

    def _emit_work_item_head(self, index_name:str, param:str)->None:
        """Emit the gid definition and the out-of-range guard."""
        self.work_item_indices = [index_name]
        self._define(index_name, "int", None)
        gid_bound:str = self._shape_var(param, 0)
        if self.backend == "opencl":
            self._emit(f"const int {index_name} = get_global_id(0);")
        else:
            self._emit(f"const int {index_name} = (int)(blockIdx.x * blockDim.x + threadIdx.x);")
        self._emit(f"if ({index_name} >= {gid_bound}) return;")

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
        module:Any = self.resolve_global(iter_node.func.value.id)
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
            param = arg.id
        elif (self.self_param and isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id == self.self_param):
            param = f"_self_{arg.attr}"
        if not param:
            return None
        holder_type:Any = self._lookup(param)
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
                holder_type:Any = self._lookup(holder)
                if not (is_ptr(holder_type) or is_image(holder_type)):
                    return None
                self._nd_shape_holder = holder
                return []
            if isinstance(arg, ast.Tuple):
                args = list(arg.elts)
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
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and self.self_param and node.value.id == self.self_param):
            # A host-side integer attribute is a constant for this launch, like a literal:
            # `np.ndindex(self.image_height, self.image_width)`. The extent is read from the
            # argument carrying the attribute (see the -2 marker in PyKernel._invoke_inner).
            attribute_type:Any = self.self_attr_types.get(node.attr)
            if attribute_type is not None and is_int(attribute_type):
                return f"_self_{node.attr}", -2
            return None
        if not isinstance(node, ast.Subscript):
            return None
        if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
            return None
        holder:Optional[str] = self._shape_holder_param(node.value)
        if holder is None:
            return None
        holder_type:Any = self._lookup(holder)
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
        return self._shape_var(name, dim)

    def _emit_nd_head(self, names:List[str], param:str)->None:
        """Multi-dimensional work-item head: names[0] maps to dimension 0 (slowest)."""
        ndim:int = len(names)
        self.work_item_indices = list(names)
        sources:List[Tuple[str, int]] = self._nd_loop_extents or [(param, d) for d in range(ndim)]
        for d, name in enumerate(names):
            self._define(name, "int", None)
            self._nd_bound_code(sources[d])
        checks:List[str] = []
        for d, name in enumerate(names):
            component:int = ndim - 1 - d
            bound:str = self._nd_bound_code(sources[d])
            if self.backend == "opencl":
                self._emit(f"const int {name} = get_global_id({component});")
            else:
                c:str = (".x" if component == 0 else ".y" if component == 1 else ".z")
                self._emit(f"const int {name} = (int)(blockIdx{c} * blockDim{c} + threadIdx{c});")
            checks.append(f"{name} >= {bound}")
        self._emit(f"if ({' || '.join(checks)}) return;")

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

    def _translate_magic_loop(self, body:List[ast.stmt])->None:
        magic_index:Optional[int] = None
        enumerate_form:Optional[Tuple[str, Any]] = None
        nd_form:Optional[Tuple[str, str, str]] = None
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, ast.For):
                continue
            if self._is_shape0_range(stmt.iter):
                magic_index = idx
                break
            enum_info:Optional[Tuple[str, Any]] = self._enumerate_magic_target(stmt.iter)
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
            # domain then has to come from somewhere: either an explicit gpu.work_size(...)
            # declaration, or the bracket at call time. Without either, the launch size cannot
            # be derived (see PyKernel._invoke).
            for stmt in body:
                self._visit_statement(stmt)
            return

        # Statements before the magic loop are translated as usual (e.g. local variable initialization).
        # The magic loop already fixes the iteration domain, so a gpu.work_size(...) declaration
        # among them would be redundant: flag it before visiting so the check sees the conflict.
        self.magic_loop_present = True
        for stmt in body[:magic_index]:
            # A creation the device cannot perform is executed by the interpreter instead
            # and becomes a parameter; that is only possible here, above the loop, because
            # below it the statement would run once per work item.
            if self._host_created_assignment(stmt):
                continue
            self._visit_statement(stmt)

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

            self.gid_param = param
            self._emit_work_item_head(index_name, param)
            # x is an element-value snapshot (matches python semantics: rebinding x does not write back to the array)
            elem_t:Any = arr_t[1]
            self._define(value_name, elem_t, target.elts[1])
            self._emit(f"{elem_t} {value_name} = {param}[{index_name}];")

            self._visit_work_item_body(loop.body)
            self._visit_region(body[magic_index + 1:], after_magic_loop=True)
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
            if not (isinstance(names_node, ast.Tuple) and all(isinstance(e, ast.Name) for e in names_node.elts)):
                self._error(
                    f"{module_name}.{form} indices must be unpacked into plain names, "
                    f"e.g. `for y, x in {module_name}.ndindex(array.shape)`", target)
            names:List[str] = [e.id for e in names_node.elts]
            ndim:int = len(names)
            if ndim == 1:
                self._error(
                    f"for 1-D iteration use `range(array.shape[0])` or `enumerate(array)`; "
                    f"{module_name}.{form} is for 2-D/3-D", target)
            if ndim > 3:
                self._error(f"{module_name}.{form} supports at most 3 dimensions, got {ndim}", target)

            # One extent per unpacked name. An explicit np.ndindex(a, b, ...) states them
            # itself, so a mismatch is a mistake worth naming; the whole-shape spelling
            # takes its extents from the array, one per name.
            explicit:Optional[List[Tuple[str, int]]] = self._nd_loop_extents
            if explicit is not None and len(explicit) != ndim:
                self._error(
                    f"this loop unpacks {ndim} name{'s' if ndim != 1 else ''} but "
                    f"{module_name}.ndindex(...) is given {len(explicit)} "
                    f"extent{'s' if len(explicit) != 1 else ''}; they must match", target)
            sources:List[Tuple[str, int]] = (explicit
                                             if explicit is not None
                                             else [(param, d) for d in range(ndim)])
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
                arr_t:Any = self._lookup(param)
                elem_t:Any = arr_t[1]
                self._define(value_name, elem_t, None)
                self._emit(f"{elem_t} {value_name} = {param}[{self._nd_flat_index(names, param)}];")

            self._visit_work_item_body(loop.body)
            self._visit_region(body[magic_index + 1:], after_magic_loop=True)
            # The launch sizes dimensions fastest-first, which is the reverse of the order
            # the loop unpacks them: gid 0 is the last index. Getting this backwards used
            # to launch a 2x3 domain as 3x2, so the last column or row was never written.
            self.work_size_dims = list(reversed(sources))
            return

        loop:ast.For = body[magic_index]
        range_call:ast.Call = loop.iter
        bound:ast.expr = range_call.args[0]

        array_name:str = (self._shape_holder_param(bound.value) or "")
        outer:Optional[Dict[str, Any]] = self.outer_vars.get(array_name)
        host:Optional[Dict[str, Any]] = self.host_values.get(array_name)
        is_outer_array:bool = ((outer is not None and is_ptr(outer["type"]))
                               or (host is not None and is_ptr(host["type"])))
        if not is_outer_array and not is_ptr(self._lookup(array_name)):
            self._error(f"'{array_name}' is not an array parameter of this kernel", bound)

        self.gid_param = array_name

        gid_bound:str = self._shape_var(array_name, 0)
        target:str = loop.target.id
        # This branch emits its own work-item head (the 1-D form), so it has to record the
        # index variable that makes a `arr[i] += v` provably race-free just as the
        # multi-dimensional head does.
        self.work_item_indices = [target]
        self._define(target, "int", loop.target)
        if self.backend == "opencl":
            self._emit(f"const int {target} = get_global_id(0);")
        else:
            self._emit(f"const int {target} = (int)(blockIdx.x * blockDim.x + threadIdx.x);")
        self._emit(f"if ({target} >= {gid_bound}) return;")

        # The loop body is the work-item body; nested range loops translate as real serial loops
        self._visit_work_item_body(loop.body)
        self._visit_region(body[magic_index + 1:], after_magic_loop=True)
        self.work_size_dims = [(array_name, 0)]

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
            produced:Any = self.host_creation_values.get(name, _UNRESOLVED)
            if produced is _UNRESOLVED:
                self._error(
                    f"'{name}' cannot be created here: the expression only builds "
                    f"data, but it could not be evaluated on the host; build it from values "
                    f"that exist at call time (a shape, a constant, another array)",
                    value)
            device_type:Any = infer_ctype(produced)
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
        """
        previous:bool = self.after_magic_loop
        previous_creation:bool = self.creation_below_loop
        self.after_magic_loop = previous
        self.creation_below_loop = True
        try:
            for stmt in statements:
                self._visit_statement(stmt)
        finally:
            self.after_magic_loop = previous
            self.creation_below_loop = previous_creation

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
            self._visit_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._visit_augassign(node)
        elif isinstance(node, ast.If):
            cond, _ = self._visit_expression(node.test)
            self._emit(f"if ({cond}) {{")
            self._push_scope_block(node.body)
            if node.orelse:
                self._emit("} else {")
                self._push_scope_block(node.orelse)
            self._emit("}")
        elif isinstance(node, ast.While):
            cond, _ = self._visit_expression(node.test)
            self._emit(f"while ({cond}) {{")
            self._push_scope_block(node.body)
            self._emit("}")
        elif isinstance(node, ast.For):
            self._visit_for(node)
        elif isinstance(node, ast.Return):
            self._visit_return(node)
        elif isinstance(node, ast.Expr):
            # Bare string literals (e.g. the function docstring) are ignored, not emitted into the kernel
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                pass
            elif self._is_work_size_decl(node.value):
                # gpu.work_size(...) is host-side only: it declares the iteration domain and
                # produces no device code at all.
                self._handle_work_size_decl(node.value)
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
            self._emit("break;")
        elif isinstance(node, ast.Continue):
            self._emit("continue;")
        elif isinstance(node, (ast.Nonlocal, ast.Global)):
            # The scope declarations themselves emit nothing: they were already read while
            # binding the outer variables (see _bind_outer_variables), so the name travels
            # as a device parameter and every use goes through it.
            pass
        elif isinstance(node, ast.Pass):
            pass
        else:
            self._error(f"unsupported statement: {node.__class__.__name__}", node)

    def _push_scope_block(self, statements:List[ast.stmt])->None:
        self.scopes.append({})
        self._indent += 1
        for stmt in statements:
            self._visit_statement(stmt)
        self._indent -= 1
        self.scopes.pop()

    def _visit_for(self, node:ast.For)->None:
        if isinstance(node.iter, ast.Name) or isinstance(node.iter, ast.Attribute):
            self._error(
                "iterating an array directly (`for x in arr`) is not supported in a kernel: "
                "a work-item has no index to write through; use `for i, x in enumerate(arr)` "
                "(x is a read-only element snapshot) or `for i in range(arr.shape[0])`", node.iter)
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

        values:List[Tuple[str, Any]] = [self._visit_expression(a) for a in args]
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
            self._emit(f"for (int {target} = {start}; {target} < {stop}; {target}++) {{")
        elif re.fullmatch(r"-?\d+", step):
            cond_op:str = (">" if step.startswith("-") else "<")
            self._emit(f"for (int {target} = {start}; {target} {cond_op} {stop}; {target} += {step}) {{")
        else:
            self._error("range() step must be a constant integer", node.iter.args[-1])
        self._push_scope_block(node.body)
        self._emit("}")

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
    RETURN_SLOT_TYPES:Dict[str, Tuple[str, Any]] = {
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
    }

    def _return_position(self, node:ast.expr)->Dict[str, Any]:
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
            return {"kind": "array", "name": name, "type": t}
        return {"kind": "value", "type": t, "code": code}

    def _return_site(self, node:ast.Return)->List[Dict[str, Any]]:
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

    def _visit_kernel_return(self, node:ast.Return)->None:
        """Translate a `return` in a kernel body into a claim plus slot stores."""
        positions:List[Dict[str, Any]] = self._return_site(node)
        if self.after_magic_loop and any(position["kind"] == "value" for position in positions):
            self._error(
                "a value-returning return cannot appear after the magic loop: the "
                "statements after the loop run once per work item, so a value accumulated "
                "there belongs to one work item, not to the whole call. Return from inside "
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
        guard:str = (f"atomic_cmpxchg((volatile __global int*)(&{claim}), 0, 1) == 0"
                     if self.backend == "opencl"
                     else f"atomicCAS(&{claim}, 0, 1) == 0")
        self._emit(f"if ({guard}) {{")
        # The winning site records what it returned, so the host rebuilds that site's
        # shape rather than a shape inferred from every site at once.
        self._emit(f"    {self.RETURN_SITE}[0] = {site + 1};")
        for slot, position in self._site_slots(site):
            if position["kind"] != "value":
                continue
            self._emit(f"    {self._return_slot(slot)}[0] = {position['code']};")
        self._emit("}")
        self._emit("return;")

    def _return_slot(self, index:int)->str:
        return f"_ret{index}"

    def _record_return_slots(self, site:int, positions:List[Dict[str, Any]],
                             node:ast.Return)->None:
        """Give each value position of this site a slot, extending the shared layout.

        A slot is keyed by its C element type, so two sites returning a float share one
        buffer while a site returning an int gets a buffer of its own. A returned array
        occupies no slot: the host hands back the very object it passed in.
        """
        for ordinal, position in enumerate(positions):
            if position["kind"] != "value":
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

    def _site_slots(self, site:int)->List[Tuple[int, Dict[str, Any]]]:
        """(slot, position) pairs this site stores into."""
        positions:List[Dict[str, Any]] = self.return_sites[site]
        pairs:List[Tuple[int, Dict[str, Any]]] = []
        for ordinal, slot in self._site_slot_map.get(site, []):
            pairs.append((slot, positions[ordinal]))
        return pairs

    def _return_param_type(self, t:Any)->str:
        """The C element type of a value slot."""
        from . import structs

        if is_vec(t):
            return t
        if isinstance(t, str) and structs.dtype_for(t) is not None:
            # A structure value travels as a one-element buffer of that structure, so the
            # slot element is the structure type itself.
            return t
        slot:Optional[Tuple[str, Any]] = self.RETURN_SLOT_TYPES.get(t)
        if slot is None:
            self._error(f"cannot return a value of type {t}")
        return self._device_type(slot[0])

    def return_slot_dtype(self, index:int)->Any:
        """The numpy dtype the host uses for one value slot.

        A vector slot uses the *host* layout (a 3-component vector has three components);
        `pygpgpu.vec_types.layout.pack` re-lays it out for the backend that needs padding,
        the same way it does for vector arguments.
        """
        from ..vec_types.layout import _host_classes
        from . import structs

        entry:Dict[str, Any] = (self.return_layout or [])[index]
        if is_vec(entry["type"]):
            return _host_classes()[entry["type"]].numpy_dtype()
        structure:Any = structs.dtype_for(entry["type"]) if isinstance(entry["type"], str) else None
        if structure is not None:
            return structure
        return self.RETURN_SLOT_TYPES[entry["type"]][1]

    def render_return_params(self)->List[str]:
        """The extra device parameters a kernel with returns needs.

        A kernel that returns only an array still needs the claim and site words, so this
        is gated on `has_returns` rather than on the value slots alone.
        """
        if not self.has_returns:
            return []
        qualifier:str = ("__global " if self.backend == "opencl" else "")
        params:List[str] = [f"{qualifier}{entry['ctype']}* {self._return_slot(index)}"
                            for index, entry in enumerate(self.return_layout)]
        params.append(f"{qualifier}int* {self.RETURN_CLAIM}")
        params.append(f"{qualifier}int* {self.RETURN_SITE}")
        return params

    def return_site_layout(self)->List[Dict[str, Any]]:
        """What the host needs to rebuild the result of each return site.

        Per site: the positions in return order, each "value" with its slot index or
        "array" with the parameter name to hand back, plus whether the site returned a
        tuple. A site with no positions is a bare `return`, i.e. None.
        """
        layout:List[Dict[str, Any]] = []
        for site, positions in enumerate(self.return_sites):
            entries:List[Dict[str, Any]] = [{"kind": "value", "ordinal": ordinal, "slot": slot}
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

        code, t = self._visit_expression(node.value)
        if is_ptr(t):
            self._error("cannot return an array from a function", node)

        if self.return_type is None:
            self.return_type = t
        elif self.return_type != t:
            unified:Any = promote(self.return_type, t)
            if unified != self.return_type:
                # The cast has to be spelled in this backend's own type names: CUDA C has
                # no `uchar`/`ushort`/`uint`/`ulong`, so a bare translator name here made
                # every device helper returning one of those fail to compile under NVRTC.
                code = f"(({self._device_type(self.return_type)})({code}))"
            self.return_type = unified
        self._emit(f"return {code};")

    def _eval_constant(self, node:Optional[ast.expr])->Any:
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
            function:Any = self._eval_constant(node.func)
            arguments:List[Any] = [self._eval_constant(argument) for argument in node.args]
            keywords:Dict[str, Any] = {keyword.arg: self._eval_constant(keyword.value)
                                       for keyword in node.keywords if keyword.arg}
            return function(*arguments, **keywords)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._eval_constant(element) for element in node.elts)
        raise SyntaxError("not a compile-time constant expression")

    def _private_element_type(self, dtype_value:Any)->Optional[Tuple[str, Any]]:
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
            shape_value:Any = self._eval_constant(positional[0]) if positional else None
            dtype_node:Optional[ast.expr] = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "dtype"), None)
            dtype_value:Any = self._eval_constant(dtype_node)
            fill_value:Any = None
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
        self._emit(f"{ctype} {name}[{total}]{initialiser};")
        self._define(name, ("bodyarr", total, shape, element_t), node)
        return True

    def _private_initialiser(self, constructor:str, total:int, fill_value:Any,
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
        value:Any = 1 if constructor == "ones" else fill_value
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

        # A creation below the magic loop runs once per work item, so its object cannot be a
        # host parameter. When the shape is a compile-time constant it is exactly a C local
        # array -- private storage per work item, which is what python's fresh array means --
        # and it is emitted as one. A runtime extent has no fixed-size declaration, so it is
        # refused instead of being silently shared between work items.
        if (self.creation_below_loop and isinstance(target, ast.Name)
                and self._is_creation_expression(node.value)):
            if self._emit_private_array(target.id, node):
                return
            self._error(
                "a value cannot be created below the magic loop unless its shape is a "
                "compile-time constant: the statements there run once per work item, so a "
                "constant shape becomes a private array and anything else would create one "
                "object per work item. Create it above the loop, or give it a constant "
                "shape (e.g. np.zeros(4, dtype=np.float32))", node)

        # self.xxx = ... : host attributes cannot be written back from device code
        if (self.self_param and isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == self.self_param):
            self._error(
                f"cannot assign to self.{target.attr}: host attributes cannot be written by device code; "
                f"write into an array element (self.{target.attr}[i] = ...) or use a local variable", node)

        # local memory declaration: buf = local(np.float32, 64)
        if (isinstance(node.value, ast.Call)
                and self._builtin_call_name(node.value) == "local"):
            self._handle_local_declaration(node)
            return

        # Tuple unpacking/swap: a, b = b, a -- also when an item is an array element,
        # as in the idiomatic swap a[i], b[i] = b[i], a[i].
        if isinstance(target, ast.Tuple):
            if not isinstance(node.value, ast.Tuple) or len(target.elts) != len(node.value.elts):
                self._error("tuple assignment requires matching tuples on both sides", node)
            if not all(isinstance(t, (ast.Name, ast.Subscript)) for t in target.elts):
                self._error(
                    "tuple assignment targets must be plain variables or array elements",
                    node)

            values:List[Tuple[str, Any]] = [self._visit_expression(v) for v in node.value.elts]
            temps:List[Tuple[str, Any]] = []
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
                existing:Any = self._lookup(name)
                if existing is not None and is_ptr(existing):
                    self._error(f"cannot assign to array parameter '{name}'", node)
                is_new:bool = (existing is None)
                self._define(name, t, node)
                decl:str = (f"{self._device_type(t)} " if is_new else "")
                self._emit(f"{decl}{name} = {tmp};")
            return
        code, t = self._visit_expression(node.value)
        if is_ptr(t):
            # Rebinding an outer array is refused earlier, while binding the outer
            # variables, so this message only covers a local/parameter target.
            if isinstance(target, ast.Name) and target.id in self.outer_vars:
                self._error(
                    f"cannot rebind the outer array '{target.id}': the array object itself "
                    f"is shared with the caller, so a device kernel can only write its elements", node)
            self._error("cannot assign an array to a variable", node)

        if isinstance(target, ast.Name):
            outer:Optional[Dict[str, Any]] = self._named_outer_target(target, node)
            if outer is not None:
                self._emit(f"{target.id}[0] = {code};")
                return
            existing:Any = self._lookup(target.id)
            if existing is not None and is_ptr(existing):
                self._error(f"cannot assign to array parameter '{target.id}'", node)
            is_new:bool = (existing is None)
            self._define(target.id, t, node)
            decl:str = ""
            if is_new:
                decl = self._device_type(t) + " "
            self._emit(f"{decl}{target.id} = {code};")
        elif isinstance(target, ast.Subscript):
            # A string subscript on a structure value writes a member: `s["i"] = 1` becomes
            # `s.i = 1`. Recognized before the array path, which rejects a structure base
            # because a structure is a value, not a pointer.
            member:Optional[Tuple[str, Any]] = self._struct_member_access(target)
            if member is not None:
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
            base_code, base_t = self._visit_array_target(target.value)
            if is_image(base_t):
                self._emit_image_write(base_code, base_t, target, code)
                return
            index_code, _ = self._visit_subscript_index(target)
            self._emit(f"{base_code}[{index_code}] = {code};")
        elif isinstance(target, ast.Attribute):
            # `p.x = ...`: a C member store on a structure value, the write twin of the
            # attribute read. The read/write analysis cannot attribute the field store to
            # the caller's argument (it sees a member expression, not a subscript), so this
            # path is only reached for synthesized object arguments, whose host side copies
            # the final member values home after the launch.
            target_code, target_t = self._visit_target(target)
            self._emit(f"{target_code} = {code};")
        else:
            self._error(f"unsupported assignment target: {target.__class__.__name__}", node)

    def _named_outer_target(self, target:ast.Name, node:ast.AST)->Optional[Dict[str, Any]]:
        """Resolve a plain-name assignment to an outer variable, refusing what cannot work.

        A `nonlocal`/`global`-declared variable is written through its pointer, so
        `total = total + x` writes the scope value and the host copies it back after the
        launch. An undeclared outer name is refused: python would create a local there,
        which would silently shadow the scope variable instead of touching it.
        """
        outer:Optional[Dict[str, Any]] = self.outer_vars.get(target.id)
        if outer is None:
            return None
        if not outer["writable"]:
            self._error(
                f"cannot assign to the outer variable '{target.id}': without a "
                f"'nonlocal {target.id}' or 'global {target.id}' declaration this would "
                f"create a local variable, shadowing the outer one", node)
        if is_ptr(outer["type"]):
            self._error(
                f"cannot rebind the outer array '{target.id}': the array object itself is "
                f"shared with the caller, so a device kernel can only write its elements", node)
        return outer

    def _handle_local_declaration(self, node:ast.Assign)->None:
        """buf = gpu.local(np.float32, 64) -> a local memory array.

        opencl: append a __local parameter to the kernel signature, pass byte size at launch;
        cuda: declare a __shared__ array at the top of the function body.
        """
        call:ast.Call = node.value
        if len(call.args) != 2 or call.keywords:
            self._error("gpu.local(dtype, count) takes exactly 2 arguments", call)

        dtype_node:ast.expr = call.args[0]
        dtype_name:str = ""
        if isinstance(dtype_node, ast.Attribute) and isinstance(dtype_node.value, ast.Name):
            module:Any = self.resolve_global(dtype_node.value.id)
            if module is np:
                dtype_name = dtype_node.attr
            else:
                self._error("gpu.local() dtype must be a numpy dtype like np.float32", dtype_node)
        elif isinstance(dtype_node, ast.Constant) and isinstance(dtype_node.value, str):
            dtype_name = dtype_node.value
        else:
            self._error("gpu.local() dtype must be a numpy dtype like np.float32", dtype_node)

        if dtype_name not in DTYPE_MAP:
            self._error(f"gpu.local() unsupported dtype: {dtype_name}", dtype_node)

        ctype:str = DTYPE_MAP[dtype_name]
        if ctype == "half":
            self._error("gpu.local() does not support half (fp16)", dtype_node)

        count_node:ast.expr = call.args[1]
        if not (isinstance(count_node, ast.Constant) and isinstance(count_node.value, int) and count_node.value > 0):
            self._error("gpu.local() count must be a constant positive integer", count_node)

        c_name:str = f"_local{len(self.local_buffers)}"
        self.local_buffers.append((c_name, ctype, count_node.value))

        target:ast.expr = node.targets[0]
        if not isinstance(target, ast.Name):
            self._error("gpu.local() result must be assigned to a plain variable", node)
        self._define(target.id, ("lptr", ctype, c_name, count_node.value), node)

    def _is_local_buffer(self, node:ast.expr)->bool:
        """True when ``node`` names a `local(...)` buffer, i.e. work-group shared memory."""
        if not isinstance(node, ast.Name):
            return False
        t:Any = self._lookup(node.id)
        return isinstance(t, tuple) and len(t) == 4 and t[0] == "lptr"

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
            owner:Any = self.resolve_global(callee.value.id)
            if owner is None:
                return None
            return builtin_name_of(getattr(owner, callee.attr, None))
        if isinstance(callee, ast.Name):
            return builtin_name_of(self.resolve_global(callee.id))
        return None

    def _is_work_size_decl(self, node:ast.expr)->bool:
        """True for a `work_size(...)` statement call."""
        return self._builtin_call_name(node) == "work_size"

    def _handle_work_size_decl(self, call:ast.Call)->None:
        """`gpu.work_size(x)` declares the iteration domain as x's full shape.

        This is the gid-style counterpart of a magic loop: a kernel that dispatches on
        `gpu.get_global_id` has no loop to read the domain from, so without this the launch
        size cannot be derived. The statement is host-side only and emits no device code.

        The single argument is an array parameter, and it contributes ALL of its dimensions,
        exactly what `for ... in np.ndindex(x.shape)` would give: a 1-D array (the usual
        reduction) declares one dimension, an (h, w) array declares two. The generated kernel
        still clamps each work item against those bounds.
        """
        if not self.is_kernel:
            self._error("gpu.work_size() is only meaningful in a @kernel function", call)
        if call.keywords:
            self._error("gpu.work_size() takes no keyword arguments", call)
        if len(call.args) != 1:
            self._error(
                "gpu.work_size() takes exactly one array parameter: that array supplies the "
                "iteration domain, dimension by dimension (like np.ndindex(x.shape)). "
                "For a 1-D reduction this is gpu.work_size(x).", call)
        if self.work_size_dims or self.magic_loop_present:
            self._error(
                "gpu.work_size() is redundant here: this kernel's magic loop already "
                "declares the iteration domain", call)
        if self.declared_work_arrays:
            self._error("gpu.work_size() must appear only once per kernel", call)

        item:ast.expr = call.args[0]
        if not isinstance(item, ast.Name):
            self._error(
                "gpu.work_size() takes an array parameter, e.g. gpu.work_size(x)", item)
        name:str = item.id
        if not is_ptr(self._lookup(name)):
            self._error(
                f"gpu.work_size({name}) requires an array parameter; "
                f"'{name}' is not an array", item)

        # Record the array name; the concrete dimension count comes from its shape at launch
        # time (a pointer type carries no rank), which PyKernel expands at call time.
        self.declared_work_arrays = [name]

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
        namespace:Any = getattr(self.func, "__globals__", None)
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
        """
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
        self._emit("else {")
        self._indent += 1
        self._emit(f"double _pgap{n} = {gap % n};")
        # A power of two has only half that gap below it.
        self._emit(f"if (_pa{n} == ldexp(1.0, ilogb(_pa{n}))) _pgap{n} *= 0.5;")
        # A subnormal value has a fixed smallest step, so ldexp would underflow it.
        self._emit(f"if (_pa{n} < {smallest_normal}) _pgap{n} = {smallest_step};")
        self._emit(f"int _pe{n} = (int)floor(log10(_pa{n}));")
        self._emit(f"double _ps{n} = pow(10.0, (double)_pe{n});")
        self._emit(f"if (_ps{n} > _pa{n}) {{ _pe{n} -= 1; _ps{n} = pow(10.0, (double)_pe{n}); }}")
        self._emit(f"else if (_ps{n} * 10.0 <= _pa{n})"
                   f" {{ _pe{n} += 1; _ps{n} = pow(10.0, (double)_pe{n}); }}")
        # Half the gap either side, then the shortest budget whose rounding fits inside it.
        self._emit(f"double _plo{n} = _pa{n} - 0.5 * _pgap{n};")
        self._emit(f"double _phi{n} = _pa{n} + 0.5 * _pgap{n};")
        self._emit(f"int _pbest{n} = {max_digits};")
        self._emit(f"for (int _pk{n} = 0; _pk{n} < {max_digits}; _pk{n}++) {{")
        self._indent += 1
        # Round to _pk+1 significant digits: divide by 10^(e-_pk), round, multiply back.
        # The exponent is split in two so no factor leaves the double range.
        self._emit(f"int _pstep{n} = _pe{n} - _pk{n};")
        self._emit(f"int _phalf{n} = _pstep{n} / 2;")
        self._emit(f"double _pdown{n} = pow(10.0, (double)_phalf{n})"
                   f" * pow(10.0, (double)(_pstep{n} - _phalf{n}));")
        self._emit(f"double _pq{n} = round(_pa{n} / _pdown{n}) * _pdown{n};")
        self._emit(f"if (_pq{n} >= _plo{n} && _pq{n} <= _phi{n}) {{ _pbest{n} = _pk{n} + 1; break; }}")
        self._indent -= 1
        self._emit("}")
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
        self._emit(f"switch ((int){index}) {{ {' '.join(cases)} }}")

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
            outer:Dict[str, Any] = self.outer_vars[name]
            self._emit_atomic_update(node, op_name, atomic_name, outer["type"],
                                     array_code=name, index_code="0",
                                     pointer_code=pointer_code)
            return

        target:ast.Subscript = node.target  # type: ignore[assignment]
        # _visit_array_target already narrows a pointer to its element type.
        base_code, element_t = self._visit_array_target(target.value)
        if is_image(element_t):
            self._error(
                "'op=' on an image element is not supported; write `img[x, y] = value` "
                "or use gpu.atomic_* on an array buffer", node)
        index_code, _ = self._visit_subscript_index(target)
        self._emit_atomic_update(node, op_name, atomic_name, element_t,
                                 array_code=base_code, index_code=index_code,
                                 is_local=self._is_local_buffer(target.value))

    def _emit_atomic_update(self, node:ast.AugAssign, op_name:str, atomic_name:str,
                            element_t:Any, array_code:str, index_code:str,
                            pointer_code:Optional[str]=None,
                            is_local:bool=False)->None:
        """The shared tail of an atomic `op=` on an array element or an outer scalar."""
        if not is_int(element_t):
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
        call:str = emit_atomic(atomic_name, array_code, index_code, value_code,
                               element_t, self.backend, pointer_code=pointer_code,
                               is_local=is_local)
        self._emit(f"{call};")

    def _augassign_symbol(self, op_name:str)->str:
        return {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//",
                "Mod": "%", "Pow": "**", "LShift": "<<", "RShift": ">>",
                "BitAnd": "&", "BitOr": "|", "BitXor": "^"}.get(op_name, op_name)

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
        outer:Optional[Dict[str, Any]] = (
            self.outer_vars.get(node.target.id)
            if isinstance(node.target, ast.Name) else None)
        racy:bool = ((isinstance(node.target, ast.Subscript)
                      and not self._augassign_is_race_free(node.target))
                     or (outer is not None and bool(outer["writable"])))
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
            pointer:Optional[str] = node.target.id if outer is not None else None
            self._visit_atomic_augassign(node, op_name, atomic_name, pointer_code=pointer)
            return

        target_code, target_t = self._visit_target(node.target)
        value_code, value_t = self._visit_expression(node.value)

        name:str = op_name
        if name == "Div":
            if is_int(target_t) and is_int(value_t):
                self._error("integer /= integer changes type (use // or float operands)", node)
            self._emit(f"{target_code} /= {value_code};")
        elif name == "FloorDiv":
            if is_int(target_t) and is_int(value_t):
                self._emit(f"{target_code} /= {value_code};")
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
        elif name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor", "Add", "Sub", "Mult", "Mod"):
            symbol:str = {"LShift": "<<", "RShift": ">>", "BitOr": "|", "BitAnd": "&", "BitXor": "^",
                          "Add": "+", "Sub": "-", "Mult": "*", "Mod": "%"}[name]
            self._emit(f"{target_code} {symbol}= {value_code};")
        else:
            self._error(f"unsupported operator: {name}", node)

    def _lookup_type_of_index(self, node:ast.AugAssign)->Any:
        return "int"

    def _visit_target(self, target:ast.expr)->Tuple[str, Any]:
        if isinstance(target, ast.Name):
            outer:Optional[Dict[str, Any]] = self.outer_vars.get(target.id)
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
            t:Any = self._lookup(target.id)
            if t is None:
                self._error(f"variable '{target.id}' used before assignment", target)
            if is_ptr(t):
                self._error(f"cannot assign to array parameter '{target.id}'", target)
            return target.id, t
        if isinstance(target, ast.Subscript):
            # A string subscript on a structure value writes a member: `s["i"] = 1` becomes
            # `s.i = 1`. It has to be recognized before the array path, which would reject a
            # structure base (it is a value, not a pointer).
            member:Optional[Tuple[str, Any]] = self._struct_member_access(target)
            if member is not None:
                return member
            base_code, base_t = self._visit_array_target(target.value)
            index_code, _ = self._visit_subscript_index(target)
            return f"{base_code}[{index_code}]", base_t
        if isinstance(target, ast.Attribute):
            # `p.x = ...` on a structure value: a C member store. The read side already
            # handles attribute access on structures; this is its write twin. A written
            # object argument travels as a one-element record array, so the store goes
            # through the element (`p[0].x = ...`), which is what makes it reach the host.
            code:str
            t:Any
            code, t = self._visit_expression(target.value)
            if isinstance(t, tuple) and is_ptr(t):
                dtype:Any = structs_dtype_for(t[1])
                if dtype is None:
                    self._error(
                        f"cannot assign to '{ast.unparse(target)}': {t[1]} is not a "
                        f"structure the translator knows", target)
                member_types:Dict[str, Any] = structs_member_types(dtype, self.backend)
                if target.attr not in member_types:
                    self._error(
                        f"structure value has no member '{target.attr}' "
                        f"(available: {', '.join(member_types)})", target)
                # The static analysis sees a member expression, not a subscript, so it never
                # marks the array as written; without this flag the host would not read the
                # stores back. `atomic_targets` is exactly the force-writable channel.
                self.atomic_targets.add(target.value.id)
                return f"({code}[0]).{target.attr}", member_types[target.attr]
            if is_vec(t) or not isinstance(t, str) or structs_dtype_for(t) is None:
                self._error(
                    f"cannot assign to '{ast.unparse(target)}': member stores need a "
                    f"structure value, and {t} is not one", target)
            member_types:Dict[str, Any] = structs_member_types(structs_dtype_for(t), self.backend)
            if target.attr not in member_types:
                self._error(
                    f"structure value has no member '{target.attr}' "
                    f"(available: {', '.join(member_types)})", target)
            return f"({code}).{target.attr}", member_types[target.attr]
        self._error(f"unsupported assignment target: {target.__class__.__name__}", target)

    def _visit_array_target(self, node:ast.expr)->Tuple[str, Any]:
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
        self.subscript_indices[name] = max(self.subscript_indices.get(name, 0), count)

    def _visit_subscript_index(self, node:ast.Subscript)->Tuple[str, Any]:
        self._record_subscript_indices(node)
        index:ast.expr = node.slice
        if isinstance(index, ast.Tuple):
            array_name:str = self._array_name(node.value)
            dims:List[Tuple[str, Any]] = [self._visit_expression(e) for e in index.elts]
            terms:List[str] = []
            for d, (code, t) in enumerate(dims):
                if not is_int(t):
                    self._error("array indices must be integers", node)
                term:str = code
                for inner in range(d + 1, len(dims)):
                    term = f"{term} * {self._shape_var(array_name, inner)}"
                terms.append(term)
            return " + ".join(terms), "int"
        code, t = self._visit_expression(index)
        if not is_int(t):
            self._error("array index must be an integer", node)
        return code, t

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

    def _visit_expression(self, node:ast.expr)->Tuple[str, Any]:
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
            parts:List[str] = [self._visit_expression(v)[0] for v in node.values]
            symbol:str = ("&&" if isinstance(node.op, ast.And) else "||")
            joined:str = f" {symbol} ".join(parts)
            return f"({joined})", "bool"
        if isinstance(node, ast.IfExp):
            cond, _ = self._visit_expression(node.test)
            then_code, tt = self._visit_expression(node.body)
            else_code, et = self._visit_expression(node.orelse)
            if tt != et:
                unified:Any = promote(tt, et)
                return f"({cond} ? ({unified})({then_code}) : ({unified})({else_code}))", unified
            return f"({cond} ? {then_code} : {else_code})", tt
        if isinstance(node, ast.Call):
            return self._visit_call(node)
        if isinstance(node, ast.Subscript):
            return self._visit_subscript(node)
        if isinstance(node, ast.Attribute):
            return self._visit_attribute(node)
        self._error(f"unsupported expression: {node.__class__.__name__}", node)

    def _visit_constant(self, node:ast.Constant)->Tuple[str, Any]:
        value:Any = node.value
        if isinstance(value, bool):
            return ("true" if value else "false"), "bool"
        if isinstance(value, int):
            return str(value), "int"
        if isinstance(value, float):
            text:str = repr(value)
            if not ("." in text or "e" in text or "E" in text or "inf" in text or "nan" in text):
                text += ".0"
            if not text.endswith(("f", "F", "y")) and "inf" not in text:
                text += "f"
            return text, "float"
        self._error(f"unsupported constant: {value!r}", node)

    def _visit_name(self, node:ast.Name)->Tuple[str, Any]:
        host:Optional[Dict[str, Any]] = self.host_values.get(node.id)
        if host is not None:
            # The host created this object; on the device it is just the parameter that
            # carries it (see _host_created_assignment).
            return node.id, host["type"]
        outer:Optional[Dict[str, Any]] = self.outer_vars.get(node.id)
        if outer is not None:
            # A writable outer variable travels as `T*`, so every use is a dereference; a
            # read-only one travels as `T` and is simply the parameter itself.
            if outer["writable"]:
                return f"(*{node.id})", outer["type"]
            return node.id, outer["type"]
        t:Any = self._lookup(node.id)
        if t is not None:
            if isinstance(t, tuple) and len(t) == 4 and t[0] == "lptr":
                return t[2], (PTR, t[1])
            if isinstance(t, tuple) and len(t) == 4 and t[0] == "bodyarr":
                # A private array created in the body is indexed like any array; its extents
                # are constants (see _shape_var).
                return node.id, (PTR, t[3])
            return node.id, t
        value:Any = self.resolve_global(node.id)
        if isinstance(value, bool):
            return self._visit_constant(ast.Constant(value=value))
        if isinstance(value, (int, float)) and not isinstance(value, np.ndarray):
            return self._visit_constant(ast.Constant(value=value))
        self._error(f"unknown name '{node.id}' (only parameters, local variables and int/float/bool global constants are supported)", node)

    def _vec_componentwise(self, op:str, left:str, vec:str, right:str="", scalar:str="")->str:
        """CUDA has no vector arithmetic/math overloads; expand into per-component form.

        When op is an infix symbol (+ - * /) generate `l.c op r.c`; when it is a
        function name (sqrt etc.) generate `op(l.c, r.c)`.
        right is a vector operand, scalar is a broadcast scalar operand.
        """
        cls:Any = _vec_registry()[vec]
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

    def _visit_binop(self, node:ast.BinOp)->Tuple[str, Any]:
        left, lt = self._visit_expression(node.left)
        right, rt = self._visit_expression(node.right)
        name:str = node.op.__class__.__name__

        if is_vec(lt) or is_vec(rt):
            symbol:str = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
                          "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
                          "Eq": "==", "NotEq": "!="}.get(name)
            if symbol is None:
                self._error(f"operator {name} is not supported on vector types; operate per-component", node)

            if is_vec(lt) and is_vec(rt):
                if lt != rt:
                    self._error(f"vector operands must have the same type, got {lt} and {rt}", node)
                if self.backend == "opencl":
                    result_t:Any = (lt if name in ("Add", "Sub", "Mult", "Div") else self._get_int_vec_type(lt))
                    return f"({left} {symbol} {right})", result_t
                if name in ("Add", "Sub", "Mult", "Div"):
                    return self._vec_componentwise(symbol, left, lt, right=right), lt
                return self._vec_componentwise_compare(symbol, left, lt, right), self._get_int_vec_type(lt)

            # vector with scalar: broadcast per component
            vec, vec_side_left = (lt, True) if is_vec(lt) else (rt, False)
            if self.backend == "opencl":
                code:str = f"({left} {symbol} {right})" if vec_side_left else f"({right} {symbol} {left})"
                return code, vec
            if vec_side_left:
                return self._vec_componentwise(symbol, left, vec, scalar=right), vec
            return self._vec_componentwise(symbol, right, vec, scalar=left), vec

        if name == "Div":
            t:Any = promote(lt, rt) if not (is_int(lt) and is_int(rt)) else "float"
            if is_int(lt) and is_int(rt):
                return f"((({t})({left})) / (({t})({right})))", t
            return f"({left} / {right})", t

        if name == "FloorDiv":
            if is_int(lt) and is_int(rt):
                return f"({left} / {right})", promote(lt, rt)
            t:Any = promote(lt, rt)
            return f"(({t})floor(({t})({left}) / ({t})({right})))", t

        if name == "Pow":
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and node.right.value == 2:
                return f"({left} * {left})", lt
            t:Any = promote(lt, rt)
            if is_int(t):
                t = "float"
            return f"(({t})pow(({t})({left}), ({t})({right})))", t

        if name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor"):
            if not (is_int(lt) and is_int(rt)):
                self._error(f"operator {name} requires integer operands", node)
            symbol:str = {"LShift": "<<", "RShift": ">>", "BitOr": "|", "BitAnd": "&", "BitXor": "^"}[name]
            return f"({left} {symbol} {right})", promote(lt, rt)

        if name == "Mod":
            t:Any = promote(lt, rt)
            if is_int(t):
                return f"({left} % {right})", t
            return f"fmod(({t})({left}), ({t})({right}))", t

        symbol:str = {"Add": "+", "Sub": "-", "Mult": "*"}.get(name)
        if symbol is None:
            self._error(f"unsupported operator: {name}", node)
        return f"({left} {symbol} {right})", promote(lt, rt)

    def _visit_unaryop(self, node:ast.UnaryOp)->Tuple[str, Any]:
        code, t = self._visit_expression(node.operand)
        name:str = node.op.__class__.__name__
        if name == "USub":
            return f"(-{code})", t
        if name == "UAdd":
            return f"(+{code})", t
        if name == "Invert":
            if not is_int(t):
                self._error("~ requires an integer operand", node)
            return f"(~{code})", t
        if name == "Not":
            return f"(!{code})", "bool"
        self._error(f"unsupported unary operator: {name}", node)

    def _visit_compare(self, node:ast.Compare)->Tuple[str, Any]:
        symbols:Dict[str, str] = {"Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=", "Eq": "==", "NotEq": "!="}

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
            right, _ = self._visit_expression(comparator)
            parts.append(f"({left} {symbols[name]} {right})")
            left = right
        code = (" && ".join(parts)) if len(parts) > 1 else parts[0]
        return f"({code})", "bool"

    def _visit_subscript(self, node:ast.Subscript)->Tuple[str, Any]:
        value:ast.expr = node.value
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            array_name:str = self._array_name(value.value)
            t:Any = self._lookup(array_name)
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
        member:Optional[Tuple[str, Any]] = self._struct_member_access(node)
        if member is not None:
            return member

        base_code, base_t = self._visit_array_target(value)
        if is_image(base_t):
            return self._emit_image_read(base_code, base_t, node)
        index_code, _ = self._visit_subscript_index(node)
        return f"{base_code}[{index_code}]", base_t

    def _struct_member_access(self, node:ast.Subscript)->Optional[Tuple[str, Any]]:
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

        dtype:Optional[Any] = structs.dtype_for(base_t) if isinstance(base_t, str) else None
        if dtype is None:
            return None
        members:Dict[str, Any] = structs.member_types(dtype, self.backend)
        if field not in members:
            self._error(
                f"structure value has no member '{field}' "
                f"(available: {', '.join(members)})", node)
        # The parenthesised member expression is what C expects; a sub-array member then
        # takes a plain index, and a vector member takes a component.
        return f"({base_code}).{field}", members[field]

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

    def _emit_image_read(self, image_code:str, image_t:Any, node:ast.Subscript)->Tuple[str, Any]:
        """Translate ``img[x, y]`` into the sampling call of the current backend.

        Coordinates are texel space, as :mod:`pygpgpu.image` documents: an integer
        coordinate names a texel and samples its centre, while a fractional one
        addresses a position between texels. CUDA expresses both with a float
        coordinate, OpenCL needs int2 for the former and float2 for the latter, so
        the coordinate type decides which form is emitted.
        """
        self._reject_hip_image(node, writing=False)
        channels:int = image_channels(image_t)
        scalar:str = image_scalar(image_t)

        indices = self._image_indices(node)
        if channels not in (1, 4):
            self._error(
                "sampling an image of %d channels is not supported: use 1 or 4"
                % channels, node)

        if self.backend == "opencl":
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
            if self.backend == "opencl":
                return "(%s).xyzw" % call, self._image_vec_type(scalar, 4)
            return call, self._image_vec_type(scalar, 4)
        if self.backend == "opencl":
            # read_imagef returns a vector; single-channel takes .x
            return "(%s).x" % call, scalar
        return call, scalar

    def _image_indices(self, node:ast.Subscript)->List[Tuple[str, Any]]:
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

    def _image_param_type(self, name:str, image_t:Any)->str:
        """The declaration type of an image parameter on the current backend.

        CUDA reads through a texture object and writes through a surface object, so
        the parameter's type follows how the kernel uses it. OpenCL uses one image
        type for both directions, with a qualifier saying which.
        """
        if self.backend == "opencl":
            written:bool = name in self.required_image_surfaces
            qualifier:str = "__write_only" if written else "__read_only"
            return f"{qualifier} image2d_t"

        if name in self.required_image_surfaces:
            return "cudaSurfaceObject_t"
        return "cudaTextureObject_t"

    def _emit_image_write(self, image_code:str, image_t:Any,
                          target:ast.Subscript, value_code:str)->None:
        """Translate ``img[x, y] = v`` into the store call of the current backend.

        A write names one texel, so the coordinate is always integral here: a
        fractional subscript is rejected rather than silently rounded, because there
        is no filtering rule for a store.
        """
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

        if self.backend == "opencl":
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

    def _visit_attribute(self, node:ast.Attribute)->Tuple[str, Any]:
        # self.xxx -> extra parameter _self_xxx
        if (self.self_param and isinstance(node.value, ast.Name) and node.value.id == self.self_param):
            if node.attr not in self.self_attr_types:
                kind:str
                _target:Any
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
            module:Any = self.resolve_global(node.value.id)
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
        if is_vec(base_t):
            component:str = _vec_component_name(node.attr)
            if len(component) > 1:
                return self._emit_vec_swizzle(base_code, base_t, component, node)
            field_key:(str, str) = (base_t, component)
            if field_key not in _C_FIELD_TYPES:
                self._error(
                    f"vector type {base_t} has no component '.{node.attr}'", node)
            return f"({base_code}.{component})", _C_FIELD_TYPES[field_key]

        # A structure member, read with python's attribute spelling. It cannot be confused
        # with a vector component: the branch above has already claimed every base that is a
        # vector, so a name reaching here is only a member if the base is a structure.
        from . import structs

        if isinstance(base_t, tuple) and is_ptr(base_t):
            # `p.x` on a written object argument: it travels as a one-element record array,
            # so the member read goes through the element (`p[0].x`).
            dtype:Optional[np.dtype] = structs.dtype_for(base_t[1])
            if dtype is not None:
                members:Dict[str, Any] = structs.member_types(dtype, self.backend)
                if node.attr not in members:
                    self._error(
                        f"structure {base_t[1]} has no member '.{node.attr}' "
                        f"(available: {', '.join(members)})", node)
                return f"({base_code}[0]).{node.attr}", members[node.attr]

        if isinstance(base_t, str):
            dtype:Optional[np.dtype] = structs.dtype_for(base_t)
            if dtype is not None:
                members:Dict[str, Any] = structs.member_types(dtype, self.backend)
                if node.attr not in members:
                    self._error(
                        f"structure {base_t} has no member '.{node.attr}' "
                        f"(available: {', '.join(members)})", node)
                return f"({base_code}).{node.attr}", members[node.attr]

        self._error(f"unsupported attribute access: .{node.attr}", node)

    # ---------- calls ----------

    def _visit_call(self, node:ast.Call)->Tuple[str, Any]:
        func:ast.expr = node.func
        if node.keywords:
            # A helper may be called with keywords; nothing below needs to know, so the
            # call is rewritten to positional order first, using the callee's signature.
            # Builtins take positional arguments only, so they keep refusing.
            node = self._reorder_keywords(node)
            if node.keywords:
                self._error("keyword arguments are not supported for this call", node)

        # A builtin is recognized by the *object* the callee resolves to, never by the
        # spelling of the name, so it does not matter whether it was imported, reached
        # through the package, or is a user function that merely shares the name.
        builtin:Optional[str] = self._builtin_call_name(node)
        if builtin is not None:
            if builtin in ("local", "work_size"):
                self._error(
                    f"{builtin}() is a statement, not a value: use it on its own, as "
                    f"`name = {builtin}(...)`", node)
            args:List[Tuple[str, Any]] = [self._visit_expression(a) for a in node.args]
            return self._emit_gpu_builtin(builtin, args, node)

        if isinstance(func, ast.Name):
            name:str = func.id
            args = [self._visit_expression(a) for a in node.args]
            arg_codes:List[str] = [c for c, _ in args]

            if name in ("int", "float", "bool"):
                if len(args) != 1:
                    self._error(f"{name}() takes exactly 1 argument", node)
                t:Any = {"int": "int", "float": "float", "bool": "bool"}[name]
                return f"(({t})({arg_codes[0]}))", t

            if name in ("abs", "min", "max", "round", "pow"):
                return self._emit_builtin(name, args, node)

            if name == "len":
                return self._emit_len(args, node)

            vec_name:Optional[str] = self._vec_type_name(name)
            if vec_name is not None:
                return self._emit_vec_constructor(vec_name, args, node)
            reason:Optional[str] = self._unsupported_vec_reason(self.resolve_global(name))
            if reason is not None:
                self._error(reason, node)

            target:Any = self.resolve_global(name)
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
                kind:str
                target:Any
                kind, target = _self_member_call_kind(self.self_owner, func.attr)
                if kind in ("method", "staticmethod"):
                    args:List[Tuple[str, Any]] = [self._visit_expression(a) for a in node.args]
                    return self._emit_self_method_call(kind, target, func.attr, args, node)
                self._error(
                    f"calling self.{func.attr}() from device code is not supported: "
                    f"self.{func.attr} is {self._why_not_callable(kind, func.attr)}", node)
            module:Any = (self.resolve_global(receiver_name)
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
                    elementwise:Optional[Tuple[str, Any]] = self._emit_numpy_elementwise(
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

        self._error("unsupported call expression", node)

    def _emit_method_on_value(self, func:ast.Attribute, node:ast.Call)->Tuple[str, Any]:
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
        owner:Optional[type] = None
        if isinstance(base_t, str):
            dtype:Optional[np.dtype] = structs.dtype_for(base_t)
            if dtype is not None:
                owner = structs.declared_class(dtype)
            elif base_t in _vec_registry():
                owner = _vec_registry()[base_t]
        if owner is None:
            self._error(
                f"cannot call '.{func.attr}()' on a value of type {base_t}: methods are "
                f"translated from the class they are defined on, and only a "
                f"ctypes.Structure subclass or a vector type carries one", node)

        kind:str
        target:Any
        kind, target = self_attr_kind(owner, func.attr)
        if kind == "method":
            args = [self._visit_expression(a) for a in node.args]
            return self._emit_device_call(func.attr, target,
                                          [(base_code, base_t)] + args, node)
        if kind == "staticmethod":
            args = [self._visit_expression(a) for a in node.args]
            return self._emit_device_call(func.attr, target, args, node)
        print('METHOD-FAIL owner:', owner.__name__, 'id:', id(owner),
              'has attr:', hasattr(owner, func.attr),
              'file:', getattr(getattr(owner, func.attr, None), '__code__', None) and owner.__dict__.get(func.attr) is not None)
        self._error(
            f"cannot call '.{func.attr}()' on {owner.__name__}: it is "
            f"{self._why_not_callable(kind, func.attr)}", node)

    def _emit_builtin(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        if name == "pow":
            # python's builtin, which differs from math.pow only in that integer arguments
            # stay exact; the device form is the same floating-point power.
            if len(args) != 2:
                self._error("pow() takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            t:Any = promote(lt, rt)
            c_func:str = _math_device_name("pow", self.backend) if is_float(t) else "pow"
            return f"{c_func}(({t})({lcode}), ({t})({rcode}))", t

        if name == "abs":
            if len(args) != 1:
                self._error("abs() takes exactly 1 argument", node)
            code, t = args[0]
            if is_float(t):
                return f"fabs({code})", t
            if is_int(t):
                return f"abs({code})", t
            self._error("abs() requires a numeric argument", node)

        if name in ("min", "max"):
            if len(args) != 2:
                self._error(f"{name}() takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            t:Any = promote(lt, rt)
            if is_float(t):
                return f"{'fmin' if name == 'min' else 'fmax'}(({t})({lcode}), ({t})({rcode}))", t
            return f"{'min' if name == 'min' else 'max'}(({t})({lcode}), ({t})({rcode}))", t

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

    def _emit_len(self, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        """``len(array)`` -> the array's extent along its only axis.

        A kernel works on one-dimensional arrays and its extents arrive as int
        parameters, so the length is already known: recovering it here keeps
        ``for i in range(len(a))`` written the way it would be in Python.
        """
        if len(args) != 1:
            self._error("len() takes exactly 1 argument", node)
        code, t = args[0]
        if not is_ptr(t):
            self._error("len() takes an array argument", node)
        holder = node.args[0]
        if not isinstance(holder, ast.Name):
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
        target:Any = None
        if isinstance(func, ast.Name):
            target = self.resolve_global(func.id)
            if not callable(target) or getattr(target, "is_pykernel", False):
                return node
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and not (self.self_param and func.value.id == self.self_param)):
            module:Any = self.resolve_global(func.value.id)
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
                                      node:ast.Call)->Optional[Tuple[str, Any]]:
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
            bounds:List[Optional[str]] = []
            for bound_node in node.args[1:]:
                if isinstance(bound_node, ast.Constant) and bound_node.value is None:
                    bounds.append(None)
                    continue
                code, t = self._visit_expression(bound_node)
                if is_ptr(t) or is_image(t):
                    self._error("np.clip() bounds must be scalars, not arrays", bound_node)
                bounds.append(code)
            lo:Optional[str] = bounds[0] if bounds else None
            hi:Optional[str] = bounds[1] if len(bounds) > 1 else None
            # np.clip keeps an integer input an integer, so the element type is not forced
            # to float: `np.clip(300, 0, 255)` is 255, not a float.
            result_t:Any = value_t if is_int(value_t) else "float"
            return self._emit_clip(value_code, result_t, lo, hi), result_t
        if attr == "where":
            if len(node.args) == 1:
                self._error(
                    "np.where(cond) with one argument returns indices, which a kernel "
                    "cannot represent; use np.where(cond, a, b)", node)
            if len(node.args) != 3:
                self._error("np.where() takes exactly 3 arguments in a kernel", node)
            args:List[Tuple[str, Any]] = [self._visit_expression(a) for a in node.args]
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

    def _emit_numpy_elementwise(self, attr:str, args:List[Tuple[str, Any]],
                                node:ast.Call)->Optional[Tuple[str, Any]]:
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
                    "minimum", "maximum", "square", "reciprocal", "round"):
            for code, t in args:
                if is_ptr(t) or is_image(t):
                    self._error_numpy_array_form(attr, node)
        if attr in ("abs", "fabs"):
            # numpy has two spellings; the device picks by type, as python's abs does.
            code, t = args[0]
            if is_float(t):
                return f"fabs({code})", t
            if is_int(t):
                return f"abs({code})", t
            self._error(f"np.{attr}() requires a numeric argument", node)
        if attr == "square":
            code, t = args[0]
            return f"(({code}) * ({code}))", t
        if attr == "reciprocal":
            code, t = args[0]
            return f"(1.0f / ((float)({code})))", "float"
        if attr in ("minimum", "maximum"):
            if len(args) != 2:
                self._error(f"np.{attr}() takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            t:Any = promote(lt, rt)
            pick:str = "min" if attr == "minimum" else "max"
            if is_float(t):
                # `fmin`/`fmax` exist on both backends (`fminf` is the C spelling OpenCL
                # does not provide); the plain min/max are the integer forms.
                return f"{'fmin' if pick == 'min' else 'fmax'}(({t})({lcode}), ({t})({rcode}))", t
            return f"{pick}(({t})({lcode}), ({t})({rcode}))", t
        if attr in ("power", "arctan2"):
            # Same operation as math.pow / math.atan2 under numpy's spelling.
            return self._emit_math_call({"power": "pow", "arctan2": "atan2"}[attr],
                                        args, node, module_name="np")
        if attr == "round":
            # numpy rounds halfway cases to even, which is python's round too.
            return self._emit_builtin("round", args, node)
        if attr == "where":
            return self._emit_where(args, node)
        return None

    def _emit_where(self, args:List[Tuple[str, Any]],
                    node:ast.Call)->Tuple[str, Any]:
        """`np.where(cond, a, b)` as a conditional expression."""
        cond_code, cond_t = args[0]
        then_code, then_t = args[1]
        else_code, else_t = args[2]
        if not is_int(cond_t) and cond_t != "bool":
            self._error(
                f"np.where() needs a boolean or integer condition, got {cond_t}", node)
        unified:Any = promote(then_t, else_t)
        return (f"(({cond_code}) ? ({unified})({then_code}) : "
                f"({unified})({else_code}))"), unified

    def _emit_clip(self, value_code:str, result_t:Any,
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
        if self.backend == "opencl":
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

    def _clip_bound(self, bound:Tuple[str, Any],
                    node:Optional[ast.expr], call:ast.Call)->Optional[str]:
        """One bound of np.clip: its code, or None when the bound is None."""
        if node is None:
            return None
        code, t = bound
        if is_ptr(t) or is_image(t):
            self._error(
                "np.clip() bounds must be scalars, not arrays", call)
        return code

    def _emit_math_call(self, attr:str, args:List[Tuple[str, Any]], node:ast.Call, module_name:str="math")->Tuple[str, Any]:
        if attr not in _MATH_FUNCS:
            self._error(f"{module_name}.{attr} is not supported", node)
        ret_override:Optional[str] = _MATH_FUNCS[attr]
        arg_codes:List[str] = []
        ret:Any = (ret_override or "float")
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
            if self.backend == "opencl":
                return f"{attr}({args[0][0]})", vec_arg
            return self._vec_componentwise(_math_device_name(attr, "cuda"), args[0][0], vec_arg), vec_arg

        if attr in ("degrees", "radians"):
            # No CUDA intrinsic for either; the conversion factor is the definition.
            factor:str = ("57.29577951308232f" if attr == "degrees"
                          else "0.017453292519943295f")
            if self.backend == "opencl":
                return f"{attr}({args[0][0]})", ret
            return f"(({args[0][0]}) * {factor})", ret

        if attr in ("isnan", "isinf", "isfinite"):
            # The predicate takes a float; casting it to the int result type is what made the
            # CUDA call ambiguous. The device answers 0/1, which is the number python's bool
            # stands for.
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
        c_func:str = _math_device_name(attr, self.backend)
        return f"{c_func}({', '.join(arg_codes)})", ret

    def _emit_numpy_cast(self, name:str, args:List[Tuple[str, Any]],
                         node:ast.Call)->Tuple[str, Any]:
        """np.float32(x) / np.int32(x): a plain C cast to the matching device type."""
        if len(args) != 1:
            self._error(f"np.{name}() takes exactly 1 argument", node)
        code, source_t = args[0]
        if is_ptr(source_t) or is_vec(source_t) or is_image(source_t):
            self._error(f"np.{name}() cannot convert a {source_t}", node)
        target_t:Any = DTYPE_MAP.get(name)
        if target_t is None:
            self._error(f"np.{name}() has no device equivalent", node)
        return f"(({self._device_type(target_t)})({code}))", target_t

    def _unsupported_vec_reason(self, target:Any)->Optional[str]:
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
        target:Any = self.resolve_global(name)
        if not isinstance(target, type):
            return None
        for base in target.__mro__:
            if base.__name__ in _vec_registry():
                return base.__name__
        return None

    def _emit_structure_constructor(self, name:str, dtype:np.dtype,
                                    args:List[Tuple[str, Any]], node:ast.expr)->Tuple[str, Any]:
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
        members:Dict[str, Any] = structs.member_types(dtype, self.backend)
        order:List[str] = list(dtype.names or ())
        supplied:Dict[str, Tuple[str, Any]] = {}

        if len(args) > len(order):
            self._error(
                f"{name}() takes at most {len(order)} arguments "
                f"({', '.join(order)}), got {len(args)}", node)
        for position, (code, t) in enumerate(args):
            supplied[order[position]] = (code, t)

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

        if self.backend == "opencl":
            # OpenCL C is C99, so a designated initialiser names the members it sets.
            return f"(({name}){{{', '.join(parts)}}})", name
        # CUDA and HIP compile as C++, where designated initialisers arrived only in C++20
        # (NVRTC is older); a braced aggregate takes its members in declaration order.
        return f"({name}{{{', '.join(positional_parts)}}})", name

    def _zero_value(self, member_type:Any)->str:
        """The zero of a member's device type, for an omitted constructor argument."""
        from . import structs

        if is_ptr(member_type):
            return "0"
        if is_vec(member_type):
            return self._vec_maker(member_type,
                                   ", ".join("0" for _ in _vec_field_names(member_type)))
        if isinstance(member_type, str) and structs.dtype_for(member_type) is not None:
            dtype:Any = structs.dtype_for(member_type)
            inner:str = ", ".join(self._zero_value(structs.member_types(dtype, self.backend)[m])
                                  for m in dtype.names or ())
            return f"({member_type}{{{inner}}})"
        return f"(({self._device_type(member_type)})(0))"

    def _structure_member_value(self, struct_name:str, member:str, member_type:Any,
                                code:str, t:Any, node:ast.expr)->str:
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
        return f"(({self._device_type(member_type)})({code}))"

    def _emit_vec_constructor(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        cls:Any = _vec_registry()[name]
        # A 3-component vector is padded to 16 bytes for the device ABI, so its ctypes struct
        # carries a trailing `_` field that is not a component: `float3(a, b, c)` takes
        # three arguments, not four. Count the real components, not the storage fields.
        components:List[Tuple[str, Any]] = [f for f in cls._fields_ if f[0] != "_"]
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
        if self.backend == "opencl":
            return f"({name})({', '.join(codes)})", name
        return f"make_{name}({', '.join(codes)})", name

    # ---------- gpu.* device builtins ----------

    def _emit_gpu_builtin(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        if name in INDEX_BUILTINS:
            if len(args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, int):
                self._error(f"gpu.{name}(dim) requires a constant integer dimension (0, 1 or 2)", node)
            if name in ("get_global_id", "get_global_size"):
                self.uses_global_id = True
            return emit_index(name, node.args[0].value, self.backend), "int"

        if name == "barrier":
            if args:
                self._error("gpu.barrier() takes no arguments", node)
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
            val_code, val_t = args[-1]
            compare_code:Optional[str] = (args[2][0] if name == "atomic_cas" else None)
            if not is_ptr(arr_t):
                shared:Optional[Dict[str, Any]] = (
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

        if name in GENERIC_BUILTINS:
            return self._emit_generic_builtin(name, args, node)

        self._error(f"unknown gpu builtin: gpu.{name}", node)

    def _emit_generic_builtin(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        arg_codes:List[str] = [c for c, _ in args]
        arg_types:List[Any] = [t for _, t in args]
        vec_t:Any = next((t for t in arg_types if is_vec(t)), None)

        if name in ("rsqrt", "exp2"):
            if len(args) != 1:
                self._error(f"gpu.{name}() takes exactly 1 argument", node)
            code, t = args[0]
            if is_vec(t):
                if self.backend == "opencl":
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
            result_t:Any = vec_t or args[0][1]
            if self.backend == "opencl":
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
                if self.backend == "opencl":
                    return f"clamp({', '.join(arg_codes)})", vec_t
                return self._emit_vec_clamp(arg_codes, arg_types, vec_t), vec_t
            if self.backend == "opencl":
                return f"clamp({', '.join(arg_codes)})", args[0][1]
            return f"fminf(fmaxf({arg_codes[0]}, {arg_codes[1]}), {arg_codes[2]})", args[0][1]

        if name == "mix":
            if len(args) != 3:
                self._error("gpu.mix(a, b, t) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self.backend == "opencl":
                    return f"mix({', '.join(arg_codes)})", vec_t
                return self._emit_vec_mix(arg_codes, arg_types, vec_t), vec_t
            a, b, t = arg_codes
            return f"(({a}) * (1.0f - ({t})) + ({b}) * ({t}))", args[0][1]

        if name == "mad":
            if len(args) != 3:
                self._error("gpu.mad(a, b, c) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self.backend == "opencl":
                    return f"mad({', '.join(arg_codes)})", vec_t
                return self._emit_vec_mad(arg_codes, arg_types, vec_t), vec_t
            a, b, c = arg_codes
            return f"((({a}) * ({b})) + ({c}))", args[0][1]

        if name == "saturate":
            if len(args) != 1:
                self._error("gpu.saturate(x) takes exactly 1 argument", node)
            code, t = args[0]
            if vec_t is not None:
                if self.backend == "opencl":
                    return f"clamp({code}, 0.0f, 1.0f)", vec_t
                return self._emit_vec_clamp([code], [t], vec_t), vec_t
            if self.backend == "opencl":
                return f"clamp({code}, 0.0f, 1.0f)", t
            return f"fminf(fmaxf({code}, 0.0f), 1.0f)", t

        if name == "step":
            if len(args) != 2:
                self._error("gpu.step(edge, x) takes exactly 2 arguments", node)
            (edge_code, edge_t), (x_code, x_t) = args
            if vec_t is not None and self.backend == "opencl":
                return f"step({edge_code}, {x_code})", vec_t
            if vec_t is not None:
                cls:Any = _vec_registry()[vec_t]
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
            if self.backend == "opencl":
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
                if self.backend == "opencl":
                    int_vec:str = self._get_int_vec_type(a_t)
                    return f"select({b_code}, {a_code}, ({c_code} == ({int_vec})(0)))", a_t
                return self._emit_vec_select(a_code, b_code, c_code, a_t), a_t
            if self.backend == "opencl":
                return f"select({b_code}, {a_code}, ({c_code} == 0))", a_t
            return f"(({c_code}) ? ({b_code}) : ({a_code}))", a_t

        if name in ("isequal", "isnotequal", "isgreater", "isgreaterequal", "isless", "islessequal", "isordered", "isunordered"):
            if len(args) != 2:
                self._error(f"gpu.{name}(a, b) takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            if is_vec(lt) or is_vec(rt):
                if not (is_vec(lt) and is_vec(rt)) or lt != rt:
                    self._error(f"gpu.{name}() vector operands must have the same type", node)
                if self.backend == "opencl":
                    # opencl vector relation functions use -1 (all-1 bits) for true; & 1 normalizes it to 1
                    return f"({name}({lcode}, {rcode}) & 1)", self._get_int_vec_type(lt)
                return self._emit_vec_relation(name, lcode, rcode, lt), self._get_int_vec_type(lt)
            if self.backend == "opencl":
                return f"{name}({lcode}, {rcode})", "int"
            return self._emit_scalar_relation(name, lcode, rcode), "int"

        if name in ("any", "all"):
            if len(args) != 1 or not is_vec(args[0][1]):
                self._error(f"gpu.{name}(cond_vec) takes exactly 1 vector argument", node)
            code, t = args[0]
            if self.backend == "opencl":
                return f"{name}({code})", "int"
            cls:Any = _vec_registry()[t]
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
            if self.backend == "opencl":
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

    def _is_scalar(self, t:Any)->bool:
        return isinstance(t, str) and not is_vec(t)

    def _component(self, code:str, t:Any, c:str)->str:
        return f"{code}.{c}" if is_vec(t) else code

    def _emit_vec_clamp(self, codes:List[str], types:List[Any], vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"fminf(fmaxf({self._component(codes[0], types[0], c)}, {self._component(codes[1], types[1], c)}), {self._component(codes[2], types[2], c)})"
            for c in components)
        return f"{maker}({inner})"

    def _emit_vec_mix(self, codes:List[str], types:List[Any], vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({self._component(codes[0], types[0], c)}) * (1.0f - ({self._component(codes[2], types[2], c)})) + ({self._component(codes[1], types[1], c)}) * ({self._component(codes[2], types[2], c)}))"
            for c in components)
        return f"{maker}({inner})"

    def _emit_vec_swizzle(self, code:str, vec:str, swizzle:str, node:ast.Attribute)->Tuple[str, Any]:
        """Read a swizzle: opencl uses native v.xy; cuda generates per-component make_float2(v.x, v.y)."""
        components:List[Tuple[str, Any]] = _vec_components(vec)
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

        if self.backend == "opencl":
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
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        int_vec:str = self._get_int_vec_type(vec)
        terms:List[str] = [f"(({left}.{c} {op} {right}.{c}) ? 1 : 0)" for c in components]
        inner:str = ", ".join(terms)
        if self.backend == "opencl":
            return f"({int_vec})({inner})"
        return f"make_{int_vec}({inner})"

    def _emit_vec_mad(self, codes:List[str], types:List[Any], vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({self._component(codes[0], types[0], c)}) * ({self._component(codes[1], types[1], c)})) + ({self._component(codes[2], types[2], c)})"
            for c in components)
        return f"{maker}({inner})"

    def _emit_vec_select(self, a_code:str, b_code:str, c_code:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({c_code}.{c}) ? ({b_code}.{c}) : ({a_code}.{c}))" for c in components)
        return f"{maker}({inner})"

    def _emit_vec_relation(self, name:str, lcode:str, rcode:str, vec:str)->str:
        op:str = {"isequal": "==", "isnotequal": "!=", "isgreater": ">",
                  "isgreaterequal": ">=", "isless": "<", "islessequal": "<="}[name]
        cls:Any = _vec_registry()[vec]
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
        if self.backend == "opencl":
            return f"({vec})({inner})"
        return f"make_{vec}({inner})"

    def _emit_vec_absdiff(self, lcode:str, rcode:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(f"abs(({lcode}.{c}) - ({rcode}.{c}))" for c in components)
        return self._vec_maker(vec, inner)

    def _emit_vec_hadd(self, lcode:str, rcode:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(
            f"((({lcode}.{c}) & ({rcode}.{c})) + ((({lcode}.{c}) ^ ({rcode}.{c})) >> 1))" for c in components)
        return self._vec_maker(vec, inner)

    def _emit_vec_rhadd(self, lcode:str, rcode:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(
            f"((({lcode}.{c}) | ({rcode}.{c})) - ((({lcode}.{c}) ^ ({rcode}.{c})) >> 1))" for c in components)
        return self._vec_maker(vec, inner)

    def _emit_vec_dot(self, lcode:str, rcode:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        terms:str = " + ".join(f"(({lcode}.{c}) * ({rcode}.{c}))" for c in components)
        return f"({terms})"

    def _emit_vec_builtin(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        opencl_name, _ = GENERIC_BUILTINS[name]
        expected_args:int = (1 if name in ("normalize", "length") else 2)
        if len(args) != expected_args:
            self._error(f"gpu.{name}() takes exactly {expected_args} arguments", node)

        vec_types:List[Any] = [t for _, t in args if is_vec(t)]
        if not vec_types:
            self._error(f"gpu.{name}() requires vector arguments "
                        f"(python has no such functions; use fabs/abs for scalars)", node)
        vec:str = vec_types[0]
        if any(t != vec for t in vec_types):
            self._error(f"gpu.{name}() operands must have the same vector type", node)

        if self.backend == "opencl":
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
        cls:Any = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        dot_expr:str = self._emit_vec_dot(code, code, vec)
        inner:str = ", ".join(f"(({code}.{c}) * rsqrtf({dot_expr}))" for c in components)
        return f"{maker}({inner})"

    def _emit_vec_cross(self, lcode:str, rcode:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
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

    def _emit_device_call(self, name:str, target:Any, args:List[Tuple[str, Any]], node:ast.Call,
                          self_param:str="")->Tuple[str, Any]:
        is_kernel_ref:bool = getattr(target, "is_pykernel", False)
        raw_func:Callable = (target.raw_func if is_kernel_ref else target)
        if is_kernel_ref:
            self._error(f"'{name}' is a @kernel function and cannot be called from device code; use a plain python function instead", node)
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
        self_types:Dict[str, Any] = (self.self_attr_types if self_param else {})
        call_codes:List[str] = [c for c, _ in args]
        if self_param:
            call_codes.extend(f"_self_{attr}" for attr in self_types)
        param_types:List[Any] = ([t for _, t in args] + list(self_types.values()))

        sig_key:Tuple[Callable, Tuple[Any, ...]] = (raw_func, tuple(param_types))
        entry:Optional[Dict[str, Any]] = self.device_cache.get(sig_key)
        if entry is not None:
            if entry["status"] == "in_progress":
                self._error(f"recursive call detected: '{name}'", node)
            argument_codes:List[str] = (call_codes + list(entry.get("shape_codes") or [])
                                        + list(entry.get("outer_codes") or []))
            return (f"{entry['c_name']}({', '.join(argument_codes)})",
                    entry["return_type"])

        self.device_cache[sig_key] = {"status": "in_progress"}
        try:
            callee:FunctionTranslator = FunctionTranslator(
                raw_func, param_types, self.backend, _helper_globals(raw_func, self.resolve_global),
                self.device_cache,
                is_kernel=False, self_param=self_param, self_attr_types=self_types,
                self_owner=self.self_owner, launch_backend=self.launch_backend)
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
                "code": callee.render_device(c_name, shape_params),
                "shape_codes": shape_codes,
                "outer_codes": outer_codes,
                # Structure types this helper's signature or body mentions: the caller emits
                # their typedefs once, ahead of every device function.
                "structs": list(callee.private_structs),
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
        return (f"{entry['c_name']}({', '.join(argument_codes)})", entry["return_type"])

    def _require_outer_parameter(self, name:str, outer:Dict[str, Any], node:ast.AST)->None:
        """Make sure the kernel carrying a helper's outer variable receives it too.

        The kernel's own scan only sees names its body mentions; a helper reached from it
        may reference more. The parameter is appended to the kernel's signature here, so
        the launch supplies it from the same scope (see PyKernel._compile).
        """
        if name in self.outer_vars:
            existing:Dict[str, Any] = self.outer_vars[name]
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

    def _outer_argument(self, name:str, outer:Dict[str, Any])->str:
        """The call-site expression that forwards an outer variable to a helper."""
        return f"&{name}" if outer["writable"] else name

    def _emit_self_method_call(self, kind:str, target:Any, attr:str,
                               args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
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

    def _device_func_name(self, func:Callable, types:List[Any])->str:
        module_file:str = getattr(getattr(func, "__code__", None), "co_filename", "")
        base:str = (os.path.basename(module_file).replace(".", "_") + "_" + func.__name__) if module_file else func.__name__
        return f"pydev_{base}_{short_hash(type_sig(types))}"

    def _device_qualifier(self)->str:
        """The storage qualifier a device function needs on this backend."""
        return "__device__ " if self.backend == "cuda" else ""

    def render_device(self, c_name:str,
                      shape_params:Optional[List[Tuple[str, int]]]=None)->str:
        """Render a device function.

        ``shape_params`` are the (array, dimension) extents this helper indexes. They travel
        as extra int parameters, because a pointer carries no extents and the helper's
        multi-dimensional subscripting needs them.
        """
        params:List[str] = []
        outer_names:Set[str] = set(self.outer_vars) | set(self.host_values)
        for name, t in self.params:
            if name in outer_names:
                continue
            if is_ptr(t):
                params.append(f"{self._device_type(t[1])}* {name}")
            else:
                params.append(f"{self._device_type(t)} {name}")
        # A structure in the signature needs its typedef emitted too, and this helper's own
        # `private_structs` is what the caller collects.
        from . import structs

        for candidate in [t[1] if is_ptr(t) else t for _, t in self.params]:
            dtype:Any = structs.dtype_for(candidate) if isinstance(candidate, str) else None
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
        ret:str = (self.return_type or "void")
        header:str = f"{self._device_qualifier()}{self._device_type(ret)} {c_name}({', '.join(params)})"
        body:str = "\n".join(self.lines)
        # No typedef block here: the caller emits every structure type once, ahead of all
        # the device functions (see PyKernel._compile). Repeating it per helper would define
        # the same struct twice, which C++ rejects.
        return f"{header} {{\n{body}\n}}"

    def _outer_param_declarations(self, device:bool=False)->List[str]:
        """C declarations for the parameters the host fills in at launch time.

        This covers both the outer-scope variables and the objects the kernel body
        creates: neither belongs to the python argument list, and both are appended to the
        device signature after everything the work size is derived from.
        """
        declarations:List[str] = []
        for name, outer in self.outer_vars.items():
            if is_ptr(outer["type"]):
                qualifier:str = ("" if device else
                                 ("__global " if self.backend == "opencl" else ""))
                declarations.append(f"{qualifier}{self._device_type(outer['type'][1])}* {name}")
            elif outer["writable"]:
                element:str = (self._return_param_type(outer["type"]) if device
                               else self._kernel_scalar_type(outer["type"]))
                qualifier = ("" if device else
                             ("__global " if self.backend == "opencl" else ""))
                declarations.append(f"{qualifier}{element}* {name}")
            else:
                declarations.append(f"{self._kernel_scalar_type(outer['type'])} {name}")
        for name, entry in self.host_values.items():
            declarations.append(f"{self._host_value_declaration(name, entry, device)}")
        return declarations

    def _host_value_declaration(self, name:str, entry:Dict[str, Any], device:bool)->str:
        """The parameter declaration for one host-created object."""
        t:Any = entry["type"]
        if is_ptr(t):
            qualifier:str = ("" if device else
                             ("__global " if self.backend == "opencl" else ""))
            return f"{qualifier}{self._device_type(t[1])}* {name}"
        return f"{self._kernel_scalar_type(t)} {name}"

    def _device_type(self, t:str)->str:
        """The backend's spelling of a scalar type name.

        The translator's internal names follow OpenCL (uchar, ushort, uint, ulong). CUDA
        C does not define those four, so the CUDA path -- which HIP shares -- writes the
        full C form instead.
        """
        if self.backend == "opencl":
            return t
        return _CUDA_TYPE_NAMES.get(t, t)

    def _kernel_scalar_type(self, t:Any)->str:
        """A kernel parameter's C type.

        `bool` is not portable as a kernel parameter: OpenCL C refuses a `_Bool` kernel
        argument outright, and the CUDA/HIP argument binding has no bool entry. It travels
        as a byte instead, which every backend accepts; the device reads 0/1.
        """
        if t == "bool":
            return self._device_type("uchar")
        return self._device_type(t)

    def render_kernel(self, c_name:str)->str:
        # Outer-scope parameters are held in self.params so the host can index launch
        # arguments by name, but they are *declared* last: the host appends them after the
        # work-size/sampler/local-size arguments it derives (see PyKernel._invoke_inner).
        outer_names:Set[str] = set(self.outer_vars) | set(self.host_values)
        params:List[str] = []
        for name, t in self.params:
            if name in outer_names:
                continue
            if is_ptr(t):
                qualifier:str = ("__global " if self.backend == "opencl" else "")
                params.append(f"{qualifier}{self._kernel_scalar_type(t[1])}* {name}")
            elif is_image(t):
                params.append(f"{self._image_param_type(name, t)} {name}")
            else:
                params.append(f"{self._kernel_scalar_type(t)} {name}")
        for array_name, dim in self.shapes_used:
            params.append(f"const int {self._shape_var(array_name, dim)}")
        # OpenCL sampling takes the sampler as a kernel argument; one per sampled
        # image, right after that image. The host binds a matching sampler per image,
        # so the Python-visible signature stays (image) only.
        if self.backend == "opencl":
            for image_name in sorted(self.required_image_samplers):
                params.append(f"const sampler_t {self._sampler_name(image_name)}")
        for lc_name, lc_type, _ in self.local_buffers:
            if self.backend == "opencl":
                params.append(f"__local {self._device_type(lc_type)}* {lc_name}")

        # Outer-scope parameters after everything the launch derives the work size from,
        # matching the order the host appends them in (see PyKernel._invoke_inner).
        params.extend(self._outer_param_declarations(device=False))

        # Return slots come last, after every parameter the work size and samplers derive
        # from, so appending them at launch time cannot disturb those positions.
        params.extend(self.render_return_params())

        prefix:str = ("__kernel " if self.backend == "opencl" else "__global__ ")
        header:str = f"{prefix}void {c_name}({', '.join(params)})"
        body_lines:List[str] = []
        if self.backend == "cuda":
            for lc_name, lc_type, lc_count in self.local_buffers:
                # `_device_type` because CUDA C has no `uint`: the shared array has to be
                # declared in the backend's spelling, not the translator's internal one.
                body_lines.append(
                    f"__shared__ {self._device_type(lc_type)} {lc_name}[{lc_count}];")
        body_lines.extend(self.lines)
        body:str = "\n".join(body_lines)
        # The structure types are emitted once by the caller, ahead of the whole source (see
        # PyKernel._compile): a helper may need them too, and two copies of the same struct
        # is an error in C++.
        return f"{header} {{\n{body}\n}}"

    def structure_dtypes(self)->List[Any]:
        """Every structure dtype this translation mentions, in no particular order.

        The caller emits one typedef block for the whole source, so the kernel and each
        helper contribute their own set and duplicates are dropped.
        """
        from . import structs

        dtypes:List[Any] = []
        for name, t in list(self.params):
            candidate:Any = t[1] if is_ptr(t) else t
            dtype:Any = structs.dtype_for(candidate) if isinstance(candidate, str) else None
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

        dtypes:List[Any] = []
        for name, t in list(self.params):
            for candidate in (t[1] if is_ptr(t) else t,):
                dtype:Any = structs.dtype_for(candidate) if isinstance(candidate, str) else None
                if dtype is not None and dtype not in dtypes:
                    dtypes.append(dtype)
        for dtype in self.private_structs:
            if dtype not in dtypes:
                dtypes.append(dtype)
        for entry in (self.return_layout or []):
            candidate:Any = entry["type"]
            dtype = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in dtypes:
                dtypes.append(dtype)
        return structs.render_structures(dtypes, self.backend)
