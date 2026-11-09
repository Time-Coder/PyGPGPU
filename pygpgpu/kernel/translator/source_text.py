# Reading a kernel's source, and the AST shims that make one tree parse everywhere.
from __future__ import annotations
import ast
import functools
import inspect
import sys
import textwrap
from typing import (Callable, Dict, List)



@functools.lru_cache(maxsize=512)
def get_dedented_source(func:Callable)->str:
    """Read the function source, drop decorator lines and dedent (class methods are indented; ast.parse needs a top-level def).

    Cached per function object, because one launch reads the same body several times -- the
    phase split, the host-region scan and the creation scan each want the text -- and
    `inspect.getsource` re-reads the file and re-tokenizes it on every call. A function's
    source cannot change after it is defined, so the answer is safe to keep; the bound keeps
    a program that creates kernels in a loop from growing the table without limit.
    """
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

    The version gate is what makes that possible. `hasattr(ast, "Num")` is *not* a usable
    test on 3.12+: the aliases still exist there, but their `__instancecheck__` raises a
    `DeprecationWarning`, so every `isinstance` below warned once per node -- thousands of
    `warnings.warn` calls per launch, which made re-parsing the source the single largest
    cost of a kernel call. From 3.9 the parser already answers the modern shapes, so the
    whole walk is skipped there.
    """
    if sys.version_info >= (3, 9):
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
    # Upward call: `walk` owns the magic-loop question and imports this module.
    from .walk import _is_magic_loop_stmt
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


@functools.lru_cache(maxsize=256)
def _parse_kernel_source(source:str)->ast.Module:
    """Parse a kernel source string into an AST normalized across Python versions.

    The result is cached by the source text, because one launch asks for the same body
    several times -- the phase split, the host-region scan and the creation scan each parse
    it -- and a kernel's source never changes. The tree is returned to callers that only
    read it; the translators that *rewrite* a body build their own tree from the same text.
    """
    return _normalize_ast_legacy_nodes(ast.parse(source, mode="exec"))
