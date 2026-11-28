# The helper closure a kernel inlines, as a content digest.
"""A kernel's device code is not its body alone.

A body that calls a plain python helper monomorphizes that helper inside the kernel's own
translation unit: the helper's text reaches the generated source, and so does the text of
every helper that helper calls. What the kernel *is* therefore reaches past its own
function, and the identities built from its own source alone -- the in-process cache key
and the generated C name -- answer "unchanged" for a kernel whose helper changed. The
device then silently answered the old helper's numbers, which is exactly the failure a
compile cache must never produce.

The digest below is the cheap, static half of that identity: it walks the kernel's body
(and, recursively, each helper's body) for the calls a translation would follow, and folds
every reachable helper's source text into one hash. It runs before the cache is consulted,
on the same budget the launch path already pays for `baked_shared_shapes`: the parse is
the cached tree, and the walk is a plain AST pass. Only the *source* is read, never
executed, and a name that cannot be resolved or whose source cannot be read contributes
nothing -- the translator reports such a call itself, with a position, when the body
really runs it.

What is deliberately not covered:

* a module constant's **value**. A constant a body reads is an outer-scope *parameter*
  (see `_bind_outer_variables`), and the launch reads it from its own scope on every
  call -- a changed value is live, exactly like a changed closure cell, and needs no
  recompile. What the digest must catch is a changed *reference* (new helper text), which
  the source fold below carries.
* a method reached through `self.` -- its class is part of the receiver's identity, and
  the attributes it reads are re-scanned per launch by `scan_self_attrs`.
* a helper name rebound to a *new* function object without the module's own code moving:
  the translator resolves such a call live, so the digest's resolution follows the same
  rule rather than a frozen snapshot.
"""
from __future__ import annotations
import ast
import types
from typing import Callable, List, Optional, Set

from ..translator import (
    get_dedented_source,
    parse_kernel_body,
    short_hash,
)


def _callee_of(node:ast.Call, func:Callable,
               resolve_global:Callable[[str], object])->Optional[Callable]:
    """The plain python function one call node names, or None when it is not one.

    The resolution mirrors the translator's own: a bare name resolves in the function's
    globals first and falls back to the caller's resolver (see `_helper_globals`), and a
    `module.attr` call resolves the module object first. A builtin, a pygpgpu-internal
    function (the dual-sided builtins among them -- their spelling never changes with the
    user's code) and a @kernel (a launch, not a helper) are all left out.
    """
    callee:ast.expr = node.func
    own:dict = getattr(func, "__globals__", None) or {}
    target:object = None
    if isinstance(callee, ast.Name):
        if callee.id in own:
            target = own[callee.id]
        else:
            target = resolve_global(callee.id)
    elif isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
        base_name:str = callee.value.id
        base:object = own[base_name] if base_name in own else resolve_global(base_name)
        if isinstance(base, types.ModuleType):
            target = getattr(base, callee.attr, None)
    if not isinstance(target, types.FunctionType):
        return None
    if getattr(target, "is_pykernel", False):
        return None
    if str(getattr(target, "__module__", "") or "").startswith("pygpgpu"):
        return None
    return target


def _helper_closure(func:Callable, resolve_global:Callable[[str], object],
                    visited:Set[int], found:List[Callable])->None:
    """Collect every transitive plain-python helper ``func`` calls, depth first.

    ``visited`` is keyed by function identity and covers the kernel itself, so a chain
    that meets a helper twice records it once. A body whose source cannot be read or
    parsed contributes no calls -- the same source would fail the translation too.
    """
    if id(func) in visited:
        return
    visited.add(id(func))
    try:
        tree:ast.Module = parse_kernel_body(func)
    except (SyntaxError, OSError):
        return
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee:Optional[Callable] = _callee_of(node, func, resolve_global)
        if callee is None or id(callee) in visited:
            continue
        found.append(callee)
        _helper_closure(callee, resolve_global, visited, found)


def _helper_identity(helper:Callable)->str:
    """A stable sort key and name mark for one helper: file, line, qualified name."""
    code:object = getattr(helper, "__code__", None)
    file_name:str = str(getattr(code, "co_filename", "") or "")
    line:int = int(getattr(code, "co_firstlineno", 0) or 0)
    qualified:str = str(getattr(helper, "__qualname__", "")
                        or getattr(helper, "__name__", "") or "")
    return f"{file_name}|{line}|{qualified}"


def helper_closure_digest(func:Callable,
                          resolve_global:Callable[[str], object])->str:
    """One digest covering every helper source this kernel's device code may inline.

    Empty when the body calls no helper of its own, so a helper-less kernel keeps the
    identity it always had. The pieces are ordered by file, line and qualified name, so
    the same closure always hashes the same way; each piece carries the helper's name
    mark and its dedented source text.
    """
    found:List[Callable] = []
    _helper_closure(func, resolve_global, set(), found)
    if not found:
        return ""
    pieces:List[str] = []
    for helper in sorted(found, key=_helper_identity):
        try:
            source:str = get_dedented_source(helper)
        except (SyntaxError, OSError):
            continue
        pieces.append(f"{_helper_identity(helper)}\n{source}")
    if not pieces:
        return ""
    return short_hash("\n|\n".join(pieces))
