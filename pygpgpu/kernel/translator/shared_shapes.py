"""Which kernel parameters a body bakes into shared-memory declarations.

`zeros_per_group(G, ...)` with `G` a kernel parameter bakes the call's value into the
generated source -- `__shared__ T name[G]` is a fixed-size declaration -- so the value
specializes the kernel the way a type does: each value compiles a kernel of its own, and the
value has to be part of the identity that decides "already compiled". The scan below is that
decision's cheap half: it runs on every launch, before the cache is consulted, walks the body
for the two shared-memory builtins, and answers the `(name, value)` pairs a parameter
supplies. The expensive half is the translation itself, which bakes the same value into the
declaration (see `FunctionTranslator._baked_shape`) and records the same pairs beside it for
the C name's hash.

Only parameters specialize. A module constant bakes to the same literal on every launch, so
it changes the source without splitting the identity, and a computed value (a host-region
binding, an expression) is refused by the translator -- a size the launch cannot know before
compiling is not a size the declaration can carry.
"""
from __future__ import annotations
import ast
from typing import (Callable, Dict, List, Optional, Tuple)

from ..builtin_registry import builtin_name_of
from .source_text import get_dedented_source, _parse_kernel_source

_SHARED_BUILTINS = ("zeros_per_group", "zeros_per_block")
_UNRESOLVED:object = object()


def _shape_parameter_names(node:ast.expr)->List[str]:
    """The plain names a shape expression reads: `G`, or `(a, b)`'s elements."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple):
        return [element.id for element in node.elts if isinstance(element, ast.Name)]
    return []


def baked_shared_shapes(func:Callable, values:List[object],
                        resolve_global:Callable[[str], object]
                        )->Tuple[Tuple[str, int], ...]:
    """The `(parameter name, value)` pairs a `zeros_per_group` shape names, in body order.

    Deterministic and cheap: the body's cached parse is walked once, a shape that names a
    parameter records this launch's value for it, and everything else (a literal, a module
    constant, an expression) records nothing because it does not split the identity. The
    answer feeds the cache key and the C name's hash, so two calls whose shapes differ
    compile different kernels and both definitions coexist in the generated file.
    """
    tree:ast.Module = _parse_kernel_source(get_dedented_source(func))
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return ()
    code:object = getattr(func, "__code__", None)
    if code is None:
        return ()
    names:List[str] = list(code.co_varnames[:code.co_argcount])
    slots:Dict[str, int] = {name: position for position, name in enumerate(names)}
    recorded:List[Tuple[str, int]] = []
    seen:set = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        target:object = resolve_global(node.func.id)
        if target is None or target is _UNRESOLVED:
            continue
        try:
            builtin:Optional[str] = builtin_name_of(target)
        except (TypeError, KeyError):
            builtin = None
        if builtin not in _SHARED_BUILTINS or not node.args:
            continue
        for name in _shape_parameter_names(node.args[0]):
            position:int = slots.get(name, -1)
            if position < 0 or position >= len(values):
                continue
            value:object = values[position]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                continue
            marker:Tuple[str, int] = (name, int(value))
            if marker not in seen:
                seen.add(marker)
                recorded.append(marker)
    return tuple(recorded)
