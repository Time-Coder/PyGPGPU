# The host stand-ins for the magic loops (`range`, `enumerate`, `ndindex`, `ndenumerate`).
from __future__ import annotations
import ast
from typing import Callable, Dict, Optional, Tuple
import numpy as np
from ..builtin_registry import builtin_name_of
from ..translator import _parse_kernel_source, get_dedented_source



# A holder is ("arg", <parameter name>) or ("self", <attribute name>): the python
# backend never rewrites self attributes into parameters, it just reads them.
Holder = Tuple[str, str]



_MAGIC_RANGE = "range"


_MAGIC_RANGE = "range"
_MAGIC_ENUMERATE = "enumerate"


_MAGIC_ENUMERATE = "enumerate"
_MAGIC_NDINDEX = "ndindex"


_MAGIC_NDINDEX = "ndindex"
_MAGIC_NDENUMERATE = "ndenumerate"



# Sentinel for "no work item returned a value yet": a kernel that returns None has to be
# distinguishable from one that has not returned at all.
_NO_RESULT: object = object()



def _dedented_ast(func: Callable) -> Optional[ast.FunctionDef]:
    """AST of the decorated function, or None when its source is unavailable.

    ``inspect.getsource`` fails for code defined through ``exec``/stdin; such a kernel
    can still run as plain serial Python, but the launch shape cannot be derived, so the
    analysis is skipped and the frame reports that when a builtin needs it.
    """
    try:
        tree = _parse_kernel_source(get_dedented_source(func))
    except Exception:
        return None
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return None
    return tree.body[0]



def _globals_of(func: Callable) -> Dict[str, object]:
    return getattr(func, "__globals__", None) or {}



def _holder_of(node: ast.expr, self_param: str) -> Optional[Holder]:
    """Resolve a plain parameter reference or a self.<attr> reference to a holder."""
    if isinstance(node, ast.Name):
        return "arg", node.id
    if (self_param and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == self_param):
        return "self", node.attr
    return None



def _shape0_holder(iter_node: ast.expr, self_param: str) -> Optional[Holder]:
    """``range(<holder>.shape[0])`` -> the holder (the 1-D magic loop)."""
    if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "range" and len(iter_node.args) == 1
            and not iter_node.keywords):
        return None
    arg: ast.expr = iter_node.args[0]
    if not (isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant)
            and arg.slice.value == 0):
        return None
    base = arg.value
    if not (isinstance(base, ast.Attribute) and base.attr == "shape"):
        return None
    return _holder_of(base.value, self_param)



def _enumerate_holder(iter_node: ast.expr, self_param: str) -> Optional[Holder]:
    """``enumerate(<holder>)`` -> the holder (index + value snapshot magic loop)."""
    if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "enumerate" and len(iter_node.args) == 1
            and not iter_node.keywords):
        return None
    return _holder_of(iter_node.args[0], self_param)



def _nd_magic(iter_node: ast.expr, self_param: str, globals_of: Dict[str, object]) -> Optional[Tuple[str, Holder]]:
    """``np.ndindex(<holder>.shape)`` / ``np.ndenumerate(<holder>)`` -> (form, holder)."""
    if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Attribute)
            and isinstance(iter_node.func.value, ast.Name) and len(iter_node.args) == 1
            and not iter_node.keywords):
        return None
    if globals_of.get(iter_node.func.value.id) is not np:
        return None
    form: str = iter_node.func.attr
    arg: ast.expr = iter_node.args[0]
    if form == _MAGIC_NDINDEX:
        if not (isinstance(arg, ast.Attribute) and arg.attr == "shape"):
            return None
        holder = _holder_of(arg.value, self_param)
    elif form == _MAGIC_NDENUMERATE:
        holder = _holder_of(arg, self_param)
    else:
        return None
    return (form, holder) if holder is not None else None



def _gpu_attr(call: ast.Call, globals_of: Dict[str, object]) -> Optional[str]:
    """The builtin name a call invokes, or None when it invokes something else.

    Identity, not spelling: the callee is resolved through the kernel's globals the way
    python would, and the resulting object is looked up in the registry. A user function
    that merely shares a builtin's name is therefore not mistaken for one.
    """
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner = globals_of.get(func.value.id)
        if owner is None:
            return None
        return builtin_name_of(getattr(owner, func.attr, None))
    if isinstance(func, ast.Name):
        return builtin_name_of(globals_of.get(func.id))
    return None
