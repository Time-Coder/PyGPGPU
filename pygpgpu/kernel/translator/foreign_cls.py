# The vocabulary for a classmethod whose cls binds to a class the caller does not
# hold: the scalar class attributes travel as compile-time constants (see
# class_constant_values), and the receiver-as-value detection that decides when a
# kernel's own receiver needs to travel at all.
from __future__ import annotations

from .source_text import _parse_kernel_source, get_dedented_source
import ast
import types
from typing import Callable, Dict, Optional, Set
import numpy as np


def class_constant_values(owner:type)->Dict[str, object]:
    """The class's own data attributes, as the compile-time values the device inlines.

    A classmethod called across classes (`BUtils.cm(x)` from a kernel) binds its
    `cls` to the class -- and the class is a python object the translation reads
    directly, so every scalar attribute the callee reads travels as a constant
    instead of as a receiver struct the caller would have to carry. Callables are
    not members (they are reached as calls); ndarrays have a device life of their
    own and are refused at the read site rather than guessed here.
    """
    constants:Dict[str, object] = {}
    for name, value in vars(owner).items():
        if name.startswith("__") or isinstance(value, (staticmethod, classmethod,
                                                       property, types.FunctionType)):
            continue
        if isinstance(value, np.ndarray):
            continue
        constants[name] = value
    return constants


def receiver_passed_as_value(func:Callable, self_param:str,
                             host_lines:Optional[Set[int]]=None)->bool:
    """Whether the body hands the receiver itself to a call as an argument.

    `helper(self, ...)` is a value use of the receiver: the callee receives the whole
    object and its own method calls resolve the members they read from it. The
    attribute scan only sees `self.<attr>` reads, so a body that ONLY passes the
    receiver on would otherwise collect no members at all -- and the receiver struct
    (and the `self` name in the body) would never come to exist.
    """
    if not self_param:
        return False
    try:
        tree:ast.Module = _parse_kernel_source(get_dedented_source(func))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if host_lines and int(getattr(node, "lineno", 0) or 0) in host_lines:
            continue
        for argument in node.args:
            if isinstance(argument, ast.Name) and argument.id == self_param:
                return True
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Name) and keyword.value.id == self_param:
                return True
    return False


