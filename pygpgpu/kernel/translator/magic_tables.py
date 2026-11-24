"""The operator/conversion name tables the magic-method dispatch reads.

Split out of `receivers.py` by subject: the tables say *which* dunder an `ast` operator or a
builtin maps to, while the rest of `receivers.py` answers *which class* carries it and what the
receiver's attributes are. Every dispatcher imports from here (`operators`,
`augassign`, `calls`, `magic_methods`, `magic_index`), so one table serves them all.
"""
from __future__ import annotations
from typing import (Callable, Dict, Tuple)


# The pair `_magic_owner` answers: the class whose method an operator must call, and that
# method. Named once because four call sites pass it around (`_emit_magic_binop`,
# `_emit_magic_compare`, `_emit_magic_conversion`, the index pair) and the second half is
# always a device function -- `Any` there would be a shape this package simply did not name.
MagicOwner = Tuple[type, Callable]


def _operator_name(magic:str)->str:
    """The `ast` operator name a magic method belongs to."""
    for name, candidate in MAGIC_METHODS.items():
        if candidate == magic:
            return name
    return ""


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
