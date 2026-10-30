"""Vector math functions, dual-sided: they run on the host and translate to device builtins.

One implementation serves both sides, and where pygf publishes a function this module *is*
pygf's object rather than a copy of it -- the translator recognizes a builtin by the identity
of the callee, so a second implementation of one name would leave whichever copy a kernel
imported translated as an ordinary device helper, and its body has no device meaning.

What is left defined below is the set with no pygf counterpart: the OpenCL-only builtins
(`mad24`/`mul24` are 24-bit, `hadd`/`rhadd` carry, the ordered comparisons are not GLSL).
"""
from __future__ import annotations

import math
from typing import Any, Callable, List, Optional

from .genVec import genVec

__all__ = [
    "dot", "cross", "normalize", "length", "distance",
    "clamp", "mix", "mad", "saturate", "step", "rsqrt", "exp2",
    "select", "any", "all",
    "isequal", "isnotequal", "isgreater", "isgreaterequal", "isless", "islessequal",
    "isordered", "isunordered", "abs_diff", "hadd", "rhadd", "mad24", "mul24",
]


def _is_vec(v:Any)->bool:
    """Whether v is a host vector: a ctypes Structure with component fields.

    Duck-typed so both PyGPGPU's own genVec types and the pygf vector types that
    the shared float/int/double/uint 2/4 widths now re-export are recognised.
    """
    return isinstance(v, genVec) or (hasattr(v, "_fields_") and hasattr(v, "__len__"))


def _components(v:Any)->Optional[List[Any]]:
    return list(v) if _is_vec(v) else None


def _check_same_shape(a:Any, b:Any)->None:
    a_components, b_components = _components(a), _components(b)
    if (a_components is not None and b_components is not None
            and len(a_components) != len(b_components)):
        raise ValueError(f"cannot operate on vectors with different dimensions: "
                         f"{len(a_components)} and {len(b_components)}")


def mad(a:Any, b:Any, c:Any)->Any:
    a_components, b_components, c_components = _components(a), _components(b), _components(c)

    def value_at(values:Any, components:Optional[List[Any]], i:int)->Any:
        return values[i] if components is not None else values

    if a_components is None:
        return a * b + c

    result:List[Any] = []
    for i in range(len(a_components)):
        result.append(value_at(a, a_components, i) * value_at(b, b_components, i)
                      + value_at(c, c_components, i))
    return type(a)(*result)


def rsqrt(x:Any)->Any:
    components = _components(x)
    if components is None:
        return 1.0 / math.sqrt(x)
    return type(x)(*[1.0 / math.sqrt(c) for c in components])


def select(a:Any, b:Any, cond:Any)->Any:
    """Component-wise selection: pick b where the cond component is nonzero, otherwise a."""
    a_components, b_components, cond_components = _components(a), _components(b), _components(cond)

    def value_at(values:Any, components:Optional[List[Any]], i:int)->Any:
        return values[i] if components is not None else values

    result_type:Any = (type(a) if _is_vec(a) else type(b) if _is_vec(b) else float)
    if a_components is None:
        return b if cond else a

    result:List[Any] = []
    for i in range(len(a_components)):
        result.append(value_at(b, b_components, i) if value_at(cond, cond_components, i) else value_at(a, a_components, i))
    return result_type(*result)


def _relation(name:str, op:Callable, a:Any, b:Any)->Any:
    _check_same_shape(a, b)
    a_components, b_components = _components(a), _components(b)
    result_type:Any = (type(a) if _is_vec(a) else type(b) if _is_vec(b) else float)

    def int_result_type()->Any:
        if isinstance(result_type, type) and (hasattr(result_type, "_fields_") or issubclass(result_type, genVec)):
            dim:str = "".join(c for c in result_type.__name__ if c.isdigit())
            from . import int2, int4
            return {"2": int2}.get(dim, int4)
        return int

    if a_components is None:
        return int(op(a, b))

    def value_at(values:Any, components:Optional[List[Any]], i:int)->Any:
        return values[i] if components is not None else values

    # `a` is a vector; `b` may be a scalar, in which case it broadcasts across components.
    int_vec:Any = int_result_type()
    return int_vec(*[int(op(x, value_at(b, b_components, i)))
                     for i, x in enumerate(a_components)])


def isequal(a:Any, b:Any)->Any:
    return _relation("isequal", lambda x, y: x == y, a, b)


def isnotequal(a:Any, b:Any)->Any:
    return _relation("isnotequal", lambda x, y: x != y, a, b)


def isgreater(a:Any, b:Any)->Any:
    return _relation("isgreater", lambda x, y: x > y, a, b)


def isgreaterequal(a:Any, b:Any)->Any:
    return _relation("isgreaterequal", lambda x, y: x >= y, a, b)


def isless(a:Any, b:Any)->Any:
    return _relation("isless", lambda x, y: x < y, a, b)


def islessequal(a:Any, b:Any)->Any:
    return _relation("islessequal", lambda x, y: x <= y, a, b)


def isordered(a:Any, b:Any)->Any:
    return _relation("isordered", lambda x, y: not (math.isnan(x) or math.isnan(y)), a, b)


def isunordered(a:Any, b:Any)->Any:
    return _relation("isunordered", lambda x, y: math.isnan(x) or math.isnan(y), a, b)


def abs_diff(a:Any, b:Any)->Any:
    _check_same_shape(a, b)
    a_components, b_components = _components(a), _components(b)
    if a_components is None:
        return abs(a - b)
    return type(a)(*[_relation_abs(x, y) for x, y in zip(a_components, b_components)])


def _relation_abs(x:Any, y:Any)->Any:
    return abs(x - y)


def hadd(a:Any, b:Any)->Any:
    _check_same_shape(a, b)
    a_components, b_components = _components(a), _components(b)
    if a_components is None:
        return (a & b) + ((a ^ b) >> 1)
    return type(a)(*[(x & y) + ((x ^ y) >> 1) for x, y in zip(a_components, b_components)])


def rhadd(a:Any, b:Any)->Any:
    _check_same_shape(a, b)
    a_components, b_components = _components(a), _components(b)
    if a_components is None:
        return (a | b) - ((a ^ b) >> 1)
    return type(a)(*[(x | y) - ((x ^ y) >> 1) for x, y in zip(a_components, b_components)])


def mul24(a:Any, b:Any)->Any:
    _check_same_shape(a, b)
    a_components, b_components = _components(a), _components(b)
    if a_components is None:
        return a * b
    return type(a)(*[(x * y) for x, y in zip(a_components, b_components)])


def mad24(a:Any, b:Any, c:Any)->Any:
    _check_same_shape(a, b)
    a_components, b_components, c_components = _components(a), _components(b), _components(c)

    def value_at(values:Any, components:Optional[List[Any]], i:int)->Any:
        return values[i] if components is not None else values

    if a_components is None:
        return a * b + c

    result:List[Any] = []
    for i in range(len(a_components)):
        result.append(value_at(a, a_components, i) * value_at(b, b_components, i)
                      + value_at(c, c_components, i))
    return type(a)(*result)


# The functions pygf also publishes are *its* objects, not copies of them: a kernel body that
# imports `normalize` from pygf and one that imports it from pygpgpu have to reach the same
# builtin, and the translator decides that by the identity of the callee. A second
# implementation of one name leaves whichever copy the user imported translated as a device
# helper, and its body (`length(x)`, a `_div` helper, ...) has no device meaning. There is no
# exception left: every name below this comment that pygf also publishes *is* pygf's object.
from pygf.funcs import (  # noqa: E402  (after the definitions it shadows)
    all,
    any,
    clamp,
    cross,
    distance,
    dot,
    exp2,
    length,
    mix,
    normalize,
    saturate,
    step,
)
