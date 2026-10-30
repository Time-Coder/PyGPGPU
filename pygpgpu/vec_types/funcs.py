"""Vector math functions, dual-sided: they run on the host and translate to device builtins.

One implementation serves both sides, and where pygf publishes a function this module *is*
pygf's object rather than a copy of it -- the translator recognizes a builtin by the identity
of the callee, so a second implementation of one name would leave whichever copy a kernel
imported translated as an ordinary device helper, and its body has no device meaning.

Several names differ only in spelling: OpenCL calls it `mad`, GLSL and pygf call it `fma`;
OpenCL's `rsqrt` is GLSL's `inversesqrt`, and OpenCL's `isequal`/`isless`/... are GLSL's
`equal`/`lessThan`/... The device still emits the OpenCL spelling -- that follows the
*registered name*, not the object -- so the import aliases them back to it.

What is left defined below is the set with no pygf counterpart at all: the OpenCL-only
builtins. `mad24`/`mul24` are 24-bit integer multiply-add, `hadd`/`rhadd` are the carrying
halving add used for image blending, `isordered`/`isunordered` are floating-point orderedness,
`abs_diff` has no GLSL form, and `select` takes a boolean selector where GLSL's `mix` takes a
weight -- a different operation, not a different spelling, so it stays here.
"""
from __future__ import annotations

import math
from typing import Any, Callable, List, Optional

from pygf.funcs import (
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
    sqrt,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    sinh,
    cosh,
    tanh,
    exp,
    log,
    log2,
    log10,
    pow,
    floor,
    ceil,
    trunc,
    fabs,
    fmod,
    hypot,
    cbrt,
)
from pygf.funcs import equal as isequal
from pygf.funcs import fma as mad
from pygf.funcs import greaterThan as isgreater
from pygf.funcs import greaterThanEqual as isgreaterequal
from pygf.funcs import inversesqrt as rsqrt
from pygf.funcs import lessThan as isless
from pygf.funcs import lessThanEqual as islessequal
from pygf.funcs import notEqual as isnotequal
from pygf.funcs import any as any
from pygf.funcs import all as all

from .genVec import genVec

__all__ = [
    "dot", "cross", "normalize", "length", "distance",
    "clamp", "mix", "mad", "saturate", "step", "rsqrt", "exp2",
    "select", "any", "all",
    "isequal", "isnotequal", "isgreater", "isgreaterequal", "isless", "islessequal",
    "isordered", "isunordered", "abs_diff", "hadd", "rhadd", "mad24", "mul24",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log2",
    "log10",
    "pow",
    "floor",
    "ceil",
    "trunc",
    "fabs",
    "fmod",
    "hypot",
    "cbrt",
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


def _int_vector_type(size:int)->Any:
    """The int vector class of a given width, for a component-wise predicate's result.

    Every width this package can see has one: pygf supplies int2/int3/int4 and the wider
    int8/int16 are PyGPGPU's own. Deriving the class from the *name* of the operand used to
    fall back to int4 for anything that was not two components, so a three-component vector
    built an `int4` from three values and raised.
    """
    from . import int2, int3, int4, int8, int16

    return {2: int2, 3: int3, 4: int4, 8: int8, 16: int16}.get(size, int4)


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


def _relation(op:Callable, a:Any, b:Any)->Any:
    """A component-wise predicate as an int vector (1/0), the OpenCL convention."""
    _check_same_shape(a, b)
    a_components, b_components = _components(a), _components(b)
    if a_components is None:
        return int(op(a, b))

    def value_at(values:Any, components:Optional[List[Any]], i:int)->Any:
        return values[i] if components is not None else values

    # `a` is a vector; `b` may be a scalar, in which case it broadcasts across components.
    int_vec:Any = _int_vector_type(len(a_components))
    return int_vec(*[int(op(x, value_at(b, b_components, i)))
                     for i, x in enumerate(a_components)])


def isordered(a:Any, b:Any)->Any:
    return _relation(lambda x, y: not (math.isnan(x) or math.isnan(y)), a, b)


def isunordered(a:Any, b:Any)->Any:
    return _relation(lambda x, y: math.isnan(x) or math.isnan(y), a, b)


def abs_diff(a:Any, b:Any)->Any:
    _check_same_shape(a, b)
    a_components, b_components = _components(a), _components(b)
    if a_components is None:
        return abs(a - b)
    return type(a)(*[abs(x - y) for x, y in zip(a_components, b_components)])


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
