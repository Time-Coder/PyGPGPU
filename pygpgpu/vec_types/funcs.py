"""向量数学函数(宿主端实现)。

这些函数可以直接在 python 解释器中运行(作用于 genVec 实例或标量);
在 @kernel 内核中使用时,翻译器会把它们映射为对应的设备内建
(与 pygpgpu.kernel.gpu 命名空间相同),因此同一份代码双端可用。
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

from .genVec import genVec

__all__ = [
    "dot", "cross", "normalize", "length", "distance",
    "clamp", "mix", "mad", "saturate", "step", "rsqrt", "exp2",
]


def _components(v:Any)->Optional[List[Any]]:
    return list(v) if isinstance(v, genVec) else None


def _check_same_shape(a:Any, b:Any)->None:
    a_components, b_components = _components(a), _components(b)
    if (a_components is not None and b_components is not None
            and len(a_components) != len(b_components)):
        raise ValueError(f"cannot operate on vectors with different dimensions: "
                         f"{len(a_components)} and {len(b_components)}")


def dot(a:Any, b:Any)->Any:
    _check_same_shape(a, b)
    a_components, b_components = _components(a), _components(b)
    if a_components is None and b_components is None:
        return a * b
    if a_components is None:
        a_components = [a] * len(b_components)
    if b_components is None:
        b_components = [b] * len(a_components)
    return float(sum(x * y for x, y in zip(a_components, b_components)))


def cross(a:genVec, b:genVec)->genVec:
    a_components, b_components = _components(a), _components(b)
    if a_components is None or b_components is None:
        raise TypeError("cross() requires vector arguments")
    if len(a_components) not in (3, 4) or len(a_components) != len(b_components):
        raise ValueError("cross() requires 3 or 4 component vectors of the same type")
    ax, ay, az = a_components[0], a_components[1], a_components[2]
    bx, by, bz = b_components[0], b_components[1], b_components[2]
    values:List[Any] = [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
    if len(a_components) == 4:
        values.append(0.0)
    return type(a)(*values)


def length(v:Any)->Any:
    components = _components(v)
    if components is None:
        return abs(v)
    return math.sqrt(sum(c * c for c in components))


def distance(a:Any, b:Any)->Any:
    _check_same_shape(a, b)
    if _components(a) is None:
        return abs(a - b)
    return length(a - b)


def normalize(v:genVec)->genVec:
    components = _components(v)
    if components is None:
        l:Any = abs(v)
        return v / l if l != 0 else 0.0
    l:Any = math.sqrt(sum(c * c for c in components))
    if l == 0:
        return type(v)(*[c * 0.0 for c in components])
    return type(v)(*[c / l for c in components])


def clamp(x:Any, lo:Any, hi:Any)->Any:
    x_components, lo_components, hi_components = _components(x), _components(lo), _components(hi)
    if x_components is None:
        return min(max(x, lo), hi)

    def value_at(values:Any, components:Optional[List[Any]], i:int)->Any:
        return values[i] if components is not None else values

    result:List[Any] = []
    for i in range(len(x_components)):
        result.append(min(max(x_components[i],
                              value_at(lo, lo_components, i)),
                          value_at(hi, hi_components, i)))
    return type(x)(*result)


def mix(a:Any, b:Any, t:Any)->Any:
    a_components, b_components, t_components = _components(a), _components(b), _components(t)

    def value_at(values:Any, components:Optional[List[Any]], i:int)->Any:
        return values[i] if components is not None else values

    if a_components is None:
        return a * (1.0 - value_at(t, t_components, 0)) + b * value_at(t, t_components, 0)

    result:List[Any] = []
    for i in range(len(a_components)):
        t_v:Any = value_at(t, t_components, i)
        result.append(a_components[i] * (1.0 - t_v) + b_components[i] * t_v)
    return type(a)(*result)


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


def saturate(x:Any)->Any:
    return clamp(x, 0.0, 1.0)


def step(edge:Any, x:Any)->Any:
    edge_components, x_components = _components(edge), _components(x)

    def value_at(values:Any, components:Optional[List[Any]], i:int)->Any:
        return values[i] if components is not None else values

    counts:List[int] = [len(c) for c in (edge_components, x_components) if c is not None]
    count:int = (max(counts) if counts else 1)
    result_type:Any = type(edge if edge_components is not None else x)
    result:List[Any] = []
    for i in range(count):
        e_v:Any = value_at(edge, edge_components, i)
        x_v:Any = value_at(x, x_components, i)
        result.append(0.0 if x_v < e_v else 1.0)
    return result_type(*result)


def rsqrt(x:Any)->Any:
    components = _components(x)
    if components is None:
        return 1.0 / math.sqrt(x)
    return type(x)(*[1.0 / math.sqrt(c) for c in components])


def exp2(x:Any)->Any:
    components = _components(x)
    if components is None:
        return math.pow(2.0, x)
    return type(x)(*[math.pow(2.0, c) for c in components])
