# The partials of python's own operators, shared by the forward and the reverse walk.
#
# A ``BinOp`` is not a call, so it has no object to key a rule on; its op class is the key
# here instead. Both modes read one table: the forward tangent of ``u * v`` is built from
# the same partials the reverse step distributes with, which is what keeps the two modes
# from drifting.
from __future__ import annotations

import ast
from typing import List, Optional

from .ast_support import (
    add,
    call,
    constant,
    multiply,
    name,
    negate,
    wrap,
)

# The zero partial is spelled ``None``: an argument the operation does not depend on.
# A refused op raises :class:`OperatorRefusal` with the op's own spelling.


class OperatorRefusal(Exception):
    """An operator the AD stage refuses, carrying its source spelling and its reason."""

    def __init__(self, spelling:str, reason:str):
        super().__init__(f"`{spelling}` {reason}")
        self.spelling:str = spelling
        self.reason:str = reason


_ARITHMETIC_REASON = ("is not differentiable: its derivative is zero almost everywhere "
                      "or undefined, which is a plausible wrong answer, not a gradient")


def _num(value:float)->ast.expr:
    return constant(value)


def binop_partials(op:ast.operator, u:ast.expr, v:ast.expr)->List[Optional[ast.expr]]:
    """The partials of ``u op v`` w.r.t. ``(u, v)``.

    The expressions read the FORWARD values, which the fused body keeps in registers, so
    a partial like ``v`` for ``u * v`` is the value the statement itself computed.
    """
    if isinstance(op, ast.Add):
        return [_num(1.0), _num(1.0)]
    if isinstance(op, ast.Sub):
        return [_num(1.0), negate(_num(1.0))]
    if isinstance(op, ast.Mult):
        return [v, u]
    if isinstance(op, ast.Div):
        return [_num(1.0) / v, negate(u) / (v * v)]
    if isinstance(op, ast.Pow):
        exponent:object = getattr(v, "value", None)
        if isinstance(v, ast.Constant) and isinstance(exponent, (int, float)) \
                and not isinstance(exponent, bool):
            return [constant(exponent) * (u ** constant(exponent - 1)), None]
        # A variable exponent needs ln(u) and refuses for its domain; the spelling is
        # `pow`'s rule, which carries the same partials for positive bases.
        return [v * (u ** (v - _num(1.0))),
                (u ** v) * call(name("log"), [wrap(u)])]
    if isinstance(op, ast.FloorDiv):
        raise OperatorRefusal("//", _ARITHMETIC_REASON)
    if isinstance(op, ast.Mod):
        raise OperatorRefusal("%", _ARITHMETIC_REASON)
    if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
        spelling:str = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
                        ast.LShift: "<<", ast.RShift: ">>"}[type(op)]
        raise OperatorRefusal(spelling,
                              "is a bitwise operator: the device result is an integer "
                              "pattern, not a differentiable quantity")
    if isinstance(op, ast.MatMult):
        raise OperatorRefusal("@",
                              "is a matrix product: its adjoint is a vector rule the "
                              "scalar table does not carry (the vector-rules milestone)")
    raise OperatorRefusal(type(op).__name__, "is not an operator this table knows")


def unaryop_partials(op:ast.unaryop, u:ast.expr)->Optional[ast.expr]:
    """The partial of ``op u`` w.r.t. ``u``; None for the boolean operators."""
    if isinstance(op, ast.USub):
        return negate(_num(1.0))
    if isinstance(op, ast.UAdd):
        return _num(1.0)
    if isinstance(op, ast.Not):
        return None    # a bool: comparisons propagate no tangent
    if isinstance(op, ast.Invert):
        raise OperatorRefusal("~", "is a bitwise operator: the result is an integer "
                                   "pattern, not a differentiable quantity")
    raise OperatorRefusal(type(op).__name__, "is not an operator this table knows")


def add_tangent(u_tangent:Optional[ast.expr], u:ast.expr,
                v_tangent:Optional[ast.expr], v:ast.expr)->Optional[ast.expr]:
    """The forward tangent of ``u * v``: sum the non-zero ``partial * tangent`` terms."""
    terms:List[ast.expr] = []
    if u_tangent is not None:
        terms.append(multiply_pair(u_tangent, v))
    if v_tangent is not None:
        terms.append(multiply_pair(u, v_tangent))
    if not terms:
        return None
    result:ast.expr = terms[0]
    for term in terms[1:]:
        result = add(result, term)
    return result


def multiply_pair(left:ast.expr, right:ast.expr)->ast.expr:
    """``left * right`` with the single-term simplifications that keep tangents readable."""
    if isinstance(left, ast.Constant) and left.value == 1:
        return right
    if isinstance(right, ast.Constant) and right.value == 1:
        return left
    return multiply(left, right)


def divide_tangent(u_tangent:Optional[ast.expr], u:ast.expr,
                   v_tangent:Optional[ast.expr], v:ast.expr)->Optional[ast.expr]:
    """The forward tangent of ``u / v``: ``(t_u * v - u * t_v) / (v * v)``."""
    numerator:Optional[ast.expr] = None
    if u_tangent is not None and v_tangent is not None:
        numerator = add(multiply_pair(u_tangent, v),
                        negate(multiply_pair(u, v_tangent)))
    elif u_tangent is not None:
        numerator = multiply_pair(u_tangent, v)
    elif v_tangent is not None:
        numerator = negate(multiply_pair(u, v_tangent))
    if numerator is None:
        return None
    return numerator / (v * v)
