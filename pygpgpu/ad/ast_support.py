# Building adjoint expressions from forward ones: the AST vocabulary both transforms share.
#
# The rule builders spell their partials with python's own operators -- ``sqrt(u) / (2.0 *
# sqrt(u))``, ``(u <= v)`` -- because that is the shape the mathematics reads in. AST nodes
# carry no operators, so every node this module builds is an :class:`E` subclass: a real
# ``ast`` node (the translator walks it unchanged) whose class implements the operators as
# node builders. ``E.wrap`` puts an original-tree node into the same shape, so a partial
# builder never sees a node that would make ``a * a`` a TypeError.
from __future__ import annotations

import ast
import copy
from typing import Dict, List, Optional, Set


class _Ops:
    """The operators, as node builders. Comparisons build :class:`ast.Compare` nodes,
    whose truth the generated code coerces to 0/1 exactly as the subgradient rules use."""

    def __add__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Add(), self, _as_node(other))

    def __radd__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Add(), _as_node(other), self)

    def __sub__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Sub(), self, _as_node(other))

    def __rsub__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Sub(), _as_node(other), self)

    def __mul__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Mult(), self, _as_node(other))

    def __rmul__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Mult(), _as_node(other), self)

    def __truediv__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Div(), self, _as_node(other))

    def __rtruediv__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Div(), _as_node(other), self)

    def __pow__(self, other:object)->"_BinOpE":
        return _BinOpE(ast.Pow(), self, _as_node(other))

    def __neg__(self)->"_UnaryOpE":
        return _UnaryOpE(ast.USub(), self)

    def __lt__(self, other:object)->"_CompareE":
        return _CompareE([ast.Lt()], self, _as_node(other))

    def __le__(self, other:object)->"_CompareE":
        return _CompareE([ast.LtE()], self, _as_node(other))

    def __gt__(self, other:object)->"_CompareE":
        return _CompareE([ast.Gt()], self, _as_node(other))

    def __ge__(self, other:object)->"_CompareE":
        return _CompareE([ast.GtE()], self, _as_node(other))


class _BinOpE(_Ops, ast.BinOp):
    def __init__(self, op:ast.operator, left:ast.expr, right:ast.expr):
        ast.BinOp.__init__(self, left=left, op=op, right=right)


class _UnaryOpE(_Ops, ast.UnaryOp):
    def __init__(self, op:ast.unaryop, operand:ast.expr):
        ast.UnaryOp.__init__(self, op=op, operand=operand)


class _CompareE(_Ops, ast.Compare):
    def __init__(self, ops:List[ast.cmpop], left:ast.expr, right:ast.expr):
        ast.Compare.__init__(self, left=left, ops=ops, comparators=[right])


class _NameE(_Ops, ast.Name):
    pass


class _ConstantE(_Ops, ast.Constant):
    pass


class _CallE(_Ops, ast.Call):
    pass


class _IfExpE(_Ops, ast.IfExp):
    pass


class _SubscriptE(_Ops, ast.Subscript):
    pass


class _AttributeE(_Ops, ast.Attribute):
    pass


class _BinOpWrapE(_Ops, ast.BinOp):
    pass


class _UnaryOpWrapE(_Ops, ast.UnaryOp):
    pass


_E_BY_TYPE:Dict[type, type] = {
    ast.Name: _NameE,
    ast.Constant: _ConstantE,
    ast.Call: _CallE,
    ast.IfExp: _IfExpE,
    ast.Subscript: _SubscriptE,
    ast.Attribute: _AttributeE,
    ast.BinOp: _BinOpWrapE,
    ast.UnaryOp: _UnaryOpWrapE,
}


def wrap(node:ast.expr)->ast.expr:
    """The same expression, in an :class:`E` class, so operators build on it.

    A copy, never the original: the builders graft wrapped nodes into adjoint statements
    that the forward tree does not own. A node type the builders do not produce (a
    comparison inside an argument, say) comes back as itself -- the operator syntax simply
    is not available on it, which the rule builders avoid by construction.
    """
    cls:Optional[type] = _E_BY_TYPE.get(type(node))
    if cls is None:
        return copy.deepcopy(node)
    rebuilt:ast.expr = cls()
    for field in node._fields:
        setattr(rebuilt, field, copy.deepcopy(getattr(node, field)))
    rebuilt.lineno = getattr(node, "lineno", 1)
    rebuilt.col_offset = getattr(node, "col_offset", 0)
    return rebuilt


def _as_node(value:object)->ast.expr:
    """The operand as an AST node; a python number becomes a constant."""
    if isinstance(value, ast.expr):
        return value
    node:ast.expr = _ConstantE(value=value)
    node.lineno = 1
    node.col_offset = 0
    return node


# ------------------------------------------------------------------ constructors

def name(id:str)->_NameE:
    """A :class:`ast.Name` load."""
    node:_NameE = _NameE(id=id, ctx=ast.Load())
    node.lineno = 1
    node.col_offset = 0
    return node


def store_name(id:str)->_NameE:
    """A :class:`ast.Name` store."""
    node:_NameE = _NameE(id=id, ctx=ast.Store())
    node.lineno = 1
    node.col_offset = 0
    return node


def constant(value:object)->_ConstantE:
    """A :class:`ast.Constant` node."""
    node:_ConstantE = _ConstantE(value=value)
    node.lineno = 1
    node.col_offset = 0
    return node


def subscript(value:ast.expr, index:ast.expr)->_SubscriptE:
    """``value[index]`` -- one index; a multi-axis read passes a tuple node."""
    node:_SubscriptE = _SubscriptE(value=value, slice=index, ctx=ast.Load())
    node.lineno = 1
    node.col_offset = 0
    return node


def store_subscript(value:ast.expr, index:ast.expr)->ast.Subscript:
    """``value[index]`` in store context."""
    node:ast.Subscript = ast.Subscript(value=value, slice=index, ctx=ast.Store())
    node.lineno = 1
    node.col_offset = 0
    return node


def call(func:ast.expr, args:List[ast.expr])->_CallE:
    """``func(args...)`` -- positional only; a keyword would be a different rule."""
    node:_CallE = _CallE(func=func, args=args, keywords=[])
    node.lineno = 1
    node.col_offset = 0
    return node


def ifexp(test:ast.expr, body:ast.expr, orelse:ast.expr)->_IfExpE:
    """``body if test else orelse``."""
    node:_IfExpE = _IfExpE(test=test, body=body, orelse=orelse)
    node.lineno = 1
    node.col_offset = 0
    return node


def binop(op:ast.operator, left:ast.expr, right:ast.expr)->_BinOpE:
    """``left op right``."""
    return _BinOpE(op, left, right)


def multiply(left:ast.expr, right:ast.expr)->_BinOpE:
    """``left * right`` (the term the chain rules assemble)."""
    return _BinOpE(ast.Mult(), left, right)


def add(left:ast.expr, right:ast.expr)->_BinOpE:
    """``left + right``."""
    return _BinOpE(ast.Add(), left, right)


def negate(value:ast.expr)->_UnaryOpE:
    """``-value``."""
    return _UnaryOpE(ast.USub(), value)


def tuple_expr(elements:List[ast.expr])->ast.Tuple:
    """``(a, b, ...)`` -- the multi-axis subscript shape the translator reads."""
    node:ast.Tuple = ast.Tuple(elts=elements, ctx=ast.Load())
    node.lineno = 1
    node.col_offset = 0
    return node


# ------------------------------------------------------------------ statements

def assign(target:ast.expr, value:ast.expr)->ast.Assign:
    """``target = value``."""
    node:ast.Assign = ast.Assign(targets=[target], value=value)
    node.lineno = 1
    node.col_offset = 0
    return node


def augassign(target:ast.expr, op:ast.operator, value:ast.expr)->ast.AugAssign:
    """``target op= value``."""
    node:ast.AugAssign = ast.AugAssign(target=target, op=op, value=value)
    node.lineno = 1
    node.col_offset = 0
    return node


def expr_stmt(value:ast.expr)->ast.Expr:
    """A statement that evaluates ``value`` for effect."""
    node:ast.Expr = ast.Expr(value=value)
    node.lineno = 1
    node.col_offset = 0
    return node


def return_stmt(value:Optional[ast.expr])->ast.Return:
    """``return value`` (or a bare return)."""
    node:ast.Return = ast.Return(value=value)
    node.lineno = 1
    node.col_offset = 0
    return node


def placed(stmt:ast.stmt, lineno:int)->ast.stmt:
    """Give a built statement the line it should report against.

    A generated statement reports at the line of the user's statement it belongs to, so a
    refusal inside the adjoint names the line the user wrote. The lineno must not collide
    with a HOST statement's line -- the translator skips statements by line (see
    ``translator.host_regions``) -- and the transforms only place generated statements at
    lines of *device* statements, which are never in that set.
    """
    stmt.lineno = max(int(lineno), 1)
    stmt.col_offset = 0
    return stmt


def copied(node:ast.AST)->ast.AST:
    """A deep copy of an original node, keeping its line numbers and end positions."""
    return copy.deepcopy(node)


def free_names(node:ast.expr)->Set[str]:
    """The plain names an expression reads."""
    found:Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            found.add(sub.id)
    return found


def substitute(node:ast.expr, mapping:Dict[str, ast.expr])->ast.expr:
    """A copy of ``node`` whose free names the mapping replaces.

    The reverse walk reads forward values that later statements may have overwritten, and
    the forward pass saves those into slots -- the mapping is `name -> the saved copy`, so
    the adjoint's partial expressions read what the statement they adjoint read.
    """
    if not mapping:
        return copied(node)
    return _substitute_copy(copied(node), mapping)


def _substitute_copy(node:ast.expr, mapping:Dict[str, ast.expr])->ast.expr:
    if isinstance(node, ast.Name) and node.id in mapping:
        return copied(mapping[node.id])
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.expr):
            setattr(node, field, _substitute_copy(value, mapping))
        elif isinstance(value, list):
            replaced:List[object] = [
                _substitute_copy(item, mapping) if isinstance(item, ast.expr) else item
                for item in value]
            setattr(node, field, replaced)
    return node


class NameAllocator:
    """Fresh names that cannot collide with the body's own or the package's reserved ones.

    The translator reads a generated name the same way it reads a user's: `_self_...`
    belongs to receiver members and `_<param>_...` to object-array members, so an adjoint
    name must avoid those prefixes as well as every name the body already spells.
    """

    def __init__(self, taken:Set[str]):
        self._taken:Set[str] = set(taken)

    def fresh(self, base:str)->str:
        """A name built on ``base`` that nothing in the kernel is using."""
        candidate:str = base
        counter:int = 0
        while candidate in self._taken:
            counter += 1
            candidate = f"{base}_{counter}"
        self._taken.add(candidate)
        return candidate

    def mark(self, used:str)->None:
        """Record a name as taken without allocating."""
        self._taken.add(used)


# ------------------------------------------------------------------ callee resolution

def resolve_callee(origin:object, node:ast.expr)->object:
    """The object a call's callee names, or the spelling when it cannot resolve.

    Mirrors the translator's resolution: a closure cell first (python resolves a nested
    function's free names through its cells before its module globals), then the module
    globals, then python's builtins.
    """
    if isinstance(node, ast.Name):
        code:object = getattr(origin, "__code__", None)
        freevars:tuple = tuple(getattr(code, "co_freevars", ()) or ())
        for cell_name, cell in zip(freevars, getattr(origin, "__closure__", None) or ()):
            if cell_name == node.id:
                try:
                    return cell.cell_contents
                except ValueError:
                    pass
            continue
        import builtins as _py
        return origin.__globals__.get(node.id, _py.__dict__.get(node.id, node.id))
    if isinstance(node, ast.Attribute):
        base:object = resolve_attribute_base(origin, node.value)
        if base is not None:
            return getattr(base, node.attr, node.attr)
    return None


def resolve_attribute_base(origin:object, node:ast.expr)->object:
    """The module-like object an attribute chain bottoms out in, or None."""
    if isinstance(node, ast.Name):
        code:object = getattr(origin, "__code__", None)
        freevars:tuple = tuple(getattr(code, "co_freevars", ()) or ())
        for cell_name, cell in zip(freevars, getattr(origin, "__closure__", None) or ()):
            if cell_name == node.id:
                try:
                    return cell.cell_contents
                except ValueError:
                    pass
        return origin.__globals__.get(node.id)
    if isinstance(node, ast.Attribute):
        base:object = resolve_attribute_base(origin, node.value)
        return getattr(base, node.attr, None) if base is not None else None
    return None
