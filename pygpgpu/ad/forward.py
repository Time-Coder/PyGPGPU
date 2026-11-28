# The forward-mode transform: a tangent variable per tracked name, one pass over the body.
#
# The transform takes the kernel's own body and emits the body of the JVP kernel: every
# statement runs as written, and each statement that computes a tracked-dependent value
# gains the statement that computes its tangent, in the same pass and the same nesting.
# The result is an ordinary Python function the existing translator emits -- no backend
# gains a code path, and the interpreter backend runs the adjoint body as written, which
# is what makes it an independent oracle.
from __future__ import annotations

import ast
import builtins as _py
from typing import Dict, List, Optional, Set

from .ast_support import (
    NameAllocator,
    assign,
    augassign,
    constant,
    copied,
    ifexp,
    name,
    placed,
    return_stmt,
    store_name,
    store_subscript,
    subscript,
    wrap,
)
from .body_scan import BodyScan
from .errors import ad_error
from .operator_rules import (
    OperatorRefusal,
    add_tangent,
    binop_partials,
    divide_tangent,
    unaryop_partials,
)
from . import rules as rule_table

class JVPBuilder:
    """The forward walk over one kernel body.

    ``env`` maps each live value name to the name of its tangent; a name leaves the env
    when a statement computes a value with no tracked dependence, because its tangent is
    zero from then on and propagating zeros would only widen the generated body.
    """

    def __init__(self, scan:BodyScan, tracked:List[str],
                 tangent_param_of:Dict[str, str], out_param:Optional[str],
                 out_tangent:Optional[str],
                 allocator:NameAllocator, origin:object,
                 filename:str, source:str, first_line:int):
        self.scan:BodyScan = scan
        self.tracked:Set[str] = set(tracked)
        self.tangent_param_of:Dict[str, str] = dict(tangent_param_of)
        self.out_param:Optional[str] = out_param
        self.allocator:NameAllocator = allocator
        self.origin:object = origin
        self.filename:str = filename
        self.source:str = source
        self.first_line:int = first_line
        self.env:Dict[str, str] = {}
        self.save_counter:int = 0
        for tracked_name, tangent_name in tangent_param_of.items():
            self.env[tracked_name] = tangent_name
        if out_param is not None and out_tangent is not None:
            # The output array's tangent buffer exists before the walk, so the stores
            # into it emit the tangent stores beside them.
            self.env[out_param] = out_tangent

    # ------------------------------------------------------------------ entry

    def build(self, funcdef:ast.FunctionDef)->ast.FunctionDef:
        """Transform ``funcdef`` in place (the caller passes its own deep copy)."""
        funcdef.body = self.transform_block(list(funcdef.body))
        return funcdef

    # ------------------------------------------------------------------ statements

    def transform_block(self, stmts:List[ast.stmt])->List[ast.stmt]:
        out:List[ast.stmt] = []
        for stmt in stmts:
            out.extend(self.transform_stmt(stmt))
        return out

    def transform_stmt(self, stmt:ast.stmt)->List[ast.stmt]:
        if isinstance(stmt, ast.Assign):
            return self.transform_assign(stmt)
        if isinstance(stmt, ast.AugAssign):
            return self.transform_augassign(stmt)
        if isinstance(stmt, ast.If):
            return self.transform_if(stmt)
        if isinstance(stmt, ast.For):
            return self.transform_for(stmt)
        if isinstance(stmt, ast.While):
            return self.transform_while(stmt)
        if isinstance(stmt, ast.Return):
            return self.transform_return(stmt)
        if isinstance(stmt, ast.Expr):
            return self.transform_expr(stmt)
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                             ast.Del, ast.With, ast.Try, ast.Raise, ast.Assert)):
            raise self._error(
                f"`{type(stmt).__name__}` is not differentiable: the adjoint of this "
                f"statement has no device meaning", stmt)
        if isinstance(stmt, ast.Pass) or isinstance(stmt, ast.Break) \
                or isinstance(stmt, ast.Continue):
            return [copied(stmt)]
        raise self._error(
            f"unsupported statement in a differentiable kernel: {type(stmt).__name__}", stmt)

    def transform_assign(self, stmt:ast.Assign)->List[ast.stmt]:
        target:ast.expr = stmt.targets[0]
        if len(stmt.targets) > 1:
            raise self._error(
                "tuple assignment is not differentiable in a kernel body", stmt)
        if isinstance(target, ast.Name):
            return self.assign_to_name(stmt, target)
        if isinstance(target, ast.Subscript):
            return self.assign_to_element(stmt, target)
        raise self._error(
            f"unsupported assignment target: {type(target).__name__}", stmt)

    def assign_to_name(self, stmt:ast.Assign, target:ast.Name)->List[ast.stmt]:
        forward:ast.stmt = copied(stmt)
        emitted:List[ast.stmt] = [forward]
        value:ast.expr = stmt.value
        # An alias of a tracked array shares its tangent; no new buffer, no new statement.
        if isinstance(value, ast.Name) and value.id in self.env:
            self.env[target.id] = self.env[value.id]
            return emitted
        tangent:Optional[ast.expr] = self.tangent_of(value)
        if target.id in self.env or tangent is not None:
            tangent_name:str = self.env.get(target.id) or self.allocator.fresh(
                f"_t_{target.id}")
            emitted.append(placed(assign(store_name(tangent_name),
                                         tangent if tangent is not None else constant(0)),
                                  stmt.lineno))
            self.env[target.id] = tangent_name
        elif target.id in self.scan.carried_scalars:
            # A carried scalar's tangent is seeded at the init even though the init is a
            # constant: the loop augments the value once per element, and the tangent has
            # to accumulate across those augmentations, not be overwritten by each.
            tangent_name:str = self.allocator.fresh(f"_t_{target.id}")
            emitted.append(placed(assign(store_name(tangent_name), constant(0)),
                                  stmt.lineno))
            self.env[target.id] = tangent_name
        else:
            self.env.pop(target.id, None)
        return emitted

    def assign_to_element(self, stmt:ast.Assign, target:ast.Subscript)->List[ast.stmt]:
        root:Optional[str] = _root_name(target.value)
        if root is None:
            raise self._error(
                "unsupported store target in a differentiable kernel", stmt)
        if root in self.tracked:
            raise self._error(
                f"`{root}` is tracked (`requires_grad`): a differentiable body reads it "
                f"but never writes it in place -- the adjoint needs the value the forward "
                f"read, which this store would overwrite", stmt)
        if root != self.out_param:
            raise self._error(
                f"a differentiable kernel writes only its output parameter; `{root}` is "
                f"neither the output nor a gradient buffer this form allocated", stmt)
        forward:ast.stmt = copied(stmt)
        emitted:List[ast.stmt] = [forward]
        tangent:Optional[ast.expr] = self.tangent_of(stmt.value)
        if tangent is None:
            return emitted
        tangent_name:str = self.env[self.out_param]
        emitted.append(placed(assign(
            store_subscript(name(tangent_name), copied(target.slice)),
            tangent), stmt.lineno))
        return emitted

    def transform_augassign(self, stmt:ast.AugAssign)->List[ast.stmt]:
        target:ast.expr = stmt.target
        if isinstance(target, ast.Subscript):
            raise self._error(
                "`arr[i] op= v` reads the element it writes: its adjoint needs the "
                "pre-state, which the fused form does not keep -- spell the read and the "
                "store as two statements on different arrays, or use a local", stmt)
        if not isinstance(target, ast.Name):
            raise self._error("unsupported augmented assignment target", stmt)
        forward:ast.stmt = copied(stmt)
        emitted:List[ast.stmt] = []
        tangent_name:Optional[str] = self.env.get(target.id)
        value_tangent:Optional[ast.expr] = self.tangent_of(stmt.value)
        op:ast.operator = stmt.op
        # An augmented multiply/divide reads its own target in the partials, so the
        # pre-store value is saved before the forward statement runs.
        saved_name:Optional[str] = None
        if isinstance(op, (ast.Mult, ast.Div)) and tangent_name is not None:
            saved_name = self.allocator.fresh(f"_ad_s_{target.id}")
            emitted.append(placed(assign(store_name(saved_name), name(target.id)),
                                  stmt.lineno))
        emitted.append(forward)
        if isinstance(op, ast.Pow):
            raise self._error("`v **= e` is not differentiable in a kernel body; spell "
                              "`v = v ** e`", stmt)
        if isinstance(op, ast.Add) or isinstance(op, ast.Sub):
            if value_tangent is None:
                return emitted          # the tangent passes through unchanged
            if tangent_name is None:
                tangent_name = self.allocator.fresh(f"_t_{target.id}")
                self.env[target.id] = tangent_name
                emitted.append(placed(assign(store_name(tangent_name), value_tangent),
                                      stmt.lineno))
                return emitted
            op_node:ast.operator = ast.Add() if isinstance(op, ast.Add) else ast.Sub()
            emitted.append(placed(augassign(store_name(tangent_name), op_node, value_tangent),
                                  stmt.lineno))
            return emitted
        # Mult / Div: both sides carry tangents through the product or quotient.
        if isinstance(op, (ast.Mult, ast.Div)):
            target_value:ast.expr = name(saved_name) if saved_name else name(target.id)
            if isinstance(op, ast.Mult):
                tangent:Optional[ast.expr] = add_tangent(
                    self.env_tangent_of(target.id), target_value,
                    value_tangent, stmt.value)
            else:
                tangent = divide_tangent(self.env_tangent_of(target.id), target_value,
                                         value_tangent, stmt.value)
            if tangent is None:
                return emitted
            if tangent_name is None:
                tangent_name = self.allocator.fresh(f"_t_{target.id}")
                self.env[target.id] = tangent_name
            emitted.append(placed(assign(store_name(tangent_name), tangent), stmt.lineno))
            return emitted
        raise self._error(f"`{type(op).__name__}` is not differentiable in a kernel body",
                          stmt)

    def transform_if(self, stmt:ast.If)->List[ast.stmt]:
        branch:ast.If = copied(stmt)
        branch.body = self.transform_block(list(stmt.body), stmt.lineno)
        branch.orelse = self.transform_block(list(stmt.orelse), stmt.lineno)
        return [branch]

    def transform_for(self, stmt:ast.For)->List[ast.stmt]:
        loop:ast.For = copied(stmt)
        loop.body = self.transform_block(list(stmt.body))
        return [loop]

    def transform_while(self, stmt:ast.While)->List[ast.stmt]:
        loop:ast.While = copied(stmt)
        loop.body = self.transform_block(list(stmt.body))
        return [loop]

    def transform_return(self, stmt:ast.Return)->List[ast.stmt]:
        if stmt.value is None:
            return [copied(stmt)]
        tangent:Optional[ast.expr] = self.tangent_of(stmt.value)
        return [placed(return_stmt(tangent if tangent is not None else constant(0)),
                       stmt.lineno)]

    def transform_expr(self, stmt:ast.Expr)->List[ast.stmt]:
        value:ast.expr = stmt.value
        if isinstance(value, ast.Call):
            self._check_call_stmt(value, stmt)
        return [copied(stmt)]

    def _check_call_stmt(self, call_node:ast.Call, stmt:ast.stmt)->None:
        target:object = self._resolve_callee(call_node)
        if isinstance(target, str):
            return                      # an unresolvable name: the translator will judge
        reason:Optional[str] = rule_table.refusal_reason(target)
        if reason is not None and "writes launch-wide state" in reason:
            raise self._error(
                "an atomic update is not differentiable here: the jvp form answers the "
                "derivative of a returned or stored result; accumulate the loss through "
                "k.grad(loss=...) instead", stmt)
        if reason is not None and "work-group storage" in reason:
            raise self._error(
                "work-group storage is not differentiable in v1: its gradient slots "
                "would need a shared-memory story per backend", stmt)
        if reason is not None and "work-item or dispatch fact" not in reason:
            raise self._error(f"this call is not differentiable: it {reason}", stmt)

    # ------------------------------------------------------------------ expressions

    def env_tangent_of(self, value_name:str)->Optional[ast.expr]:
        tangent_name:Optional[str] = self.env.get(value_name)
        return name(tangent_name) if tangent_name else None

    def tangent_of(self, node:ast.expr)->Optional[ast.expr]:
        """The tangent expression of ``node``, or None when it is identically zero."""
        if isinstance(node, ast.Constant):
            return None
        if isinstance(node, ast.Name):
            return self.env_tangent_of(node.id)
        if isinstance(node, ast.Subscript):
            root:Optional[str] = _root_name(node.value)
            if root is not None and root in self.env and isinstance(node.value, ast.Name):
                # The tangent of `x[i]` is the direction's own element: the direction
                # array has the tracked array's shape, so the indices carry over as-is.
                return subscript(name(self.env[root]), copied(node.slice))
            return None
        if isinstance(node, ast.BinOp):
            left:ast.expr = wrap(node.left)
            right:ast.expr = wrap(node.right)
            try:
                partials:List[Optional[ast.expr]] = binop_partials(node.op, left, right)
            except OperatorRefusal as refused:
                raise self._error(str(refused), node) from None
            return self._combine(node, [node.left, node.right], partials)
        if isinstance(node, ast.UnaryOp):
            operand:ast.expr = wrap(node.operand)
            try:
                partial:Optional[ast.expr] = unaryop_partials(node.op, operand)
            except OperatorRefusal as refused:
                raise self._error(str(refused), node) from None
            if partial is None:
                return None
            operand_tangent:Optional[ast.expr] = self.tangent_of(node.operand)
            if operand_tangent is None:
                return None
            return operand_tangent * partial
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return None                  # a bool: comparisons propagate no tangent
        if isinstance(node, ast.IfExp):
            body_tangent:Optional[ast.expr] = self.tangent_of(node.body)
            else_tangent:Optional[ast.expr] = self.tangent_of(node.orelse)
            if body_tangent is None and else_tangent is None:
                return None
            return ifexp(copied(node.test),
                         body_tangent if body_tangent is not None else constant(0),
                         else_tangent if else_tangent is not None else constant(0))
        if isinstance(node, ast.Call):
            return self.call_tangent(node)
        if isinstance(node, ast.Attribute):
            return None                  # an index fact or a component read: no tangent
        raise self._error(
            f"unsupported expression in a differentiable kernel: "
            f"{type(node).__name__}", node)

    def _combine(self, node:ast.expr, operands:List[ast.expr],
                 partials:List[Optional[ast.expr]])->Optional[ast.expr]:
        """Sum the ``partial_i * tangent_i`` terms; None when every term is zero."""
        result:Optional[ast.expr] = None
        for operand, partial in zip(operands, partials):
            if partial is None:
                continue
            operand_tangent:Optional[ast.expr] = self.tangent_of(operand)
            if operand_tangent is None:
                continue
            term:ast.expr = _term_of(operand_tangent, partial)
            result = term if result is None else result + term
        return result

    def call_tangent(self, node:ast.Call)->Optional[ast.expr]:
        target:object = self._resolve_callee(node)
        if isinstance(target, str):
            raise self._error(
                "a device helper is not differentiable in v1: register an adjoint for it "
                "with pygpgpu.ad.grad_for (reverse mode), or inline its body", node)
        rule:Optional[rule_table.DiffRule] = rule_table.rule_for(target)
        if rule is None:
            reason:Optional[str] = rule_table.refusal_reason(target)
            if reason is None:
                raise self._error(
                    f"the callee {getattr(target, '__name__', target)!r} has no AD rule; "
                    f"every differentiable call must resolve to a ruled or explicitly "
                    f"refused name", node)
            raise self._error(f"this call is not differentiable: it {reason}", node)
        if rule.arity is not None and len(node.args) != rule.arity:
            raise self._error(
                f"this call takes {rule.arity} differentiable argument(s) in the AD "
                f"rules; the call passes {len(node.args)}", node)
        wrapped:List[ast.expr] = [wrap(argument) for argument in node.args]
        try:
            partials:List[Optional[ast.expr]] = rule.partials(wrapped)
        except Exception as e:
            raise self._error(
                f"the AD rule for {getattr(target, '__name__', target)!r} failed to build "
                f"its partials: {e}", node) from None
        return self._combine(node, list(node.args), partials)

    def _resolve_callee(self, node:ast.Call)->object:
        """The object a call's callee names, or the spelling when it cannot resolve.

        Mirrors the translator's resolution: a bare name resolves in the kernel's globals
        and falls back to python's builtins; an attribute chain resolves by getattr.
        """
        func:ast.expr = node.func
        if isinstance(func, ast.Name):
            return self.origin.__globals__.get(func.id, _py.__dict__.get(func.id, func.id))
        if isinstance(func, ast.Attribute):
            base:object = self._resolve_attribute_base(func.value)
            if base is not None:
                return getattr(base, func.attr, func.attr)
        return None

    def _resolve_attribute_base(self, node:ast.expr)->object:
        if isinstance(node, ast.Name):
            return self.origin.__globals__.get(node.id)
        if isinstance(node, ast.Attribute):
            base:object = self._resolve_attribute_base(node.value)
            return getattr(base, node.attr, None) if base is not None else None
        return None

    # ------------------------------------------------------------------ errors

    def _error(self, message:str, node:Optional[ast.AST])->SyntaxError:
        return ad_error(message, node, self.filename, self.source, self.first_line)


def _term_of(tangent:ast.expr, partial:ast.expr)->ast.expr:
    """``partial * tangent`` with the constant-one simplification."""
    if isinstance(partial, ast.Constant) and partial.value == 1:
        return tangent
    if isinstance(tangent, ast.Constant) and tangent.value == 1:
        return partial
    return tangent * partial


def _root_name(node:ast.expr)->Optional[str]:
    """The name a subscript bottoms out in, or None."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None
