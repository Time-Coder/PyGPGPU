"""The suspended-recursion half of the tail conversion.

A recursive call whose result lands in a slot -- `r = f(...)` -- or is wrapped in
an expression -- `return g(f(...))` -- suspends the caller's remaining work. The
conversion re-runs that work once per level after the loop bottoms out (a counter
counts the levels), which is exactly the order the original nesting spells
(innermost suspension first). The machinery here is what the scan and the rewrite
share: the substitution that replaces the recursive call with the slot name, the
read/binding helpers that decide whether a suspended block can run per-level (it
may read only the slot, loop-invariant globals, and its own locals -- a parameter
the recursion rebinds differs per level, and the counter loop would show every
level the final one's), and the rebinding emission the sites use.
"""
from __future__ import annotations

import ast
from typing import List, Optional, Set, Tuple

class _Plan:
    """A forward declaration for the annotation only: the real class lives in
    tail_recursion (the scan and the entry), whose import of this module's
    helpers runs first."""


class TailRecursionError(SyntaxError):
    """The body recurses, but not into the shape this pass converts."""


def _fail(reason:str, node:Optional[ast.AST])->TailRecursionError:
    """The refusal, carrying the offending node for the caller's own positioning."""
    error:TailRecursionError = TailRecursionError(reason)
    error.node = node  # type: ignore[attr-defined]
    return error


def _binds_local(node:object)->Set[str]:
    """The names a statement (or a statement list) binds, anywhere in its subtree.

    python's own rule: an assignment anywhere in a block makes the name local to the
    whole function, so a suspended computation that reads a name bound in the same
    block reads it on *every* pass -- the name cannot be treated as loop-invariant
    without proving the binding happens before the recursion, which this pass does
    not attempt.
    """
    bound:Set[str] = set()
    nodes:List[ast.AST] = (list(node) if isinstance(node, list) else [node])
    for root in nodes:
        for child in ast.walk(root):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                bound.add(child.id)
    return bound


def _reads_of(body:List[ast.stmt])->Set[str]:
    """Every name the statement list reads (loads), anywhere in its subtree."""
    reads:Set[str] = set()
    for node in body:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                reads.add(child.id)
    return reads


def _rebinding_assignment_from(pairs:List[tuple], node:ast.AST)->ast.stmt:
    """The simultaneous rebinding of every parameter a recursive call passes.

    Several pairs become one tuple assignment, whose right-hand sides the device
    translation evaluates into temporaries before any of them lands (see the
    translator's tuple branch) -- python's own tuple-assignment rule, which is
    what makes `n, acc = n - 1, acc + n` read both old values.
    """
    if len(pairs) == 1:
        target:ast.Name = ast.Name(id=pairs[0][0], ctx=ast.Store())
        ast.copy_location(target, node)
        assign:ast.Assign = ast.Assign(targets=[target], value=pairs[0][1])
        ast.copy_location(assign, node)
        return assign
    targets:List[ast.Name] = [ast.Name(id=name, ctx=ast.Store()) for name, _ in pairs]
    value:ast.Tuple = ast.Tuple(elts=[expr for _, expr in pairs], ctx=ast.Load())
    assign = ast.Assign(targets=[ast.Tuple(elts=targets, ctx=ast.Store())], value=value)
    ast.copy_location(assign, node)
    return assign


def _substitute_call(value:ast.expr, call:ast.Call,
                     slot:str, node:ast.AST)->ast.expr:
    """A copy of `value` with the one recursive call replaced by the slot name.

    The descent replaces only the node that *is* the recursive call and never
    stops early: every other child is visited in full, so a call nested several
    wrappers deep (`double_(triple_(f(x)))`) is reached through all of them.
    """
    if value is call:
        name:ast.Name = ast.Name(id=slot, ctx=ast.Load())
        ast.copy_location(name, call)
        return name
    for field, old in list(ast.iter_fields(value)):
        if isinstance(old, list):
            for position, item in enumerate(old):
                if isinstance(item, ast.expr):
                    old[position] = _substitute_call(item, call, slot, node)
        elif isinstance(old, ast.expr):
            setattr(value, field, _substitute_call(old, call, slot, node))
    return value


def _check_suspension_reads(reads:Set[str], slot:str, params:List[str],
                            rebound_params:Set[str], globals_map:Optional[dict],
                            node:ast.AST, description:str)->None:
    """Refuse a suspended computation that reads a per-level value.

    The block re-runs once per level; a name the recursion rebinds (or any local
    the levels write) would show every level the final one's. The allowed reads
    are the slot itself, the block's own locals, and any module-level name (a
    function, class, or constant resolved through the module's globals: one value
    for the whole walk), plus the parameters no recursive call rebinds.
    """
    offenders:Set[str] = set()
    for read in reads:
        if read == slot:
            continue
        if read in params and read not in rebound_params:
            continue
        if globals_map is not None and read in globals_map:
            continue
        offenders.add(read)
    if offenders:
        names:str = ", ".join(sorted(offenders))
        raise _fail(
            f"{description} reads '{names}', whose value differs per recursion "
            f"level -- the loop form runs the remaining statements once per level "
            f"and can only show them the final level's values. Move the reads "
            f"before the call, fold the value into an accumulator parameter, or "
            f"carry it through the call as a parameter", node)


def convert_statements_full(body:List[ast.stmt], plan:_Plan)->List[ast.stmt]:
    """The loop-body statements one statement list becomes, in the generalized form.

    Three kinds of statement, each rewritten where it recurses:

    * a tail `return f(...)` -- the rebinding and the `continue` (as v1 spelled it);
    * a suspended `r = f(...)` -- the rebinding, one `k += 1`, and the `continue`;
      the statements after it (the suspended block) move OUT of the loop to the
      post-loop section, which re-runs them once per level;
    * a nested `return g(f(...))` -- the rebinding, one `k += 1`, and the
      `continue` (the wrapped expression lives in the post-loop block);
    * a plain `return expr` -- the bottom-out: `r = expr` and a `break`, because
      the post-loop section (the suspended blocks) runs after the loop with the
      value that return would have produced;
    * an `if` keeps its node and converts its arms.
    """
    from .tail_recursion import _resolve_callee, _site_rebinds, _tail_calls

    statements:List[ast.stmt] = []
    for node in body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) \
                and node.value is not None and isinstance(node.value, ast.Call) \
                and _resolve_callee(node.value, plan.name, plan.self_name) \
                and isinstance(node.targets[0] if isinstance(node, ast.Assign)
                               else node.target, ast.Name) \
                and (node.targets[0].id if isinstance(node, ast.Assign)
                     else node.target.id) in plan.suspended:
            from .suspended_recursion import _rebinding_assignment_from  # noqa: PLC0415

            pairs:List[Tuple[str, ast.expr]] = _site_rebinds(node.value, plan.params, plan.self_name)
            if pairs:
                statements.append(_rebinding_assignment_from(pairs, node))
            slot_name:str = (node.targets[0].id if isinstance(node, ast.Assign)
                             else node.target.id)
            counter:ast.Name = ast.Name(id=f"_pg_recursion_k_{slot_name}",
                                        ctx=ast.Store())
            increase:ast.AugAssign = ast.AugAssign(
                target=counter, op=ast.Add(), value=ast.Constant(value=1))
            ast.copy_location(increase, node)
            statements.append(increase)
            cont:ast.Continue = ast.Continue()
            ast.copy_location(cont, node)
            ast.fix_missing_locations(cont)
            statements.append(cont)
            continue
        if isinstance(node, ast.Return) and node.value is not None:
            calls:List[ast.Call] = _tail_calls(node.value, plan.name)
            if calls:
                pairs:List[Tuple[str, ast.expr]] = _site_rebinds(calls[0], plan.params)
                if pairs:
                    from .suspended_recursion import (  # noqa: PLC0415
                        _rebinding_assignment_from)

                    statements.append(_rebinding_assignment_from(pairs, calls[0]))
                cont:ast.Continue = ast.Continue()
                ast.copy_location(cont, calls[0])
                ast.fix_missing_locations(cont)
                statements.append(cont)
                continue
            slot:Optional[str] = plan.nested_returns.get(id(node))
            if slot is not None:
                # A nested site: the call's arguments are the level's rebindings;
                # the counter advances and the next pass descends.
                from .suspended_recursion import (  # noqa: PLC0415
                    _rebinding_assignment_from)

                pairs:List[Tuple[str, ast.expr]] = plan.nested_pairs[slot]
                if pairs:
                    statements.append(_rebinding_assignment_from(pairs, node))
                counter:ast.Name = ast.Name(id=f"_pg_recursion_k_{slot}",
                                            ctx=ast.Store())
                increase:ast.AugAssign = ast.AugAssign(
                    target=counter, op=ast.Add(), value=ast.Constant(value=1))
                ast.copy_location(increase, node)
                statements.append(increase)
                cont:ast.Continue = ast.Continue()
                ast.copy_location(cont, node)
                ast.fix_missing_locations(cont)
                statements.append(cont)
                continue
            if plan.suspended:
                # A bottom-out: the value this return would produce is what the
                # suspended blocks consume -- assign the slot and leave the loop.
                from .suspended_recursion import (  # noqa: PLC0415
                    _rebinding_assignment_from)

                assign:ast.Assign = ast.Assign(
                    targets=[ast.Name(id=sorted(plan.suspended)[0], ctx=ast.Store())],
                    value=node.value)
                ast.copy_location(assign, node)
                statements.append(assign)
                brk:ast.Break = ast.Break()
                ast.copy_location(brk, node)
                ast.fix_missing_locations(brk)
                statements.append(brk)
                continue
        if isinstance(node, ast.If):
            arm:ast.If = ast.If(test=node.test,
                                body=convert_statements_full(node.body, plan),
                                orelse=convert_statements_full(node.orelse, plan))
            ast.copy_location(arm, node)
            ast.fix_missing_locations(arm)
            statements.append(arm)
            continue
        statements.append(node)
    return statements


