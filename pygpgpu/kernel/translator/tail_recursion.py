"""Tail recursion compiled into a loop, before the helper's device translation begins.

A device helper is monomorphized per call site and cached by (function, argument types,
exception base); a recursive call would revisit the site that is being translated, so the
translator refuses cycles (`recursive call detected`). The device languages bound the
refusal harder than python does: MSL and OpenCL C do not allow recursion at all, and CUDA's
per-thread stack overflows silently -- a wrong answer rather than an error. The one shape a
device *can* take is a loop, and a self-recursive function whose recursive call sits in
tail position is exactly a loop in different notation: calling the function again with new
arguments is assigning the parameters and running the body from the top.

This module rewrites that shape at the AST, over the tree the translator already owns
(`FunctionTranslator.__init__` deep-copies the parse, so the rewrite here owns its input).
The rewrite is three moves:

* the whole body is wrapped in `while True:`;
* every tail-recursive `return f(...)` -- at the top level or nested in an `if` arm --
  becomes one simultaneous parameter rebinding (`n, acc = n - 1, acc + n`) followed by
  `continue`;
* every other statement, `return` included, is kept as written: a plain device `return`
  already means "leave the helper", which is what the source's `return` meant.

The translation then sees an ordinary iterative body and emits ordinary device code; the
python backend is untouched by construction -- it executes the user's original function,
where python recursion works.

The pass covers one shape beyond the pure tail call: a recursive call whose result is
*consumed* -- `r = f(...)` with the statements after it reading the slot, or
`return g(f(...))` wrapping the call in an expression. Those become a level counter
(`k += 1` per descent) plus a post-loop section that re-runs the suspended block once
per level, deepest first -- exactly the order the original nesting spells. The
suspended block may read only the slot, loop-invariant globals, and its own locals;
anything else refuses with the reason (see suspended_recursion).

What is refused, with a reason and a position (the hook turns it into the error the user
sees):

* a recursive call inside a `for`/`while` -- the `continue` would bind to *that* loop,
  and one call per iteration is not one return anyway;
* a recursive call inside `try`/`with` -- the handler would cover the callee's whole
  execution in python, and the conversion moves the callee out;
* keyword or starred arguments, or more arguments than parameters -- no fixed slot to
  rebind;
* a default argument that calls the function itself -- the call runs at *definition* time
  on the host, never on the device, but the static check cannot tell.

Omitting a trailing argument is *accepted*: tail-call semantics pass the current value for
a parameter the call does not spell, and the loop form keeps that parameter's variable
untouched at that site -- the same value.

One divergence is deliberately accepted, the package's own documented convention for a
hoisted declaration a branch never wrote: a local read before its first assignment raises
`UnboundLocalError` in the original on the inputs that reach it, while the loop form reads
the stale value an earlier iteration left.
"""
from __future__ import annotations

import ast
from typing import List, Optional, Set

from .site_scan import (  # noqa: F401  (the scan moves with its subject)
    _Plan as _Plan,
    _binds_own_name as _binds_own_name,
    _calls_itself as _calls_itself,
    _resolve_callee as _resolve_callee,
    _site_rebinds as _site_rebinds,
    _tail_calls as _tail_calls,
)
from .suspended_recursion import (
    TailRecursionError as TailRecursionError,
)


def tail_loop_tree(tree:ast.FunctionDef, identity:Optional[object] = None,
                   globals_map:Optional[dict] = None,
                   pointer_params:Optional[Set[str]] = None,
                   self_name:Optional[str] = None)->Optional[ast.FunctionDef]:
    """The loop form of a recursive body, or None when the body does not recurse.

    The input is the translator's own deep-copied tree, and the converted tree moves its
    statement subtrees out of that copy -- nothing is shared with the cached parse. The
    original tree is left reusable (the scan only reads it), so a caller that keeps both
    pays nothing extra. Raises `TailRecursionError`, with the offending node attached,
    when the body recurses but does not convert.

    `identity` is the function object the tree was parsed from, and `globals_map` its
    `__globals__`: when both are given, a bare-name recursive call is only recognized
    while the module-level name still binds *that* object -- a name rebound to another
    function leaves the recursion to the translator's own live resolution.
    `pointer_params` names the parameters that travel as pointers. One a
    recursive call rebinds is supported: the conversion renames the parameter and
    snapshots its value into a local under the original name (the copy-on-rebind
    rule makes that first binding a value copy), so the body's reads see the
    current level's value and the rebind is an ordinary struct-value store.
    """
    if self_name is None and identity is not None and globals_map is not None \
            and globals_map.get(tree.name) is not identity:
        # The bare-name gate: a module function whose module name was rebound to
        # something else is not recognized (the translator's own live resolution
        # answers those). A METHOD skips the gate: its name lives on the class, not
        # in the module globals, and its call sites resolve through the method
        # table, which is the live identity there.
        return None
    plan:_Plan = _Plan(tree, pointer_params, globals_map, self_name)
    plan.scan()
    if plan.sites == 0:
        return None
    # A pointer parameter a recursive call rebinds (a structure argument the walk
    # replaces per level) gets a local value under the parameter's own name: the
    # parameter is renamed, and one initializer snapshots the caller's value into
    # the local -- the body's reads and every rebind then work on the local, which
    # is an ordinary struct value the converted loop can store. The snapshot is
    # what the copy-on-rebind rule wants: the local's name is assigned again (the
    # rebind), so the first binding from the pointer is a value copy, never an
    # alias.
    initializers:List[ast.stmt] = []
    for param_name in sorted(plan.rebind_pointer_params):
        hidden:str = f"_pg_param_{param_name}"
        for index, arg in enumerate(tree.args.posonlyargs + tree.args.args):
            if arg.arg == param_name:
                arg.arg = hidden
                break
        snapshot:ast.Assign = ast.Assign(
            targets=[ast.Name(id=param_name, ctx=ast.Store())],
            value=ast.Name(id=hidden, ctx=ast.Load()))
        ast.copy_location(snapshot, tree)
        ast.fix_missing_locations(snapshot)
        initializers.append(snapshot)
    # The level counters start at zero before the loop: an AugAssign reads the
    # name, and the translator's use-before-assignment check would fire on a
    # counter whose first assignment is the AugAssign itself. The result slots
    # need no initializer: the main loop's bottom-out always assigns the slot
    # before the k-loop reads it, and a second declaration would collide with
    # the site assignment's own hoist.
    for slot in sorted(plan.suspended):
        counter_init:ast.Assign = ast.Assign(
            targets=[ast.Name(id=f"_pg_recursion_k_{slot}", ctx=ast.Store())],
            value=ast.Constant(value=0))
        ast.copy_location(counter_init, tree)
        ast.fix_missing_locations(counter_init)
        initializers.append(counter_init)
    from .suspended_recursion import convert_statements_full  # noqa: PLC0415

    statements:List[ast.stmt] = convert_statements_full(tree.body, plan)
    loop:ast.While = ast.While(test=ast.Constant(value=True), body=statements,
                               orelse=[])
    if statements:
        ast.copy_location(loop, tree.body[0])
    converted_body:List[ast.stmt] = [*initializers, loop]
    if plan.suspended:
        # The post-loop section: every suspended block re-runs once per level,
        # deepest first (the original recursion runs the innermost suspension
        # before the outer ones, and the k-loop's first pass is the innermost
        # level's). The counter starts where the loop left it -- the descending
        # passes incremented it once per level -- and the guard runs the block
        # while it still owes passes. Each pass decrements, so the block runs
        # exactly k times. A suspension whose block is a bare `return expr` (the
        # nested-site form) becomes the assignment the k-loop needs -- the value
        # flows into the slot, and the function's own final `return slot` hands
        # it back after every level has been unwound.
        for slot in sorted(plan.suspended):
            rest:List[ast.stmt] = plan.suspended[slot]
            counter:str = f"_pg_recursion_k_{slot}"
            block:List[ast.stmt] = []
            for statement in rest:
                if isinstance(statement, ast.Return) and statement.value is not None:
                    assign:ast.Assign = ast.Assign(
                        targets=[ast.Name(id=slot, ctx=ast.Store())],
                        value=statement.value)
                    ast.copy_location(assign, statement)
                    ast.fix_missing_locations(assign)
                    block.append(assign)
                    continue
                block.append(statement)
            while_guard:ast.While = ast.While(
                test=ast.Compare(
                    left=ast.Name(id=counter, ctx=ast.Load()),
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=0)]),
                body=[*block,
                      ast.AugAssign(target=ast.Name(id=counter, ctx=ast.Store()),
                                    op=ast.Sub(), value=ast.Constant(value=1))],
                orelse=[])
            ast.copy_location(while_guard, rest[0] if rest else tree)
            ast.fix_missing_locations(while_guard)
            converted_body.append(while_guard)
        final:ast.Return = ast.Return(
            value=ast.Name(id=sorted(plan.suspended)[0], ctx=ast.Load()))
        ast.copy_location(final, tree)
        ast.fix_missing_locations(final)
        converted_body.append(final)
    converted:ast.FunctionDef = ast.FunctionDef(
        name=tree.name, args=tree.args, body=converted_body,
        decorator_list=[], returns=tree.returns, type_comment=None)
    ast.copy_location(converted, tree)
    ast.fix_missing_locations(converted)
    return converted