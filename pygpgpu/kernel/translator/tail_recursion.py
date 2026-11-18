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
from typing import Dict, List, Optional, Set, Tuple

from .suspended_recursion import (
    TailRecursionError as TailRecursionError,
    _binds_local,
    _fail as _fail,
    _reads_of,
    _rebinding_assignment_from,
    _substitute_call,
)


def _resolve_callee(node:ast.Call, func_name:str)->bool:
    """Whether `node` calls the function by its bare name (`name(...)`).

    Only that spelling is recognized. A method's recursive call is spelled
    `self.name(...)`, an attribute lookup the translator resolves to the same object --
    leaving it unrecognized sends it to the translator's own cycle detection, which is
    the deliberate v1 scope: the attribute path can also name a rebound method, and the
    static check does not guess. The *identity* of the name -- whether the module has
    rebound it to another function since the tree was parsed -- is the hook's question
    (it holds the function object; see FunctionTranslator.__init__).
    """
    return (isinstance(node.func, ast.Name) and isinstance(node.func.ctx, ast.Load)
            and node.func.id == func_name)


def _binds_own_name(tree:ast.FunctionDef)->bool:
    """Whether the body binds the function's own name as a local.

    A local `fact = ...` shadows the module-level function, so a later bare `fact(...)`
    call resolves to the *local* -- python's own rule -- and converting the call into a
    self-recursion would change what the call names. The conversion is refused; the
    translator's cycle detection answers such a body on its own.
    """
    return any(isinstance(child, ast.Name) and child.id == tree.name
               and isinstance(child.ctx, (ast.Store, ast.Del))
               for child in ast.walk(tree))


def _calls_itself(node:ast.AST, func_name:str)->bool:
    return any(isinstance(child, ast.Call) and _resolve_callee(child, func_name)
               for child in ast.walk(node))


def _tail_calls(node:ast.expr, func_name:str)->List[ast.Call]:
    """The recursive calls whose value *is* this expression, whole.

    A call inside a larger expression (`1 + fact(n-1)`) is not whole -- the caller still
    has work after the callee answers. An `and`/`or` right operand is not whole either:
    the operator may short-circuit, so the caller keeps work. `not` and parentheses
    change nothing about when the callee's work ends, so they keep the position.
    """
    if isinstance(node, ast.Call):
        return [node] if _resolve_callee(node, func_name) else []
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _tail_calls(node.operand, func_name)
    if isinstance(node, ast.IfExp):
        return _tail_calls(node.body, func_name) + _tail_calls(node.orelse, func_name)
    return []


def _site_rebinds(call:ast.Call, params:List[str])->List[Tuple[str, ast.expr]]:
    """The (parameter, new value) pairs one recursive call site rebinds.

    Positional only: a keyword or starred argument has no fixed slot, and more arguments
    than parameters is a bug the conversion would hide. A parameter the call does not
    spell keeps its current value -- tail-call semantics pass exactly that -- so it is
    simply absent from the pairs, and the loop form leaves the variable alone.
    """
    if call.keywords:
        raise _fail("a recursive call with keyword arguments has no fixed parameter "
                    "slot to rebind", call)
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            raise _fail("a starred argument has no fixed parameter slot to rebind", call)
    if len(call.args) > len(params):
        raise _fail("the recursive call passes more arguments than the function "
                    "declares", call)
    return [(params[position], arg) for position, arg in enumerate(call.args)]


class _Plan:
    """The static check: does this body recurse, and only into the convertible shape?"""

    def __init__(self, tree:ast.FunctionDef, pointer_params:Optional[Set[str]] = None,
                 globals_map:Optional[dict] = None):
        self.tree:ast.FunctionDef = tree
        self.name:str = tree.name
        self.globals_map:Optional[dict] = globals_map
        # The positional parameters, in declaration order (posonly first). A function
        # the translator accepts has no vararg/kwarg/kwonly -- those are refused before
        # this pass runs -- so this list is the whole parameter world.
        self.params:List[str] = ([arg.arg for arg in tree.args.posonlyargs]
                                 + [arg.arg for arg in tree.args.args])
        # The parameters that travel as pointers (arrays, structures). A rebinding
        # assignment cannot take one -- `T*` cannot be re-stored through the tuple
        # path ("cannot swap array parameters") -- so a tail site passing one is not
        # a shape this pass converts; the native unlock is the caller's next stop.
        self.pointer_params:Set[str] = pointer_params or set()
        self.sites:int = 0
        # The suspended-result sites: `r = f(...)` assignments whose value is exactly
        # one recursive call, keyed by the slot variable. Each carries the statements
        # after the assignment in the same block (the suspended computation) -- the
        # block the k-loop re-runs once per level.
        self.suspended:Dict[str, List[ast.stmt]] = {}
        # The nested sites keyed by the id of their Return node: the recursive call
        # is wrapped inside the return's expression, and the conversion rewrites the
        # whole return into the descend path (rebind, count, continue).
        self.nested_returns:Dict[int, str] = {}
        # The nested sites' rebind pairs, keyed by the slot: the call's arguments
        # recorded at scan time (the convert phase sees the substituted return,
        # whose call is already the slot name).
        self.nested_pairs:Dict[str, List[Tuple[str, ast.expr]]] = {}
        # The parameters the recursive calls rebind (tail sites and suspended sites
        # alike): their values differ per level, so a suspended computation reading
        # one would see only the final value in the k-loop -- refused.
        self.rebound_params:Set[str] = set()

    def scan(self)->None:
        """Walk the body, refusing every shape the conversion cannot model.

        Raises `TailRecursionError` with a position when a premise fails; `sites` says
        how many recursive sites the body carries (zero when it does not recurse at
        all), `suspended` the suspended-result slots.
        """
        if _binds_own_name(self.tree):
            raise _fail(
                f"'{self.name}' is bound as a local inside its own body, so a bare "
                f"call of that name names the local and not this function", self.tree)
        if _calls_itself(self.tree.args, self.name):
            raise _fail("a default argument calls the function itself", self.tree.args)
        self._scan_statements(self.tree.body)

    def _scan_statements(self, body:List[ast.stmt])->None:
        for node in body:
            self._scan_statement(node)

    def _scan_statement(self, node:ast.stmt)->None:
        if isinstance(node, ast.Return) and node.value is not None:
            calls:List[ast.Call] = _tail_calls(node.value, self.name)
            if calls:
                # Validate the site now, so a refusal names its reason and position
                # even though the conversion re-plans each site while rewriting.
                pairs:List[Tuple[str, ast.expr]] = _site_rebinds(calls[0], self.params)
                for param_name, _expr in pairs:
                    if param_name in self.pointer_params:
                        raise _fail(
                            f"the tail call rebinds '{param_name}', which travels as a "
                            f"pointer: a loop conversion cannot re-store it (the "
                            f"native unlock is the caller's next stop)", calls[0])
                    self.rebound_params.add(param_name)
                self.sites += 1
                return
            # A nested site: the return value *contains* the recursive call (once),
            # wrapped in whatever expression the caller spelled -- `return g(f(...))`.
            # The conversion splits it: the call lands in a slot, and the return
            # expression with the slot substituted becomes the suspended block.
            nested:List[ast.Call] = [child for child in ast.walk(node.value)
                                     if isinstance(child, ast.Call)
                                     and _resolve_callee(child, self.name)]
            if len(nested) == 1:
                slot:str = f"_pg_recursion_r_{len(self.suspended)}"
                rest:List[ast.stmt] = self._statements_after(node)
                if rest:
                    raise _fail(
                        "statements follow a return whose value wraps the recursive "
                        "call in an expression -- the wrapped spelling carries the "
                        "suspension in the expression itself", node)
                for param_name, _expr in _site_rebinds(nested[0], self.params):
                    if param_name in self.pointer_params:
                        raise _fail(
                            f"the call rebinds '{param_name}', which travels as a "
                            f"pointer: a loop conversion cannot re-store it (the "
                            f"native unlock is the caller's next stop)", nested[0])
                    self.rebound_params.add(param_name)
                # The suspended expression is the return value with the call
                # replaced by the slot: the same per-level rule applies -- it may
                # read only the slot and the parameters no recursive call rebinds
                # (`n * f(...)` reads `n`, which the call rebinds: the k-loop
                # would show every level the final one's). The substitution
                # mutates the return's value in place (the identity match needs
                # the live nodes); a refusal restores the saved original, so the
                # tree stays exactly as parsed and the native unlock translates
                # the user's body without leaking the slot name.
                import copy  # noqa: PLC0415

                saved:ast.expr = copy.deepcopy(node.value)
                substituted:ast.expr = _substitute_call(node.value, nested[0],
                                                        slot, node)
                reads:Set[str] = _reads_of([ast.Return(value=substituted)])
                from .suspended_recursion import (  # noqa: PLC0415
                    _check_suspension_reads, TailRecursionError as _SuspendedError)

                try:
                    _check_suspension_reads(reads, slot, self.params,
                                            self.rebound_params, self.globals_map,
                                            node,
                                            "the expression wrapping the recursive call")
                except _SuspendedError:
                    # A refusal leaves the tree exactly as parsed: the hook
                    # swallows it on the recursive backends and the native unlock
                    # translates the user's body -- a mutated tree would leak the
                    # slot name into that body.
                    node.value = saved
                    raise
                self.nested_pairs[slot] = _site_rebinds(nested[0], self.params)
                self.suspended[slot] = [ast.Return(value=substituted)]
                self.nested_returns[id(node)] = slot
                self.sites += 1
            return
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and _resolve_callee(node.value, self.name) \
                and isinstance(node.targets[0], ast.Name):
            # A suspended-result site: `r = f(...)` -- the recursive call's value
            # lands in a slot, and the statements after it in this block are what
            # runs once the callee (and every deeper level) has answered. The
            # conversion re-runs that block once per level, so the block may read
            # only the slot and the parameters no recursive call rebinds -- a
            # rebound parameter's value differs per level, and the k-loop would
            # show every level the final one's.
            slot:str = node.targets[0].id
            rest:List[ast.stmt] = self._statements_after(node)
            if not rest:
                raise _fail(
                    f"'{slot} = {self.name}(...)' assigns the recursive result but "
                    f"nothing follows it: the call is in tail position and should be "
                    f"spelled as a plain 'return {self.name}(...)'", node)
            for param_name, _expr in _site_rebinds(node.value, self.params):
                if param_name in self.pointer_params:
                    raise _fail(
                        f"the call rebinds '{param_name}', which travels as a "
                        f"pointer: a loop conversion cannot re-store it (the "
                        f"native unlock is the caller's next stop)", node)
                self.rebound_params.add(param_name)
            reads:Set[str] = _reads_of(rest)
            # A name the block assigns before reading is the block's own local:
            # every pass re-runs the assignment, so the reads never see a stale
            # value -- only names bound OUTSIDE the block are per-level facts.
            bound_in_block:Set[str] = _binds_local(rest)
            from .suspended_recursion import _check_suspension_reads  # noqa: PLC0415

            _check_suspension_reads(reads - bound_in_block, slot, self.params,
                                    self.rebound_params, self.globals_map, node,
                                    "the computation after the recursive call")
            self.suspended[slot] = rest
            self.sites += 1
            return
        if isinstance(node, (ast.For, ast.While)):
            if _calls_itself(node, self.name):
                raise _fail(
                    "a recursive call inside a loop is one call per iteration, and its "
                    "'continue' would bind to that loop instead of the converted one",
                    node)
            self._scan_statements(node.body)
            self._scan_statements(node.orelse)
            return
        if isinstance(node, ast.If):
            self._scan_statements(node.body)
            self._scan_statements(node.orelse)
            return
        if isinstance(node, (ast.With, ast.Try)):
            if _calls_itself(node, self.name):
                raise _fail(
                    f"a recursive call inside a '{type(node).__name__}' statement is not "
                    f"in tail position: the handler would cover the callee's execution",
                    node)

    def _statements_after(self, node:ast.stmt)->List[ast.stmt]:
        """The statements that follow `node` in the block that directly holds it.

        The suspended computation of a `r = f(...)` site is exactly this tail of the
        block -- python runs it after the call answers, and no earlier statement can
        interleave. The lookup walks the tree to find the enclosing list, so the site
        may sit nested in `if` arms.
        """
        return self._statements_after_in(self.tree.body, node)

    def _statements_after_in(self, body:List[ast.stmt],
                             node:ast.stmt)->List[ast.stmt]:
        for position, item in enumerate(body):
            if item is node:
                return body[position + 1:]
            found:List[ast.stmt] = self._statements_after_in(
                getattr(item, "body", []) + getattr(item, "orelse", []), node)
            if found:
                return found
        return []


def _rebinding_assignment(call:ast.Call, pairs:List[Tuple[str, ast.expr]])->ast.stmt:
    """One statement that rebinds every parameter the call passes, simultaneously."""

    return _rebinding_assignment_from(pairs, call)


def tail_loop_tree(tree:ast.FunctionDef, identity:Optional[object] = None,
                   globals_map:Optional[dict] = None,
                   pointer_params:Optional[Set[str]] = None)->Optional[ast.FunctionDef]:
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
    `pointer_params` names the parameters that travel as pointers: a tail site
    rebinding one is refused, because the loop form cannot re-store a pointer.
    """
    if identity is not None and globals_map is not None \
            and globals_map.get(tree.name) is not identity:
        return None
    plan:_Plan = _Plan(tree, pointer_params, globals_map)
    plan.scan()
    if plan.sites == 0:
        return None
    # The level counters start at zero before the loop: an AugAssign reads the
    # name, and the translator's use-before-assignment check would fire on a
    # counter whose first assignment is the AugAssign itself. The result slots
    # need no initializer: the main loop's bottom-out always assigns the slot
    # before the k-loop reads it, and a second declaration would collide with
    # the site assignment's own hoist.
    initializers:List[ast.stmt] = []
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