"""The static site scan for the tail conversion: does a body recurse, and only
into the shape the rewrite can take?

The scan walks the body once and answers three questions: how many recursive
sites the body carries, which of them are *suspended* (a result the caller
consumes after the callee answers, spelled as an assignment or wrapped in the
return expression), and what each site's rebinds touch. A shape the rewrite
cannot model refuses here, with the reason and the offending node attached --
the hook turns that into the error the user sees, or swallows it on the
backends whose device functions may recurse natively.

The vocabulary lives beside the scan: the call-spelling recognition (a bare
name, or the method's own `self.<name>`), the tail-position test, the rebind
pairs one call spells, and the alias/identity rules those pairs follow.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

from .suspended_recursion import (
    TailRecursionError as TailRecursionError,
    _binds_local as _binds_local,
    _fail as _fail,
    _reads_of as _reads_of,
    _substitute_call,
)


def _resolve_callee(node:ast.Call, func_name:str,
                    self_name:Optional[str] = None)->bool:
    """Whether `node` calls the function by its bare name (`name(...)`), or --
    when `self_name` is given -- by the method spelling (`self.name(...)`).

    A method's recursive call is an attribute lookup the translator resolves to the
    same object through the method table, so the self-spelling is recognized only
    under two gates the static check can hold: the base is the receiver parameter
    itself (`self`, as the hook names it), and the attribute is the function's own
    name. A rebound instance attribute would still slip past the spelling check --
    accepted, the same way the method table's dispatch already treats the
    registered method as the attribute's answer; the *identity* of a bare module
    name is the hook's question (it holds the function object).
    """
    if isinstance(node.func, ast.Name) and isinstance(node.func.ctx, ast.Load) \
            and node.func.id == func_name:
        return True
    if self_name is not None and isinstance(node.func, ast.Attribute) \
            and isinstance(node.func.ctx, ast.Load) \
            and isinstance(node.func.value, ast.Name) \
            and node.func.value.id == self_name \
            and node.func.attr == func_name:
        return True
    return False


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


def _site_rebinds(call:ast.Call, params:List[str],
                  self_name:Optional[str] = None)->List[Tuple[str, ast.expr]]:
    """The (parameter, new value) pairs one recursive call site rebinds.

    Positional only: a keyword or starred argument has no fixed slot, and more arguments
    than parameters is a bug the conversion would hide. A parameter the call does not
    spell keeps its current value -- tail-call semantics pass exactly that -- so it is
    simply absent from the pairs, and the loop form leaves the variable alone.

    A method-spelling call (`self.name(...)`) never rebinds the receiver: the
    receiver is what the attribute names, not an argument -- the call's arguments
    bind the parameters *after* it, and the loop form keeps the receiver's variable
    (the `self` pointer) exactly as it is.
    """
    if call.keywords:
        raise _fail("a recursive call with keyword arguments has no fixed parameter "
                    "slot to rebind", call)
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            raise _fail("a starred argument has no fixed parameter slot to rebind", call)
    offset:int = 1 if self_name else 0
    bindable:List[str] = params[offset:]
    if len(call.args) > len(bindable):
        raise _fail("the recursive call passes more arguments than the function "
                    "declares", call)
    pairs:List[Tuple[str, ast.expr]] = [(bindable[position], arg)
                                        for position, arg in enumerate(call.args)]
    # A parameter the call passes *as itself* (`f(n - 1, step)`) is an identity:
    # the rebinding would be `step = step`, which stores nothing and -- after the
    # pointer-param rename -- would read the hidden parameter a second time. The
    # pair is dropped: the loop form leaves the variable alone, which is exactly
    # the value the call passes.
    return [(name, expr) for name, expr in pairs
            if not (isinstance(expr, ast.Name) and expr.id == name)]


class _Plan:
    """The static check: does this body recurse, and only into the convertible shape?"""

    def __init__(self, tree:ast.FunctionDef, pointer_params:Optional[Set[str]] = None,
                 globals_map:Optional[dict] = None, self_name:Optional[str] = None):
        self.tree:ast.FunctionDef = tree
        self.name:str = tree.name
        self.globals_map:Optional[dict] = globals_map
        # The receiver parameter's name, for the method spelling (`self.name(...)`) of
        # a recursive call: the hook supplies it when the function is a method, and a
        # `None` keeps the bare-name-only recognition.
        self.self_name:Optional[str] = self_name
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
        # The pointer parameters a recursive call rebinds (a structure argument the
        # walk replaces per level, the bounce ray of a ray tracer): the loop form
        # cannot re-store the parameter, so the conversion gives each one a local
        # value under the parameter's own name -- the body's reads see the current
        # level's value, the parameter itself is renamed and read only by the one
        # initializer, and the rebind is an ordinary struct-value store.
        self.rebind_pointer_params:Set[str] = set()

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
                pairs:List[Tuple[str, ast.expr]] = _site_rebinds(calls[0], self.params, self.self_name)
                for param_name, _expr in pairs:
                    if param_name in self.pointer_params:
                        self.rebind_pointer_params.add(param_name)
                    self.rebound_params.add(param_name)
                self.sites += 1
                return
            # A nested site: the return value *contains* the recursive call (once),
            # wrapped in whatever expression the caller spelled -- `return g(f(...))`.
            # The conversion splits it: the call lands in a slot, and the return
            # expression with the slot substituted becomes the suspended block.
            nested:List[ast.Call] = [child for child in ast.walk(node.value)
                                     if isinstance(child, ast.Call)
                                     and _resolve_callee(child, self.name, self.self_name)]
            if len(nested) > 1:
                raise _fail(
                    "a return whose value wraps the recursive call in an expression "
                    f"carries it {len(nested)} times -- the counter loop unwinds one "
                    "level per pass, and two independent walks in one expression have "
                    "no order a single counter can reproduce. Name one call's result "
                    "and re-run the other as a loop, or fold both into accumulators",
                    node)
            if len(nested) == 1:
                slot:str = f"_pg_recursion_r_{len(self.suspended)}"
                rest:List[ast.stmt] = self._statements_after(node)
                if rest:
                    raise _fail(
                        "statements follow a return whose value wraps the recursive "
                        "call in an expression -- the wrapped spelling carries the "
                        "suspension in the expression itself", node)
                for param_name, expr in _site_rebinds(nested[0], self.params, self.self_name):
                    if param_name in self.pointer_params:
                        self.rebind_pointer_params.add(param_name)
                    # A parameter the call passes *as itself* (`f(n - 1, step)`) keeps
                    # one value for the whole walk: the rebinding is an identity, and
                    # the suspended expression may read it like any loop-invariant
                    # name. Only a call that gives the parameter a NEW expression
                    # makes its value differ per level.
                    if isinstance(expr, ast.Name) and expr.id == param_name:
                        continue
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
                self.nested_pairs[slot] = _site_rebinds(nested[0], self.params, self.self_name)
                self.suspended[slot] = [ast.Return(value=substituted)]
                self.nested_returns[id(node)] = slot
                self.sites += 1
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)) \
                and node.value is not None and isinstance(node.value, ast.Call) \
                and _resolve_callee(node.value, self.name, self.self_name) \
                and isinstance(node.targets[0] if isinstance(node, ast.Assign)
                               else node.target, ast.Name):
            # A suspended-result site: `r = f(...)` (annotated or not) -- the
            # recursive call's value lands in a slot, and the statements after it
            # in this block are what runs once the callee (and every deeper level)
            # has answered. The conversion re-runs that block once per level, so
            # the block may read only the slot and the parameters no recursive
            # call rebinds -- a rebound parameter's value differs per level, and
            # the k-loop would show every level the final one's.
            slot:str = (node.targets[0].id if isinstance(node, ast.Assign)
                        else node.target.id)
            rest:List[ast.stmt] = self._statements_after(node)
            if not rest:
                raise _fail(
                    f"'{slot} = {self.name}(...)' assigns the recursive result but "
                    f"nothing follows it: the call is in tail position and should be "
                    f"spelled as a plain 'return {self.name}(...)'", node)
            for param_name, expr in _site_rebinds(node.value, self.params, self.self_name):
                if param_name in self.pointer_params:
                    self.rebind_pointer_params.add(param_name)
                # A parameter the call passes *as itself* keeps one value for the
                # whole walk -- see the nested-site branch for the full note.
                if isinstance(expr, ast.Name) and expr.id == param_name:
                    continue
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


