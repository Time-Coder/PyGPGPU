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

Why the rebinding is one tuple assignment: the new argument values are all computed from
the *old* parameter values, so they have to be evaluated before any of them lands --
python's tuple assignment is simultaneous, and the translator's tuple branch is too (it
evaluates every right-hand side into temporaries first, then stores). A sequential spelling
would let a later argument read an earlier parameter's new value.

Why `continue` is safe at any nesting: a tail site inside an `if` arm keeps its `if` (and
its braces) and the `continue` binds to the converted loop, exactly as python's `return`
left every enclosing block at once. A tail site inside an original `for`/`while` is
refused instead -- there the `continue` would bind to *that* loop, and one call per
iteration is not one return anyway.

What is refused, with a reason and a position (the hook turns it into the error the user
sees):

* a recursive call whose result is consumed (`1 + fact(n-1)`) -- tail position only;
* a recursive call inside a `for`/`while` -- see above;
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
from typing import List, Optional, Set, Tuple


class TailRecursionError(SyntaxError):
    """The body recurses, but not into the shape this pass converts."""


def _fail(reason:str, node:Optional[ast.AST])->TailRecursionError:
    """The refusal, carrying the offending node for the caller's own positioning."""
    error:TailRecursionError = TailRecursionError(reason)
    error.node = node  # type: ignore[attr-defined]
    return error


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

    def __init__(self, tree:ast.FunctionDef, pointer_params:Optional[Set[str]] = None):
        self.tree:ast.FunctionDef = tree
        self.name:str = tree.name
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

    def scan(self)->None:
        """Walk the body, refusing every shape the conversion cannot model.

        Raises `TailRecursionError` with a position when a premise fails; `sites` says
        how many tail sites the body carries (zero when it does not recurse at all).
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


def _rebinding_assignment(call:ast.Call, pairs:List[Tuple[str, ast.expr]])->ast.stmt:
    """One statement that rebinds every parameter the call passes, simultaneously.

    Exactly one pair is a plain assignment; several are one tuple assignment, whose
    right-hand sides the device translation evaluates into temporaries before any of
    them lands (see the translator's tuple branch) -- python's own tuple-assignment
    rule, which is what makes `n, acc = n - 1, acc + n` read both old values.
    """
    if len(pairs) == 1:
        target:ast.Name = ast.Name(id=pairs[0][0], ctx=ast.Store())
        ast.copy_location(target, call)
        assign:ast.Assign = ast.Assign(targets=[target], value=pairs[0][1])
        ast.copy_location(assign, call)
        return assign
    targets:List[ast.Name] = [ast.Name(id=name, ctx=ast.Store()) for name, _ in pairs]
    value:ast.Tuple = ast.Tuple(elts=[expr for _, expr in pairs], ctx=ast.Load())
    assign = ast.Assign(targets=[ast.Tuple(elts=targets, ctx=ast.Store())], value=value)
    ast.copy_location(assign, call)
    return assign


def _convert_statements(body:List[ast.stmt], func_name:str,
                        params:List[str])->List[ast.stmt]:
    """The loop-body statements one statement list becomes.

    A tail `return` is replaced by the rebinding and the `continue`; an `if` keeps its
    node and converts its arms (a tail site nested in an arm must be rewritten too);
    everything else -- a plain `return`, a `for`/`while` (its own recursion was refused,
    so nothing inside can be a site) -- is kept as written.
    """
    statements:List[ast.stmt] = []
    for node in body:
        if isinstance(node, ast.Return) and node.value is not None:
            calls:List[ast.Call] = _tail_calls(node.value, func_name)
            if calls:
                pairs:List[Tuple[str, ast.expr]] = _site_rebinds(calls[0], params)
                if pairs:
                    statements.append(_rebinding_assignment(calls[0], pairs))
                cont:ast.Continue = ast.Continue()
                ast.copy_location(cont, calls[0])
                ast.fix_missing_locations(cont)
                statements.append(cont)
                continue
        if isinstance(node, ast.If):
            arm:ast.If = ast.If(test=node.test,
                                body=_convert_statements(node.body, func_name, params),
                                orelse=_convert_statements(node.orelse, func_name,
                                                           params))
            ast.copy_location(arm, node)
            ast.fix_missing_locations(arm)
            statements.append(arm)
            continue
        statements.append(node)
    return statements


def tail_loop_tree(tree:ast.FunctionDef, identity:Optional[object] = None,
                   globals_map:Optional[dict] = None,
                   pointer_params:Optional[Set[str]] = None)->Optional[ast.FunctionDef]:
    """The loop form of a tail-recursive body, or None when the body does not recurse.

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
    plan:_Plan = _Plan(tree, pointer_params)
    plan.scan()
    if plan.sites == 0:
        return None
    statements:List[ast.stmt] = _convert_statements(tree.body, tree.name, plan.params)
    loop:ast.While = ast.While(test=ast.Constant(value=True), body=statements,
                               orelse=[])
    if statements:
        ast.copy_location(loop, tree.body[0])
    converted:ast.FunctionDef = ast.FunctionDef(
        name=tree.name, args=tree.args, body=[loop],
        decorator_list=[], returns=tree.returns, type_comment=None)
    ast.copy_location(converted, tree)
    ast.fix_missing_locations(converted)
    return converted
