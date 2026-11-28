# The body scan both AD transforms start from: where the dispatch is, what the body touches.
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

from ..kernel.translator.dispatch_domain import (
    _is_magic_loop_stmt,
    _parameter_names_of,
)


class BodyScan:
    """The facts about a kernel body that decide what its adjoint can look like.

    Both transforms walk the same tree the translator will walk, so they classify the body
    through the same rule the translator uses (`_is_magic_loop_stmt` with the kernel's own
    parameter names) -- the reason a transform's host region and the translator's can never
    disagree about a statement.
    """

    def __init__(self, tree:ast.FunctionDef, params:List[str],
                 receiver_param:Optional[str]=None):
        self.params:List[str] = list(params)
        self.param_set:Set[str] = set(params) | ({receiver_param} if receiver_param else set())
        self.tree:ast.FunctionDef = tree
        self.body:List[ast.stmt] = list(tree.body)
        # The work-item dispatch: the one magic loop a single-phase body carries.
        self.dispatch:Optional[ast.For] = None
        self.dispatch_index:int = -1
        for index, stmt in enumerate(self.body):
            if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt, self.param_set):
                self.dispatch = stmt
                self.dispatch_index = index
                break
        # Every top-level magic loop, dispatch included: a body with several launches as
        # several kernels (the phase machinery), and the fused single-kernel adjoint
        # cannot see the later passes -- the second loop's stores would silently
        # contribute nothing to the gradient -- so the transforms refuse up front.
        self.dispatches:List[ast.For] = [stmt for stmt in self.body
                                         if isinstance(stmt, ast.For)
                                         and _is_magic_loop_stmt(stmt, self.param_set)]
        # The dispatch's index variables, in unpack order -- the names a race-free gather
        # subscript is built from (the same set `_augassign_is_race_free` proves against).
        self.work_item_indices:List[str] = self._dispatch_indices(self.dispatch)
        # Locals initialized above the dispatch and written inside it: the launch-wide
        # carried scalars, whose value is one number for the whole launch.
        self.carried_scalars:List[str] = self._carried()
        # Every name the body BINDS (assignments and loop targets): a name that is neither
        # bound here nor a parameter comes from the enclosing scope, and the AD stage
        # neither tracks it nor allocates dead cotangent registers for it.
        self.assigned_names:Set[str] = set()
        for sub in ast.walk(tree):
            if isinstance(sub, (ast.Assign, ast.AugAssign)):
                for target in _assign_targets(sub):
                    if isinstance(target, ast.Name):
                        self.assigned_names.add(target.id)
            elif isinstance(sub, ast.For):
                for element in ([sub.target] if isinstance(sub.target, ast.Name)
                                else list(getattr(sub.target, "elts", []))):
                    if isinstance(element, ast.Name):
                        self.assigned_names.add(element.id)
        # Every plain name the body spells, so generated names cannot collide.
        self.all_names:Set[str] = {sub.id for sub in ast.walk(tree)
                                   if isinstance(sub, ast.Name)}
        self.all_names.update(self.params)

    def is_outer_name(self, name:str)->bool:
        """True when ``name`` comes from the enclosing scope, not the call's arguments.

        A name the body never binds and the signature never declares is an outer value
        (`_outer_scope_names`'s set, answered without the translator): the gradient of an
        outer scalar was not requested, and an outer array is refused with its own
        message naming the parameter spelling.
        """
        return name not in self.param_set and name not in self.assigned_names

    def is_member_root(self, root:str)->bool:
        """True when a store or read root bottoms out in an attribute chain."""
        return "." in root

    @staticmethod
    def _dispatch_indices(dispatch:Optional[ast.For])->List[str]:
        """The index names a dispatch loop unpacks."""
        if dispatch is None:
            return []
        target:ast.expr = dispatch.target
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Tuple):
            return [element.id for element in target.elts
                    if isinstance(element, ast.Name)]
        return []

    def _carried(self)->List[str]:
        """Names assigned before the dispatch and written inside it."""
        if self.dispatch is None:
            return []
        above:Set[str] = set()
        written_inside:Set[str] = set()
        for stmt in self.body[:self.dispatch_index]:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Assign, ast.AugAssign)):
                    for target in _assign_targets(sub):
                        if isinstance(target, ast.Name):
                            above.add(target.id)
        for sub in ast.walk(self.dispatch):
            if isinstance(sub, (ast.Assign, ast.AugAssign, ast.For)):
                if isinstance(sub, ast.For):
                    continue
                for target in _assign_targets(sub):
                    if isinstance(target, ast.Name):
                        written_inside.add(target.id)
        return [name for name in self.params_order(above & written_inside)]

    def params_order(self, names:Set[str])->List[str]:
        """The given names, in the body's own first-use order (stable output)."""
        found:List[str] = []
        for sub in ast.walk(self.tree):
            if isinstance(sub, ast.Name) and sub.id in names and sub.id not in found:
                found.append(sub.id)
        return found

    # ---------- per-parameter read/write facts ----------

    def value_reads(self)->Dict[str, int]:
        """Parameter -> the number of VALUE reads the DISPATCH gives it.

        The question this answers is the pre-state one: which arrays does the device code
        read that a store in the same code overwrites. The host tail's reads run after the
        launch and flow no gradient, so they are not counted -- `return out` after the
        stores is the saxpy shape the plan keeps legal, not an update in place. A `.shape`
        access is metadata and is pruned outright.
        """
        nodes:List[ast.AST] = [self.tree if self.dispatch is None else self.dispatch]
        counts:Dict[str, int] = {}
        for node in nodes:
            for sub in walk_skipping_shape(node):
                if isinstance(sub, ast.Name) and sub.id in self.param_set \
                        and isinstance(sub.ctx, ast.Load):
                    counts[sub.id] = counts.get(sub.id, 0) + 1
        return counts

    def store_targets(self)->Dict[str, List[ast.stmt]]:
        """Parameter -> the store statements that write into it.

        The tape's output question ("which arguments did this launch write?") and the
        transform's write-analysis refusals both read this; shape metadata again excluded.
        """
        found:Dict[str, List[ast.stmt]] = {}
        for stmt in ast.walk(self.tree):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    root:_StoreRoot = _store_root(target)
                    # A member root (`self.out`) is recorded too: it is not a parameter,
                    # and dropping it would make the store invisible to the refusals --
                    # exactly the silent-miss the scans exist to prevent.
                    if root is not None and (root.name in self.param_set
                                             or root.dotted):
                        found.setdefault(root.name, []).append(stmt)
            elif isinstance(stmt, ast.AugAssign):
                root = _store_root(stmt.target)
                if root is not None and (root.name in self.param_set or root.dotted):
                    found.setdefault(root.name, []).append(stmt)
        return found

    def body_created_stores(self)->Dict[str, List[ast.stmt]]:
        """Store roots the signature does not declare (`scratch[i] = ...` on a body-local).

        The multi-pass adjoint refuses these before anything is built: a later pass's
        adjoint would see fresh zeros where the forward wrote bytes. `store_targets`
        deliberately filters them out (its consumers answer the parameter question), so
        this is the one reader that wants them.
        """
        found:Dict[str, List[ast.stmt]] = {}
        for stmt in ast.walk(self.tree):
            targets:List[ast.expr] = []
            if isinstance(stmt, ast.Assign):
                targets = list(stmt.targets)
            elif isinstance(stmt, ast.AugAssign):
                targets = [stmt.target]
            for target in targets:
                root:Optional[_StoreRoot] = _store_root(target)
                if root is not None and root.name not in self.param_set                         and not root.dotted:
                    found.setdefault(root.name, []).append(stmt)
        return found

    def atomic_calls(self)->List[ast.Call]:
        """Every `atomic_*` call in the body, in source order."""
        found:List[ast.Call] = []
        for sub in ast.walk(self.tree):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                    and sub.func.id.startswith("atomic_"):
                found.append(sub)
        return found

    def returns(self)->List[ast.Return]:
        """The body's `return` statements, in source order."""
        return [sub for sub in ast.walk(self.tree) if isinstance(sub, ast.Return)]

    def is_host_statement(self, stmt:ast.stmt)->bool:
        """True when this top-level statement runs on the host, outside the dispatch.

        The same classification the launcher's host-region split makes (`_outside_loops`):
        everything before the dispatch is host work, the dispatch's own body is not, and
        statements after it are the host tail.
        """
        if self.dispatch is None:
            return False
        index:int = next((i for i, s in enumerate(self.body) if s is stmt), -1)
        if index < 0:
            return False
        return index != self.dispatch_index

    def uses_wrt_values_above_dispatch(self, wrt:Set[str])->Optional[ast.AST]:
        """A host-region statement reading a tracked array's values, or None.

        The host region runs python, and this package does not do numpy AD (a design
        decision, not an oversight) -- so a host statement that computes from a tracked
        array is refused rather than half-differentiated. A `.shape` read stays legal.
        """
        for stmt in self.body:
            if not self.is_host_statement(stmt):
                continue
            for sub in walk_skipping_shape(stmt):
                if isinstance(sub, ast.Name) and sub.id in wrt \
                        and isinstance(sub.ctx, ast.Load):
                    return sub
        return None


def walk_skipping_shape(node:ast.AST):
    """The subtree of ``node``, with every ``<expr>.shape`` metadata access pruned.

    A `.shape` read is a fact about the launch geometry, not a value read, so both the
    value-read counts and the host-region refusals prune the whole attribute subtree --
    `ast.walk` would descend into the attribute it skipped and count the array name under
    it, which is exactly the read that never happens.
    """
    todo:List[ast.AST] = [node]
    while todo:
        current:ast.AST = todo.pop()
        if isinstance(current, ast.Attribute) and current.attr == "shape":
            continue
        if isinstance(current, ast.Subscript) and isinstance(current.ctx, ast.Store):
            # A store target's base is not a value read: CPython spells `out[i] = v` with
            # the base Name in Load context (the Subscript carries Store), so counting it
            # would read the saxpy shape's `out` as an update in place. The indices still
            # walk -- `out[x[i]] = v` really does read `x[i]`.
            yield current
            todo.append(current.slice)
            continue
        if isinstance(current, ast.Call) and isinstance(current.func, ast.Name)                 and current.func.id.startswith("atomic_") and current.args:
            # An atomic's first argument is the slot the update lands in: a write, not a
            # read -- counting it would refuse the loss accumulation as a mid-flight read
            # of the loss. The payload arguments still walk (`atomic_add(a, i, x[i]*v)`).
            yield current
            todo.append(current.func)
            todo.extend(current.args[1:])
            continue
        yield current
        todo.extend(ast.iter_child_nodes(current))


def _assign_targets(stmt:ast.stmt)->List[ast.expr]:
    """The targets of an assignment statement."""
    if isinstance(stmt, ast.Assign):
        return list(stmt.targets)
    if isinstance(stmt, ast.AugAssign):
        return [stmt.target]
    return []


class _StoreRoot:
    """The parameter a store statement's target bottoms out in.

    ``dotted`` says the target bottoms out in an attribute chain (`self.out[i] = ...`
    roots at `"self.out"`): such a store lands in a receiver or object member, which the
    AD stage does not carry a gradient slot for -- the scans record it so the refusals
    can name it, because a store the analysis cannot see would make the adjoint silently
    drop its contribution.
    """

    def __init__(self, name:str, dotted:bool = False):
        self.name:str = name
        self.dotted:bool = dotted


def _store_root(target:ast.expr)->Optional[_StoreRoot]:
    """The root of a store target (`a[i] = ...` -> `a`, `self.out[i] = ...` -> `self.out`)."""
    node:ast.expr = target
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return _StoreRoot(node.id)
    if isinstance(node, ast.Attribute):
        parts:List[str] = []
        walker:ast.expr = node
        while isinstance(walker, ast.Attribute):
            parts.append(walker.attr)
            walker = walker.value
        if isinstance(walker, ast.Name) and parts:
            parts.reverse()
            return _StoreRoot(".".join([walker.id] + parts), dotted=True)
    return None


def member_root_name(node:ast.expr)->Optional[str]:
    """The dotted name an attribute-chain read bottoms out in (`self.pts` from
    `self.pts[i]`), or None when the value is not an attribute chain."""
    walker:ast.expr = node
    while isinstance(walker, ast.Subscript):
        walker = walker.value
    if isinstance(walker, ast.Attribute):
        parts:List[str] = []
        top:ast.expr = walker
        while isinstance(top, ast.Attribute):
            parts.append(top.attr)
            top = top.value
        if isinstance(top, ast.Name) and parts:
            parts.reverse()
            return ".".join([top.id] + parts)
    return None


def parameter_names_of(func:object)->List[str]:
    """The kernel's positional parameter names, receiver excluded, in order."""
    code:object = getattr(func, "__code__", None)
    if code is None:
        return []
    all_names:Tuple[str, ...] = tuple(getattr(code, "co_varnames", ()) or ())
    count:int = int(getattr(code, "co_argcount", 0) or 0)
    names:List[str] = list(all_names[:count])
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return names


def scan_parameters(func:object)->Set[str]:
    """The full parameter set (receiver included) the dispatch classification needs."""
    return _parameter_names_of(func)
