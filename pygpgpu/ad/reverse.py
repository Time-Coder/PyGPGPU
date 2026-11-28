# The reverse-mode transform: the statement list walked backwards, accumulating gradients.
#
# The fused shape this emits is the M2 decision: one kernel, where each work item first
# recomputes its forward values in registers and then walks its statement list backwards,
# accumulating cotangents. The forward values stay in registers -- nothing is saved to
# memory -- so the only statements the forward pass gains are the copies that keep a value
# a partial reads before a store overwrites it.
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

from .ast_support import (
    resolve_callee,
    NameAllocator,
    assign,
    augassign,
    binop,
    constant,
    copied,
    free_names,
    name,
    placed,
    store_name,
    substitute,
    subscript,
)
from .body_scan import BodyScan, _assign_targets
from .errors import ad_error
from .forward import _root_name
from . import grad_for
from . import rules as rule_table
from .reverse_distribute import (Contribution, OutCotangent, SEED_COT, SEED_ONE,
                                 SEED_PARAM, _DistributeMixin)
from .reverse_guards import _ScanGuardsMixin
from .reverse_while import _WhileReplayMixin

#: A `for`/`while` loop's `else` block has no adjoint in v1: the walk cannot attribute
#: the else block's stores to the loop's iterations, so a value assigned only there
#: would silently lose its cotangent. One message for the dispatch loop (checked in
#: `build`, whose walk starts at the dispatch body) and every nested loop (checked in
#: `walk_stmt`).
LOOP_ELSE_REFUSAL:str = ("the `else` block of a `for`/`while` loop is not differentiable "
                         "in v1: the reverse walk cannot attribute the else block's "
                         "stores to the loop's iterations -- move the else work after "
                         "the loop")

class GradBuilder(_DistributeMixin, _ScanGuardsMixin, _WhileReplayMixin):
    """The reverse walk over one kernel body.

    ``slots`` maps each live local to its cotangent register; ``gradient_of`` maps each
    tracked array parameter to its gradient buffer parameter. A slot is demanded lazily,
    from the read side: the reverse list is built backwards, so by the time the walk
    reaches the statement that defines a value, every later statement that reads it has
    already demanded the slot -- which is exactly the "does the loss reach this value"
    question, answered where it is cheap.
    """

    def __init__(self, scan:BodyScan, wrt:List[str], gradient_of:Dict[str, str],
                 loss_name:Optional[str], seed_mode:str,
                 cot_of_out:Optional[OutCotangent], cot_param:Optional[str],
                 ret_cot_param:Optional[str],
                 allocator:NameAllocator, origin:object,
                 filename:str, source:str, first_line:int,
                 allow_scatter:bool = False, drop_launch_wide:bool = False):
        self.scan:BodyScan = scan
        self.wrt:Set[str] = set(wrt)
        self.gradient_of:Dict[str, str] = dict(gradient_of)
        self.loss_name:Optional[str] = loss_name
        self.seed_mode:str = seed_mode
        self.cot_of_out:Optional[OutCotangent] = cot_of_out
        self.cot_param:Optional[str] = cot_param
        self.ret_cot_param:Optional[str] = ret_cot_param
        self.allocator:NameAllocator = allocator
        self.origin:object = origin
        self.filename:str = filename
        self.source:str = source
        self.first_line:int = first_line
        self.allow_scatter:bool = allow_scatter
        self.drop_launch_wide:bool = drop_launch_wide
        self.slots:Dict[str, str] = {}
        self.slot_inits:List[str] = []
        # Statements the distribution needs emitted before the contributions commit --
        # a registered adjoint's unpacking call is the one producer of them in v1.
        self.pending:List[ast.stmt] = []
        # Registered helper adjoints this kernel's body reached, keyed by the mangled
        # spelling their call sites spell; the assembled namespace carries the objects.
        self.helper_injects:Dict[str, object] = {}
        # The saved copies of the statement being distributed (see _save_destroyed); the
        # registered-adjoint call sites read them so their arguments match the statement's.
        self._current_saves:Dict[str, str] = {}
        # Save slot -> the original variable it snapshots, so a contribution aimed at a
        # snapshot lands on the original's cotangent instead.
        self._slot_to_original:Dict[str, str] = {}
        # The carried scalar the loss returns, whose cotangent is the seed itself.
        self.seed_name:Optional[str] = None
        if seed_mode in (SEED_ONE, SEED_PARAM) and loss_name is not None:
            self.seed_name = loss_name

    # ------------------------------------------------------------------ entry

    def build(self, funcdef:ast.FunctionDef)->ast.FunctionDef:
        """Transform ``funcdef`` in place (the caller passes its own deep copy)."""
        if self.scan.dispatch is None:
            raise self._error(
                "the fused reverse walk needs a magic loop over the domain: its gradient "
                "accumulation is provably private to the work item, which a "
                "`get_global_id` body cannot show -- iterate "
                "`for i in range(<array>.shape[0])`", funcdef)
        self._check_scanned()
        # The dispatch loop itself never passes through walk_stmt (the walk starts at
        # its body), so its own `else` block is checked here.
        if self.scan.dispatch.orelse:
            raise self._error(LOOP_ELSE_REFUSAL, self.scan.dispatch)
        prefix:List[ast.stmt] = []
        tail:List[ast.stmt] = []
        seen_dispatch:bool = False
        for stmt in funcdef.body:
            if stmt is self.scan.dispatch:
                seen_dispatch = True
                continue
            (tail if seen_dispatch else prefix).append(stmt)
        self._check_host_region(prefix)
        # The cotangent slots are demanded from the read side, and the walk runs
        # backwards over the statements -- so a value read by an EARLY statement has its
        # slot demanded only when the walk reaches that statement, which can be after the
        # later statement whose adjoint needed the slot. The demand is also transitive
        # (a's slot demands b's, whose demands a's). So the walk runs dry first, to a
        # fixpoint over the slot set, and the answers it seeded -- which slots exist --
        # are what the real walk reads. The dry passes' statements are thrown away; the
        # slots, the save registry and the helper injections they built are kept.
        previous:int = -1
        for _pass in range(len(self.scan.dispatch.body) + 1):
            self.walk_block(list(self.scan.dispatch.body), set())
            if len(self.slots) == previous:
                break
            previous = len(self.slots)
        forward, reverse = self.walk_block(list(self.scan.dispatch.body), set())
        dispatch:ast.For = copied(self.scan.dispatch)
        dispatch.body = forward + self._reverse_section(reverse)
        funcdef.body = [copied(stmt) for stmt in prefix] + [dispatch] \
            + [copied(stmt) for stmt in tail]
        return funcdef

    # ------------------------------------------------------------------ blocks

    def walk_block(self, stmts:List[ast.stmt],
                   enclosing:Set[str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        """One block: the forward statements, and the adjoints in reverse order.

        The walk runs BACKWARDS over the statements -- the adjoints are then already in
        reverse order, and, load-bearing, every later statement's cotangent demands have
        happened by the time an earlier statement's adjoint asks whether the value it
        defines reaches the loss. A forward-order walk would answer that question before
        the reads existed and drop every contribution through an intermediate.

        ``enclosing`` is the set of names statements *after* this block assign; a nested
        block's saves must cover the outer block too, because the adjoint of a statement
        inside an `if` runs after the whole forward pass, not after the block.
        """
        forward_groups:List[List[ast.stmt]] = []
        reverse:List[ast.stmt] = []
        for index in range(len(stmts) - 1, -1, -1):
            stmt:ast.stmt = stmts[index]
            destroyed:Set[str] = set(enclosing)
            for later_stmt in stmts[index + 1:]:
                for target in _assign_targets(later_stmt):
                    if isinstance(target, ast.Name):
                        destroyed.add(target.id)
            saved:Dict[str, str] = self._save_destroyed(stmt, destroyed)
            forward_group, reverse_group = self.walk_stmt(stmt, saved, destroyed)
            forward_groups.append(forward_group)
            reverse.extend(reverse_group)
        forward:List[ast.stmt] = []
        for group in reversed(forward_groups):
            forward.extend(group)
        return forward, reverse

    def _save_destroyed(self, stmt:ast.stmt, destroyed:Set[str])->Dict[str, str]:
        """Save the local scalars a later statement overwrites and this one's partials read.

        The adjoint of a statement reads the FORWARD values its partials are built from; a
        local reassigned later no longer holds that value when the adjoint runs, so the
        forward pass copies it into a private slot first. Array parameters are never
        copied: a differentiable body never writes them, so they cannot change.
        """
        if not destroyed:
            return {}
        reads:Set[str] = free_names(stmt) if not isinstance(stmt, ast.Assign) \
            else free_names(stmt.value)
        saved:Dict[str, str] = {}
        for value_name in sorted(reads & destroyed):
            if value_name in self.wrt:
                continue
            saved[value_name] = self.allocator.fresh(f"_ad_s_{value_name}")
        return saved

    # ------------------------------------------------------------------ statements

    def walk_stmt(self, stmt:ast.stmt, saved:Dict[str, str],
                  destroyed:Set[str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        if isinstance(stmt, ast.Assign):
            return self.walk_assign(stmt, saved)
        if isinstance(stmt, ast.AugAssign):
            return self.walk_augassign(stmt, saved)
        if isinstance(stmt, ast.If):
            return self.walk_if(stmt, destroyed)
        if isinstance(stmt, (ast.For, ast.While)) and stmt.orelse:
            raise self._error(LOOP_ELSE_REFUSAL, stmt)
        if isinstance(stmt, ast.For):
            return self.walk_for(stmt, destroyed)
        if isinstance(stmt, ast.While):
            return self.walk_while(stmt, destroyed)
        if isinstance(stmt, ast.Return):
            raise self._error(
                "a `return` inside the work-item dispatch cannot be reversed: the "
                "adjoint statements after it would never run -- return after the loop, "
                "or name `loss=` for the reduction shape", stmt)
        if isinstance(stmt, ast.Expr):
            return self.walk_expr(stmt, saved)
        if isinstance(stmt, ast.Pass):
            return [copied(stmt)], []
        if isinstance(stmt, (ast.Break, ast.Continue)):
            raise self._error(
                f"`{type(stmt).__name__}` cannot be reversed: the adjoint of a partial "
                f"iteration is not defined", stmt)
        raise self._error(
            f"unsupported statement in a differentiable kernel: {type(stmt).__name__}",
            stmt)

    def walk_assign(self, stmt:ast.Assign,
                    saved:Dict[str, str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        target:ast.expr = stmt.targets[0]
        if isinstance(target, ast.Subscript):
            return self._saves(saved, stmt.lineno) + [copied(stmt)], \
                self.reverse_store(target, stmt.value, saved)
        if not isinstance(target, ast.Name):
            raise self._error(f"unsupported assignment target: {type(target).__name__}",
                              stmt)
        incoming:Optional[ast.expr] = self.incoming_for(target.id)
        forward:List[ast.stmt] = []
        if incoming is None:
            return [copied(stmt)], []
        # A plain assign that reads its own target needs the pre-store value saved.
        if target.id in free_names(stmt.value):
            saved = dict(saved)
            saved.setdefault(target.id, self.allocator.fresh(f"_ad_s_{target.id}"))
        forward = self._saves(saved, stmt.lineno) + [copied(stmt)]
        replaced:ast.expr = self._substituted(stmt.value, saved)
        contributions:List[Contribution] = self.distribute(replaced, incoming)
        slot:Optional[str] = self.slots.get(target.id)
        return forward, self._commit(contributions, reset_target=slot)

    def walk_augassign(self, stmt:ast.AugAssign,
                       saved:Dict[str, str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        target:ast.expr = stmt.target
        if isinstance(target, ast.Subscript):
            root:Optional[str] = _root_name(target.value)
            if root is not None and self.loss_name is not None                     and root == self.loss_name:
                # The loss's own accumulation: `loss[i] += E` is the racy-augassign
                # spelling the translator turns into the device atomic (including the
                # float element type), and its adjoint scatters the cotangent onto E --
                # the same shape an `atomic_add(loss, i, E)` call gets.
                if not isinstance(target.slice, ast.Constant):
                    raise self._error(
                        "the loss slot's index must be a constant in v1: a scatter "
                        "index would need the atomic-scatter machinery to reach the "
                        "right slot", target.slice)
                seed:ast.expr = self._seed_expr(target.slice)
                replaced:ast.expr = self._substituted(stmt.value, saved)
                self.pending = []
                self._current_saves = dict(saved)
                contributions:List[Contribution] = self.distribute(replaced, seed)
                forward:List[ast.stmt] = ([] if self.drop_launch_wide
                                          else [copied(stmt)])
                return forward, self.pending + self._commit(contributions,
                                                            reset_target=None)
            raise self._error(
                "`arr[i] op= v` reads the element it writes: its adjoint needs the "
                "pre-state, which the fused form does not keep -- spell the read and the "
                "store as two statements on different arrays, or use a local", stmt)
        if not isinstance(target, ast.Name):
            raise self._error("unsupported augmented assignment target", stmt)
        value_name:str = target.id
        incoming:Optional[ast.expr] = self.incoming_for(value_name)
        if incoming is None:
            return [copied(stmt)], []
        op:ast.operator = stmt.op
        if isinstance(op, ast.Pow):
            raise self._error("`v **= e` is not differentiable in a kernel body; spell "
                              "`v = v ** e`", stmt)
        if isinstance(op, (ast.Mult, ast.Div, ast.BitAnd, ast.BitOr, ast.BitXor,
                           ast.FloorDiv, ast.Mod, ast.MatMult)):
            raise self._error(
                f"`v {type(op).__name__}= e` reads the value it overwrites, and the "
                f"augmented spelling keeps the pre-state from the adjoint: spell "
                f"`v = v ... e` so the plain-assign walk saves what the partials read",
                stmt)
        forward:List[ast.stmt] = []
        if value_name in free_names(stmt.value):
            saved = dict(saved)
            saved.setdefault(value_name, self.allocator.fresh(f"_ad_s_{value_name}"))
        forward = self._saves(saved, stmt.lineno) + [copied(stmt)]
        sign:ast.expr = incoming if isinstance(op, ast.Add) else -incoming
        replaced:ast.expr = self._substituted(stmt.value, saved)
        contributions:List[Contribution] = self.distribute(replaced, sign)
        # The target's cotangent passes through unchanged: the old value's partial is 1.
        return forward, self._commit(contributions, reset_target=None)

    def walk_if(self, stmt:ast.If, destroyed:Set[str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        forward_body, reverse_body = self.walk_block(list(stmt.body), destroyed)
        forward_else, reverse_else = self.walk_block(list(stmt.orelse), destroyed)
        forward:ast.If = copied(stmt)
        forward.body = forward_body
        forward.orelse = forward_else
        if not reverse_body and not reverse_else:
            return [forward], []
        reverse:ast.If = copied(stmt)
        reverse.body = reverse_body or [ast.Pass()]
        reverse.orelse = reverse_else or [ast.Pass()]
        return [forward], [reverse]

    def walk_for(self, stmt:ast.For, destroyed:Set[str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        forward_body, reverse_body = self.walk_block(list(stmt.body), destroyed)
        if stmt is self.scan.dispatch:
            loop:ast.For = copied(stmt)
            loop.body = forward_body
            return [loop], reverse_body
        forward:ast.For = copied(stmt)
        forward.body = forward_body
        reverse_range:Optional[ast.Call] = _reversed_range(stmt.iter)
        if reverse_range is None:
            raise self._error(
                "this serial loop's bound cannot be reversed: only `range(<count>)` "
                "loops reverse in v1", stmt.iter)
        reverse:ast.For = copied(stmt)
        reverse.iter = reverse_range
        reverse.body = reverse_body
        return [forward], [reverse]

    def walk_expr(self, stmt:ast.Expr,
                  saved:Dict[str, str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        value:ast.expr = stmt.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            spell:str = value.func.id
            if spell == "atomic_add" and self.loss_name is not None \
                    and isinstance(value.args[0], ast.Name) \
                    and value.args[0].id == self.loss_name:
                return self.walk_loss_add(value, stmt, saved)
            if spell == "barrier" or spell.startswith("get_"):
                return [copied(stmt)], []
            if spell == "print":
                raise self._error("`print` is not differentiable", stmt)
        if isinstance(value, ast.Call):
            self._check_value_call(value, stmt)
        return [copied(stmt)], []

    def walk_loss_add(self, call_node:ast.Call, stmt:ast.stmt,
                      saved:Dict[str, str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        """`atomic_add(loss, i, v)`: the adjoint scatters the cotangent the slot carried."""
        index:ast.expr = call_node.args[1]
        if not isinstance(index, ast.Constant):
            raise self._error(
                "the loss slot's index must be a constant in v1: a scatter index would "
                "need the atomic-scatter machinery to reach the right slot", index)
        payload:ast.expr = call_node.args[2]
        seed:ast.expr = self._seed_expr(index)
        extended:Dict[str, str] = dict(self._save_destroyed(payload, set()))
        forward:List[ast.stmt] = self._saves(extended, stmt.lineno)
        if not self.drop_launch_wide:
            # The accumulation is spelled as the augmented store, not as the call: the
            # translator's own atomic conversion covers it, including the float element
            # type the user-facing `atomic_add` call refuses -- and it is the spelling
            # the carried-scalar machinery already emits, so one path serves both.
            target:ast.Subscript = ast.Subscript(
                value=ast.Name(id=call_node.args[0].id, ctx=ast.Load()),
                slice=copied(index), ctx=ast.Store())
            target.lineno = stmt.lineno
            target.col_offset = 0
            forward = forward + [placed(augassign(target, ast.Add(), copied(payload)),
                                         stmt.lineno)]
        replaced:ast.expr = self._substituted(payload, extended)
        contributions:List[Contribution] = self.distribute(replaced, seed)
        return forward, self._commit(contributions, reset_target=None)

    def _seed_expr(self, index:ast.expr)->ast.expr:
        """The cotangent of the loss slot: 1.0 for `k.grad`, the weight array for a tape."""
        if self.seed_mode == SEED_COT and self.cot_param is not None:
            return subscript(name(self.cot_param), copied(index))
        return constant(1.0)

    def _check_value_call(self, call_node:ast.Call, stmt:ast.stmt)->None:
        target:object = self._resolve_callee(call_node)
        if isinstance(target, str):
            raise self._error(
                "a device helper is not differentiable in v1: register an adjoint for it "
                "with pygpgpu.ad.grad_for, or inline its body", stmt)
        if grad_for.adjoint_of(target) is not None:
            return          # a discarded result contributes nothing; the call passes
        if rule_table.rule_for(target) is None:
            reason:Optional[str] = rule_table.refusal_reason(target)
            if reason is None:
                raise self._error(
                    f"the callee {getattr(target, '__name__', target)!r} has no AD rule; "
                    f"every differentiable call must resolve to a ruled or explicitly "
                    f"refused name", stmt)
            if "work-item or dispatch fact" not in reason \
                    and "work-group storage" not in reason:
                raise self._error(f"this call is not differentiable: it {reason}", stmt)

    # ------------------------------------------------------------------ misc

    def _substituted(self, node:ast.expr, saved:Dict[str, str])->ast.expr:
        """The expression with the saved copies read in place of overwritten values.

        The mapping is remembered the other way round as well: when the distribution
        meets a save slot as a contribution TARGET, it routes the contribution to the
        original variable's own cotangent (see `_distribute_name`).
        """
        for original, slot in saved.items():
            self._slot_to_original.setdefault(slot, original)
        if not saved:
            return node
        return substitute(node, {key: name(slot) for key, slot in saved.items()})

    def _saves(self, saved:Dict[str, str], lineno:int)->List[ast.stmt]:
        return [placed(assign(store_name(slot), name(value_name)), lineno)
                for value_name, slot in sorted(saved.items())]

    def _reverse_section(self, reverse:List[ast.stmt])->List[ast.stmt]:
        inits:List[ast.stmt] = [placed(assign(store_name(slot), constant(0.0)), 1)
                                for slot in self.slot_inits]
        return inits + reverse

    def _resolve_callee(self, node:ast.Call)->object:
        return resolve_callee(self.origin, node.func)

    def _error(self, message:str, node:Optional[ast.AST])->SyntaxError:
        return ad_error(message, node, self.filename, self.source, self.first_line)


def _reversed_range(iterator:ast.expr)->Optional[ast.Call]:
    """`range(m)` -> `range(m - 1, -1, -1)`; anything else is refused by the caller."""
    if not (isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range" and len(iterator.args) == 1
            and not iterator.keywords):
        return None
    bound:ast.expr = iterator.args[0]
    return ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                    args=[binop(ast.Sub(), copied(bound), constant(1)),
                          constant(-1), constant(-1)],
                    keywords=[])
