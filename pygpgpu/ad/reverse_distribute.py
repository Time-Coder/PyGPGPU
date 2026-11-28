# The cotangent distribution: how an expression hands its incoming cotangent to its
# constituents, and how the contributions are committed.
#
# Split from the statement walk by subject, per the mixin extraction the file-size policy
# names: `reverse.py` decides the order statements reverse in; this module decides what one
# expression's adjoint is and how its contributions land without one read of a slot
# happening after a commit that rewrites it.
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from ..kernel.translator.augassign import race_free_gather
from .ast_support import (
    assign,
    augassign,
    call,
    constant,
    copied,
    ifexp,
    name,
    placed,
    store_name,
    store_subscript,
    subscript,
    wrap,
)
from . import grad_for
from .body_scan import member_root_name
from .forward import _root_name
from .operator_rules import OperatorRefusal, binop_partials, unaryop_partials
from . import rules as rule_table

# A contribution target: a gradient buffer (param name, index expressions) or a register
# slot (its name). Grouped and committed per statement, never emitted mid-walk.
Contribution = Tuple[Tuple[str, object], ast.expr]

# Where the cotangent of the kernel's answer comes from, and how it is spelled.
SEED_ONE = "one"            # k.grad: the loss is THE number, its cotangent is 1.0
SEED_PARAM = "param"        # vjp on a scalar return: the pullback's weight parameter
SEED_COT = "cot"            # vjp/tape on an array result: the caller's weight array


class OutCotangent:
    """The cotangent of one output array: ``_c_<out>[index]``, spelled per element."""

    def __init__(self, out_param:str, cot_param:str):
        self.out_param:str = out_param
        self.cot_param:str = cot_param

    def __call__(self, index:ast.expr)->ast.expr:
        return subscript(name(self.cot_param), index)


class _DistributeMixin:
    """The expression-level half of the reverse walk, composed into ``GradBuilder``."""

    def incoming_for(self, value_name:str)->Optional[ast.expr]:
        """The cotangent expression of a value, as the reverse walk sees it.

        The loss value's cotangent is the seed; a local's is its register; a value the
        loss never reaches has none, and the statements producing it get no adjoint.
        """
        if value_name == self.seed_name:
            if self.seed_mode == SEED_PARAM and self.ret_cot_param is not None:
                return name(self.ret_cot_param)
            return constant(1.0)
        slot:Optional[str] = self.slots.get(value_name)
        return name(slot) if slot else None

    def demand_slot(self, value_name:str)->str:
        """Create the cotangent register of a local, marking it for the zero-init."""
        slot:Optional[str] = self.slots.get(value_name)
        if slot is None:
            slot = self.allocator.fresh(f"_g_{value_name}")
            self.slots[value_name] = slot
            self.slot_inits.append(slot)
        return slot

    def reverse_store(self, target:ast.Subscript, value:ast.expr,
                      saved:Dict[str, str])->List[ast.stmt]:
        """The adjoint of `out[i] = E`: distribute the element's cotangent onto E."""
        cot:Optional[OutCotangent] = self.cot_of_out
        if cot is None:
            return []
        root:Optional[str] = _root_name(target.value)
        if root is None:
            root = member_root_name(target.value)
        if root != cot.out_param:
            # The store lands somewhere the cotangent does not flow from (a member, a
            # different array): _check_scanned refuses every legal-looking case, and
            # this guard keeps a member store from silently spending the out's weights.
            raise self._error(
                f"this store writes `{root}`, which is not the output parameter "
                f"(`{cot.out_param}`): member stores have no gradient slot in the fused "
                f"form -- pass the member as its own array argument", target)
        incoming:ast.expr = cot(copied(target.slice))
        replaced:ast.expr = self._substituted(value, saved)
        self.pending = []
        self._current_saves = dict(saved)
        contributions:List[Contribution] = self.distribute(replaced, incoming)
        return self.pending + self._commit(contributions, reset_target=None)

    # ------------------------------------------------------------------ distribution

    def distribute(self, node:ast.expr,
                   incoming:ast.expr)->List[Contribution]:
        """Walk ``node`` and answer the cotangent contributions it hands out.

        Nothing is emitted here -- the caller commits the grouped contributions, so every
        read of a slot happens before any commit writes one.
        """
        out:List[Contribution] = []
        self._distribute_into(node, incoming, out)
        return out

    def _distribute_into(self, node:ast.expr, incoming:ast.expr,
                         out:List[Contribution])->None:
        if incoming is None:
            return
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, ast.Name):
            self._distribute_name(node, incoming, out)
            return
        if isinstance(node, ast.Subscript):
            self._distribute_subscript(node, incoming, out)
            return
        if isinstance(node, ast.BinOp):
            left:ast.expr = wrap(node.left)
            right:ast.expr = wrap(node.right)
            try:
                partials:List[Optional[ast.expr]] = binop_partials(node.op, left, right)
            except OperatorRefusal as refused:
                raise self._error(str(refused), node) from None
            for operand, partial in zip((node.left, node.right), partials):
                if partial is None:
                    continue
                self._distribute_into(operand, incoming * partial, out)
            return
        if isinstance(node, ast.UnaryOp):
            operand:ast.expr = wrap(node.operand)
            try:
                partial:Optional[ast.expr] = unaryop_partials(node.op, operand)
            except OperatorRefusal as refused:
                raise self._error(str(refused), node) from None
            if partial is not None:
                self._distribute_into(node.operand, incoming * partial, out)
            return
        if isinstance(node, ast.IfExp):
            test:ast.expr = copied(node.test)
            self._distribute_into(node.body, incoming * ifexp(test, constant(1.0),
                                                              constant(0.0)), out)
            self._distribute_into(node.orelse, incoming * ifexp(copied(node.test),
                                                                constant(0.0),
                                                                constant(1.0)), out)
            return
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return                          # a bool carries no cotangent
        if isinstance(node, ast.Call):
            self._distribute_call(node, incoming, out)
            return
        if isinstance(node, ast.Attribute):
            return                          # an index fact or a component read
        raise self._error(
            f"unsupported expression in a differentiable kernel: {type(node).__name__}",
            node)

    def _distribute_name(self, node:ast.Name, incoming:ast.expr,
                         out:List[Contribution])->None:
        value_name:str = node.id
        # A saved copy stands for the value at the statement being adjointed: its
        # contribution lands on the ORIGINAL variable's cotangent, never on the copy's --
        # the copy is a register snapshot, not a value the loss reaches on its own.
        value_name = self._slot_to_original.get(value_name, value_name)
        if self.scan.is_outer_name(value_name):
            # An enclosing-scope scalar: its gradient was not requested (wrt names call
            # arguments), so the contribution is dropped without allocating a dead
            # register -- and without silently claiming the value was tracked.
            return
        if value_name in self.wrt:
            raise self._error(
                f"`{value_name}` is an array: read the element you mean "
                f"(`{value_name}[i]`), not the whole buffer", node)
        if value_name == self.seed_name:
            return          # the seed is a constant; it takes no contributions
        if value_name not in self.scan.carried_scalars \
                and value_name in self.scan.param_set:
            return          # a scalar parameter's gradient was not requested
        slot:str = self.demand_slot(value_name)
        out.append((("slot", slot), incoming))

    def _distribute_subscript(self, node:ast.Subscript, incoming:ast.expr,
                              out:List[Contribution])->None:
        root:Optional[str] = _root_name(node.value)
        if root is None:
            root = member_root_name(node.value)
        if root is None:
            raise self._error("unsupported read target", node)
        if self.scan.is_member_root(root):
            # A receiver or object member (`self.pts[i]`, `obj.arr[j]`): the value is
            # real, but the fused form carries no gradient slot for a member -- and a
            # member flagged `requires_grad` is refused at the call, so a read here is
            # either a constant to the adjoint (correct to drop) or a mistake. Refuse the
            # read when the cotangent reaches it, naming the parameter spelling.
            raise self._error(
                f"`{root}` is a receiver or object member: the fused adjoint carries no "
                f"gradient slot for a member -- pass the member as its own array "
                f"argument and differentiate that", node)
        if root in self.wrt:
            if not self.allow_scatter \
                    and not race_free_gather(node.slice, self.scan.work_item_indices):
                raise self._error(
                    f"this read of `{root}` is a scatter (its index is not the work "
                    f"item's own): the gradient contribution needs an atomic add, which "
                    f"is the atomic-scatter milestone -- in v1 the fused walk answers "
                    f"gather adjoints only", node)
            gradient:str = self.gradient_of[root]
            out.append((("buffer", gradient, node.slice), incoming))
            return
        if self.cot_of_out is not None and root == self.cot_of_out.out_param:
            raise self._error(
                f"`{self.cot_of_out.out_param}` is read after being written: the adjoint "
                f"needs the pre-state the store overwrote", node)
        if root in self.scan.param_set:
            # A parameter whose gradient was not requested: the read is real, but no
            # cotangent was asked for, so nothing accumulates into it.
            return
        if self.scan.is_outer_name(root):
            raise self._error(
                f"`{root}` comes from the enclosing scope: outer-scope arrays are not "
                f"differentiable in v1 -- pass the array as a parameter to the kernel",
                node)
        raise self._error(
            f"an element of `{root}` feeds a tracked value, but `{root}` is a local "
            f"array: element gradients need a slot per element, which v1 does not carry "
            f"-- pass the array as a parameter instead", node)

    def _distribute_call(self, node:ast.Call, incoming:ast.expr,
                         out:List[Contribution])->None:
        target:object = self._resolve_callee(node)
        if isinstance(target, str):
            raise self._error(
                "a device helper is not differentiable in v1: register an adjoint for it "
                "with pygpgpu.ad.grad_for, or inline its body", node)
        registered:object = grad_for.adjoint_of(target)
        if registered is not None:
            self._distribute_registered(node, incoming, out, registered)
            return
        rule:Optional[rule_table.DiffRule] = rule_table.rule_for(target)
        if rule is None:
            reason:Optional[str] = rule_table.refusal_reason(target)
            if reason is None:
                raise self._error(
                    f"the callee {getattr(target, '__name__', target)!r} has no AD rule; "
                    f"every differentiable call must resolve to a ruled or explicitly "
                    f"refused name -- register an adjoint for your own helper with "
                    f"pygpgpu.ad.grad_for, or inline its body", node)
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
                f"the AD rule for {getattr(target, '__name__', target)!r} failed to "
                f"build its partials: {e}", node) from None
        for operand, partial in zip(node.args, partials):
            if partial is None:
                continue
            self._distribute_into(operand, incoming * partial, out)

    def _distribute_registered(self, node:ast.Call, incoming:ast.expr,
                               out:List[Contribution], registered:object)->None:
        # A helper with a @grad_for adjoint: call it, and hand out its contributions.
        # The adjoint takes the helper's own arguments plus the cotangent of its result,
        # and returns one contribution per argument -- so the call is an unpacking
        # statement the distribution parks in `pending`, and each contribution accumulates
        # into the argument's own cotangent exactly as a rule's partial would.
        spell:str = self.allocator.fresh("_ad_helper")
        self.helper_injects[spell] = registered
        results:List[str] = [self.allocator.fresh("_ad_c") for _ in node.args]
        call_args:List[ast.expr] = [self._substituted(argument, self._current_saves)
                                    for argument in node.args]
        call_args.append(incoming)
        if len(results) == 1:
            self.pending.append(placed(assign(store_name(results[0]),
                                              call(name(spell), call_args)), 1))
        else:
            targets:List[ast.expr] = [store_name(result) for result in results]
            self.pending.append(placed(assign(
                ast.Tuple(elts=targets, ctx=ast.Store()),
                call(name(spell), call_args)), 1))
        for index, argument in enumerate(node.args):
            self._hand_out(argument, name(results[index]), out)

    def _hand_out(self, argument:ast.expr, contribution:ast.expr,
                  out:List[Contribution])->None:
        """Route one registered adjoint's contribution to its argument's target."""
        if isinstance(argument, ast.Name):
            value_name:str = argument.id
            if value_name in self.wrt:
                raise self._error(
                    f"`{value_name}` is an array: read the element you mean "
                    f"(`{value_name}[i]`), not the whole buffer", argument)
            if value_name == self.seed_name:
                return          # the seed is a constant; it takes no contributions
            if value_name not in self.scan.carried_scalars \
                    and value_name in self.scan.param_set:
                return          # a scalar parameter's gradient was not requested
            out.append((("slot", self.demand_slot(value_name)), contribution))
            return
        if isinstance(argument, ast.Subscript):
            root:Optional[str] = _root_name(argument.value)
            if root is not None and root in self.wrt:
                if not self.allow_scatter \
                        and not race_free_gather(argument.slice,
                                                 self.scan.work_item_indices):
                    raise self._error(
                        f"this read of `{root}` is a scatter: the gradient contribution "
                        f"needs an atomic add (the atomic-scatter milestone)", argument)
                out.append((("buffer", self.gradient_of[root], argument.slice),
                            contribution))
                return
        raise self._error(
            "a registered adjoint's contribution to this argument has no cotangent "
            "target: only local scalars and tracked array elements carry one", argument)

    # ------------------------------------------------------------------ committing

    def _commit(self, contributions:List[Contribution],
                reset_target:Optional[str])->List[ast.stmt]:
        """Group the contributions and emit them: buffers first, then slots, target last.

        The group order is load-bearing: a slot's commit must not run before another
        commit that reads it, and the target slot's own commit -- which rewrites what the
        slot means -- reads its own pre-commit value, so it goes last.
        """
        emitted:List[ast.stmt] = []
        buffers:Dict[Tuple[str, ast.expr], List[ast.expr]] = {}
        slots:Dict[str, List[ast.expr]] = {}
        for target, expression in contributions:
            if target[0] == "buffer":
                buffers.setdefault((target[1], target[2]), []).append(expression)
            else:
                slots.setdefault(target[1], []).append(expression)
        for (gradient, index), terms in buffers.items():
            emitted.append(placed(augassign(
                store_subscript(name(gradient), copied(index)), ast.Add(),
                _sum_of(terms)), 1))
        target_slot_terms:Optional[List[ast.expr]] = None
        if reset_target is not None:
            target_slot_terms = slots.pop(reset_target, None)
        for slot, terms in slots.items():
            emitted.append(placed(augassign(store_name(slot), ast.Add(), _sum_of(terms)), 1))
        if reset_target is not None:
            if target_slot_terms:
                emitted.append(placed(assign(store_name(reset_target),
                                             _sum_of(target_slot_terms)), 1))
            else:
                emitted.append(placed(assign(store_name(reset_target), constant(0.0)), 1))
        return emitted



def _sum_of(terms:List[ast.expr])->ast.expr:
    """The terms as one expression."""
    result:ast.expr = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


