# The AD rule registry: one entry per differentiable name, keyed by the callee's object.
from __future__ import annotations

import ast
from typing import Callable, Dict, List, Optional

from ..ast_support import constant
from ...kernel.translator import short_hash

# The partial derivative of a result w.r.t. one of its arguments, as an expression over the
# forward values the call already computed. ``None`` means the argument's partial is zero,
# which is a different statement from "this name is not differentiable" -- a zero partial
# comes from a closed rule, a refusal comes with a reason the user can act on.
PartialBuilder = Callable[[List[ast.expr]], List[Optional[ast.expr]]]


class DiffRule:
    """One differentiable callee, in both directions.

    ``partials`` serves forward and reverse mode from one table: the forward tangent of
    ``f(u0, u1)`` is ``sum(partial_i * tangent_i)``, and the reverse step distributes an
    incoming cotangent ``g`` as ``distribute(u_i, g * partial_i)``. Both walks read the
    same entry, which is what keeps the two modes from drifting -- a mode that disagreed
    with the other would answer a different derivative for the same kernel.

    ``injects`` are the names the partial expressions may spell (``sqrt``, ``sign``, ...)
    mapped to the objects they must resolve to. The transform copies the kernel's own
    globals for the adjoint and adds these, so a rule works whether or not the user's
    module imported the name; the translator then recognizes the call by the object, as it
    does for the user's own imports.
    """

    def __init__(self, partials:PartialBuilder,
                 injects:Optional[Dict[str, object]] = None,
                 arity:Optional[int] = None):
        self.partials:PartialBuilder = partials
        self.injects:Dict[str, object] = dict(injects or {})
        # The argument count the partials are built for; None lets the builder decide.
        # A rule built for two operands refuses a five-operand `min(...)` rather than
        # differentiating the wrong operand positions.
        self.arity:Optional[int] = arity


# The differentiable names, keyed by the callable object a body resolves to.
RULES_BY_OBJECT:Dict[object, DiffRule] = {}

# The names a body may call that the AD stage refuses, with the reason spelled out. The
# sentence this table prevents: "a new builtin was added and it silently has a zero
# gradient" -- the forward answer is right, the expression type-checks, and only the
# number backward produces is wrong.
NON_DIFFERENTIABLE:Dict[object, str] = {}


def register(target:object, rule:DiffRule)->None:
    """Add one differentiable entry; a duplicate is a table bug and fails here."""
    if target in RULES_BY_OBJECT or target in NON_DIFFERENTIABLE:
        raise ValueError(f"AD rule registered twice for {target!r}")
    RULES_BY_OBJECT[target] = rule


def refuse(target:object, reason:str)->None:
    """Add one NON_DIFFERENTIABLE entry with its reason."""
    if target in RULES_BY_OBJECT or target in NON_DIFFERENTIABLE:
        raise ValueError(f"AD rule registered twice for {target!r}")
    NON_DIFFERENTIABLE[target] = reason


def rule_for(target:object)->Optional[DiffRule]:
    """The rule a callee object carries, or None when it is not differentiable."""
    return RULES_BY_OBJECT.get(target)


def refusal_reason(target:object)->Optional[str]:
    """Why a callee is refused, or None when it is not known to this table at all."""
    return NON_DIFFERENTIABLE.get(target)


def knows(target:object)->bool:
    """True when the table has an opinion about this object (either kind)."""
    return target in RULES_BY_OBJECT or target in NON_DIFFERENTIABLE


def collect_injects()->Dict[str, object]:
    """Every name the whole table may spell, mapped to its object.

    The adjoint namespace is built once from this, so a rule's spelling always resolves no
    matter which subset of rules the body used.
    """
    merged:Dict[str, object] = {}
    for rule in RULES_BY_OBJECT.values():
        merged.update(rule.injects)
    return merged


RULES_VERSION:str = ""


def seal() -> str:
    """Freeze the table's identity: a digest over its contents.

    A VJP change moves the gradient kernel's body while the user's ``.py`` has not moved,
    so the rules' version goes into the adjoint kernel's identity (its source signature
    and its C name) beside ``type_map`` -- without it a rule fix would be served the
    binary compiled from the old rule.
    """
    global RULES_VERSION
    if not RULES_VERSION:
        parts:List[str] = []
        for target in sorted(RULES_BY_OBJECT, key=lambda t: repr(type(t)) + repr(getattr(t, "__name__", t))):
            parts.append(f"r:{getattr(target, '__qualname__', None) or getattr(target, '__name__', repr(target))}")
        for target, reason in sorted(NON_DIFFERENTIABLE.items(),
                                     key=lambda t: repr(getattr(t[0], "__qualname__", None) or getattr(t[0], "__name__", repr(t[0])))):
            parts.append(f"n:{getattr(target, '__qualname__', None) or getattr(target, '__name__', repr(target))}:{reason}")
        RULES_VERSION = short_hash("\n".join(parts))
    return RULES_VERSION


# The concrete entries, imported for their registration side effects.
from . import scalar  # noqa: E402, F401  (registration)
from .scalar import INJECTS  # noqa: E402  (the rule-provided namespace)

seal()
