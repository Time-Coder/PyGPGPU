# `grad_for`: a user-registered adjoint for a device helper.
#
# A helper is where an iteration lives -- a Newton step, an integrator -- and an iteration
# is not a place to differentiate through. Warp's `@wp.func_grad` and Taichi's
# `ti.ad.grad_for` exist for the same reason. The shape that fits this package is a
# decorator on the helper that names its adjoint's body, so the adjoint is itself a Python
# function the translator emits, never a host-side closure the device cannot call:
#
#     @grad_for(newton_raphson)
#     def newton_raphson_grad(x, steps, g):
#         ...                                  # recompute what it needs from (x, steps)
#         return gx, gsteps                    # one contribution per argument, in order
#
# The registered function takes the helper's own arguments plus the cotangent of the
# helper's result, and returns one contribution per argument -- the reverse walk
# accumulates each into the argument's own cotangent, exactly as it would a rule's
# partial.
from __future__ import annotations

import inspect
from typing import Callable, Dict, Optional

# Keyed by the helper's *object*, like the rules table: a call is differentiable because
# the callee resolves to an object that has an adjoint, never because its spelling
# matches.
ADJOINTS_BY_FUNCTION:Dict[object, object] = {}


def grad_for(helper:Callable)->Callable:
    """Register the decorated function as ``helper``'s adjoint.

    The decorated function's signature is checked at registration: it takes the helper's
    own arguments plus one cotangent, and returns one contribution per argument. A
    mismatch is a wrong number waiting to happen, so it fails here, where the user is
    looking at the decorator.
    """
    def register(adjoint:Callable)->Callable:
        if helper in ADJOINTS_BY_FUNCTION:
            raise ValueError(f"an AD adjoint is already registered for {helper!r}")
        helper_arity:int = _arity(helper)
        adjoint_arity:int = _arity(adjoint)
        if adjoint_arity != helper_arity + 1:
            raise ValueError(
                f"the adjoint of {getattr(helper, '__name__', helper)!r} must take the "
                f"helper's {helper_arity} argument(s) plus the cotangent "
                f"({helper_arity + 1}); it takes {adjoint_arity}")
        adjoint.__name__ = f"{getattr(helper, '__name__', 'helper')}_adjoint"
        ADJOINTS_BY_FUNCTION[helper] = adjoint
        return adjoint
    return register


def adjoint_of(helper:object)->Optional[object]:
    """The registered adjoint of a helper, or None."""
    return ADJOINTS_BY_FUNCTION.get(helper)


def _arity(func:Callable)->int:
    """The positional arity of a function, defaults not counted separately."""
    signature = inspect.signature(func)
    return len([parameter for parameter in signature.parameters.values()
                if parameter.kind in (parameter.POSITIONAL_ONLY,
                                      parameter.POSITIONAL_OR_KEYWORD)])
