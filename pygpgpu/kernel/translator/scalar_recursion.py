"""The native-recursion bar: which self-recursive bodies the unlock admits.

The monomorphizer refuses every cycle because a device helper is compiled per call site
and cached by (function, argument types, exception base): a recursive call would revisit
the very cache slot that is still being filled (`recursive call detected`). Two of the
device languages cannot help -- MSL and OpenCL C forbid recursion outright -- but CUDA and
HIP device functions *may* recurse: the per-thread stack is a documented, adjustable limit
(`cudaLimitStackSize`), not a language ban. The refusal therefore discarded a capability
the backend actually has.

The unlock is deliberately narrow: a self-recursive helper whose signature the *call
site* can spell completely before the body is translated. That is exactly

* **scalars or plain array pointers** -- a plain `T*` travels with the *call site's own*
  shape arguments, and a self-call forwards the same parameter variable back (the
  identity the same-function forwarding gives); a work-group slice, an image or any
  other internal marker is per-launch state the site cannot spell;
* **no raise** -- a raising helper's exception base is baked into its generated code
  and into its cache key, and a self-call's own site would grow the base every visit;
* **a plain function or a method of its own class** -- a method's recursion spells
  `self.<name>` and forwards the receiver struct plus its `_self_<attr>` slots back to
  the same function (the same identity); a *foreign* object's method can name a rebound
  method, which the static facts cannot rule out;
* **positional calls only, spelling every parameter** -- a keyword or starred argument
  has no fixed slot to rebind, and the device call has no python-default machinery, so
  an omitted trailing argument would leave the call one slot short;
* **no outer-scope value reads** -- an outer value becomes an extra trailing parameter
  whose type comes from the caller's own scope, so the self-call site would have to
  know a signature the callee has not derived yet. A bare call of another *function*
  is not an outer value (the translator resolves callable names itself), and that is
  what makes mutual recursion expressible.

The cycle is recognized by a **frame on the in-progress entry**, not by the cache key. A
recursive site's key carries the *caller's* exception base, which is not the key the
callee's own translation marked `in_progress` (the base differs whenever an outer caller
raises), so a key-based test would miss the cycle and translate the body again -- forever.
The in-progress entry therefore carries the translating helper's identity pair
(`(id(func), parameter types)`), and a site whose pair matches any in-progress entry is
the cycle. **Mutual recursion** rides the same check: A's translation marks A, A calls B
(a fresh helper, translated normally), B calls A -- whose frame is active -- and the site
emits a call to A's own generated name.

The return type is the one fact the call site cannot derive from its own arguments; the
scan that makes it static lives in `return_type_scan` (a sibling module, split by
subject: this file owns *admission*, that one owns *typing*).
"""
from __future__ import annotations

import ast
import types
from typing import Dict, List, Optional, Set, Tuple

from .notation import PTR
from .return_type_scan import (
    _bare_calls,
    _parameter_names,
    _receiver_name_of,
)


# The code-generation backends whose generated code may call itself. Metal has no
# recursion in MSL and OpenCL C does not either. HIP compiles through the cuda
# generation and its runtime recurses too, so both land here by that name.
RECURSIVE_BACKENDS:tuple = ("cuda",)

# The in_progress entry carries the translation frame itself -- {"status":
# "in_progress", "frame": (id(func), types), "owner": translator} -- so the cycle
# check is a scan of the shared cache for a matching frame. No separate structure is
# threaded through the constructors, and every value in the cache stays the entry
# dict the host-side writers already assume when they iterate it. `owner` is the
# translator walking the body: a recursive site inside it reports the recursion to
# every frame's owner, which is what the launch's stack raise reads.


def mark_in_progress(cache:Dict[object, object], key:object, func:object,
                     param_types:List[object], owner:Optional[object] = None)->None:
    """Mark a helper as being translated, with its identity pair on the entry."""
    cache[key] = {"status": "in_progress",
                  "frame": (id(func), tuple(param_types)),
                  "owner": owner}


def unmark_in_progress(cache:Dict[object, object], key:object)->None:
    """Drop an in-progress mark (the failure path; the success path replaces the entry)."""
    cache.pop(key, None)


def frame_is_active(cache:Dict[object, object], func:object,
                    param_types:List[object])->bool:
    """Whether some translation in this unit is mid-flight over the identity pair.

    The *innermost* match is the one a recursive call site means, but for the admit
    decision any match is enough: a pair on the stack is a cycle, and the parameter
    types of the innermost frame are the ones the site's own arguments carry (the
    site's signature IS the pair it tests).
    """
    pair:Tuple[object, ...] = (id(func), tuple(param_types))
    for value in cache.values():
        if isinstance(value, dict) and value.get("status") == "in_progress" \
                and value.get("frame") == pair:
            return True
    return False


def scalar_bar_met(tree:ast.FunctionDef, name:str, param_types:List[object],
                   globals_map:Optional[dict] = None)->bool:
    """Whether the body meets the bar the native-recursion unlock requires.

    The bar is what lets a *call site* name the callee's full signature without the
    body being translated. It reads only the tree, so it runs before any translation,
    and it is deliberately conservative: a body that fails any clause keeps the
    monomorphizer's refusal unchanged.

    The recursion clause accepts a body that calls *any* plain name or its own
    `self.<name>` spelling -- self-recursion names the function itself (a method
    spells it through the receiver), mutual recursion names a partner whose own
    translation will reach back here. The cycle check (the frame on the in-progress
    entry) is what decides which of the two a given site closes; the bar only demands
    that the call the site cannot translate *inline* is one the cycle machinery will
    meet again.
    """
    receiver:str = _receiver_name_of(tree)
    calls:List[tuple] = _bare_calls(tree, receiver)
    if not calls:
        return False
    param_count:int = len(_parameter_names(tree, receiver))
    for call_name, call, method_spelled in calls:
        if call.keywords:
            return False
        for arg in call.args:
            if isinstance(arg, ast.Starred):
                return False
        required:int = (param_count - 1 if method_spelled else param_count)
        if call_name == name and len(call.args) < required:
            # The device call spells every parameter: the launch machinery has no
            # python-default pass-through, so an omitted trailing argument would
            # leave the call one slot short ("takes N device arguments but M types
            # were given"). The bar refuses the shape; the monomorphizer's cycle
            # detection answers it with its own message instead. Only the *cycle*
            # calls are checked -- `range(1)` and every other foreign call spell
            # their own signatures.
            return False
    for index, t in enumerate(param_types):
        if not isinstance(t, tuple):
            continue
        # A plain array pointer (`("*", base)`) and a structure pointer travel as
        # `T*` with the *call site's own* shape arguments: a self-call passes the
        # same parameter variable back, and `_bound_shape_name` then maps the
        # callee's extent name to the caller's -- which is the same name, because
        # the two frames are one function. The forwarding is the identity, so the
        # pointer shapes the self-call can spell are exactly these.
        if len(t) == 2 and t[0] == PTR:
            continue
        # A work-group slice or binding, an image or any other internal marker:
        # the spelling the site would have to produce is per-launch state.
        return False
    if _outer_value_names(tree, globals_map):
        # An outer *value* is a trailing parameter whose type comes from the caller's
        # scope; the self-call site would have to know a signature the callee has not
        # derived yet.
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            return False
    return True


def _outer_value_names(tree:ast.FunctionDef, globals_map:Optional[dict])->Set[str]:
    """The names that would become outer-scope *parameters* of this helper.

    The rule mirrors `_bind_outer_variables`: a local (assigned without a scope
    declaration) shadows any outer name, so it is not a parameter; a parameter is
    already bound and is never one; a name declared `global`/`nonlocal` is one; a
    closure cell (`co_freevars`) is one. A module, class, builtin or helper *function*
    is not a value a parameter can carry -- the translator resolves those names by its
    own dispatch (see `_outer_var_entry`'s callable branch), so they stay out. The
    remainder is exactly what the signature would carry that the self-call site cannot
    spell. `globals_map` is the parsed function's own `__globals__`, which is what the
    callable test answers against; without it (a standalone parse) every unresolved
    load is treated as an outer *value* -- the conservative answer.
    """
    from .walk import _BUILTIN_NAMES, _declared_scope_names  # noqa: PLC0415

    declared:Set[str] = _declared_scope_names(tree)
    receiver:str = _receiver_name_of(tree)
    bound:Set[str] = set(_parameter_names(tree, receiver))
    if receiver:
        bound.add(receiver)
    loads:Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loads.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
    candidates:Set[str] = (loads | declared) - bound - _BUILTIN_NAMES
    if globals_map is None:
        return candidates
    resolved:Set[str] = set()
    for name in candidates:
        value:object = globals_map.get(name)
        if isinstance(value, (type, types.ModuleType)) or callable(value):
            # The translator resolves this name itself; it is not a parameter.
            continue
        resolved.add(name)
    return resolved
