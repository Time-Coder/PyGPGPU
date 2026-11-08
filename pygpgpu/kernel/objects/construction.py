# The structures a body constructs inline, and the classes behind them.
from __future__ import annotations
from .scan import _parse, _tree_key
from .state import (
    _CONSTRUCTED,
    _CONSTRUCTED_PARAMETERS,
    _LIFTED_MEMBERS,
    _LIFTED_TREES,
    _LIFTED_VALUES,
    INLINE_PREFIX,
)
from .kinds import _argument_kind, _template_from
import ast
import ctypes
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple



def _constructs_a_class(func:object, resolve:Callable[[str], object],
                         _seen:Optional[set] = None)->bool:
    """Whether the body -- or a helper it calls -- builds a value from a class it defines.

    Both spellings count: `ray = Ray(...)` assigns it to a name, and `hit(ray, Interval(...))`
    builds it straight into a call, which is the one the lift has to handle.

    The helper case is what makes this scan worth running for a kernel body that builds
    nothing itself: `Camera.render` hands its work to `scene.ray_color`, and *that* is where
    `Interval(0, math.inf)` is written. Without following the call, the gate said "nothing is
    built" and the whole scan was skipped, so the construction was never lifted.

    A body already lifted answers from the recorded values: its tree holds the name the lift
    gave it, not the construction, so reading that tree would say "no class is built" and
    skip the very scan that produced the names.
    """
    if _LIFTED_VALUES.get(_tree_key(func)):
        return True
    seen:set = set() if _seen is None else _seen
    if _tree_key(func) in seen:
        return False
    seen.add(_tree_key(func))
    tree:Optional[ast.Module] = _parse(func)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            # `scene.ray_color(ray)`: the receiver is an argument, so the method is the
            # only thing that can be followed -- and it is where the construction is.
            receiver:object = node.func.value
            if not isinstance(receiver, ast.Name):
                continue
            owner:object = resolve(receiver.id)
            if not isinstance(owner, type):
                continue
            method:object = getattr(owner, node.func.attr, None)
            # Only a function has globals to resolve its own names against; an
            # attribute that is a class (`v.float3(...)`) is not a body to walk.
            method_globals:Optional[Dict[str, object]] = getattr(method, "__globals__", None)
            if method_globals is not None and _constructs_a_class(
                    method, method_globals.get, seen):
                return True
            continue
        if not isinstance(node.func, ast.Name):
            continue
        cls:object = resolve(node.func.id)
        if isinstance(cls, type):
            if not issubclass(cls, (ctypes.Structure, list, tuple)):
                return True
            continue
        # A helper is an ordinary function: its own body may be the one that constructs.
        if callable(cls) and _constructs_a_class(cls, resolve, seen):
            return True
    return False



def _inline_constructions(tree:ast.Module, find_class:Callable[[str], object],
                          objects:Dict[str, object],
                          resolve:Callable[[str], object],
                          type_map:Optional[Dict[str, str]]=None)->Dict[str, object]:
    """Lift every unnamed construction into a local, and answer the templates it built.

    `self.hit(ray, Interval(0, math.inf))` builds a value straight into a call, so the scan
    has no *name* to bind it to, and a plain class was refused with "device code has no
    objects". The value means the same as `bounds = Interval(0, math.inf)`, so the tree is
    rewritten to that shape in place: every step after this one -- the member collection, the
    method walk, the translator -- then sees a construction it already handles.

    Rewriting is safe because the lifted name is never emitted: the translator translates
    the *body*, where the constructor call is, so only the scan's view carries the name.
    """
    lifted:Dict[str, object] = {}
    counter:List[int] = [0]

    def lift(node:ast.AST)->ast.AST:
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and not getattr(node, "_named_construction", False)):
            cls:object = find_class(node.func.id)
            if isinstance(cls, type) and not issubclass(cls, (ctypes.Structure, list, tuple)):
                node.args = [lift(child) for child in node.args]
                kinds:Optional[List[object]] = []
                for argument in node.args:
                    # `find_class` resolves the *body's own* names first: `math.inf` is a
                    # module the body's module imported, which the kernel's resolver need
                    # not see.
                    kind:Optional[object] = _argument_kind(argument, objects, find_class,
                                                   tree, type_map)
                    if kind is None:
                        kinds = None
                        break
                    kinds.append(kind)
                template:object = _template_from(kinds, cls)
                if template is not None:
                    name:str = f"{INLINE_PREFIX}{counter[0]}"
                    counter[0] += 1
                    lifted[name] = template
                    # The members this construction supplies, in the constructor's own
                    # argument order: `__init__`'s defaults may name a member nothing can
                    # put on a device (a string no body class ever reads), so only an
                    # argument the call passed is a candidate. The order is kept because the
                    # synthesized structure sorts its fields by name, and an aggregate
                    # literal filled in field order would swap the values.
                    supplied:List[Tuple[str, type]] = []
                    parameter_names:List[str] = _init_parameters(cls)
                    for position, kind in enumerate(kinds or ()):
                        if position < len(parameter_names):
                            supplied.append((parameter_names[position], kind))
                    _LIFTED_MEMBERS.setdefault(cls, list(supplied))
                    return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            # A construction that already has a name is handled by the loop in
            # `constructed_locals`; lifting it again would give the same value a second,
            # unnamed path, which the scan then resolves against an empty template.
            lift_arguments(node.value)
            return node
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                if value and isinstance(value[0], ast.stmt):
                    for item in value:
                        lift(item)
                    continue
                setattr(node, field, [lift(item) if isinstance(item, ast.expr) else item
                                      for item in value])
            elif isinstance(value, ast.expr):
                setattr(node, field, lift(value))
            elif isinstance(value, ast.AST):
                lift(value)
        return node

    def lift_arguments(call:ast.Call)->None:
        """Lift the constructions nested inside a named one's own arguments."""
        call.args = [lift(child) for child in call.args]

    for statement in tree.body:
        lift(statement)
    return lifted



def constructed_locals(func:object, objects:Dict[str, object],
                       resolve:Callable[[str], object],
                       type_map:Optional[Dict[str, str]]=None) -> Dict[str, object]:
    """The values the body builds itself, as though they were arguments.

    `ray = Ray(a, b)` in a kernel body is a value the device will hold, and what it has to
    carry is decided exactly like an object argument's shape: by the members the body -- and
    the methods it hands the value to -- read off it. So it is registered under the body\'s
    own local name and the scan walks it like any other path.

    The value itself is a template: the class is called with placeholders of the call\'s own
    argument types, which is where every member\'s type comes from. A class whose
    constructor refuses them, or which the body never reads, contributes nothing.

    Only a class defined outside this module counts -- a local class has no device meaning --
    and only a call assigned to a plain local name, because an unnamed temporary has no path
    for the scan to bind.

    The class resolves in the *function's own* globals first, then through ``resolve``: a
    method's body uses the names its own module imported (`Sphere.hit` names `HitRecord`,
    which the kernel's module never imported), and the kernel-module fallback keeps the
    caller-visible spellings working.
    """
    found:Dict[str, object] = {}
    cached:Optional[Dict[str, object]] = _LIFTED_VALUES.get(_tree_key(func))
    if cached is not None:
        # This function was already read once, and the tree kept is the *lifted* one: it
        # holds no construction any more, so scanning it again would answer nothing. The
        # values the first pass produced are the answer.
        return dict(cached)
    tree:Optional[ast.Module] = _parse(func)
    if tree is None:
        return found
    own_globals:Dict[str, object] = getattr(func, "__globals__", {}) or {}

    def find_class(name:str)->type:
        if name in own_globals:
            return own_globals[name]
        return resolve(name)

    for statement in ast.walk(tree):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target:ast.expr = statement.targets[0]
        value:ast.expr = statement.value
        if not (isinstance(target, ast.Name) and isinstance(value, ast.Call)):
            continue
        callee:ast.expr = value.func
        if not isinstance(callee, ast.Name):
            continue
        cls:object = find_class(callee.id)
        if not isinstance(cls, type) or issubclass(cls, (ctypes.Structure, list, tuple)):
            continue
        kinds:Optional[List[object]] = []
        for argument in value.args:
            kind:Optional[object] = _argument_kind(argument, objects, resolve, tree, type_map)
            if kind is None:
                # One untyped argument does not sink the construction: the member *set* is
                # decided by the body's reads either way, and the template falls back to the
                # class's own defaults for the types. A type pinned wrong here would put a
                # member of the wrong width on the device; leaving the width to `__init__`'s
                # own defaults is the more conservative answer.
                kinds = None
                break
            kinds.append(kind)
        template:object = _template_from(kinds, cls)
        if template is not None:
            found[target.id] = template
    # A construction written straight into a call -- `self.hit(ray, Interval(0, math.inf))`
    # -- has no name for the scan to bind, so the value it builds was refused as "device
    # code has no objects". The tree is rewritten as though the body had written
    # `bounds = Interval(...)` and passed `bounds`, which is the shape every later step
    # already handles; the lifted names answer from `found` and never reach the generated
    # source, because the translator reaches the constructor in the body itself.
    lifted:Dict[str, object] = _inline_constructions(tree, find_class, objects, resolve)
    if lifted:
        # Every later step reads the same tree (see _parse), or the lifted name would be
        # nowhere in the body the scan walks next. The values are cached with it: a second
        # call to this function would read the *lifted* tree, which no longer holds a
        # construction to lift, and would answer with nothing.
        _LIFTED_TREES[_tree_key(func)] = tree
        _LIFTED_VALUES[_tree_key(func)] = dict(lifted)
    found.update(lifted)
    return found



def _init_parameters(cls:type)->List[str]:
    """The class's constructor parameter order, which is the order a construction supplies."""
    import inspect as _inspect  # noqa: PLC0415

    try:
        parameters:object = _inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return []
    return [name for name, parameter in parameters.items()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]



def constructed_parameters(cls:type)->List[str]:
    """The constructor's positional parameter names for a class the scan constructed."""
    return _CONSTRUCTED_PARAMETERS.get(cls, [])



def constructed_structure(cls:type)->Optional[object]:
    """The structure dtype a class the body constructs was given, or None."""
    return _CONSTRUCTED.get(cls)



def forget_constructed()->None:
    """Drop the recorded constructions, so a new scan cannot see a previous one."""
    _CONSTRUCTED.clear()
    _LIFTED_TREES.clear()
    _LIFTED_VALUES.clear()
    _LIFTED_MEMBERS.clear()
