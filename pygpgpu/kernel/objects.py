"""Arbitrary python objects as structure arguments.

Anywhere this package accepts a structure -- a by-value argument, a nested member, an element
of a record array -- an arbitrary python object is accepted too. The structure is synthesized
at launch, and its members are the attributes the *kernel body actually reads*: a class with
fifty attributes contributes only the two its reader mentions. That set is decided by a static
scan of the kernel and the helpers it calls, because the device type has to exist before the
argument can be typed.

The scan follows the narrow shapes that can be answered:

* ``p.x`` on a parameter bound to an object;
* ``q = p; q.x`` (a local alias);
* ``helper(p)`` -- the helper's parameter joins the scan, recursively;
* ``p.method()`` -- the method's own ``self.xxx`` reads join the set through the same
  ``scan_self_attrs`` the ``self`` feature uses, and the method is copied onto the synthesized
  class so the translator's existing receiver support finds it.

Anything else is refused with a message rather than guessed at: a nested object member, an
array member, a write to the object, ``getattr``, or an object reached through a container.

Two objects of the same class are merged into one structure *when they can be*: the union of
their members, with ``int`` widened to ``float`` where the two disagree. They stay two
structures when merging would be dishonest -- one instance lacks a member the other needs, or
the same member has types that do not promote.
"""
from __future__ import annotations

import ast
import ctypes
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "looks_like_object",
    "convert_object_arguments",
]


# ---------------------------------------------------------------------------
# What counts as an object argument
# ---------------------------------------------------------------------------


def looks_like_object(value:object)->bool:
    """Whether an argument is an arbitrary python object rather than a value we already know.

    Everything the translator already handles is excluded: arrays, numpy scalars, plain
    numbers, strings, device handles, and anything that is a `ctypes` structure *or union* --
    a caller's declared layout and a vector type included, all of which already have a device
    type. A union is not a `ctypes.Structure` subclass, so testing for that alone sent every
    union down this path, where the members are guessed from what the body reads and a
    `p["i"]` spelling has no layout behind it to answer it.
    """
    from .structs import is_ctypes_record  # noqa: PLC0415 (import cycle)

    if value is None:
        return False
    if isinstance(value, (np.ndarray, np.generic)) or is_ctypes_record(value):
        return False
    if isinstance(value, (bool, int, float, complex, str, bytes)):
        return False
    if not isinstance(value, (type, Callable)):
        from .translator import _is_device_handle  # noqa: PLC0415 (import cycle)

        if _is_device_handle(value):
            return False
    return True


# ---------------------------------------------------------------------------
# The scan: which members does the body read?
# ---------------------------------------------------------------------------


class Scan:
    """The members and methods the body reaches for, per object *path*.

    A path is a dotted chain rooted at a kernel parameter -- ``item``, or ``item.inner`` when
    a member itself is an object. Every path gets its own member set and its own synthesized
    structure, keyed by the path so two members with the same attribute name never collide.
    A path can also name a *list* of objects (``scene.hittables``); its element class is
    taken from the first instance, the loop variable the body iterates with joins that
    class, and the list becomes an array of the element structure.
    """

    def __init__(self, resolve:Callable[[str], object],
                 type_map:Optional[Dict[str, str]]=None)->None:
        self.resolve = resolve
        # The kernel's `int=`/`float=` mapping, so every member type the scan decides carries
        # the width the kernel's own scalar parameters carry. It lives here rather than in
        # every helper's signature: the scan is already threaded through all of them, and a
        # member's width is a property of the kernel this scan belongs to.
        self.type_map:Dict[str, str] = dict(type_map or {})
        self.classes:Dict[str, Optional[type]] = {}
        self.values:Dict[str, object] = {}
        self.members:Dict[str, List[str]] = {}
        self.methods:Dict[str, List[Tuple[str, Callable[..., object]]]] = {}
        self.lists:Dict[str, List[type]] = {}
        self.written:List[str] = []
        # The values the body builds itself and hands around, keyed by its own local name.
        self.constructed:Dict[str, object] = {}
        self.visited:set = set()
        # Every function the walk reached: their bodies may construct a class too, and each
        # of those constructions needs a structure the translator can hand the call.
        self.functions_reached:List["ast.Call"] = []

    def add_path(self, path:str, value:object)->None:
        self.classes.setdefault(path, type(value))
        self.values.setdefault(path, value)
        self.members.setdefault(path, [])
        self.methods.setdefault(path, [])

    def add_list(self, path:str, items:List[type])->None:
        """A path that is a list of objects: element values, one class among them."""
        if not items:
            raise SyntaxError(f"'{path}' is an empty list, so no element structure can be "
                              f"built from it")
        classes:{type} = {type(item) for item in items}
        if len(classes) > 1:
            raise SyntaxError(
                f"'{path}' holds objects of {len(classes)} classes "
                f"({', '.join(sorted(c.__name__ for c in classes))}), and a device loop over "
                f"them would have to dispatch to a different method per element. Use one "
                f"class per list, or a plain array of structures")
        self.lists[path] = list(items)
        self.add_path(path, items[0])
        self.classes[path] = classes.pop()

    def note(self, origin:str, attr:str)->None:
        names:List[str] = self.members.setdefault(origin, [])
        if attr not in names:
            names.append(attr)

    def note_method(self, origin:str, attr:str,
                function:Callable[..., object])->None:
        methods:List[Tuple[str, Callable[..., object]]] = self.methods.setdefault(origin, [])
        if all(name != attr for name, _ in methods):
            methods.append((attr, function))


def _parse(func:Callable[..., object])->Optional[ast.Module]:
    from .translator import _parse_kernel_source, get_dedented_source  # noqa: PLC0415

    cached:Optional[ast.Module] = _LIFTED_TREES.get(_tree_key(func))
    if cached is not None:
        # The scan lifts an inline construction into a local (see _inline_constructions),
        # and every later step has to read that same tree: a fresh parse would give back
        # the original `self.hit(ray, Interval(...))`, with the lifted name nowhere in it.
        import copy as _copy  # noqa: PLC0415

        return _copy.deepcopy(cached)
    try:
        return _parse_kernel_source(get_dedented_source(func))
    except Exception:
        return None


_LIFTED_TREES:Dict[str, ast.Module] = {}
_LIFTED_VALUES:Dict[str, Dict[str, object]] = {}
# class -> the members a lifted construction supplied, in the constructor order.
_LIFTED_MEMBERS:Dict[type, List[Tuple[str, type]]] = {}


def _tree_key(func:Callable[..., object])->str:
    """A key naming one function's source, so a lifted tree can be found again."""
    return f"{getattr(func, '__module__', '')}:{getattr(func, '__qualname__', '')}"


def _follow_aliases(tree:ast.Module, tracked:Dict[str, str])->None:
    """Follow `q = p` so `q.x` counts as `p.x`. A fixed point, because aliases chain.

    A tracked name maps to an object *path*: the parameter it came from, plus any member
    hops, so `inner = p.inner` maps `inner.x` onto the path `p.inner`.
    """
    changed:bool = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Name) and node.value.id in tracked):
                continue
            origin:str = tracked[node.value.id]
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in tracked:
                    tracked[target.id] = origin
                    changed = True


def _member_path(base_path:str, attr:str)->str:
    """The path of a member read: `item.inner` from base `item` and member `inner`."""
    return f"{base_path}.{attr}"


def _element_path_of(scan:Scan, list_path:str)->Optional[str]:
    """The element path a list path's loop bound, or None when nothing iterated it yet."""
    for path in scan.members:
        if path != list_path and _list_root(path) == list_path:
            return path
    return None


def _resolve_list_path(base_path:str, attr:str, scan:Scan,
                       tracked:Dict[str, str])->Optional[str]:
    """The list path `base_path.attr` names, when that member is a list of objects.

    Opening it here is what makes `bag.items[j].v` work without a `for ... in bag.items:`
    anywhere: the subscript's chain is walked, the member list is registered as a list path,
    and the element reads describe the element structure.
    """
    member_path:str = _member_path(base_path, attr)
    if member_path in scan.lists:
        return member_path
    value:object = getattr(scan.values.get(base_path), attr, None)
    if isinstance(value, list) and value and (
            all(looks_like_object(item) or isinstance(item, ctypes.Structure)
                for item in value)):
        scan.add_list(member_path, value)
        scan.note(base_path, attr)
        return member_path
    return None


def self_chain(cursor:ast.Attribute, tracked:Dict[str, str], scan:Scan,
               called:set, node:ast.AST)->Optional[str]:
    """The path a nested attribute chain (`bag.items`) resolves to, opening paths en route."""
    hops:List[str] = []
    walker:ast.AST = cursor
    while isinstance(walker, ast.Attribute):
        hops.append(walker.attr)
        walker = walker.value
    if not (isinstance(walker, ast.Name) and walker.id in tracked):
        return None
    current:str = tracked[walker.id]
    for attr in reversed(hops):
        member_path:str = _member_path(current, attr)
        value:object = getattr(scan.values.get(current), attr, None)
        if isinstance(value, list) and value:
            opened:str = _resolve_list_path(current, attr, scan, tracked)
            if opened is not None:
                current = opened
                continue
        if looks_like_object(value):
            if member_path not in scan.members:
                scan.add_path(member_path, value)
            scan.note(current, attr)
            current = member_path
            continue
        scan.note(current, attr)
        return None
    return current


def _list_root(path:str)->str:
    """The list path an element path belongs to: `items.element#<id>`'s root is `items`.

    An element path is the list path plus the `.element` hop and the `#<node id>` that keeps
    two loops over one list apart; both suffixes come off here.
    """
    root:str = path.split("#", 1)[0]
    if root.endswith(".element"):
        root = root[:-len(".element")]
    return root


def _called_members(tree:ast.Module, tracked:Dict[str, str])->set:
    called:set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver:ast.AST = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id in tracked:
                called.add(node.func.attr)
    return called


def _scan_function(func:object, tracked:Dict[str, str], scan:Scan)->None:
    """Walk one function body, adding the members it reads and recursing into helpers.

    The visit key is the (function, tracked-context) pair: the same helper reached with a
    nested path bound to a parameter scans *that* path's members, which is a different walk
    from the top-level one even though the function is the same.
    """
    visit_key:object = (func, tuple(sorted(tracked.items())))
    if visit_key in scan.visited:
        return
    scan.visited.add(visit_key)
    tree:Optional[ast.Module] = _parse(func)
    if tree is None:
        return
    # A class this body constructs (`record = HitRecord(...)` inside Sphere.hit) is a value
    # like any other: it joins the walk under the body's own name for it, so its member reads
    # land on a path and the translator gets a structure for the constructor call. The
    # template is built from the call's own argument types, which is where each member's type
    # comes from. Doing this per *function* -- not once for the kernel body -- is what makes a
    # construction inside a method or helper visible at all.
    for local_name, template in constructed_locals(func, scan.values, scan.resolve,
                                                         scan.type_map).items():
        if local_name not in scan.constructed:
            scan.constructed[local_name] = template
            scan.add_path(local_name, template)
            # The constructor call itself names members (`HitRecord(is_hitted=False)`), and
            # so does the template's own state -- a `__init__` that assigns every member
            # unconditionally describes the whole layout in its defaults. Both join the
            # reads: a value returned out of this function (`return record`) is read by the
            # *caller*, which this walk never sees, so the construction's own names are the
            # only evidence here.
            for attribute in vars(template):
                scan.note(local_name, attribute)
    _follow_aliases(tree, tracked)
    called:set = _called_members(tree, tracked)
    # `for element in p.items:` over a tracked list path: the loop variable is another name
    # for one element, so its member reads belong to the element structure. (A list of
    # objects is not itself a structure member; the two are kept apart by `add_list`.)
    loop_paths:Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # The iterable is either a tracked name (`for item in items:`) or a member chain
        # rooted at one (`for h in scene.hittables:`).
        iter_path:Optional[str] = None
        if isinstance(node.iter, ast.Name) and node.iter.id in tracked:
            iter_path = tracked[node.iter.id]
        elif isinstance(node.iter, ast.Attribute) and isinstance(node.iter.value, ast.Name) \
                and node.iter.value.id in tracked:
            root_path:str = tracked[node.iter.value.id]
            value:object = getattr(scan.values.get(root_path), node.iter.attr, None)
            if isinstance(value, list):
                member_path:str = _member_path(root_path, node.iter.attr)
                if member_path not in scan.lists:
                    scan.add_list(member_path, value)
                    scan.note(root_path, node.iter.attr)
                iter_path = member_path
        if iter_path is None or iter_path not in scan.lists:
            continue
        if not (isinstance(node.target, ast.Name)):
            continue
        # One element path per (list, loop variable): a re-scan of the same body must reuse
        # it, or the re-scan makes a fresh path every time and never terminates.
        element_path:str = f"{_member_path(iter_path, 'element')}#{node.target.id}"
        element_value:object = scan.lists[iter_path][0]
        scan.add_path(element_path, element_value)
        loop_paths[node.target.id] = element_path

    if loop_paths:
        # Re-scan with the loop variables bound: their attribute reads and method calls
        # describe the element structure. The element path's class is the list's class.
        for name, element_path in loop_paths.items():
            scan.classes[element_path] = scan.classes[_list_root(element_path)]
            scan.values[element_path] = scan.lists[_list_root(element_path)][0]
        merged_tracked:Dict[str, str] = {**tracked, **loop_paths}
        _scan_function(func, merged_tracked, scan)



    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        # Walk out to the root of the attribute chain: `a.b.c` yields the hops
        # [a, b, c] from the outside in, and the root Name decides whether the chain is
        # rooted in a tracked object at all. A subscript hop (`items[j].v`) crosses a list
        # of objects: the element class the list was scanned with describes what follows.
        hops:List[str] = []
        cursor:ast.AST = node
        crossed_list:bool = False
        while isinstance(cursor, ast.Attribute):
            hops.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Subscript):
            # `items[j].v` -- or deeper, `bag.items[j].v`: walk the subscript's value chain
            # to its root, opening any object and list paths along the way, and the member
            # read belongs to the element structure the chain ends on.
            while isinstance(cursor, ast.Subscript):
                cursor = cursor.value
            if not isinstance(cursor, ast.Attribute):
                if not (isinstance(cursor, ast.Name) and cursor.id in tracked):
                    continue
                base_path:str = tracked[cursor.id]
            else:
                base_path:str = self_chain(cursor, tracked, scan, called, node)
                if base_path is None:
                    continue
            # The chain under the subscript names the LIST: either it already ends on one
            # (ag.items[j], where base_path IS the list path), or the last hop names it
            # (items[j] on a top-level list).
            if base_path in scan.lists:
                list_path:str = base_path
            else:
                list_path:str = _resolve_list_path(base_path, node.attr, scan, tracked)
            if list_path is None:
                continue
            element_path:str = _element_path_of(scan, list_path)
            if element_path is None:
                # An indexed read (`items[j].v`) with no `for ... in items:` anywhere: the
                # element path starts here, one per list.
                element_path:str = _member_path(list_path, "element#items")
                scan.add_path(element_path, scan.lists[list_path][0])
                scan.classes[element_path] = scan.classes[list_path]
            # The element's member is the *innermost* attribute: `hops` is collected outside
            # in, so `items[j].center.z` gives `['z', 'center']` and `center` is the member
            # read off the element (`z` belongs to the vector). Noting the outer one asked a
            # Sphere for `z` and failed.
            scan.note(element_path, hops[-1] if hops else node.attr)
            crossed_list = True
            continue
        if crossed_list:
            continue
        if not (isinstance(cursor, ast.Name) and cursor.id in tracked):
            continue
        root_path:str = tracked[cursor.id]
        # Resolve hop by hop: every hop whose value is itself an object becomes its own path,
        # and the members after it belong to that path. A hop whose value is a *list of
        # objects* opens a list path, exactly as a top-level list argument does. Anything
        # else is a plain member read and ends the chain.
        current_path:str = root_path
        for index, attr in enumerate(reversed(hops)):
            member_path:str = _member_path(current_path, attr)
            parent_value:object = scan.values.get(current_path)
            value:object = (getattr(parent_value, attr, None) if parent_value is not None else None)
            if index < len(hops) - 1 and isinstance(value, ctypes.Structure):
                # A vector member read through (`ray.origin.x`): the member *is* the payload,
                # with a device type of its own, so it is noted and the walk stops here.
                # Opening a path inside it would treat the vector as a nested structure,
                # which it is not -- and that refusal is what broke a constructed value whose
                # members are vectors.
                scan.note(current_path, attr)
                break
            if index < len(hops) - 1 and looks_like_object(value):
                # An object member that is read *through*: its path opens and the walk
                # continues inside it.
                if member_path not in scan.members:
                    scan.add_path(member_path, value)
                scan.note(current_path, attr)
                current_path = member_path
                continue
            if index < len(hops) - 1 and isinstance(value, list) and value:
                # A member that is a list of objects, read *through* (`scene.hittables` and
                # then `h.hit(...)` on the elements): it becomes a list path of its own, and
                # the walk continues inside the elements.
                if member_path not in scan.lists:
                    scan.add_list(member_path, value)
                    scan.note(current_path, attr)
                current_path = member_path
                continue
            if index < len(hops) - 1:
                raise SyntaxError(
                    f"'{ast.unparse(node)}' reads through '{attr}', which is a "
                    f"{type(value).__name__} and cannot be turned into a nested structure")
            if attr not in called:
                scan.note(current_path, attr)

    # `p.method()`: the method's own self.xxx reads are members of the same object. The
    # receiver may be a tracked name, or an element of a tracked list (`items[j].doubled()`),
    # in which case the method belongs to the element structure.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        receiver:ast.AST = node.func.value
        list_path:Optional[str] = None
        if isinstance(receiver, ast.Subscript):
            base:ast.AST = receiver.value
            if isinstance(base, ast.Name) and base.id in tracked:
                list_path = tracked[base.id]
            if list_path is None or list_path not in scan.lists:
                continue
            origin:str = _element_path_of(scan, list_path) or \
                _member_path(list_path, "element#items")
            if origin not in scan.members:
                scan.add_path(origin, scan.lists[list_path][0])
                scan.classes[origin] = scan.classes[list_path]
        elif isinstance(receiver, ast.Name) and receiver.id in tracked:
            origin:str = tracked[receiver.id]
        else:
            continue
        owner:Optional[type] = scan.classes.get(origin)
        if owner is None:
            raise SyntaxError(
                f"'{receiver.id}.{node.func.attr}()' is a method call on an object whose class "
                f"is not known, so the method cannot be translated")
        function:object = getattr(owner, node.func.attr, None)
        if not isinstance(function, Callable) or isinstance(function, (type, ctypes.Structure)):
            raise SyntaxError(
                f"{owner.__name__}.{node.func.attr} is not a method that can be translated into "
                f"device code")

        # The method body is scanned with the FULL walker, bound to the same object: its own
        # `for e in self.items:` loops open list paths, its `self.other()` calls are followed
        # recursively, and its member reads land on the same origin -- `ray_color` calling
        # `self.hit` reading `self.hittables` is one chain, and the element reads inside
        # `hit` describe the element structure of `self.hittables`.
        #
        # The arguments are forwarded as well: `scene.ray_color(ray)` hands the caller's own
        # value to `ray_color`'s `ray` parameter, and what that method -- and the ones it
        # calls in turn -- reads through it is what the value has to carry. Without this,
        # `Sphere.hit`'s `ray.origin` had no path to land on.
        bound:Dict[str, str] = {"self": origin}
        try:
            parameters:List[str] = list(
                function.__code__.co_varnames[:function.__code__.co_argcount])
        except AttributeError:
            parameters = []
        # `parameters[0]` is the receiver, which `node.args` does not carry, so the call's
        # first argument lines up with `parameters[1]` -- skipping index 0 here aligned the
        # two wrong ways and `scene.ray_color(ray)` never reached its `ray` parameter.
        call_parameters:List[str] = parameters[1:] if parameters else []
        for index, argument in enumerate(node.args):
            if index >= len(call_parameters):
                continue
            if isinstance(argument, ast.Name) and argument.id in tracked:
                bound[call_parameters[index]] = tracked[argument.id]
        for keyword in node.keywords:
            if keyword.arg and isinstance(keyword.value, ast.Name) \
                    and keyword.value.id in tracked:
                bound[keyword.arg] = tracked[keyword.value.id]
        _scan_function(function, bound, scan)
        scan.note_method(origin, node.func.attr, function)

    # `helper(p)`: the helper's parameter becomes another name for the same object.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        target:object = scan.resolve(node.func.id)
        if not isinstance(target, Callable) or isinstance(target, type):
            continue
        try:
            parameters:List[str] = list(target.__code__.co_varnames[:target.__code__.co_argcount])
        except AttributeError:
            continue
        forwarded:Dict[str, str] = {}
        for index, argument in enumerate(node.args):
            if index >= len(parameters):
                break
            if isinstance(argument, ast.Name) and argument.id in tracked:
                forwarded[parameters[index]] = tracked[argument.id]
        for keyword in node.keywords:
            if (keyword.arg and isinstance(keyword.value, ast.Name)
                    and keyword.value.id in tracked):
                forwarded[keyword.arg] = tracked[keyword.value.id]
        if forwarded:
            _scan_function(target, forwarded, scan)
        # The helper's own body can build a value inline too (`sum_of(Pair(1, 2))`), and
        # that tree is lifted on its own: the walk above only follows names, so the helper
        # is scanned for its own constructions here.
        helper_constructions:Dict[str, object] = constructed_locals(
            target, scan.values, scan.resolve)
        for name, value in helper_constructions.items():
            if name not in scan.members:
                scan.add_path(name, value)
        if helper_constructions:
            _scan_function(target, {**forwarded, **{n: n
                                                    for n in helper_constructions}}, scan)


def _writes_to(tree:ast.Module, tracked:Dict[str, str])->List[str]:
    """Every `p.x = ...` target on a tracked object, as `path.member` strings."""
    written:List[str] = []
    for node in ast.walk(tree):
        targets:List[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id in tracked):
                description:str = f"{tracked[target.value.id]}.{target.attr}"
                if description not in written:
                    written.append(description)
                continue
            # items[j].x = ...: a store into an element of a tracked list.
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Subscript)
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id in tracked):
                description:str = f"{tracked[target.value.value.id]}.{target.attr}"
                if description not in written:
                    written.append(description)
    return written


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
            if owner is None:
                # The receiver is a parameter, so the annotation is the only record of its
                # class: `_argument_holder` answers the class itself, not an instance.
                owner = _argument_holder(func, receiver.id)
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


def _argument_holder(func:Callable[..., object], name:str)->object:
    """A placeholder value for a parameter, when its class can be read from the annotation.

    A method call on an argument (`scene.ray_color(...)`) needs the receiver's class to find
    the method, and the annotation is the only record of it before a launch supplies the real
    object. The annotation is evaluated best-effort: an unresolvable name answers None.
    """
    try:
        import inspect as _inspect  # noqa: PLC0415

        parameters:object = _inspect.signature(func).parameters
        annotation:object = parameters[name].annotation if name in parameters else None
        if isinstance(annotation, type):
            return annotation
        if isinstance(annotation, str):
            resolved:object = func.__globals__.get(annotation)
            if isinstance(resolved, type):
                return resolved
    except (TypeError, ValueError, AttributeError):
        return None
    return None


# The prefix an inline construction is lifted to. It never reaches the generated source --
# only the scan's copy of the tree -- so it only has to be a name the body cannot spell.
INLINE_PREFIX:str = "_inline_"


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


def _argument_kind(argument:ast.expr, objects:Dict[str, object],
                   resolve:Callable[[str], object],
                   tree:Optional[ast.Module] = None,
                   type_map:Optional[Dict[str, str]]=None) -> Optional[object]:
    """The `ctypes` type an argument expression carries, when that can be told.

    A kernel parameter's type is the value the caller passed; a member of one is that
    member's type; a literal is a float. Anything else is unknown, and the construction is
    left alone rather than guessed at.
    """
    if isinstance(argument, ast.Name):
        value:object = objects.get(argument.id)
        if value is not None:
            return member_type(value, argument.id, type_map)
        # A local the body assigned carries whatever its right-hand side did.
        return _assigned_kind(argument.id, objects, resolve, tree)
    if isinstance(argument, ast.Attribute) and isinstance(argument.value, ast.Name):
        holder:object = objects.get(argument.value.id)
        if holder is not None and hasattr(holder, argument.attr):
            return member_type(getattr(holder, argument.attr), argument.attr, type_map)
        # A module attribute (math.inf): the value lives in the globals, and its type is
        # known even though no object argument carries it.
        module:object = resolve(argument.value.id)
        module_value:object = getattr(module, argument.attr, None)
        if isinstance(module_value, (bool, int, float)):
            return member_type(module_value, argument.attr, type_map)
        return None
    if isinstance(argument, ast.Constant) and isinstance(argument.value, (int, float)):
        # An int literal is an int member: the width has to follow the value, or
        # `Interval(0, 3)` gives both members the float width and the device type no
        # longer matches what `__init__` assigned. `bool` has its own width too.
        if isinstance(argument.value, bool):
            return ctypes.c_bool
        return _numeric_member_type(argument.value, type_map) or scalar_member_type(type_map, float)
    # Anything else -- an inline construction, an arithmetic expression over known operands --
    # is answered by the same walk `_expression_kind` does for an assigned local.
    return _expression_kind(argument, objects, resolve, tree)


# The ctypes class each device type name stands for. A synthesized structure member is
# declared with one of these, so a mapping's device type name has to come back as the class
# that has that width -- `long` is `c_int64`, not `c_int32`.
_DEVICE_SCALAR_CLASSES:Dict[str, type] = {
    "bool": ctypes.c_bool,
    "char": ctypes.c_int8, "short": ctypes.c_int16,
    "int": ctypes.c_int32, "long": ctypes.c_int64,
    "uchar": ctypes.c_uint8, "ushort": ctypes.c_uint16,
    "uint": ctypes.c_uint32, "ulong": ctypes.c_uint64,
    "float": ctypes.c_float, "double": ctypes.c_double,
}

# The member a python `int`/`float` becomes when the kernel declares no mapping. These are
# the widths every existing kernel already has, so the default is unchanged.
_DEFAULT_INT_MEMBER:type = ctypes.c_int32
_DEFAULT_FLOAT_MEMBER:type = ctypes.c_float


def scalar_member_type(type_map:Optional[Dict[str, str]], python_type:type)->type:
    """The ctypes class a python `int`/`float` becomes on a synthesized structure member.

    A synthesized structure is the type of an argument, so the width its members carry has to
    follow the same `int=`/`float=` mapping the kernel's own scalar parameters follow. It used
    to be fixed at `c_int32`/`c_float`, which made the mapping inert for exactly the values
    that most often overflow: `@kernel(int=np.int64)` with `obj.n = 2 ** 31` still reached the
    device as a negative 32-bit number, silently.

    Only the widths the mapping may name are reachable, and `validate_type_map` already
    restricted those, so an unknown name falls back to the built-in width rather than raising
    from a translation-time scan.
    """
    mapping:Dict[str, str] = type_map or {}
    if python_type is float:
        name:str = mapping.get("float", "")
        return _DEVICE_SCALAR_CLASSES.get(name, _DEFAULT_FLOAT_MEMBER)
    if python_type is int:
        name = mapping.get("int", "")
        return _DEVICE_SCALAR_CLASSES.get(name, _DEFAULT_INT_MEMBER)
    if python_type is bool:
        return ctypes.c_bool
    return ctypes.c_float


def _ctypes_name_of(cls:type)->str:
    """The device type name a ctypes scalar class stands for, for a range check."""
    for name, known in _DEVICE_SCALAR_CLASSES.items():
        if known is cls:
            return name
    return "int"

def _numeric_member_type(value:object, type_map:Optional[Dict[str, str]],
                         attr:str="") -> Optional[type]:
    """The ctypes class a plain python number becomes, under the mapping.

    An integer that the chosen width cannot hold is refused here rather than wrapped: a
    structure member travels in the dtype's own bytes, so `obj.n = 2 ** 31` under the default
    32-bit width used to reach the device as a negative number with nothing reported. The
    check is the same one `infer_ctype` applies to a scalar argument, so a value and the same
    value held in an object's member behave identically.
    """
    if isinstance(value, (bool, np.bool_)):
        return ctypes.c_bool
    if isinstance(value, (int, np.integer)):
        chosen:type = scalar_member_type(type_map, int)
        from .translator import check_integer_range  # noqa: PLC0415 (import cycle)

        check_integer_range(int(value), _ctypes_name_of(chosen), attr)
        return chosen
    if isinstance(value, (float, np.floating)):
        return scalar_member_type(type_map, float)
    return None


def _assigned_kind(name:str, objects:Dict[str, object],
                   resolve:Callable[[str], object],
                   tree:Optional[ast.Module],
                   type_map:Optional[Dict[str, str]]=None)->Optional[object]:
    """The `ctypes` type a local name carries, from what the body assigned to it.

    Only the shapes a construction actually uses are followed: the result of a binary
    operation over known types (which is what `ray_dir = a - b` is), a member of the
    receiver, or another known name. The first assignment to the name in the body is the one
    that counts, and a type that cannot be pinned leaves the construction alone -- guessing
    would put a member of the wrong width on the device.
    """
    if tree is None:
        return None
    for statement in ast.walk(tree):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target:ast.expr = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        return _expression_kind(statement.value, objects, resolve, tree, type_map)
    return None


def _expression_kind(node:ast.expr, objects:Dict[str, object],
                     resolve:Callable[[str], object],
                     tree:Optional[ast.Module],
                     type_map:Optional[Dict[str, str]]=None)->Optional[object]:
    """The `ctypes` type an expression yields, for the shapes a construction uses."""
    if isinstance(node, ast.Name):
        value:object = objects.get(node.id)
        if value is not None:
            numeric:Optional[type] = _numeric_member_type(value, type_map)
            if numeric is not None:
                return numeric
            return type(value)
        return _assigned_kind(node.id, objects, resolve, tree, type_map)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        holder:object = objects.get(node.value.id)
        if holder is not None:
            member:object = getattr(holder, node.attr, None)
            if member is not None:
                numeric = _numeric_member_type(member, type_map)
                if numeric is not None:
                    return numeric
                return type(member)
        return None
    if isinstance(node, ast.BinOp):
        # An operation over known operands keeps the wider one: a float3 minus a point is a
        # vector of the same width, and that is all the member's layout needs.
        left:Optional[object] = _expression_kind(node.left, objects, resolve, tree, type_map)
        right:Optional[object] = _expression_kind(node.right, objects, resolve, tree, type_map)
        for candidate in (left, right):
            if isinstance(candidate, type) and issubclass(candidate, ctypes.Structure):
                return candidate
        if left is not None or right is not None:
            # An arithmetic result keeps the wider of its operands when both are scalars:
            # `a - b` over two `long` members is a `long`, not a `float`.
            for candidate in (left, right):
                if isinstance(candidate, type) and candidate in _DEVICE_SCALAR_CLASSES.values():
                    if candidate in (ctypes.c_float, ctypes.c_double):
                        return candidate
            return scalar_member_type(type_map, float)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _numeric_member_type(node.value, type_map) or scalar_member_type(type_map, float)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        # A construction written inline (`Ray(p, point3f(0, 0, -1))`): the callee *is* the
        # type, when it is a struct or vector the device can carry.
        cls:object = resolve(node.func.id)
        if isinstance(cls, type) and issubclass(cls, ctypes.Structure):
            return cls
        return None
    return None


def _template_from(kinds:Sequence[object], cls:type)->type:
    """An instance of `cls` built from placeholders of the given types.

    When the call's arguments cannot all be typed -- a local the body assigned, whose type
    the scan cannot pin -- a default-constructed instance is the template instead: its
    members are whatever `__init__` assigns when called with nothing, which is the type each
    member carries in the body too. The class is still *used* by the body the same way, so
    the member set comes from the reads either way; only the types fall back to the
    constructor's own defaults.
    """
    if kinds is not None:
        values:List[object] = []
        for kind in kinds:
            try:
                values.append(kind())
            except Exception:  # noqa: BLE001 -- an unusable placeholder means "no template"
                return None
        try:
            return cls(*values)
        except Exception:  # noqa: BLE001
            return None
    try:
        return cls()
    except Exception:  # noqa: BLE001
        return None


# The structures built for values a body constructs, keyed by the class: the translator asks
# this when it reaches the constructor call, and the scan above is what filled it in.
_CONSTRUCTED:Dict[type, type] = {}


_CONSTRUCTED_PARAMETERS:Dict[type, List[str]] = {}


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


def scan_object(func:object, objects:Dict[str, object],
                resolve:Callable[[str], object],
                type_map:Optional[Dict[str, str]]=None)->Scan:
    """Scan `func` for the members and methods each object parameter needs."""
    scan:Scan = Scan(resolve, type_map)
    transitive:Dict[str, str] = {}
    # A value the body builds itself joins the same scan, so a member read one method further
    # down -- `scene.ray_color(ray)` handing `ray` to `Sphere.hit` -- lands on the value\'s
    # own path instead of vanishing.
    constructed:Dict[str, object] = constructed_locals(func, objects, resolve, type_map)
    scan.constructed = dict(constructed)
    # The registry describes this scan only: a stale entry would make a later kernel see a
    # structure from an earlier one.
    _CONSTRUCTED.clear()
    for name, value in {**constructed, **objects}.items():
        if isinstance(value, list):
            scan.add_list(name, value)
        else:
            scan.add_path(name, value)
        transitive[name] = name
    _scan_function(func, transitive, scan)

    tree:Optional[ast.Module] = _parse(func)
    if tree is not None:
        for description in _writes_to(tree, transitive):
            path, member = description.rsplit(".", 1)
            if path in scan.lists:
                raise SyntaxError(
                    f"the kernel body assigns to '{description}', and a write to an element "
                    f"of a list of objects cannot be copied back to the caller's list yet. "
                    f"Write into an array argument instead")
            source:object = scan.values.get(path)
            if source is not None:
                cls:type = type(source)
                try:
                    attribute:object = getattr(cls, member)
                    if isinstance(attribute, property):
                        raise SyntaxError(
                            f"the kernel body assigns to '{description}', which is a "
                            f"property of {cls.__name__}: a computed attribute has no "
                            f"storage to write back to")
                except AttributeError:
                    pass
                if hasattr(cls, "__slots__") and member not in (
                        getattr(cls, "__slots__") or ()):
                    raise SyntaxError(
                        f"the kernel body assigns to '{description}', but {cls.__name__} "
                        f"declares __slots__ without '{member}', so the write has nowhere "
                        f"to land")
    scan.written = _writes_to(_parse(func) or ast.Module(body=[], type_ignores=[]), transitive) \
        if tree is not None else []
    return scan


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


# `int` widens to `float`; nothing else mixes.
_PROMOTIONS:Dict[Tuple[type, type], type] = {
    (ctypes.c_int32, ctypes.c_float): ctypes.c_float,
    (ctypes.c_int32, ctypes.c_double): ctypes.c_double,
    (ctypes.c_float, ctypes.c_double): ctypes.c_double,
}


def promote(left:type, right:type)->Optional[type]:
    """The type both can widen to, or None when they cannot be mixed."""
    if left is right:
        return left
    return _PROMOTIONS.get((left, right)) or _PROMOTIONS.get((right, left))


def _is_number(value:object)->bool:
    """True for a value that becomes a scalar structure member."""
    return isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating))


def numeric_list_as_array(value:object)->object:
    """A python list/tuple of numbers as the ndarray a kernel argument can carry.

    Anything else is returned unchanged, so this is safe to run over every argument. A nested
    list of numbers also converts (`[[1, 2], [3, 4]]` is a 2-D array), which is the shape a
    2-D python list should have. A list that holds objects is left alone on purpose: it is an
    object collection, and `objects.py` gives it a structure element type instead.
    """
    if not isinstance(value, (list, tuple)) or not value:
        return value
    try:
        array:np.ndarray = np.asarray(value)
    except ValueError:
        return value
    if array.dtype.kind not in "biufc":
        return value                 # object/str/void: not a numeric buffer
    return array

def numeric_list_type(value:Sequence[object], attr:str,
                      type_map:Optional[Dict[str, str]]=None)->Optional[type]:
    """The fixed-size array type a list of numbers becomes, or None if it is not one.

    A list member used to be refused outright ("cannot become a structure member yet"), which
    left `bag.values[0]` -- a list of numbers inside an object -- with no spelling at all. It
    is a **fixed-size array** member (`T * N`), and numpy already records that as a sub-array
    field (`dtype.subdtype`), so the C side needs no new machinery: `structs` puts the extent
    on the member name (`float items[4]`).

    Two limits are load-bearing and both are the caller's own values, not a policy:

    * **the length is fixed by the value the caller passed**, because a structure's C layout is
      decided once, per dtype -- a length read at launch time cannot declare a field;
    * **every element must become the same scalar type**, or the field has no one element
      type. An int and a float mix promotes (that is what `Interval(0, 3)` does for two
      members), but a list that mixes them is promoted the same way rather than refused.

    An empty list is refused: it describes no element type, and a zero-length array member is
    not a shape C can declare.
    """
    if not isinstance(value, (list, tuple)):
        return None
    items:List[object] = list(value)
    if not items:
        raise SyntaxError(
            f"'{attr}' is an empty list, so the structure member it becomes has no element "
            f"type. Give it at least one number, or leave the member out") from None
    element:Optional[type] = None
    for item in items:
        if isinstance(item, (list, tuple)):
            raise SyntaxError(
                f"'{attr}' is a list of lists, and only a list of numbers can become an "
                f"array member (a nested list would need a nested array field)") from None
        if not _is_number(item):
            raise SyntaxError(
                f"'{attr}' is a list of {type(item).__name__} values, and only a list of "
                f"numbers can become an array member") from None
        item_type:type = member_type(item, attr, type_map)
        element = item_type if element is None else promote(element, item_type)
        if element is None:
            raise SyntaxError(
                f"'{attr}' mixes member types that have no common width "
                f"({item_type.__name__} among them)") from None
    return element * len(items)


def member_type(value:object, attr:str,
                type_map:Optional[Dict[str, str]]=None)->type:
    """The ctypes type for one member, decided from the value the caller passed.

    A plain python object member is allowed only through the nested path in
    ``convert_object_arguments``, which replaces its type with the synthesized structure; this
    function is where that substitution is applied.

    A python `int`/`float` member follows the kernel's `int=`/`float=` mapping, because the
    structure *is* the argument's device type: fixing the width at 32 bits here made the
    mapping silently inert for object members (`obj.n = 2 ** 31` still wrapped). A declared
    `ctypes` member keeps its own class -- the caller wrote that width down.
    """
    numeric:Optional[type] = _numeric_member_type(value, type_map, attr)
    if numeric is not None:
        return numeric
    if isinstance(value, (list, tuple)):
        # A list of numbers is a fixed-size array member (`T * N`); numpy records it as a
        # sub-array field, so the C declaration carries the extent on the member name.
        numeric = numeric_list_type(value, attr, type_map)
        if numeric is not None:
            return numeric
    if isinstance(value, np.ndarray):
        raise SyntaxError(
            f"'{attr}' is an array, and an array member of a structure is not supported yet. "
            f"Pass the array as its own kernel argument, or pass a list of objects if the "
            f"elements are instances of one class")
    if isinstance(value, ctypes.Structure):
        # A declared structure or a vector type already has a device type: use it as it is.
        return type(value)
    # The scan's template is an instance built from placeholder types (`c_int32()`), so a
    # member can arrive as a ctypes scalar rather than a python number. Its own class is
    # the width the constructor stored, which is exactly what the member has to be.
    if isinstance(value, ctypes._SimpleCData):  # noqa: SLF001 (the documented base)
        return type(value)
    if isinstance(value, (type, Callable)):
        raise SyntaxError(
            f"'{attr}' is a function or a class, which has no device representation")
    raise SyntaxError(
        f"'{attr}' is a {type(value).__name__}, which cannot become a structure member yet "
        f"(only numbers, booleans and vector values can)")


# ---------------------------------------------------------------------------
# Building the structure
# ---------------------------------------------------------------------------


_STRUCT_CACHE:Dict[Tuple[object, ...], type] = {}


def _prefer_class_name(structure:type, cls:Optional[type]) -> None:
    """Let the C structure this dtype describes be named after the python class.

    The device type is generated from the dtype, and its C name comes from
    `structs.struct_name`; a hash says nothing a reader recognizes, while the class the
    value came from does. The hint is advisory and first-seen wins, because two classes
    with one layout share a dtype.
    """
    if cls is None or getattr(cls, "__module__", "") == "builtins":
        return
    from .structs import register_name_hint  # noqa: PLC0415 (import cycle)

    register_name_hint(np.dtype(structure), cls)


def _structure_for(fields:Sequence[Tuple[str, type]],
                   methods:Sequence[Tuple[str, Any]])->type:
    # Keyed by the member *set* alone -- not their order, and not the method list. The walk
    # collects members through `ast.walk` (order not source order), and one path may see a
    # method where another does not (`hitable.hit` is called in one place, only read in
    # another). Two classes with one layout are still two ctypes types, so the class has to
    # be one: a later call adds any method the first one did not know about.
    key:Tuple[object, ...] = frozenset((member, kind) for member, kind in fields)
    cached:Optional[type] = _STRUCT_CACHE.get(key)
    if cached is not None:
        for method_name, function in methods:
            if method_name not in cached.__dict__:
                setattr(cached, method_name, function)
        return cached
    # The field order is fixed by name, so the same member set always describes the same
    # layout -- the device type is generated from this dtype, and a differing order would
    # make one value's structure incompatible with another's.
    ordered:List[Tuple[str, type]] = sorted(fields, key=lambda f: f[0])
    name:str = "Object" + "".join(member.capitalize() for member, _ in ordered)
    namespace:Dict[str, object] = {"_fields_": [(member, kind) for member, kind in ordered]}
    structure:type = type(name, (ctypes.Structure,), namespace)
    # A method the body calls has to be reachable from the synthesized class: the translator
    # resolves `p.method` from the class its dtype came from.
    for method_name, function in methods:
        if method_name not in structure.__dict__:
            setattr(structure, method_name, function)
    _STRUCT_CACHE[key] = structure
    return structure


def _fields_for(obj:object, names:Sequence[str],
                type_map:Optional[Dict[str, str]]=None)->List[Tuple[str, type]]:
    fields:List[Tuple[str, type]] = []
    for name in names:
        if not hasattr(obj, name):
            raise SyntaxError(
                f"object of type {type(obj).__name__} has no attribute '{name}', which the "
                f"kernel body reads")
        value:object = getattr(obj, name)
        if isinstance(value, (type, Callable)) and not isinstance(value, ctypes.Structure):
            raise SyntaxError(
                f"{type(obj).__name__}.{name} is a function or a class, so it cannot be a "
                f"structure member. Call it as a method instead of reading it")
        fields.append((name, member_type(value, name, type_map)))
    return fields


def merge_fields(group:Sequence[List[Tuple[str, type]]]
                 )->Optional[List[Tuple[str, type]]]:
    """One field list for several instances, or None when merging would be dishonest.

    The union of the members is used, each widened across the instances that carry it. If an
    instance lacks a member another one needs, or one member's types do not promote, the
    instances keep separate structures -- the device type is per argument, so nothing is lost
    but a shared name.
    """
    names:List[str] = []
    for fields in group:
        for name, _ in fields:
            if name not in names:
                names.append(name)
    merged:List[Tuple[str, type]] = []
    for name in names:
        kind:Optional[type] = None
        for fields in group:
            found:Optional[type] = next((k for n, k in fields if n == name), None)
            if found is None:
                return None
            kind = found if kind is None else promote(kind, found)
            if kind is None:
                return None
        if kind is None:
            return None
        merged.append((name, kind))
    return merged


class ObjectBinding:
    """The pairing a write-back needs: the caller's object and the instance that replaced it.

    written_arg: the path.member descriptions the body assigns to; values[path] is
    the caller object the writes land back on, and instances[path] the structure that
    travelled to the device.
    """

    def __init__(self)->None:
        self.written:List[str] = []
        self.values:Dict[str, object] = {}
        self.instances:Dict[str, object] = {}

    def write_back(self)->None:
        """Copy the final member values home, for every object the body wrote to."""
        for description in self.written:
            path, member = description.rsplit('.', 1)
            instance:object = self.instances.get(path)
            source:object = self.values.get(path)
            if instance is None or source is None:
                continue
            if isinstance(instance, np.ndarray):
                # The object travelled as a one-element record array; the device wrote its
                # member stores there, and the host read the array back after the launch.
                try:
                    value:object = instance[0][member]
                except (IndexError, ValueError):
                    continue
                setattr(source, member, value.item() if hasattr(value, "item") else value)
                continue
            try:
                value:object = getattr(instance, member)
            except AttributeError:
                continue
            setattr(source, member, value)


def convert_object_arguments(func:object, args:Sequence[object],
                             resolve:Callable[[str], object],
                             self_param:str = '',
                             binding:object = None,
                             self_value:object = None,
                             type_map:Optional[Dict[str, str]]=None)->Tuple[object, ...]:
    """Replace every object argument with an instance of its synthesized structure.

    Called before the argument types are inferred, because the structure *is* the type: the
    translator then sees a `ctypes.Structure`, so the device typedef, the member access, the
    layout and the packing all work unchanged.

    ``self_param`` is the kernel's receiver name, when the kernel is a method: the receiver
    travels outside ``args``, so it is skipped when the parameters are lined up with the
    arguments, or `scene` in `render(self, scene)` would line up with the name `self`.

    ``self_value`` is that receiver, when the kernel has one. It is not a structure argument,
    but a body that builds its own value types the construction's arguments from what the
    receiver's members are -- `Ray(self.position, ray_dir)` needs to know what `self.position`
    *is* -- so the receiver joins the scan as a source of types, never as an argument.
    """
    hits:List[int] = [index for index, value in enumerate(args) if looks_like_object(value)]
    # A python list of numbers is a kernel ARRAY argument, not a collection of objects: the
    # device has one element type per buffer, and `for x in items: total += x` over a numeric
    # list is a serial loop over that buffer. Converting it here -- before anything looks for
    # object members -- is what makes a bare `list[int]`/`list[float]` argument reach the
    # translator as the ndarray it is. `np.asarray` already picks numpy's own dtype
    # (`int64`, `float64`, `complex128`), which is the same width rule every other array
    # argument follows.
    converted_args:List[object] = [numeric_list_as_array(value) for value in args]
    args = tuple(converted_args)
    hits = [index for index, value in enumerate(args) if looks_like_object(value)]
    # A body that builds its own value still needs the scan, even with no object argument:
    # what that value carries is what the scan discovers.
    if not hits and not _constructs_a_class(func, resolve):
        return tuple(args)

    parameters:List[str] = list(func.__code__.co_varnames[:func.__code__.co_argcount])
    if self_param and parameters and parameters[0] == self_param:
        # The receiver travels outside args, so it is dropped from the names that index
        # into it -- and all_parameters is the same list, used only for the positions.
        parameters = parameters[1:]
    all_parameters:List[str] = list(parameters)
    named:Dict[str, object] = {}
    for index in hits:
        if index >= len(parameters):
            raise SyntaxError(
                f"an object passed in position {index} does not line up with a named "
                f"parameter, so its members cannot be discovered")
        named[parameters[index]] = args[index]

    # The constructor's arguments are typed from the caller's values, so the scan needs those
    # parameters too -- but only the ones that could be a structure member. An array is not
    # one, and registering it would make the scan ask for members of it. The receiver joins
    # under its own name: reading `self.position` off it answers the construction's argument
    # types, and the scan's own path machinery decides what else the body reads through it.
    scan_inputs:Dict[str, object] = {}
    for index, name in enumerate(parameters):
        if index >= len(args) or name in named:
            continue
        value:object = args[index]
        if isinstance(value, (int, float, bool, ctypes.Structure)):
            scan_inputs[name] = value
    if self_value is not None and self_param:
        scan_inputs.setdefault(self_param, self_value)
    scan_inputs.update(named)

    scan:Scan = scan_object(func, scan_inputs, resolve, type_map)
    out:List[object] = list(args)

    # The receiver joined the scan as a source of types, but the structure it *is* here is
    # still the declared class the body method mechanism already handles. A member the body
    # reads off it that is an array (`self.image`) travels the self-attribute way, not as a
    # structure field, so it leaves the synthesized member list -- `member_type` would refuse
    # it, and the refusal is right for a real object argument but not for the receiver.
    if self_value is not None and self_param:
        receiver_members:List[str] = scan.members.get(self_param, [])
        instance_values:object = self_value
        for member in list(receiver_members):
            attribute:object = getattr(instance_values, member, None)
            if isinstance(attribute, np.ndarray):
                receiver_members.remove(member)

    # An object nothing reads has no structure to build: an empty one is not a portable
    # device type (CUDA accepts it, OpenCL rejects the launch), and there would be nothing to
    # pass. Say so instead of shipping an empty aggregate. A nested path is exempt, because a
    # parent that is read only for its nested object still has a structure to carry, and so is
    # a list path, whose element reads are what carry it.
    for name in named:
        if name in scan.lists:
            continue
        if not scan.members[name]:
            raise SyntaxError(
                f"'{name}' is a {type(named[name]).__name__}, but the kernel body reads none of "
                f"its attributes and calls none of its methods, so there is nothing to put in a "
                f"structure. Pass the values it needs as scalar or vector arguments")

    # Innermost first: a nested member's structure has to exist before the parent's field can
    # name it. Paths sort naturally, so `item` precedes `item.inner` -- reverse the order.
    # A list path (`scene.hittables#element...`) builds its ELEMENT structure, which the
    # list argument then becomes an array of.
    structure_of:Dict[str, type] = {}
    nested_of:Dict[str, List[Tuple[str, type]]] = {}
    list_element_paths:Dict[str, str] = {}
    # A member that is a list of objects becomes a fixed-length array member of the parent's
    # structure: element structure * element count, both known from the caller's list.
    list_array_types:Dict[str, object] = {}
    for list_path in scan.lists:
        element_paths:List[str] = [path for path in scan.members
                                   if _list_root(path) == list_path and path != list_path]
        if not element_paths:
            raise SyntaxError(
                f"'{list_path}' is a list of objects, but the body never reads a member of "
                f"or calls a method on its elements")
        element_path:str = element_paths[0]
        list_element_paths[list_path] = element_path
        for path in element_paths[1:]:
            # Ordered, not set-unioned: the member order decides the structure's field order,
            # and two paths must merge to the *same* class -- an unordered merge built
            # `ObjectRadiusCenter` in one place and `ObjectCenterRadius` in another, so a
            # method looked absent on the copy the translator saw.
            for member in scan.members[path]:
                if member not in scan.members[element_path]:
                    scan.members[element_path].append(member)
        # The element structure must exist before any parent that carries the list as a
        # fixed-length array of it.
        fields:List[Tuple[str, type]] = []
        called_methods:set = {name for name, _ in scan.methods.get(element_path, [])}
        for member in scan.members[element_path]:
            if member in called_methods:
                continue
            value:object = getattr(scan.values[element_path], member)
            member_path:str = _member_path(element_path, member)
            if member_path in structure_of:
                fields.append((member, structure_of[member_path]))
            else:
                fields.append((member, member_type(value, member, scan.type_map)))
        structure_of[element_path] = _structure_for(fields, scan.methods[element_path])
        _prefer_class_name(structure_of[element_path], scan.classes.get(element_path))

    # Second pass for the list members: their element structures exist now, so the parent's
    # field can be a fixed-length array of them.
    for list_path, element_path in list_element_paths.items():
        if '#' in list_path:
            continue                     # a list inside a list is not supported
        element_structure:type = structure_of[element_path]
        count:int = len(scan.lists[list_path])
        list_array_types[list_path] = element_structure * count

    # Nested objects and member chains: built innermost first, so a parent's field can name
    # the structure of the member it carries.
    nested_of:Dict[str, List[Tuple[str, type]]] = {}
    for path in sorted(scan.members, key=len, reverse=True):
        if (path in named or path in scan.constructed) and "." not in path \
                and path not in scan.lists:
            continue                     # top-level paths are built below, with merging
        if _list_root(path) in scan.lists and path != _list_root(path):
            pass                         # an element path; built with its list above
        elif path in scan.lists:
            continue                     # the list itself becomes an array, not a structure
        if path in structure_of:
            continue                     # an element path, already built
        fields:List[Tuple[str, type]] = []
        called_methods:set = {name for name, _ in scan.methods.get(path, [])}
        for member in scan.members[path]:
            if member in called_methods:
                continue                 # a translated method, copied onto the class
            value:object = getattr(scan.values[path], member)
            member_path:str = _member_path(path, member)
            if member_path in structure_of:
                fields.append((member, structure_of[member_path]))
            elif member_path in list_array_types:
                fields.append((member, list_array_types[member_path]))
            else:
                fields.append((member, member_type(value, member, scan.type_map)))
        structure_of[path] = _structure_for(fields, scan.methods[path])
        _prefer_class_name(structure_of[path], scan.classes.get(path))
        if "." in path:
            parent, member = path.rsplit(".", 1)
            parent = _list_root(parent)
            if parent not in scan.lists:
                nested_of.setdefault(parent, []).append((member, structure_of[path]))

    # Both a caller\'s object and a value the body built describe a structure the same way:
    # the members the scan found, typed from the template the value carries. They differ only
    # in what happens next -- an argument is replaced in place, a constructed value is looked
    # up by the translator at its call site -- so the structure itself is built together.
    holders:Dict[str, object] = dict(scan.constructed)
    holders.update(named)
    by_class:Dict[type, List[str]] = {}
    for name in holders:
        by_class.setdefault(type(holders[name]), []).append(name)
    for _cls, names in by_class.items():
        specs:List[List[Tuple[str, type]]] = []
        for name in names:
            fields:List[Tuple[str, type]] = []
            for member in scan.members[name]:
                member_path:str = _member_path(name, member)
                if member_path in structure_of:
                    fields.append((member, structure_of[member_path]))
                elif member_path in list_array_types:
                    fields.append((member, list_array_types[member_path]))
                else:
                    fields.append((member, member_type(getattr(holders[name], member), member,
                                     type_map)))
            fields.extend((m, k) for m, k in nested_of.get(name, [])
                          if m not in {n for n, _ in fields})
            specs.append(fields)
        merged:Optional[List[Tuple[str, type]]] = (merge_fields(specs)
                                                   if len(specs) > 1 else specs[0])
        for name, fields in zip(names, specs):
            chosen:List[Tuple[str, type]] = merged if merged is not None else fields
            structure:type = _structure_for(chosen, scan.methods[name])
            _prefer_class_name(structure, type(holders[name]))
            if name in scan.constructed:
                # A value the body built has no argument to replace: the structure is all it
                # needs, and the translator looks it up at the constructor call. The class is
                # recorded as the dtype\'s declaring class, so a method the body calls on the
                # value -- `ray.at(t)` -- resolves to the class it is defined on.
                cls:type = type(scan.constructed[name])
                dtype:np.dtype = np.dtype(structure)
                from .structs import register_declared_class  # noqa: PLC0415

                register_declared_class(dtype, cls)
                _CONSTRUCTED[cls] = dtype
                # The order the constructor's arguments arrive in, which is the `__init__`
                # parameter order -- *not* the member order. The synthesized structure sorts
                # its fields by name, so a positional aggregate literal filled in field
                # order would put `direction` where `origin` belongs.
                _CONSTRUCTED_PARAMETERS[cls] = _init_parameters(cls)
                continue

            def build(source:object)->object:
                """An instance of ``structure`` carrying ``source``'s values, recursively."""
                values:Dict[str, object] = {}
                for member, kind in chosen:
                    value:object = getattr(source, member)
                    if (isinstance(kind, type) and issubclass(kind, ctypes.Structure)
                            and not isinstance(value, kind)):
                        # A nested member is rebuilt in the synthesized type from the values
                        # the caller's object carries.
                        value = kind(**{n: getattr(value, n) for n, _ in kind._fields_})
                    elif isinstance(value, list) and isinstance(kind, type) \
                            and issubclass(kind, ctypes.Array):
                        # A list member: the field is a fixed-length array of the element
                        # structure, filled from the caller's list in order.
                        filled:object = kind()
                        element_kind:object = kind._type_
                        for position, item in enumerate(value):
                            if isinstance(element_kind, type) \
                                    and issubclass(element_kind, ctypes.Structure) \
                                    and not isinstance(item, element_kind):
                                item = element_kind(**{
                                    n: getattr(item, n) for n, _ in element_kind._fields_})
                            filled[position] = item
                        value = filled
                    values[member] = value
                return structure(**values)

            instance:object = build(named[name])
            written_paths:set = {description.rsplit(".", 1)[0] for description in scan.written}
            if name in written_paths:
                # The body writes this object's members: by value the device would write to
                # its own copy and the stores would be lost. A one-element record array
                # travels as a pointer, so the stores land in device memory that the host
                # reads back, and write_back then copies them onto the caller's object.
                records:np.ndarray = np.zeros(1, dtype=np.dtype(structure))
                for member, _ in structure._fields_:
                    value:object = getattr(named[name], member)
                    if isinstance(value, ctypes.Structure):
                        value = np.array(value, dtype=np.dtype(type(value)))
                    records[0][member] = value
                out[all_parameters.index(name)] = records
                if binding is not None:
                    binding.instances[name] = records
                    binding.values[name] = named[name]
            else:
                out[all_parameters.index(name)] = instance
                if binding is not None:
                    binding.instances[name] = instance
                    binding.values[name] = named[name]

    # A top-level list-of-objects argument becomes a numpy record array of the element
    # structure; a member list was already embedded in its parent's structure above. The
    # element class is registered first, so a method the body calls on an element resolves
    # from the class its dtype came from.
    from .structs import reregister_class  # noqa: PLC0415

    for list_path, element_path in list_element_paths.items():
        element_structure:type = structure_of[element_path]
        reregister_class(element_structure)
        root:str = _list_root(list_path)
        if "." in list_path or "#" in list_path:
            # A member list (ag.items): the parent structure already carries it as a
            # fixed-length array, so there is no argument to replace. (An element path
            # never reaches this loop; # marks one.)
            continue
        index:int = all_parameters.index(root)
        records:np.ndarray = np.zeros(len(scan.lists[list_path]), dtype=np.dtype(element_structure))
        for position, item in enumerate(scan.lists[list_path]):
            for member, _ in element_structure._fields_:
                value:object = getattr(item, member)
                if isinstance(value, ctypes.Structure):
                    value = np.array(value, dtype=np.dtype(type(value)))
                records[position][member] = value
        out[index] = records
    if binding is not None:
        binding.written = list(scan.written)

    # Record what each constructed value became, so the translator can answer the constructor
    # call it meets: the class was scanned here, and its structure is `structure_of` under the
    # body's own name for the value.
    for name, value in scan.constructed.items():
        cls:type = type(value)
        path:str = name
        if path in structure_of:
            _CONSTRUCTED[cls] = np.dtype(structure_of[path])
    # A construction lifted inside a *helper* was scanned by the helper's own walk, so its
    # name never reached this scan's `constructed` map -- and the translator still meets the
    # constructor call when it translates that helper. The lift recorded the class with the
    # tree, so the structure it needs can be built here from the same members.
    # A construction lifted inside a *helper* was scanned by the helper's own walk, so its
    # name never reached this scan's `constructed` map -- and the translator still meets the
    # constructor call when it translates that helper. The lift recorded the class with the
    # tree, and the members it may carry are the ones the body fills: only an argument the
    # construction actually supplied is a candidate, because `__init__`'s own defaults can
    # be types a device has no room for (a string that nothing ever reads).
    for source_function, lifted in _LIFTED_VALUES.items():
        for name, value in lifted.items():
            cls = type(value)
            if cls in _CONSTRUCTED:
                continue
            fields = list(_LIFTED_MEMBERS.get(cls) or ())
            if not fields:
                continue
            structure = _structure_for(fields, [])
            _prefer_class_name(structure, cls)
            from .structs import register_declared_class  # noqa: PLC0415

            dtype = np.dtype(structure)
            register_declared_class(dtype, cls)
            _CONSTRUCTED[cls] = dtype
            # The constructor's argument order, which the aggregate literal follows: the
            # synthesized structure sorts its fields by name, so filling it in field order
            # would put the second argument in the first member.
            _CONSTRUCTED_PARAMETERS[cls] = _init_parameters(cls)
    return tuple(out)


# ---------------------------------------------------------------------------
# A class the body builds itself
# ---------------------------------------------------------------------------
