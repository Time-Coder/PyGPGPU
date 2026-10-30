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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "looks_like_object",
    "convert_object_arguments",
]


# ---------------------------------------------------------------------------
# What counts as an object argument
# ---------------------------------------------------------------------------


def looks_like_object(value:Any)->bool:
    """Whether an argument is an arbitrary python object rather than a value we already know.

    Everything the translator already handles is excluded: arrays, numpy scalars, plain
    numbers, strings, device handles, and anything that is a `ctypes.Structure` -- a caller's
    declared structure and a vector type included, both of which already have a device type.
    """
    if value is None:
        return False
    if isinstance(value, (np.ndarray, np.generic, ctypes.Structure)):
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

    def __init__(self, resolve:Callable[[str], Any])->None:
        self.resolve = resolve
        self.classes:Dict[str, Optional[type]] = {}
        self.values:Dict[str, Any] = {}
        self.members:Dict[str, List[str]] = {}
        self.methods:Dict[str, List[Tuple[str, Any]]] = {}
        self.lists:Dict[str, List[Any]] = {}
        self.written:List[str] = []
        self.visited:set = set()

    def add_path(self, path:str, value:Any)->None:
        self.classes.setdefault(path, type(value))
        self.values.setdefault(path, value)
        self.members.setdefault(path, [])
        self.methods.setdefault(path, [])

    def add_list(self, path:str, items:List[Any])->None:
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

    def note_method(self, origin:str, attr:str, function:Any)->None:
        methods:List[Tuple[str, Any]] = self.methods.setdefault(origin, [])
        if all(name != attr for name, _ in methods):
            methods.append((attr, function))


def _parse(func:Any)->Optional[ast.Module]:
    from .translator import _parse_kernel_source, get_dedented_source  # noqa: PLC0415

    try:
        return _parse_kernel_source(get_dedented_source(func))
    except Exception:
        return None


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


def _scan_function(func:Any, tracked:Dict[str, str], scan:Scan)->None:
    """Walk one function body, adding the members it reads and recursing into helpers.

    The visit key is the (function, tracked-context) pair: the same helper reached with a
    nested path bound to a parameter scans *that* path's members, which is a different walk
    from the top-level one even though the function is the same.
    """
    visit_key:Any = (func, tuple(sorted(tracked.items())))
    if visit_key in scan.visited:
        return
    scan.visited.add(visit_key)
    tree:Optional[ast.Module] = _parse(func)
    if tree is None:
        return
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
            value:Any = getattr(scan.values.get(root_path), node.iter.attr, None)
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
        element_value:Any = scan.lists[iter_path][0]
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
            # `items[j].v`: the subscript's value chain decides the list, and everything
            # after the subscript belongs to the element structure.
            while isinstance(cursor, ast.Subscript):
                cursor = cursor.value
            if not (isinstance(cursor, (ast.Name, ast.Attribute))):
                continue
            if isinstance(cursor, ast.Name) and cursor.id in tracked:
                list_path:str = tracked[cursor.id]
            elif isinstance(cursor, ast.Attribute):
                continue
            else:
                continue
            if list_path not in scan.lists:
                continue
            element_path:str = _element_path_of(scan, list_path)
            if element_path is None:
                # An indexed read (`items[j].v`) with no `for ... in items:` anywhere: the
                # element path starts here, one per list.
                element_path:str = _member_path(list_path, "element#items")
                scan.add_path(element_path, scan.lists[list_path][0])
                scan.classes[element_path] = scan.classes[list_path]
            scan.note(element_path, node.attr)
            crossed_list = True
            continue
        if crossed_list:
            continue
        if not (isinstance(cursor, ast.Name) and cursor.id in tracked):
            continue
        root_path:str = tracked[cursor.id]
        # Resolve hop by hop: every hop whose value is itself an object becomes its own path,
        # and the members after it belong to that path. A hop whose value is not an object is
        # a plain member read and ends the chain.
        current_path:str = root_path
        for index, attr in enumerate(reversed(hops)):
            member_path:str = _member_path(current_path, attr)
            parent_value:Any = scan.values.get(current_path)
            value:Any = (getattr(parent_value, attr, None) if parent_value is not None else None)
            if index < len(hops) - 1 and looks_like_object(value):
                # An object member that is read *through*: its path opens and the walk
                # continues inside it.
                if member_path not in scan.members:
                    scan.add_path(member_path, value)
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
        function:Any = getattr(owner, node.func.attr, None)
        if not isinstance(function, Callable) or isinstance(function, (type, ctypes.Structure)):
            raise SyntaxError(
                f"{owner.__name__}.{node.func.attr} is not a method that can be translated into "
                f"device code")
        from .translator import scan_self_attrs  # noqa: PLC0415

        # A method may itself call another method of the same object, and that one's reads
        # belong to the same member set -- `ray_color` calling `self.hit` reading
        # `self.hittables` is one chain. `scan_self_attrs` only returns the *data* a body
        # reads, so the method calls are followed from here, recursively.
        def follow(method:Any, seen:Set[Any])->None:
            if method in seen:
                return
            seen.add(method)
            method_tree:Optional[ast.Module] = _parse(method)
            if method_tree is None:
                return
            called:Set[str] = set()
            for inner_node in ast.walk(method_tree):
                if (isinstance(inner_node, ast.Call) and isinstance(inner_node.func, ast.Attribute)
                        and isinstance(inner_node.func.value, ast.Name)
                        and inner_node.func.value.id == "self"):
                    called.add(inner_node.func.attr)
            for name in scan_self_attrs(method, "self", owner):
                scan.note(origin, name)
            for name in called:
                inner:Any = getattr(owner, name, None)
                if isinstance(inner, Callable) and not isinstance(inner, (type, ctypes.Structure)):
                    scan.note_method(origin, name, inner)
                    follow(inner, seen)

        follow(function, set())
        scan.note_method(origin, node.func.attr, function)

    # `helper(p)`: the helper's parameter becomes another name for the same object.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        target:Any = scan.resolve(node.func.id)
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


def scan_object(func:Any, objects:Dict[str, Any],
                resolve:Callable[[str], Any])->Scan:
    """Scan `func` for the members and methods each object parameter needs."""
    scan:Scan = Scan(resolve)
    transitive:Dict[str, str] = {}
    for name, value in objects.items():
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
            source:Any = scan.values.get(path)
            if source is not None:
                cls:type = type(source)
                try:
                    attribute:Any = getattr(cls, member)
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


def member_type(value:Any, attr:str)->type:
    """The ctypes type for one member, decided from the value the caller passed.

    A plain python object member is allowed only through the nested path in
    ``convert_object_arguments``, which replaces its type with the synthesized structure; this
    function is where that substitution is applied.
    """
    if isinstance(value, (bool, np.bool_)):
        return ctypes.c_bool
    if isinstance(value, (int, np.integer)):
        return ctypes.c_int32
    if isinstance(value, (float, np.floating)):
        return ctypes.c_float
    if isinstance(value, np.ndarray):
        raise SyntaxError(
            f"'{attr}' is an array, and an array member of a structure is not supported yet. "
            f"Pass the array as its own kernel argument, or pass a list of objects if the "
            f"elements are instances of one class")
    if isinstance(value, ctypes.Structure):
        # A declared structure or a vector type already has a device type: use it as it is.
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


_STRUCT_CACHE:Dict[Tuple[Any, ...], type] = {}


def _structure_for(fields:Sequence[Tuple[str, type]],
                   methods:Sequence[Tuple[str, Any]])->type:
    key:Tuple[Any, ...] = (tuple(fields), tuple(name for name, _ in methods))
    cached:Optional[type] = _STRUCT_CACHE.get(key)
    if cached is not None:
        return cached
    name:str = "Object" + "".join(member.capitalize() for member, _ in fields)
    namespace:Dict[str, Any] = {"_fields_": [(member, kind) for member, kind in fields]}
    structure:type = type(name, (ctypes.Structure,), namespace)
    # A method the body calls has to be reachable from the synthesized class: the translator
    # resolves `p.method` from the class its dtype came from.
    for method_name, function in methods:
        if method_name not in structure.__dict__:
            setattr(structure, method_name, function)
    _STRUCT_CACHE[key] = structure
    return structure


def _fields_for(obj:Any, names:Sequence[str])->List[Tuple[str, type]]:
    fields:List[Tuple[str, type]] = []
    for name in names:
        if not hasattr(obj, name):
            raise SyntaxError(
                f"object of type {type(obj).__name__} has no attribute '{name}', which the "
                f"kernel body reads")
        value:Any = getattr(obj, name)
        if isinstance(value, (type, Callable)) and not isinstance(value, ctypes.Structure):
            raise SyntaxError(
                f"{type(obj).__name__}.{name} is a function or a class, so it cannot be a "
                f"structure member. Call it as a method instead of reading it")
        fields.append((name, member_type(value, name)))
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
        self.values:Dict[str, Any] = {}
        self.instances:Dict[str, Any] = {}

    def write_back(self)->None:
        """Copy the final member values home, for every object the body wrote to."""
        for description in self.written:
            path, member = description.rsplit('.', 1)
            instance:Any = self.instances.get(path)
            source:Any = self.values.get(path)
            if instance is None or source is None:
                continue
            if isinstance(instance, np.ndarray):
                # The object travelled as a one-element record array; the device wrote its
                # member stores there, and the host read the array back after the launch.
                try:
                    value:Any = instance[0][member]
                except (IndexError, ValueError):
                    continue
                setattr(source, member, value.item() if hasattr(value, "item") else value)
                continue
            try:
                value:Any = getattr(instance, member)
            except AttributeError:
                continue
            setattr(source, member, value)


def convert_object_arguments(func:Any, args:Sequence[Any],
                             resolve:Callable[[str], Any],
                             self_param:str = '',
                             binding:Any = None)->Tuple[Any, ...]:
    """Replace every object argument with an instance of its synthesized structure.

    Called before the argument types are inferred, because the structure *is* the type: the
    translator then sees a `ctypes.Structure`, so the device typedef, the member access, the
    layout and the packing all work unchanged.

    ``self_param`` is the kernel's receiver name, when the kernel is a method: the receiver
    travels outside ``args``, so it is skipped when the parameters are lined up with the
    arguments, or `scene` in `render(self, scene)` would line up with the name `self`.
    """
    hits:List[int] = [index for index, value in enumerate(args) if looks_like_object(value)]
    if not hits:
        return tuple(args)

    parameters:List[str] = list(func.__code__.co_varnames[:func.__code__.co_argcount])
    all_parameters:List[str] = list(parameters)
    if self_param and parameters and parameters[0] == self_param:
        parameters = parameters[1:]
    named:Dict[str, Any] = {}
    for index in hits:
        if index >= len(parameters):
            raise SyntaxError(
                f"an object passed in position {index} does not line up with a named "
                f"parameter, so its members cannot be discovered")
        named[parameters[index]] = args[index]

    scan:Scan = scan_object(func, named, resolve)
    out:List[Any] = list(args)

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
            scan.members[element_path] = list({
                *scan.members[element_path], *scan.members[path]})
    for path in sorted(scan.members, key=len, reverse=True):
        if path in named and "." not in path and path not in scan.lists:
            continue                     # top-level paths are built below, with merging
        if _list_root(path) in scan.lists and path != _list_root(path):
            pass                         # an element path; built here like any other
        elif path in scan.lists:
            continue                     # the list itself becomes an array, not a structure
        fields:List[Tuple[str, type]] = []
        called_methods:set = {name for name, _ in scan.methods.get(path, [])}
        for member in scan.members[path]:
            if member in called_methods:
                # A method the body calls: it is copied onto the synthesized class by
                # `_structure_for`, and it is not data, so it takes no field.
                continue
            value:Any = getattr(scan.values[path], member)
            member_path:str = _member_path(path, member)
            if member_path in structure_of:
                # The member was scanned as a nested object: its synthesized structure is
                # the field's type.
                fields.append((member, structure_of[member_path]))
            else:
                fields.append((member, member_type(value, member)))
        structure_of[path] = _structure_for(fields, scan.methods[path])
        if "." in path:
            parent, member = path.rsplit(".", 1)
            parent = _list_root(parent)
            if parent not in scan.lists:
                nested_of.setdefault(parent, []).append((member, structure_of[path]))

    by_class:Dict[type, List[str]] = {}
    for name in named:
        by_class.setdefault(type(named[name]), []).append(name)
    for _cls, names in by_class.items():
        specs:List[List[Tuple[str, type]]] = []
        for name in names:
            fields:List[Tuple[str, type]] = []
            for member in scan.members[name]:
                member_path:str = _member_path(name, member)
                if member_path in structure_of:
                    fields.append((member, structure_of[member_path]))
                else:
                    fields.append((member, member_type(getattr(named[name], member), member)))
            fields.extend((m, k) for m, k in nested_of.get(name, [])
                          if m not in {n for n, _ in fields})
            specs.append(fields)
        merged:Optional[List[Tuple[str, type]]] = (merge_fields(specs)
                                                   if len(specs) > 1 else specs[0])
        for name, fields in zip(names, specs):
            chosen:List[Tuple[str, type]] = merged if merged is not None else fields
            structure:type = _structure_for(chosen, scan.methods[name])

            def build(source:Any)->Any:
                """An instance of ``structure`` carrying ``source``'s values, recursively."""
                values:Dict[str, Any] = {}
                for member, kind in chosen:
                    value:Any = getattr(source, member)
                    if (isinstance(kind, type) and issubclass(kind, ctypes.Structure)
                            and not isinstance(value, kind)):
                        # A nested member is rebuilt in the synthesized type from the values
                        # the caller's object carries.
                        value = kind(**{n: getattr(value, n) for n, _ in kind._fields_})
                    values[member] = value
                return structure(**values)

            instance:Any = build(named[name])
            written_paths:set = {description.rsplit(".", 1)[0] for description in scan.written}
            if name in written_paths:
                # The body writes this object's members: by value the device would write to
                # its own copy and the stores would be lost. A one-element record array
                # travels as a pointer, so the stores land in device memory that the host
                # reads back, and write_back then copies them onto the caller's object.
                records:np.ndarray = np.zeros(1, dtype=np.dtype(structure))
                for member, _ in structure._fields_:
                    value:Any = getattr(named[name], member)
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

    # A list-of-objects argument becomes a numpy record array of the element structure: a
    # device loop over it indexes the array, exactly as a record array argument would. The
    # element class is registered first, so a method the body calls on an element resolves
    # from the class its dtype came from.
    from .structs import reregister_class  # noqa: PLC0415

    for list_path, element_path in list_element_paths.items():
        index:int = all_parameters.index(_list_root(list_path))
        element_structure:type = structure_of[element_path]
        reregister_class(element_structure)
        records:np.ndarray = np.zeros(len(scan.lists[list_path]), dtype=np.dtype(element_structure))
        for position, item in enumerate(scan.lists[list_path]):
            for member, _ in element_structure._fields_:
                value:Any = getattr(item, member)
                if isinstance(value, ctypes.Structure):
                    value = np.array(value, dtype=np.dtype(type(value)))
                records[position][member] = value
        out[index] = records
    if binding is not None:
        binding.written = list(scan.written)
    return tuple(out)
