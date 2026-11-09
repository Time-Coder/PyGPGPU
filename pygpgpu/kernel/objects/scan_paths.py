from __future__ import annotations
from .state import _LIFTED_TREES
import ast
import ctypes
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple
import numpy as np



def looks_like_object(value:object)->bool:
    """Whether an argument is an arbitrary python object rather than a value we already know.

    Everything the translator already handles is excluded: arrays, numpy scalars, plain
    numbers, strings, device handles, and anything that is a `ctypes` structure *or union* --
    a caller's declared layout and a vector type included, all of which already have a device
    type. A union is not a `ctypes.Structure` subclass, so testing for that alone sent every
    union down this path, where the members are guessed from what the body reads and a
    `p["i"]` spelling has no layout behind it to answer it.
    """
    from ..structs import is_ctypes_record  # noqa: PLC0415 (import cycle)

    if value is None:
        return False
    if isinstance(value, (np.ndarray, np.generic)) or is_ctypes_record(value):
        return False
    if isinstance(value, (bool, int, float, complex, str, bytes)):
        return False
    if not isinstance(value, (type, Callable)):
        from ..translator import _is_device_handle  # noqa: PLC0415 (import cycle)

        if _is_device_handle(value):
            return False
    return True



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
    import copy as _copy  # noqa: PLC0415
    from ..translator import _parse_kernel_source, get_dedented_source  # noqa: PLC0415

    cached:Optional[ast.Module] = _LIFTED_TREES.get(_tree_key(func))
    if cached is not None:
        # The scan lifts an inline construction into a local (see _inline_constructions),
        # and every later step has to read that same tree: a fresh parse would give back
        # the original `self.hit(ray, Interval(...))`, with the lifted name nowhere in it.
        return _copy.deepcopy(cached)
    try:
        # The scan *rewrites* the tree it is given -- that is what lifting a construction
        # is -- so it takes a copy of the parsed one. `_parse_kernel_source` answers a
        # cached tree shared by every reader, and inserting a statement into that would
        # hand the next launch a body with a lifted name no scan put there.
        return _copy.deepcopy(_parse_kernel_source(get_dedented_source(func)))
    except Exception:
        return None



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
