# Re-export shim: `scan` keeps the walk, the path helpers moved to `scan_paths`, and
# both stay importable from here because `objects/__init__` and `convert` name them.
from __future__ import annotations

from .scan_paths import (  # noqa: F401  (the facade re-exports the path helpers)
    Scan,
    _called_members,
    _element_path_of,
    _follow_aliases,
    _list_root,
    _member_path,
    _parse,
    _resolve_list_path,
    _tree_key,
    looks_like_object,
    self_chain,
)



import ast
import ctypes
from collections.abc import Callable
from typing import Dict, List, Optional
import numpy as np



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
    # Upward call: `construction` imports this module, so a top-level import here would
    # make the pair a cycle.
    from .construction import constructed_locals
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
            if index < len(hops) - 1 and isinstance(value, (complex, np.complexfloating)):
                # A complex member read through (`holder.center.real`): the member *is* the
                # payload, with a device type of its own, so it is noted and the walk stops
                # here. Opening a path inside it would treat the complex value as a nested
                # structure, which it is not -- the same rule the vector branch above states.
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
        # Upward call, as in the loop above.
        from .construction import constructed_locals
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