# The entry points: the scan a compile runs, and the conversion a launch runs.
from __future__ import annotations
from .binding import _complex_member_value, _is_complex_member_class
from .construction import (
    _constructs_a_class,
    _init_parameters,
    constructed_locals,
)
from .kinds import (
    _prefer_class_name,
    _structure_for,
    member_type,
    merge_fields,
    numeric_list_as_array,
)
from .scan import (
    Scan,
    _list_root,
    _member_path,
    _parse,
    _scan_function,
    _writes_to,
    looks_like_object,
)
from .state import (
    _CONSTRUCTED,
    _CONSTRUCTED_PARAMETERS,
    _LIFTED_MEMBERS,
    _LIFTED_VALUES,
)
import ast
import ctypes
from collections.abc import Callable
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np



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
                from ..structs import register_declared_class  # noqa: PLC0415

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
                    if _is_complex_member_class(kind) and not isinstance(value, kind):
                        # `ctypes` has no complex field, so the value is split into the
                        # record's two parts at the class's own width.
                        value = _complex_member_value(value, kind)
                    elif (isinstance(kind, type) and issubclass(kind, ctypes.Structure)
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
                for member, field_type in structure._fields_:
                    value:object = getattr(named[name], member)
                    if _is_complex_member_class(field_type) \
                            and not isinstance(value, field_type):
                        # The record's field is the two-part storage, so a python complex is
                        # split into it before numpy sees the assignment (numpy would raise
                        # "float() argument must be ... not 'complex'").
                        value = _complex_member_value(value, field_type)
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
    from ..structs import reregister_class  # noqa: PLC0415

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
            from ..structs import register_declared_class  # noqa: PLC0415

            dtype = np.dtype(structure)
            register_declared_class(dtype, cls)
            _CONSTRUCTED[cls] = dtype
            # The constructor's argument order, which the aggregate literal follows: the
            # synthesized structure sorts its fields by name, so filling it in field order
            # would put the second argument in the first member.
            _CONSTRUCTED_PARAMETERS[cls] = _init_parameters(cls)
    return tuple(out)
