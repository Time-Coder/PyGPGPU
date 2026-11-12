# What kind of value an argument or a member is, and the structure a group becomes.
from __future__ import annotations
from .state import (
    _DEVICE_SCALAR_CLASSES,
    _DEFAULT_FLOAT_MEMBER,
    _DEFAULT_INT_MEMBER,
    _PROMOTIONS,
    _STRUCT_CACHE,
)
import ast
import ctypes
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np



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
        from ..translator import check_integer_range  # noqa: PLC0415 (import cycle)

        check_integer_range(int(value), _ctypes_name_of(chosen), attr)
        return chosen
    if isinstance(value, (float, np.floating)):
        return scalar_member_type(type_map, float)
    return None



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
    if isinstance(value, (complex, np.complexfloating)):
        # A complex member is one of the fixed records the complex header declares (`c64` /
        # `c128`), exactly as a quaternion member is one of the quaternion header's. A numpy
        # complex scalar keeps its own width; a python `complex` has none, so it follows the
        # `float=` mapping the same way a python `float` member does.
        from ..complex import member_class  # noqa: PLC0415 (import cycle)
        from ..translator import python_complex_type  # noqa: PLC0415 (import cycle)

        if isinstance(value, np.complexfloating):
            dtype:np.dtype = np.dtype(type(value))
        else:
            dtype = np.dtype(np.complex64 if python_complex_type(
                (type_map or {}).get("float", "float")) == "c64" else np.complex128)
        cls:Optional[type] = member_class(dtype)
        if cls is None:
            raise SyntaxError(
                f"'{attr}' is a {np.dtype(dtype).name}, which has no device member type")
        return cls
    raise SyntaxError(
        f"'{attr}' is a {type(value).__name__}, which cannot become a structure member yet "
        f"(only numbers, booleans, complex values and vector values can)")



def promote(left:type, right:type)->Optional[type]:
    """The type both can widen to, or None when they cannot be mixed."""
    if left is right:
        return left
    return _PROMOTIONS.get((left, right)) or _PROMOTIONS.get((right, left))



def _is_number(value:object)->bool:
    """True for a value that becomes a scalar structure member."""
    return isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating))



def _prefer_class_name(structure:type, cls:Optional[type]) -> None:
    """Let the C structure this dtype describes be named after the python class.

    The device type is generated from the dtype, and its C name comes from
    `structs.struct_name`; a hash says nothing a reader recognizes, while the class the
    value came from does. The hint is advisory and first-seen wins, because two classes
    with one layout share a dtype.
    """
    if cls is None or getattr(cls, "__module__", "") == "builtins":
        return
    from ..structs import register_name_hint  # noqa: PLC0415 (import cycle)

    register_name_hint(np.dtype(structure), cls)



def _structure_for(fields:Sequence[Tuple[str, type]],
                   methods:Sequence[Tuple[str, Any]], backend:str)->type:
    # Keyed by the member *set* and the backend -- not their order, and not the method
    # list. The walk collects members through `ast.walk` (order not source order), and
    # one path may see a method where another does not (`hitable.hit` is called in one
    # place, only read in another). Two classes with one layout are still two ctypes
    # types, so the class has to be one: a later call adds any method the first one did
    # not know about. The backend is part of the layout (a three-component vector
    # occupies twelve bytes in CUDA C and sixteen in OpenCL C), so one member set under
    # two backends is two structures.
    key:Tuple[object, ...] = (frozenset((member, kind) for member, kind in fields), backend)
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
    from ..structs import padded_member_fields  # noqa: PLC0415 (import cycle)

    # The members are placed the way the backend's C would place them, with explicit
    # padding where the two disagree: a vector member's device alignment (float4 sits
    # at offset 16 after an int on CUDA) is what the launch packs and what the rendered
    # declaration carries, so one layout computation serves both sides.
    namespace:Dict[str, object] = {"_fields_": padded_member_fields(ordered, backend)}
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
