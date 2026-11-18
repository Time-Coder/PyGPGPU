"""The static return-type scan a native-recursion verdict reads.

The unlock's one unknown is the callee's return type: the call site must report it
before the body is translated, and the scan is what makes it static. Every `return` of
the body must name the *same* scalar -- and the expression rules are the translation's
own, read off the AST:

* a literal names its own width (`bool`, `int`, or the kernel's `float=` width for a
  real, exactly as `_visit_constant` spells it);
* a bare read of a scalar parameter names the parameter's declared type;
* a comparison answers `bool`; a unary sign keeps its operand's type; `~` needs an
  integer and keeps it;
* arithmetic is `promote` over its operands' *known* types, with `Div`'s true-division
  rule (an integer pair answers the real width) and `Pow`'s integer-power promotion;
  the bitwise family needs two integers and keeps the promotion;
* `abs` keeps a real/int argument, `round` answers `int` from a real, `min`/`max`
  promote, `pow` promotes an integer power to the real width, and `int()`/`float()`/
  `bool()` name their own conversions;
* a math/np call answers the promoted real width, with the table overrides for
  python's own return types (`floor`/`ceil`/`trunc` answer `int`, the predicates
  answer `bool`) and numpy's float64-for-integer-input rule;
* a vector constructor names the registered device spelling the class's MRO carries
  (`pygf.vector3f` compiles into the `float3` constructor through the alias walk);
* a call inside the *cycle* -- the function's own name, or a mutual partner the
  globals resolve to a plain function, spelled as a bare name or as a method's
  `self.<name>` -- is the induction that closes it: its type is the cycle's own,
  which the other returns fix.

An operand whose type the scan does not know (a local, a foreign call) leaves the
whole return undecided, and the refusal stands: the type would come from the
translation of *this* helper, which cannot run before its own signature exists.
"""
from __future__ import annotations

import ast
import types
from typing import Dict, List, Optional

from .vectors import _vec_registry


# The type worlds the scan reads the translation's rules from: INT_TYPES and
# FLOAT_TYPES (see type_map), and the real pair beside them.
INT_TYPES:frozenset = frozenset({"bool", "char", "uchar", "short", "ushort", "int",
                                 "uint", "long", "ulong"})
REAL_TYPES:frozenset = frozenset({"float", "double"})

# math/np functions whose return type the emission table overrides (see _MATH_FUNCS
# in builtins.py and _emit_math_call): python's own return type is the rule.
MATH_RETURN_OVERRIDES:Dict[str, str] = {
    "floor": "int", "ceil": "int", "trunc": "int",
    "isnan": "bool", "isinf": "bool", "isfinite": "bool",
}

# The marker `_expression_type` answers for a call inside the cycle: the seed skips
# it (the induction), because the cycle's own type is what the other returns fix.
SELF_TYPE:str = "@self"


def _parameter_names(tree:ast.FunctionDef, receiver:str = "")->List[str]:
    """The device parameters one method signature carries, receiver excluded.

    A plain function's parameters are its `def` list. A method's device signature
    carries the receiver struct first, then its `_self_<attr>` slots, then the
    declared parameters -- but the *scan* counts only what the recursive call spells
    positionally, which is the declared list (the receiver is spelled by the `self.`
    prefix, the attribute slots are forwarded implicitly).
    """
    names:List[str] = [arg.arg for arg in tree.args.posonlyargs]
    names.extend(arg.arg for arg in tree.args.args)
    return [name for name in names if name != receiver]


def _receiver_name_of(tree:ast.FunctionDef)->str:
    """The receiver parameter's name, when the body reads one (`self`)."""
    args:List[ast.arg] = tree.args.posonlyargs + tree.args.args
    if args and args[0].arg == "self":
        return "self"
    return ""


def _bare_calls(node:ast.AST, receiver_name:Optional[str] = None
                )->List[tuple]:
    """Every call the body spells as a plain name or as `self.<name>`, named.

    A method's recursive call is spelled `self.walk(...)` -- an attribute lookup the
    translator resolves to the same object (see `_self_member_call_kind`). When
    `receiver_name` is given (the translated function's own receiver parameter), that
    spelling is recognized too: the pair it names is the method's own name, which is
    what the bar and the seed need the call to say. Any other receiver (`other.obj`,
    a rebound method) stays unrecognized, exactly as the bare-name rule leaves it.
    """
    found:List[tuple] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func:ast.expr = child.func
        if isinstance(func, ast.Name):
            if isinstance(func.ctx, ast.Load):
                found.append((func.id, child))
        elif (isinstance(func, ast.Attribute) and receiver_name is not None
              and isinstance(func.value, ast.Name)
              and func.value.id == receiver_name
              and isinstance(func.value.ctx, ast.Load)):
            found.append((func.attr, child))
    return found


def _constant_kind(value:object)->Optional[str]:
    """The device scalar type one python literal denotes, or None.

    The same three answers `_visit_constant` gives -- `bool`, `int`, and the `float=`
    width (`float` or `double`); the width choice itself is the caller's, because the
    mapping is a per-kernel fact this module does not hold (`"@float-width"` is the
    marker the caller resolves).
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "@float-width"
    return None


def seed_return_type(tree:ast.FunctionDef, param_types:List[object],
                     float_width:str,
                     globals_map:Optional[dict] = None)->Optional[str]:
    """The return type every `return` names statically, or None when they disagree.

    The expression forms are decided by the *same rules the translation applies*, read
    off the AST -- see `_expression_type`. A call inside the cycle (the function's own
    name, or a mutual partner the globals resolve to a plain function) is the
    induction that closes it. Every value return must agree, and any disagreement
    answers None -- the driver keeps the refusal.
    """
    params:List[str] = _parameter_names(tree, _receiver_name_of(tree))
    declared:Dict[str, object] = dict(zip(params, param_types[:len(params)]))
    cycle:frozenset = _cycle_names(tree, globals_map)
    answer:Optional[str] = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Return) and node.value is not None):
            continue
        kind:Optional[str] = _expression_type(node.value, tree.name, declared,
                                              float_width, depth=0,
                                              cycle_names=cycle,
                                              globals_map=globals_map)
        if kind == SELF_TYPE:
            # The induction: this return's type is the cycle's own, which the other
            # returns fix. It contributes no fact and no disagreement.
            continue
        if kind is None:
            return None
        if answer is not None and answer != kind:
            return None
        answer = kind
    return answer


def _cycle_names(tree:ast.FunctionDef, globals_map:Optional[dict])->frozenset:
    """The calls this body spells as the cycle the seed may assume.

    The function's own name is always in the set. A call of *another* bare name is in
    it only when the name resolves to a plain python function through `globals_map` AND
    the function is defined in the same module the caller reads -- the mutual partner
    whose own translation will follow this one. A helper a body merely *uses* (pygf's
    `mix`, `normalize`, a random sampler) is also a plain function, but its return type
    is its own annotation's business, not the cycle's: treating every imported function
    as a partner made `mix(color3f(...), ...)` answer "@self" (the induction) instead of
    the vector its arguments carry. Same-module is the line: a recursive partner has to
    live where the recursion is written, while vocabulary comes from imports. A method's
    `self.<name>` spelling names the method itself when the attribute is the function's
    own name (its receiver resolves to the same object).
    """
    receiver:str = _receiver_name_of(tree)
    names:set = {tree.name}
    if globals_map is not None:
        own_module:str = str(globals_map.get("__name__", "") or "")
        for call_name, _call in _bare_calls(tree, receiver):
            if call_name == tree.name:
                continue
            value:object = globals_map.get(call_name)
            if not isinstance(value, types.FunctionType):
                continue
            module:str = getattr(value, "__module__", "") or ""
            if module == own_module or module == "__main__":
                names.add(call_name)
    return frozenset(names)


def _expression_type(node:ast.expr, func_name:str, declared:Dict[str, object],
                     float_width:str, depth:int,
                     cycle_names:Optional[frozenset] = None,
                     globals_map:Optional[dict] = None)->Optional[str]:
    """The scalar type one returned expression denotes, by the translation's own rules.

    `depth` bounds the recursion: an expression tree is finite and shallow in practice,
    and the cap keeps a pathologically deep nesting from walking the whole tree twice
    per scan.
    """
    if depth > 24:
        return None
    if isinstance(node, ast.Constant):
        kind:Optional[str] = _constant_kind(node.value)
        return float_width if kind == "@float-width" else kind
    if isinstance(node, ast.Name) and node.id in declared \
            and isinstance(declared[node.id], str):
        return declared[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        # `_visit_unaryop` spells `(-x)` and keeps the operand's type; `not` answers
        # a bool, `~` an int, and everything else is refused by the translation.
        if isinstance(node.op, ast.Not):
            return "bool"
        if isinstance(node.op, ast.Invert):
            operand:str = _expression_type(node.operand, func_name, declared,
                                           float_width, depth + 1, cycle_names,
                                           globals_map)
            return operand if operand in INT_TYPES else None
        return _expression_type(node.operand, func_name, declared, float_width,
                                depth + 1, cycle_names, globals_map)
    if isinstance(node, ast.Compare):
        return "bool"
    if isinstance(node, ast.Call) and cycle_names is not None \
            and ((isinstance(node.func, ast.Name) and node.func.id in cycle_names)
                 or (isinstance(node.func, ast.Attribute)
                     and node.func.attr in cycle_names)):
        # The induction: a call inside the cycle -- spelled as a bare name or as a
        # method's `self.<name>` -- answers the cycle's own type, which the other
        # returns fix.
        return SELF_TYPE
    if isinstance(node, ast.BoolOp):
        # `a and b` answers an *operand* (python's rule, spelled as a conditional
        # expression): the type is the promoted pair, and `not`-free arms only.
        kinds:List[str] = [_expression_type(value, func_name, declared, float_width,
                                            depth + 1, cycle_names, globals_map)
                           for value in node.values]
        if any(kind is None for kind in kinds):
            return None
        answer:Optional[str] = kinds[0]
        for kind in kinds[1:]:
            if answer != kind:
                answer = _promoted(answer, kind)
        return answer
    if isinstance(node, ast.IfExp):
        # The conditional answers the promoted pair, exactly as _visit_expression's
        # IfExp branch spells it.
        body:str = _expression_type(node.body, func_name, declared, float_width,
                                    depth + 1, cycle_names, globals_map)
        orelse:str = _expression_type(node.orelse, func_name, declared, float_width,
                                      depth + 1, cycle_names, globals_map)
        if body is None or orelse is None:
            return None
        return body if body == orelse else _promoted(body, orelse)
    if isinstance(node, ast.BinOp):
        left:str = _expression_type(node.left, func_name, declared, float_width,
                                    depth + 1, cycle_names, globals_map)
        right:str = _expression_type(node.right, func_name, declared, float_width,
                                     depth + 1, cycle_names, globals_map)
        if left is None or right is None:
            return None
        name:str = node.op.__class__.__name__
        # A vector operand carries the operation: `0.5 * ray_color(...)` is the scalar-
        # times-vector form the emission handles component-wise, and the answer is the
        # vector's own type -- the same rule `_visit_binop` applies when one operand is
        # a vector. `@self` (the induction) rides along: the cycle's own type is what
        # the other returns fix, and the multiply keeps it.
        for operand in (left, right):
            if operand in _vec_registry():
                return operand
            if operand == SELF_TYPE:
                return SELF_TYPE
        if name == "Div":
            # `x / y` is python's true division: an integer pair answers a real
            # (see _visit_binop's Div branch), anything else promotes.
            if left in INT_TYPES and right in INT_TYPES:
                return float_width
            return _promoted(left, right)
        if name == "Pow":
            # The `**` emission promotes an integer power to the real width (the
            # builtin does too); a real power promotes as usual.
            promoted:str = _promoted(left, right)
            return float_width if promoted in INT_TYPES else promoted
        if name == "FloorDiv":
            return _promoted(left, right)
        if name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor"):
            if left in INT_TYPES and right in INT_TYPES:
                return _promoted(left, right)
            return None
        if name == "Mod":
            return _promoted(left, right)
        if name in ("Add", "Sub", "Mult"):
            return _promoted(left, right)
        return None
    if isinstance(node, ast.Call):
        # `abs(x)` keeps its argument's real/int type; `round(x)` answers int from a
        # real and keeps an int; `min`/`max`/`pow` follow their emission rules. A math
        # call answers the promoted real width unless the table overrides it. Anything
        # else is decided by the translation.
        return _call_type(node, func_name, declared, float_width, depth, cycle_names,
                          globals_map)
    return None


def _call_type(node:ast.Call, func_name:str, declared:Dict[str, object],
               float_width:str, depth:int,
               cycle_names:Optional[frozenset] = None,
               globals_map:Optional[dict] = None)->Optional[str]:
    """The static type of the builtin and math vocabulary a returned call may spell."""
    if not (isinstance(node.func, ast.Name) or (isinstance(node.func, ast.Attribute)
                                                and isinstance(node.func.value, ast.Name))):
        return None
    # A vector constructor names its own device type: `vector3f(...)` compiles into
    # the `float3` constructor (the host aliases subclass the same vector), so the
    # returned type is the registered device spelling the class's MRO carries. The
    # registry answers only the device names, and the aliases are host classes, so
    # the base walk is what `pygf.vector3f` needs. Structures refuse here (a dtype
    # the scan would have to read members for is the translation's answer).
    if globals_map is not None and isinstance(node.func, ast.Name):
        target:object = globals_map.get(node.func.id)
        if isinstance(target, type):
            if target.__name__ in _vec_registry():
                return target.__name__
            for base in target.__mro__:
                # The alias's *registered base* is the answer, not the host name: the
                # seed speaks device types, and `color3f` is a host alias whose device
                # spelling is `float3` (the MRO walk is what finds it).
                if base.__name__ in _vec_registry():
                    return base.__name__
            return None
    # The argument types first: every rule below reads them.
    kinds:List[str] = [_expression_type(arg, func_name, declared, float_width,
                                        depth + 1, cycle_names, globals_map)
                       for arg in node.args]
    # A vector-family helper (pygf's mix/normalize/... spell `-> genType`) answers its
    # first *vector* argument's own type, which is what the emission does -- and the
    # answer stands even when a later scalar argument's type the scan cannot see (a
    # local weight). Try it before the any-None refusal below.
    if globals_map is not None and isinstance(node.func, ast.Name):
        target:object = globals_map.get(node.func.id)
        if callable(target) and not isinstance(target, type):
            for kind in kinds:
                if kind in _vec_registry():
                    return kind
    if any(kind is None for kind in kinds):
        return None
    if globals_map is not None and isinstance(node.func, ast.Name) \
            and callable(globals_map.get(node.func.id)) \
            and not isinstance(globals_map.get(node.func.id), type):
        # A helper call names the type the helper's own return annotation declares: a
        # pygf vector function is annotated `-> genType`, the *family* the concrete type
        # belongs to rather than a device name -- the concrete answer follows the first
        # vector argument (mix(color3f, color3f, w) answers color3f), which is what the
        # emission does. Only a *vector* resolution is taken here -- a scalar annotation
        # lies about widths the mapping may have moved, and a structure return is the
        # translation's decision, not a static fact.
        target:object = globals_map.get(node.func.id)
        annotation:object = getattr(target, "__annotations__", {}).get("return")
        # The first argument's own kind, normalized to the registered device name: a
        # constructor call of a host alias answers its base (`color3f` -> `float3`),
        # which is what kinds carries after the MRO walk above.
        if kinds and kinds[0] in _vec_registry():
            return kinds[0]
        for base in getattr(annotation, "__mro__", ()):
            if getattr(base, "__name__", "") in _vec_registry():
                return base.__name__
    if isinstance(node.func, ast.Attribute):
        module:str = node.func.value.id
        attr:str = node.func.attr
        if module in ("math", "np") and attr in MATH_RETURN_OVERRIDES:
            return MATH_RETURN_OVERRIDES[attr]
        if module in ("math", "np"):
            # The promoted real width, with numpy's float64-for-integer-input rule
            # (see _emit_math_call): math keeps the plain width, np widens ints.
            promoted:Optional[str] = None
            for kind in kinds:
                promoted = kind if promoted is None else _promoted(promoted, kind)
            if promoted is None:
                return None
            if module == "np" and all(kind in INT_TYPES for kind in kinds):
                return "double"
            return promoted if promoted in REAL_TYPES else float_width
        return None
    name:str = node.func.id
    if name == "abs":
        return kinds[0] if kinds[0] in (REAL_TYPES | INT_TYPES) else None
    if name == "round":
        return "int" if kinds[0] in REAL_TYPES else kinds[0]
    if name in ("min", "max"):
        answer:Optional[str] = kinds[0]
        for kind in kinds[1:]:
            answer = answer if answer == kind else _promoted(answer, kind)
        return answer
    if name == "pow":
        promoted:str = kinds[0]
        for kind in kinds[1:]:
            promoted = _promoted(promoted, kind)
        return float_width if promoted in INT_TYPES else promoted
    if name in ("int", "float", "bool") and len(kinds) == 1:
        return {"int": "int", "float": float_width, "bool": "bool"}[name]
    return None


def _promoted(left:str, right:str)->str:
    """The type `promote` answers for two known scalars, or `int` for its fallback."""
    if left in REAL_TYPES or right in REAL_TYPES:
        return "double" if "double" in (left, right) else "float"
    rank:List[str] = ["bool", "char", "uchar", "short", "ushort", "int", "uint",
                      "long", "ulong"]
    if left in rank and right in rank:
        return left if rank.index(left) >= rank.index(right) else right
    return "int"
