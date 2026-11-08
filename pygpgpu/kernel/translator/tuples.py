from __future__ import annotations
import ast
import inspect
import textwrap
import types
from typing import (Dict, List, Optional, Set, Tuple)
import numpy as np
from .type_map import (
    DTYPE_MAP,
)
from .notation import (
    DeviceType,
    device_name,
)
from .predicates import (
    is_float,
    is_ptr,
)


class _TuplesMixin:

    def _tuple_call_element_count(self, call:ast.Call)->Optional[int]:
        """How many values a call answers, when it answers a tuple -- or None.

        The answer is known only for a *helper* call whose body returns a tuple; the
        interpreter backend runs the same call and python unpacks the pair there, so this
        is the device side of one spelling. The count comes from the function's return
        statements -- `return a, b` has a fixed arity, whatever the types turn out to be
        at the call.
        """
        count:Optional[int] = self._tuple_call_return_info(call)[0]
        if count is not None:
            # The element types, read from the returns as well: each one is the device
            # type the callee's own translation produces (a np.float64 cast is a double,
            # a python float result is the mapping's width). They decide the
            # destinations' declarations, so a plain name unpacks like an assignment.
            types:List[object] = self._tuple_call_element_types(call)
            if types is not None:
                self._pending_tuple_types = types
        return count


    def _tuple_call_return_info(self, call:ast.Call)->Tuple[Optional[int],
                                                             List[object]]:
        """A tuple-returning helper's arity and element types (empty when unknown).

        The arity comes from the function's return statements. The element types are the
        ones a *first pass* of the call would produce: the cheapest reliable source is a
        dry translation of the helper at this call's argument types, which is exactly
        what `_emit_device_call` does -- so the caller unpacks after asking here, and the
        types come back through the cache entry.
        """
        if not isinstance(call.func, ast.Name):
            return None, []
        target:object = self.resolve_global(call.func.id)
        if not isinstance(target, types.FunctionType):
            return None, []
        counts:Set[int] = set()
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
        except (OSError, TypeError):
            return None, []
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
                counts.add(len(sub.value.elts))
            elif isinstance(sub, ast.Return) and sub.value is not None:
                return None, []
        if len(counts) != 1:
            return None, []
        return counts.pop(), []


    def _tuple_call_element_types(self, call:ast.Call)->Optional[List[object]]:
        """The device type of each element a tuple-returning helper answers, or None.

        Read from the helper's own returns: a `np.<dtype>(...)` cast names its width, a
        python float result names the mapping's width, a python int result is an int, and
        arithmetic over its own parameters (`return n, n + 2`) takes the type of the
        parameter it starts from -- the same rule the kernel's own expressions follow. That
        last case is what makes an ordinary helper usable: `lo, hi = bounds(i)` is how
        python spells "give me two numbers", and a helper written that way has no cast to
        read. Anything whose type is not statically known is refused rather than guessed.
        """
        if not isinstance(call.func, ast.Name):
            return None
        target:object = self.resolve_global(call.func.id)
        if not isinstance(target, types.FunctionType):
            return None
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
        except (OSError, TypeError):
            return None
        python_float_kind:str = self.type_map.get("float", "float")
        # The helper's own parameters, at the types its call site passes: `n` in
        # `def bounds(n)` is whatever the caller handed over, which is what the arithmetic
        # below has to start from.
        parameter_types:Dict[str, object] = {}
        try:
            names:List[str] = list(target.__code__.co_varnames[
                :target.__code__.co_argcount])
        except AttributeError:
            names = []
        for position, name in enumerate(names):
            if position < len(call.args):
                parameter_types[name] = self._visit_expression(call.args[position])[1]
        sites:List[List[object]] = []
        for sub in ast.walk(tree):
            if not (isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple)):
                continue
            element_types:List[object] = []
            for element in sub.value.elts:
                inferred:Optional[object] = self._tuple_element_type(
                    element, python_float_kind, parameter_types)
                if inferred is None:
                    return None
                element_types.append(inferred)
            sites.append(element_types)
        if len({tuple(site) for site in sites}) != 1 or not sites:
            return None
        return sites[0]


    def _tuple_element_type(self, element:ast.expr, python_float_kind:str,
                            parameter_types:Dict[str, object])->Optional[object]:
        """One element of a helper's tuple return, at its device type, or None.

        The shapes an ordinary two-value helper uses: a cast, a float/int constant, one of
        the helper's own parameters, or arithmetic over those. Everything else -- an
        argument's member, another call -- is not statically known here.
        """
        if isinstance(element, ast.Call) \
                and isinstance(element.func, ast.Attribute) \
                and isinstance(element.func.value, ast.Name) \
                and self.resolve_global(element.func.value.id) is np:
            return DTYPE_MAP.get(element.func.attr)
        if isinstance(element, ast.Constant):
            if isinstance(element.value, bool):
                return "bool"
            if isinstance(element.value, float):
                return python_float_kind
            if isinstance(element.value, int):
                return "int"
            return None
        if isinstance(element, ast.Name):
            return parameter_types.get(element.id)
        if isinstance(element, ast.UnaryOp):
            return self._tuple_element_type(element.operand, python_float_kind,
                                            parameter_types)
        if isinstance(element, ast.BinOp) and isinstance(
                element.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            left:Optional[object] = self._tuple_element_type(
                element.left, python_float_kind, parameter_types)
            right:Optional[object] = self._tuple_element_type(
                element.right, python_float_kind, parameter_types)
            if isinstance(element.op, ast.Div):
                # True division of two ints answers a float in python, so the result is the
                # mapping's float width rather than an int.
                return python_float_kind if left is not None and right is not None else None
            if left is None or right is None:
                return None
            if is_float(left) or is_float(right):
                return "double" if "double" in (left, right) else python_float_kind
            return "int"
        return None


    def _emit_tuple_call_unpack(self, targets:List[ast.expr], call:ast.Call)->None:
        """`x, y = pair(...)`: translate the call, passing the destinations as outputs.

        The helper's signature ends in one `T*` output per element (see `_visit_return`'s
        tuple branch and `render_device`), so the destinations travel by address and the
        helper's stores land in them -- the device shape of python's unpacking. A plain
        name is declared at its element type first, so the call site reads like the
        assignment it is; the types come from the helper's own returns
        (`_tuple_call_element_types`).
        """
        element_types:List[object] = (self._pending_tuple_types
                                      or self._tuple_call_element_types(call) or [])
        # The destinations are evaluated first, exactly as a tuple swap evaluates every
        # right-hand side before storing: an element destination captures its own address.
        address_codes:List[str] = []
        for ordinal, target_item in enumerate(targets):
            if isinstance(target_item, ast.Subscript):
                base_code, _base_t = self._visit_array_target(target_item.value)
                index_code, _ = self._visit_subscript_index(target_item)
                if isinstance(target_item.value, ast.Name):
                    self.atomic_targets.add(target_item.value.id)
                address_codes.append(f"&({base_code}[{index_code}])")
                continue
            if not isinstance(target_item, ast.Name):
                self._error(
                    "a tuple-returning call unpacks into plain variables or array "
                    "elements", target_item)
            name:str = target_item.id
            existing:DeviceType = self._lookup(name)
            if existing is not None and is_ptr(existing):
                self._error(f"cannot assign to array parameter '{name}'", target_item)
            if existing is None:
                if ordinal >= len(element_types):
                    self._error(
                        "cannot tell the type of the value this helper answers for "
                        f"'{name}': its returns are not all `np.<dtype>(...)` casts, "
                        f"float results, or the same shape -- declare the variable "
                        f"yourself (`x: np.float64 = 0.0`) and unpack into it", target_item)
                self._define(name, element_types[ordinal], target_item)
                self._emit(f"{self._device_type(element_types[ordinal])} {name};")
            address_codes.append(f"&({device_name(name)})")
        self._pending_tuple_types = None
        call_code, call_t = self._emit_device_call(
            call.func.id, self.resolve_global(call.func.id),
            [self._visit_expression(a) for a in call.args], call,
            tuple_targets=address_codes)
        if call_t != "void":
            self._error("the call does not answer a tuple; unpack a tuple-returning "
                        "helper", call)
        # The call is a statement here: the helper's own body stores through the
        # destinations, and the (void) answer carries nothing to assign.
        self._emit(f"{call_code};")
