# Vector `%` and `//`: python's floored rule, applied per component.
from __future__ import annotations
import ast
from typing import List, Tuple

from ..device_builtins import (
    PYTHON_FLOOR_QUALIFIERS,
    UNSIGNED_C_TYPES,
    python_floor_name,
    python_floor_source,
)
from ..device_builtins.vector_helpers import helper_name
from .notation import (
    Translated,
)
from .vectors import (
    _vec_field_names,
    _vec_registry,
    is_vec,
)
from .predicates import (
    is_int,
    promote,
)


class _VectorFloorMixin:

    def _emit_vec_floored(self, name:str, left:str, lt:object,
                          right:str, rt:object, node:ast.AST)->Translated:
        """`%` and `//` over a vector operand: python's floored rule, per component.

        python floors both operators and C truncates, so the scalar spellings route
        through `_emit_floored`, whose `__pygpgpu_*` helpers carry the one-step
        correction (see `device_builtins.floor`). The vector form calls the same
        helpers per component -- the correction is never restated here -- and the
        whole per-component expression lives in one device function, so an operand
        expression appears exactly once in the generated source (the reason the
        arithmetic family takes `_vector_device_call` instead of an expansion; a
        nested `uv % scale` would otherwise copy its operand once per component).
        A scalar operand is shared by every component, the broadcast the arithmetic
        family takes, and the vector operand decides the result type either way.

        The width the rule runs at is the element's own for two vectors, the
        promotion of the element and the scalar otherwise -- the same answer the
        scalar spellings pass to `_emit_floored`. An unsigned width needs no
        correction (its remainder is never negative), so its own operator is
        spelled per component and no helper is declared, exactly as the scalar
        path withholds one; a floating `//` has no helper on the scalar path
        either (it is the inline `floor` of the quotient) and stays one here.
        """
        if is_vec(lt) and is_vec(rt):
            if lt != rt:
                self._error(f"vector operands must have the same type, got {lt} and {rt}", node)
        vec:str = lt if is_vec(lt) else rt
        elem:str = self._vec_element_type(vec)
        other:object = rt if is_vec(lt) else lt
        width:str = elem if is_vec(other) else promote(elem, other)
        device_width:str = self._device_type(width)
        unsigned:bool = device_width in UNSIGNED_C_TYPES

        # The per-component spelling, decided once -- by the *promoted* width, the
        # same driver the scalar spellings hand to `_emit_floored`: an int vector
        # with a float scalar (`int2 % 2.5`) promotes to a float width and takes
        # the float form, exactly as `7 % 2.5` does on the scalar path.
        # ("helper", operation) calls one `__pygpgpu_*` floor helper,
        # ("operator", symbol) spells the width's own C operator, and
        # ("floor", "") is the floating `//` -- the inline floor of the quotient,
        # which is what the scalar path spells for it.
        kind:str
        spelling:str
        if name == "FloorDiv" and is_int(width):
            kind, spelling = ("operator", "/") if unsigned else ("helper", "floor_div")
        elif name == "Mod" and is_int(width):
            kind, spelling = ("operator", "%") if unsigned else ("helper", "floor_mod")
        elif name == "Mod":
            kind, spelling = "helper", "float_mod"
        else:
            kind, spelling = "floor", ""

        if kind == "helper":
            helper:str = python_floor_name(spelling, device_width)
            source:str = python_floor_source(spelling, device_width, self.backend)
            if source and source not in self.device_helpers:
                self.device_helpers.append(source)

        # The wrapper's name is the semantic operation plus the declared parameter
        # types, the way the arithmetic family names its helpers (`vec_add_float2_float2`).
        # `floor_div` names the floating form too: the parameter types make the name
        # unique, and no integer wrapper can carry float parameters.
        param_types:Tuple[str, ...] = tuple(self._device_type(t) for t in (lt, rt))
        wrapper:str = helper_name("floor_div" if name == "FloorDiv" else "floor_mod",
                                  param_types)
        letters:List[str] = _vec_field_names(_vec_registry()[vec])
        # The wrapper body reads its own parameters, never the call-site expressions:
        # the arguments travel in the call, where each appears exactly once.
        parameter_codes:Tuple[str, str] = ("a", "b")
        vector_sides:Tuple[bool, ...] = (is_vec(lt), is_vec(rt))
        terms:List[str] = []
        for letter in letters:
            first:str = (f"{parameter_codes[0]}.{letter}" if vector_sides[0]
                         else parameter_codes[0])
            second:str = (f"{parameter_codes[1]}.{letter}" if vector_sides[1]
                          else parameter_codes[1])
            if kind == "helper":
                terms.append(f"{helper}({first}, {second})")
            elif kind == "operator":
                terms.append(f"({first} {spelling} {second})")
            else:
                terms.append(f"(({width})floor(({width})({first}) / ({width})({second})))")
        params:str = ", ".join(f"{t} {p}" for t, p in zip(param_types, ("a", "b")))
        qualifier:str = PYTHON_FLOOR_QUALIFIERS.get(self.backend,
                                                    PYTHON_FLOOR_QUALIFIERS["opencl"])
        source:str = (f"{qualifier} {vec} {wrapper}({params})\n"
                      "{\n"
                      f"    return {self._vec_maker(vec, ', '.join(terms))};\n"
                      "}\n")
        if source not in self.device_helpers:
            self.device_helpers.append(source)
        return f"{wrapper}({', '.join((left, right))})", vec
