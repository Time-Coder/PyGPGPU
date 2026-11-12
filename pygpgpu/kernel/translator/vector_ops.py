from __future__ import annotations
import ast
from typing import (List, Optional, Set)
from ..device_builtins import (
    GENERIC_BUILTINS,
)
from .notation import (
    Translated,
)
from .vectors import (
    _C_FIELD_TYPES,
    _vec_components,
    _vec_component_name,
    _vec_field_names,
    _vec_registry,
    is_vec,
    kernel_vector_refusal,
)
from .predicates import (
    is_int,
    is_ptr,
)


class _VectorOpsMixin:

    def _is_scalar(self, t:object)->bool:
        return isinstance(t, str) and not is_vec(t)


    def _vec_element_type(self, vec:str)->str:
        """The C scalar type of one vector component (`float3` -> `float`)."""
        return _C_FIELD_TYPES[(vec, _vec_components(vec)[0][0])]


    def _emit_vec_abs(self, code:str, vec:str)->str:
        """`abs(v)` over a whole vector, per component.

        GLSL spells it on the vector and pygf's host answers per slot, so the device
        expands into the per-component form the +/* paths generate: the floating widths
        take `fabs`, the integer widths `abs`.
        """
        device:str = "abs" if is_int(self._vec_element_type(vec)) else "fabs"
        return self._vec_componentwise(device, code, vec)


    def _emit_vec_minmax(self, name:str, args:List[Translated], vec:str,
                         node:ast.Call)->Translated:
        """`min(v, w)` / `max(v, 0.0)` over a vector operand, per component.

        The scalar spelling's own rule decides the callee: the floating widths take the
        `fmin`/`fmax` pair, the integer widths `min`/`max` -- the same pick the scalar
        chain makes, so one name means one thing on both paths. python's builtin takes
        any arity and folds left to right, which this chain keeps.
        """
        for _, t in args:
            if is_vec(t) and t != vec:
                self._error(
                    f"{name}() vector operands must have the same type, got {vec} "
                    f"and {t}", node)
        element:str = self._vec_element_type(vec)
        pick:str = (("fmin" if name == "min" else "fmax")
                    if element in ("float", "double")
                    else ("min" if name == "min" else "max"))
        left:str = ""
        left_t:object = None
        for rcode, rt in args:
            if is_vec(rt):
                # The mismatch check above already proved every vector arg is `vec`.
                if is_vec(left_t):
                    left = self._vec_componentwise(pick, left, vec, right=rcode)
                elif left:
                    left = self._vec_componentwise(pick, rcode, vec, scalar=left)
                else:
                    left = rcode
            elif is_vec(left_t):
                left = self._vec_componentwise(pick, left, vec, scalar=rcode)
            else:
                # A leading scalar pair waits for the vector that decides the shape
                # (`min(2.0, v, w)` starts folding on the vector's terms).
                left = rcode
                continue
            left_t = vec
        return left, vec


    def _component(self, code:str, t:object, c:str)->str:
        return f"{code}.{c}" if is_vec(t) else code


    def _emit_vector_inplace(self, target_code:str, op_name:str,
                             left:str, lt:object, right:str, rt:object)->Optional[str]:
        """`v += w` on a vector as `v = v + w`, or None when that form does not apply.

        A vector has no C in-place operator: CUDA rejects `v += w` outright. The ordinary
        `+` over vectors already produces the per-component form the vector dispatch knows
        (see `_visit_binop`), so the store is written out instead.
        """
        symbol:str = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/"}.get(op_name, "")
        if not symbol or not (is_vec(lt) or is_vec(rt)):
            return None
        if is_vec(lt) and is_vec(rt):
            if lt != rt:
                return None      # the caller reports the mismatched types
            if self._opencl_syntax:
                return f"{target_code} = ({left} {symbol} {right});"
            # The right operand can be a call with side effects (`pixel_color +=
            # scene.ray_color(...)`), and the per-component form names it once per
            # component: evaluating it three times would be wrong and wasteful, so it goes
            # into a temporary first, which the caller emits to get a name back.
            return f"{target_code} = {self._vec_componentwise(symbol, left, lt, right=right)};"
        # A vector with a scalar broadcast: the vector side decides which is which.
        vector_type:object = lt if is_vec(lt) else rt
        vector_operand:str = left if is_vec(lt) else right
        scalar_operand:str = right if is_vec(lt) else left
        if self._opencl_syntax:
            return f"{target_code} = ({left} {symbol} {right});"
        expanded:str = self._vec_componentwise(symbol, vector_operand, vector_type,
                                               scalar=scalar_operand)
        return f"{target_code} = {expanded};"


    def _vec_componentwise(self, op:str, left:str, vec:str, right:str="", scalar:str="")->str:
        """CUDA has no vector arithmetic/math overloads; expand into per-component form.

        When op is an infix symbol (+ - * /) generate `l.c op r.c`; when it is a
        function name (sqrt etc.) generate `op(l.c, r.c)`.
        right is a vector operand, scalar is a broadcast scalar operand.
        """
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        infix:bool = (op in ("+", "-", "*", "/"))
        terms:List[str] = []
        for c in components:
            if scalar:
                terms.append(f"({left}.{c} {op} {scalar})" if infix else f"{op}({left}.{c}, {scalar})")
            elif right:
                terms.append(f"({left}.{c} {op} {right}.{c})" if infix else f"{op}({left}.{c}, {right}.{c})")
            else:
                terms.append(f"({op}{left}.{c})" if infix else f"{op}({left}.{c})")
        return self._vec_maker(vec, ", ".join(terms))


    def _vec_maker(self, vec:str, inner:str)->str:
        """Assemble the vector per backend: opencl (vec)(...) / cuda make_vec(...)."""
        if self._opencl_syntax:
            return f"({vec})({inner})"
        return f"make_{vec}({inner})"


    def _emit_vec_clamp(self, codes:List[str], types:List[object], vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"fminf(fmaxf({self._component(codes[0], types[0], c)}, {self._component(codes[1], types[1], c)}), {self._component(codes[2], types[2], c)})"
            for c in components)
        return f"{maker}({inner})"


    def _emit_vec_mix(self, codes:List[str], types:List[object], vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({self._component(codes[0], types[0], c)}) * (1.0f - ({self._component(codes[2], types[2], c)})) + ({self._component(codes[1], types[1], c)}) * ({self._component(codes[2], types[2], c)}))"
            for c in components)
        return f"{maker}({inner})"


    def _emit_vec_member_store(self, target:ast.Attribute, value_code:str,
                               value_t:object, node:ast.AST)->bool:
        """Emit a CUDA swizzle store (`c.xy = uv`), or False when this is not one.

        CUDA C has no swizzle lvalue, so the components are stored one by one, in the
        swizzle's own order -- exactly the writes python's spelling means. The plain
        component store is one C statement and takes the `_visit_target` path (where the
        assignment and both augassign spellings meet it); OpenCL C takes the swizzle as an
        lvalue natively, so its stores never reach here.
        """
        if self._opencl_syntax or not isinstance(target, ast.Attribute):
            return False
        base_code:str
        base_t:object
        base_code, base_t = self._visit_expression(target.value)
        if not (isinstance(base_t, str) and is_vec(base_t)):
            return False
        component:str = _vec_component_name(target.attr)
        if len(component) == 1:
            # One component: `_visit_target`'s own case handles it with the element type
            # it knows; emitting here would bypass its bool/other member care.
            return False
        valid:Set[str] = set(_vec_field_names(_vec_registry()[base_t]))
        if not set(component) <= valid:
            self._error(
                f"vector type {base_t} has no components '{target.attr}' "
                f"(available: {''.join(sorted(valid))})", node)
        if len(set(component)) != len(component):
            self._error(
                f"swizzle '{target.attr}' repeats a component: a store would assign one "
                f"component twice and python gives no order for that", node)
        if not is_vec(value_t) or len(_vec_field_names(_vec_registry()[value_t])) < len(component):
            self._error(
                f"a swizzle store needs a vector value of at least {len(component)} "
                f"component(s) (this one is a {value_t})", node)
        # The value lands in a temporary FIRST: python evaluates the right-hand side once,
        # and a swizzle whose value reads the vector it writes (`p.xy = p.yx`) corrupts
        # every store after the first without it -- the later statements would read the
        # components the earlier ones just overwrote, turning a swap into a duplication.
        index:int = getattr(self, "_swizzle_store_temps", 0) + 1
        self._swizzle_store_temps = index
        temporary:str = f"_swizzle_store_{index}"
        self._emit(f"{value_t} {temporary} = {value_code};")
        for char in component:
            self._emit(f"{base_code}.{char} = {temporary}.{char};")
        return True


    def _emit_vec_swizzle(self, code:str, vec:str, swizzle:str, node:ast.Attribute)->Translated:
        """Read a swizzle: opencl uses native v.xy; cuda generates per-component make_float2(v.x, v.y)."""
        refusal:Optional[str] = kernel_vector_refusal(vec)
        if refusal is not None:
            self._error(refusal, node)
        components:List[Translated] = _vec_components(vec)
        valid:Set[str] = {f_name for f_name, _ in components}
        chars:Set[str] = set(swizzle)
        if not chars <= valid:
            self._error(f"vector type {vec} has no components '{swizzle}' "
                        f"(available: {''.join(sorted(valid))})", node)

        # Result type = vector with the same element type and the component count (.zyx of float4 -> float3, .xy -> float2)
        result_vec:str = vec
        if len(swizzle) != len(components):
            elem_ctype:str = _C_FIELD_TYPES[(vec, components[0][0])]
            candidate:str = f"{elem_ctype}{len(swizzle)}"
            if candidate in _vec_registry():
                result_vec = candidate
            else:
                self._error(f"no vector type '{candidate}' for swizzle '.{swizzle}'", node)

        if self._opencl_syntax:
            return f"({code}.{swizzle})", result_vec
        maker:str = f"make_{result_vec}"
        inner:str = ", ".join(f"({code}.{c})" for c in swizzle)
        return f"{maker}({inner})", result_vec


    def _get_int_vec_type(self, vec:str)->str:
        """Result type of vector comparisons (same-shape integer vector, same semantics as genVec._compare_op)."""
        prefix:str = "".join(c for c in vec if not c.isdigit())
        if vec[-1].isdigit():
            return "int" + (vec[len(prefix):] if prefix else "")
        return "int" + vec


    def _vec_componentwise_compare(self, op:str, left:str, vec:str, right:str)->str:
        """Vector comparison: normalize the result to per-component 0/1 (device comparisons use -1,
        all-1 bits, for true, unlike genVec._compare_op's int(True)=1; here it is unified to 0/1)."""
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        int_vec:str = self._get_int_vec_type(vec)
        terms:List[str] = [f"(({left}.{c} {op} {right}.{c}) ? 1 : 0)" for c in components]
        inner:str = ", ".join(terms)
        if self._opencl_syntax:
            return f"({int_vec})({inner})"
        return f"make_{int_vec}({inner})"


    def _emit_vec_mad(self, codes:List[str], types:List[object], vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({self._component(codes[0], types[0], c)}) * ({self._component(codes[1], types[1], c)})) + ({self._component(codes[2], types[2], c)})"
            for c in components)
        return f"{maker}({inner})"


    def _emit_vec_select(self, a_code:str, b_code:str, c_code:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({c_code}.{c}) ? ({b_code}.{c}) : ({a_code}.{c}))" for c in components)
        return f"{maker}({inner})"


    def _emit_vec_relation(self, name:str, lcode:str, rcode:str, vec:str)->str:
        op:str = {"isequal": "==", "isnotequal": "!=", "isgreater": ">",
                  "isgreaterequal": ">=", "isless": "<", "islessequal": "<="}[name]
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        int_vec:str = self._get_int_vec_type(vec)
        maker:str = f"make_{int_vec}"
        if name == "isordered":
            inner:str = ", ".join(f"((int)((!isnan({lcode}.{c})) && (!isnan({rcode}.{c}))))" for c in components)
        elif name == "isunordered":
            inner:str = ", ".join(f"((int)(isnan({lcode}.{c}) || isnan({rcode}.{c})))" for c in components)
        else:
            inner:str = ", ".join(f"((int)(({lcode}.{c}) {op} ({rcode}.{c})))" for c in components)
        return f"{maker}({inner})"


    def _emit_scalar_relation(self, name:str, lcode:str, rcode:str)->str:
        op:str = {"isequal": "==", "isnotequal": "!=", "isgreater": ">",
                  "isgreaterequal": ">=", "isless": "<", "islessequal": "<="}[name]
        if name == "isordered":
            return f"((int)((!isnan({lcode})) && (!isnan({rcode}))))"
        if name == "isunordered":
            return f"((int)(isnan({lcode}) || isnan({rcode})))"
        return f"((int)(({lcode}) {op} ({rcode})))"


    def _emit_vec_absdiff(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(f"abs(({lcode}.{c}) - ({rcode}.{c}))" for c in components)
        return self._vec_maker(vec, inner)


    def _emit_vec_hadd(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(
            f"((({lcode}.{c}) & ({rcode}.{c})) + ((({lcode}.{c}) ^ ({rcode}.{c})) >> 1))" for c in components)
        return self._vec_maker(vec, inner)


    def _emit_vec_rhadd(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        inner:str = ", ".join(
            f"((({lcode}.{c}) | ({rcode}.{c})) - ((({lcode}.{c}) ^ ({rcode}.{c})) >> 1))" for c in components)
        return self._vec_maker(vec, inner)


    def _emit_vec_dot(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        terms:str = " + ".join(f"(({lcode}.{c}) * ({rcode}.{c}))" for c in components)
        return f"({terms})"


    def _emit_vec_builtin(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        opencl_name, _ = GENERIC_BUILTINS[name]
        expected_args:int = (1 if name in ("normalize", "length") else 2)
        if len(args) != expected_args:
            self._error(f"gpu.{name}() takes exactly {expected_args} arguments", node)

        vec_types:List[object] = [t for _, t in args if is_vec(t)]
        if not vec_types:
            self._error(f"gpu.{name}() requires vector arguments "
                        f"(python has no such functions; use fabs/abs for scalars)", node)
        vec:str = vec_types[0]
        if any(t != vec for t in vec_types):
            self._error(f"gpu.{name}() operands must have the same vector type", node)

        if self._opencl_syntax:
            return f"{opencl_name}({', '.join(c for c, _ in args)})", vec

        codes:List[str] = [c for c, _ in args]
        if name == "normalize":
            return self._emit_vec_normalize(codes[0], vec), vec
        if name == "length":
            return f"sqrtf({self._emit_vec_dot(codes[0], codes[0], vec)})", "float"
        if name == "distance":
            diff:str = self._vec_componentwise("-", codes[0], vec, right=codes[1])
            return f"sqrtf({self._emit_vec_dot(diff, diff, vec)})", "float"
        if name == "cross":
            return self._emit_vec_cross(codes[0], codes[1], vec), vec
        self._error(f"gpu.{name} is not supported", node)


    def _emit_vec_normalize(self, code:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        dot_expr:str = self._emit_vec_dot(code, code, vec)
        inner:str = ", ".join(f"(({code}.{c}) * rsqrtf({dot_expr}))" for c in components)
        return f"{maker}({inner})"


    def _emit_vec_cross(self, lcode:str, rcode:str, vec:str)->str:
        cls:object = _vec_registry()[vec]
        components:List[str] = _vec_field_names(cls)
        maker:str = f"make_{vec}"
        x, y, z = components[0], components[1], components[2]
        inner:str = (f"(({lcode}.{y}) * ({rcode}.{z}) - ({lcode}.{z}) * ({rcode}.{y})), "
                     f"(({lcode}.{z}) * ({rcode}.{x}) - ({lcode}.{x}) * ({rcode}.{z})), "
                     f"(({lcode}.{x}) * ({rcode}.{y}) - ({lcode}.{y}) * ({rcode}.{x}))")
        if len(components) == 4:
            inner += ", 0.0f"
        elif len(components) != 3:
            self._error(f"gpu.cross() requires a 3 or 4 component vector, got {vec}")
        return f"{maker}({inner})"


    def _emit_vec_constructor(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        cls:object = _vec_registry()[name]
        # A 3-component vector is padded to 16 bytes for the device ABI, so its ctypes struct
        # carries a trailing `_` field that is not a component: `float3(a, b, c)` takes
        # three arguments, not four. Count the real components, not the storage fields.
        components:List[Translated] = [f for f in cls._fields_ if f[0] != "_"]
        expected:int = len(components)
        field_type:str = _C_FIELD_TYPES[(name, components[0][0])]
        # `pygf` accepts three forms, and the device call has to mean the same thing:
        # no arguments (all zeros), one argument (broadcast), or one per component.
        if len(args) == 0:
            codes:List[str] = [f"(({field_type})(0))"] * expected
        elif len(args) == 1:
            code, t = args[0]
            if is_ptr(t) or is_vec(t):
                self._error(f"{name}() components must be scalars", node)
            codes = [code if t == field_type else f"(({field_type})({code}))"] * expected
        elif len(args) == expected:
            codes = []
            for i, (code, t) in enumerate(args):
                if is_ptr(t) or is_vec(t):
                    self._error(f"{name}() components must be scalars", node)
                component_type:str = _C_FIELD_TYPES[(name, components[i][0])]
                codes.append(code if t == component_type else f"(({component_type})({code}))")
        else:
            self._error(
                f"{name}() takes no arguments, one scalar to broadcast, or {expected} "
                f"component the vector has; {len(args)} given", node)
        if self._opencl_syntax:
            return f"({name})({', '.join(codes)})", name
        return f"make_{name}({', '.join(codes)})", name
