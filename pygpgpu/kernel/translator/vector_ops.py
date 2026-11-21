from __future__ import annotations
import ast
from typing import (List, Optional, Set, Tuple)
from ..device_builtins import (
    GENERIC_BUILTINS,
)
from ..device_builtins.vector_helpers import (ARITH_HELPERS, geometric_source, helper_call,
                                              helper_name, helper_source)
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


    def _emit_vec_abs(self, code:str, t:object, vec:str)->Translated:
        """`abs(v)` over a whole vector, per component.

        GLSL spells it on the vector and pygf's host answers per slot, so the device
        answers the same way: OpenCL and Metal negate through their native overloads
        (the component spelling there, as the +/* paths keep), while CUDA/HIP have no
        vector `abs` and take the same device-function form the arithmetic family
        does -- the body spells `fabs`/`abs` per component, the floating widths
        `fabs`, the integer widths `abs`.
        """
        device:str = "abs" if is_int(self._vec_element_type(vec)) else "fabs"
        if self._opencl_syntax:
            return self._vec_componentwise(device, code, vec), vec
        return self._vector_device_call(device, "call", device,
                                        [(code, t)], vec, vec)


    def _emit_vec_minmax(self, name:str, args:List[Translated], vec:str,
                         node:ast.Call)->Translated:
        """`min(v, w)` / `max(v, 0.0)` over a vector operand, per component.

        The scalar spelling's own rule decides the callee: the floating widths take the
        `fmin`/`fmax` pair, the integer widths `min`/`max` -- the same pick the scalar
        chain makes, so one name means one thing on both paths. python's builtin takes
        any arity and folds left to right, which this chain keeps: on CUDA/HIP every
        pair folds through the same device-function form the arithmetic family takes
        (the pair helper is emitted once and reused), and OpenCL/Metal keep their
        native per-component expansion.
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
                    left = self._fold_minmax_pair(name, pick, left, vec, rcode, rt)
                elif left:
                    # `left` is a leading scalar: the vector decides the shape, and
                    # the fold keeps the scalar on the right -- the operand order the
                    # expansion spelled (`min(2.0, v)` is `min(v, 2.0)`).
                    left = self._fold_minmax_pair(name, pick, rcode, vec, left, left_t)
                else:
                    left = rcode
                    left_t = rt
                    continue
            elif is_vec(left_t):
                left = self._fold_minmax_pair(name, pick, left, vec, rcode, rt)
            else:
                # A leading scalar pair waits for the vector that decides the shape
                # (`min(2.0, v, w)` starts folding on the vector's terms).
                left = rcode
                left_t = rt
                continue
            left_t = vec
        return left, vec


    def _fold_minmax_pair(self, name:str, pick:str, left:str, lt:object,
                          right:str, rt:object)->str:
        """One left-to-right folding step of a vector min/max.

        CUDA/HIP spell the pair through the same device-function form the arithmetic
        family takes (one helper per operand shape, emitted once and reused);
        OpenCL/Metal keep their native per-component expansion -- reached only by an
        arity above the native overloads' own (two- and three-argument calls never
        land here).
        """
        if self._opencl_syntax:
            if is_vec(rt):
                return self._vec_componentwise(pick, left, lt, right=right)
            return self._vec_componentwise(pick, left, lt, scalar=right)
        return self._vector_device_call(name, "call", pick,
                                        [(left, lt), (right, rt)], lt, lt)[0]


    def _component(self, code:str, t:object, c:str)->str:
        return f"{code}.{c}" if is_vec(t) else code


    def _geometric_device_call(self, name:str, args:List[Translated],
                               vec:str, result_t:object)->Translated:
        """Call one geometric helper (`dot`, `length`, ...), emitting its definition.

        The geometric family's answer is a scalar summed over every component, or a
        formula that mixes them (cross's `a.y * b.z - a.z * b.y`), so its body is
        composed whole (see `device_builtins.vector_helpers.geometric_source`) --
        the one shape the per-component term form cannot spell. The definition is
        built from the same formulas the inline expansion spelled and travels in
        the `device_helpers` channel beside the per-component family's.
        """
        param_types:Tuple[str, ...] = tuple(t for _, t in args)
        source:str = geometric_source(name, vec, str(result_t), param_types)
        if source not in self.device_helpers:
            self.device_helpers.append(source)
        return helper_call(helper_name(name, param_types),
                           [code for code, _ in args]), result_t


    def _vector_device_call(self, name:str, kind:str, spell:str,
                            args:List[Translated], vec:str, result_t:object)->Translated:
        """Call one `pygpgpu_*` device helper, emitting its definition on first use.

        Every vector operation the CUDA/HIP spellings have no native form of -- the
        math builtins, the clamp family, arithmetic, comparison -- goes through one
        overloaded device function instead of a per-component expansion, so an
        argument expression appears exactly once in the generated source and the
        compiler inlines the helper back to the same instructions. `vec` is the
        vector whose components the body walks (the operand's own type; a
        comparison returns the int vector of it, which `result_t` carries). The
        definition is built from the same formulas the expansion spelled (see
        `device_builtins.vector_helpers`) and travels in the `device_helpers`
        channel, the one the HIP round-toward-zero helpers use.
        """
        shape:Tuple[str, ...] = tuple("v" if is_vec(t) else "s" for _, t in args)
        param_types:Tuple[str, ...] = tuple(t for _, t in args)
        call_name:str = helper_name(name, param_types)
        source:str = helper_source(name, spell, kind, result_t, vec, shape, param_types)
        if source not in self.device_helpers:
            self.device_helpers.append(source)
        return helper_call(call_name, [code for code, _ in args]), result_t


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
            # scene.ray_color(...)`), so the operation goes through the device
            # helper: the operand expression appears exactly once, in the call.
            return (f"{target_code} = "
                    f"{self._vector_device_call(ARITH_HELPERS[symbol], 'arith', symbol, [(left, lt), (right, rt)], lt, lt)[0]};")
        # A vector with a scalar broadcast: the vector side decides which is which.
        vector_type:object = lt if is_vec(lt) else rt
        if self._opencl_syntax:
            return f"{target_code} = ({left} {symbol} {right});"
        return (f"{target_code} = "
                f"{self._vector_device_call(ARITH_HELPERS[symbol], 'arith', symbol, [(left, lt), (right, rt)], vector_type, vector_type)[0]};")


    def _vec_componentwise(self, op:str, left:str, vec:str, right:str="", scalar:str="")->str:
        """CUDA has no vector arithmetic/math overloads; expand into per-component form.

        Kept for the OpenCL/Metal spellings; the CUDA/HIP call sites go through
        `_vector_device_call` instead, so an argument expression is not duplicated
        once per component.
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
        # The store pairs the swizzle's OWN letters positionally: `p.xz = w` means
        # p.x = w[0], p.z = w[1] -- the value's second component is read through its
        # field at position 1 (`w.y`), never through the target's letters (`w.z`, which
        # a two-component value does not even have).
        value_fields:str = "".join(_vec_field_names(_vec_registry()[value_t]))
        self._emit(f"{value_t} {temporary} = {value_code};")
        for position, char in enumerate(component):
            self._emit(f"{base_code}.{char} = {temporary}.{value_fields[position]};")
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
            if name in ("length", "distance"):
                # OpenCL C's `length`/`distance` answer a *scalar* (the vector's own
                # element type), not the vector: typing the call as the vector made
                # `float3 len_result = length(result)` legal-looking C that the
                # compiler then refuses (`if` demands a scalar), and every comparison
                # against the result with it. The element type is what the geometric
                # device helpers on the CUDA path declare for the same calls.
                return (f"{opencl_name}({', '.join(c for c, _ in args)})",
                        self._vec_element_type(vec))
            return f"{opencl_name}({', '.join(c for c, _ in args)})", vec
        # The geometric family's answer is a scalar summed over every component, or
        # a formula that mixes them, so its device function is composed whole (see
        # `_geometric_device_call`); the argument expression appears exactly once,
        # the way the arithmetic family's call keeps it.
        result:object = ("float" if name in ("length", "distance") else vec)
        return self._geometric_device_call(name, args, vec, result)


    def _emit_vec_constructor(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        cls:object = _vec_registry()[name]
        # A 3-component vector is padded to 16 bytes for the device ABI, so its ctypes struct
        # carries a trailing `_` field that is not a component: `float3(a, b, c)` takes
        # three arguments, not four. Count the real components, not the storage fields.
        components:List[Translated] = [f for f in cls._fields_ if f[0] != "_"]
        expected:int = len(components)
        field_type:str = _C_FIELD_TYPES[(name, components[0][0])]
        # `pygf` accepts these forms, and the device call has to mean the same thing:
        # no arguments (all zeros), one scalar (broadcast), or components filling the
        # vector left to right -- GLSL's widening construction, where a vector argument
        # supplies its components in order from the next slot and scalars fill the rest
        # (`float3(float2(1.0, 2.0), 3.0)`, `float4(v, 1.0, 2.0)`). A single vector
        # argument alone is the copy form: pygf truncates a *wider* one to the first
        # `expected` components (`float2(float4(...))` answers `float2(x, y)`), while
        # any other over- or under-supply is a ValueError there and refused here.
        if len(args) == 0:
            codes:List[str] = [f"(({self._device_type(field_type)})(0))"] * expected
        elif len(args) == 1 and not is_vec(args[0][1]) and not is_ptr(args[0][1]):
            code, t = args[0]
            codes = [code if t == field_type
                     else f"(({self._device_type(field_type)})({code}))"] * expected
        else:
            pieces:List[Tuple[str, str]] = []
            for code, t in args:
                if is_ptr(t):
                    self._error(f"{name}() components must be scalars or vectors", node)
                if is_vec(t):
                    elem:str = _C_FIELD_TYPES[(t, _vec_components(t)[0][0])]
                    for component in _vec_field_names(_vec_registry()[t]):
                        pieces.append((f"{code}.{component}", elem))
                else:
                    pieces.append((code, t))
            if len(args) == 1 and is_vec(args[0][1]):
                if len(pieces) < expected:
                    self._error(
                        f"{name}() from one {args[0][1]} gets {len(pieces)} component(s) "
                        f"but the vector has {expected}", node)
                pieces = pieces[:expected]
            elif len(pieces) != expected:
                self._error(
                    f"{name}() takes no arguments, one scalar to broadcast, or components "
                    f"filling exactly {expected} slots: a vector argument supplies its "
                    f"components in order and scalars fill the rest; the arguments given "
                    f"supply {len(pieces)}", node)
            codes = []
            for i, (code, source_type) in enumerate(pieces):
                # The cast names the type the way the *backend* spells it: the internal
                # names follow OpenCL (`uint`, `uchar`), and CUDA C -- which HIP shares --
                # does not define those, so a `uint2(7, 9)` cast has to say
                # `unsigned int` there.
                slot_type:str = _C_FIELD_TYPES[(name, components[i][0])]
                codes.append(code if source_type == slot_type
                             else f"(({self._device_type(slot_type)})({code}))")
        if self._opencl_syntax:
            # Metal compiles this C++-style constructor call, but Apple's clang
            # miscalculates a constructor whose arguments are long parenthesised
            # expressions (a `float3(exp(...), exp(...), exp(...))` palette answered
            # one argument's value in every component on MSL 3.2). Hoisting each
            # argument into a local first keeps every initialiser trivial -- the
            # same workaround the structure constructor applies on Metal (see
            # structures.py's _metal_syntax branch).
            if self._metal_syntax:
                # Apple's clang (MSL 3.2, -fno-fast-math) miscalculates a
                # `(floatN)(expr, expr, expr)` constructor whose arguments are long
                # parenthesised expressions: every component answered the same value
                # (observed with a `float3(exp(...), exp(...), exp(...))` palette that
                # returned one channel's value in all three slots; verified natively
                # against the host). Component-wise assignment compiles correctly, so
                # each argument goes into its own hoisted float local first, and the
                # vector is built by component assignment from those locals.
                field_names:List[str] = _vec_field_names(_vec_registry()[name])
                locals_:List[str] = []
                for index, code in enumerate(codes):
                    temp:str = f"_{name}_c{index}"
                    slot_type:str = _C_FIELD_TYPES[(name, components[index][0])]
                    self._hoist_declaration(temp, slot_type, zero="0")
                    self._emit(f"{temp} = {code};")
                    locals_.append(temp)
                temp_vec:str = f"_{name}_vec"
                self._hoist_declaration(temp_vec, name)
                for index, local in enumerate(locals_):
                    self._emit(f"{temp_vec}.{field_names[index]} = {local};")
                return f"({temp_vec})", name
            return f"({name})({', '.join(codes)})", name
        return f"make_{name}({', '.join(codes)})", name
