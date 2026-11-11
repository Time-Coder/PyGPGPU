# python's own builtins inside a kernel body: len and the generic fallback.
from __future__ import annotations
import ast
from typing import (Any, List, Optional, Tuple)
import numpy as np
from ..structs import dtype_for as structs_dtype_for
from ..device_builtins import (
    CUDA_ONLY_BUILTINS,
    CUDA_ONLY_CONVERSIONS,
    GENERIC_BUILTINS,
    MATH_BUILTINS,
    SHARED_MATH_BUILTINS,
    SHARED_MATH_TEMPLATES,
)
from .notation import (
    DeviceType,
    Translated,
)
from .receivers import (
    _receiver_param_of,
)
from .vectors import (
    _vec_field_names,
    _vec_registry,
    is_vec,
)
from .builtins import (
    _math_device_name,
)
from .predicates import (
    is_float,
    is_int,
    is_ptr,
    promote,
)


class _BuiltinPythonMixin:

    def _emit_builtin(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        # Python's own builtins (`abs`, `pow`, `min`, `max`, `round`) take a complex value
        # in the interpreter for some of these and refuse it for others, and the device has
        # to agree with that rather than with a table keyed by type (see
        # _emit_complex_builtin).
        if any(self._complex_dtype_of(t) is not None for _, t in args):
            return self._emit_complex_builtin(name, args, node)

        if name == "pow":
            # python's builtin, which differs from math.pow only in that integer arguments
            # stay exact; the device form is the same floating-point power. A pair of ints
            # promotes to `int` above, and spelling `pow((int)2, (int)10)` is ambiguous in
            # C++ -- NVRTC and OpenCL's clang refuse it ("more than one instance of
            # overloaded function pow"), while HIP's <cmath> happens to resolve it, so the
            # identical source answered on one backend and refused on the others. The `**`
            # operator already promotes an integer power to float (see _visit_binop); the
            # builtin follows the same rule.
            if len(args) != 2:
                self._error("pow() takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            t:object = promote(lt, rt)
            if is_int(t):
                t = "float"
            c_func:str = _math_device_name("pow", self.backend, t)
            return f"{c_func}(({t})({lcode}), ({t})({rcode}))", t

        if name == "abs":
            if len(args) != 1:
                self._error("abs() takes exactly 1 argument", node)
            code, t = args[0]
            if is_vec(t):
                # GLSL's spelling on a whole vector, and pygf's host answers per slot:
                # the device expands component-wise like the +/* paths do.
                return self._emit_vec_abs(code, t), t
            if is_float(t):
                return f"fabs({code})", t
            if is_int(t):
                return f"abs({code})", t
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is not None:
                # numpy's `abs(complex)` is the magnitude, a *real*: the same rule the
                # module-level `np.abs`/`np.absolute` follows (see _emit_complex_math).
                kind:str = self._complex_side(t, complex_dtype)
                self._require_complex(complex_dtype)
                return f"{kind}_abs({code})", ("double" if kind == "c128" else "float")
            self._error("abs() requires a numeric argument", node)

        if name in ("min", "max"):
            if len(args) < 2:
                self._error(f"{name}() takes at least 2 arguments", node)
            vec_t:object = next((t for _, t in args if is_vec(t)), None)
            if vec_t is not None:
                # GLSL spells the component-wise comparison on whole vectors, and pygf's
                # host answers one the same way; expand like the +/* paths (the scalar
                # chain below keeps refusing a vector through promote()'s rule).
                return self._emit_vec_minmax(name, args, vec_t, node)
            # python's builtin takes any arity and folds left to right
            # (`min(a, b, c)` is `min(min(a, b), c)`), so the pairs chain. The mixed
            # complex pair reaches _emit_complex_builtin before this branch.
            pick:str = ("fmin" if name == "min" else "fmax") \
                if is_float(promote(args[0][1], args[1][1])) \
                else ("min" if name == "min" else "max")
            left, lt = args[0]
            for rcode, rt in args[1:]:
                t:object = promote(lt, rt)
                if is_float(t):
                    left = f"{pick}(({t})({left}), ({t})({rcode}))"
                else:
                    left = f"{pick}(({t})({left}), ({t})({rcode}))"
                lt = t
            return left, lt

        if name == "round":
            if len(args) not in (1, 2):
                self._error("round() takes 1 or 2 arguments", node)
            code, t = args[0]
            if len(args) == 2:
                # round(x, digits) with a constant digit count: python rounds the value,
                # not the digits it prints, so this scales, rounds and unscales.
                digits = node.args[1]
                if not (isinstance(digits, ast.Constant)
                        and isinstance(digits.value, int)):
                    self._error(
                        "round() digits must be a constant integer: the scale factor is "
                        "a literal in the generated code", node)
                scale = 10.0 ** int(digits.value)
                inner = f"((float)({code}))"
                return (f"(({t})rint({inner} * {scale}f) / {scale}f)"
                        if is_float(t) else f"({code})"), t
            if is_float(t):
                return f"((int)rint(({t})({code})))", "int"
            return f"({code})", t

        self._error(f"unsupported builtin: {name}", node)


    def _emit_len(self, args:List[Translated], node:ast.Call)->Translated:
        """``len(array)`` -> the array's extent along its only axis.

        A kernel works on one-dimensional arrays and its extents arrive as int
        parameters, so the length is already known: recovering it here keeps
        ``for i in range(len(a))`` written the way it would be in Python.
        """
        if len(args) != 1:
            self._error("len() takes exactly 1 argument", node)
        code, t = args[0]
        # A class that defines `__len__` states its own length, which the array rule below
        # knows nothing about.
        owner:Optional[Tuple[type, Any]] = self._magic_owner(t, "__len__")
        if owner is not None:
            _cls, method = owner
            return self._emit_device_call(
                "__len__", method, [(code, t)], node,
                self_param=_receiver_param_of(method), receiver_as_argument=True)
        if not is_ptr(t):
            self._error("len() takes an array argument", node)
        holder = node.args[0]
        if not isinstance(holder, ast.Name):
            # `len(bag.items)` on a member array: the argument spelling differs but the
            # length is the same compile-time shape variable, keyed on the base's name.
            if isinstance(holder, ast.Attribute) and isinstance(holder.value, ast.Name):
                base_t:DeviceType = self._lookup(holder.value.id)
                element_t:object = base_t[1] if is_ptr(base_t) else base_t
                dtype:Optional[np.dtype] = (structs_dtype_for(element_t)
                                            if isinstance(element_t, str) else None)
                if dtype is not None and holder.attr in (dtype.names or ()):
                    field_dtype:np.dtype = np.dtype(dtype.fields[holder.attr][0])
                    if field_dtype.subdtype is not None:
                        return (str(field_dtype.subdtype[1][0]), "int")
            self._error("len() takes an array parameter name", node)
        return self._shape_var(holder.id, 0), "int"


    def _emit_generic_builtin(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        arg_codes:List[str] = [c for c, _ in args]
        arg_types:List[object] = [t for _, t in args]
        vec_t:object = next((t for t in arg_types if is_vec(t)), None)

        if name in CUDA_ONLY_BUILTINS or name in CUDA_ONLY_CONVERSIONS:
            return self._emit_cuda_only_builtin(name, args, node)

        if name in SHARED_MATH_BUILTINS or name in SHARED_MATH_TEMPLATES:
            return self._emit_shared_math_builtin(name, args, node)

        if name in ("rsqrt", "exp2"):
            if len(args) != 1:
                self._error(f"gpu.{name}() takes exactly 1 argument", node)
            code, t = args[0]
            if is_vec(t):
                if self._opencl_syntax:
                    return f"{name}({code})", t
                return self._vec_componentwise(name, code, t), t
            code = f"{GENERIC_BUILTINS[name][1]}({code})" if self.backend == "cuda" else f"{name}({code})"
            return code, t

        if name in MATH_BUILTINS:
            # A scalar math intrinsic. OpenCL has a native vector overload for every one of
            # them, so the call passes straight through; CUDA has only the `f`-suffixed
            # scalar spelling, so a vector argument is expanded per component here.
            if len(args) not in (1, 2):
                self._error(f"gpu.{name}() takes 1 or 2 arguments", node)
            cuda_name:str = MATH_BUILTINS[name][1]
            result_t:object = vec_t or args[0][1]
            if self._opencl_syntax:
                return f"{name}({', '.join(arg_codes)})", result_t
            if vec_t is not None:
                if len(args) == 1:
                    return self._vec_componentwise(cuda_name, arg_codes[0], vec_t), vec_t
                if is_vec(arg_types[0]):
                    if is_vec(arg_types[1]):
                        return self._vec_componentwise(cuda_name, arg_codes[0], vec_t,
                                                       right=arg_codes[1]), vec_t
                    return self._vec_componentwise(cuda_name, arg_codes[0], vec_t,
                                                   scalar=arg_codes[1]), vec_t
                # The vector is the second operand (`pow(2.0, v)`): expand on that side, and
                # keep the operand order, which a non-commutative pair depends on.
                components:List[str] = _vec_field_names(_vec_registry()[vec_t])
                terms:str = ", ".join(
                    f"{cuda_name}({arg_codes[0]}, {arg_codes[1]}.{c})" for c in components)
                return self._vec_maker(vec_t, terms), vec_t
            return f"{cuda_name}({', '.join(arg_codes)})", result_t

        if name == "clamp":
            if len(args) != 3:
                self._error("gpu.clamp(a, lo, hi) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self._opencl_syntax:
                    return f"clamp({', '.join(arg_codes)})", vec_t
                return self._emit_vec_clamp(arg_codes, arg_types, vec_t), vec_t
            if self._opencl_syntax:
                return f"clamp({', '.join(arg_codes)})", args[0][1]
            # The CUDA spelling is the fmin/fmax pair, and the pair follows the promoted
            # width: a double operand beside a float one left the call spelled fminf with
            # a double inside, and the value that came out fed a `pow(float, double)` --
            # two half-width candidates no overload matches. Casting to the width and
            # spelling the pair for it keeps one answer per width.
            width:str = "double" if any(t == "double" for _, t in args) else "float"
            minimum:str = "fmin" if width == "double" else "fminf"
            maximum:str = "fmax" if width == "double" else "fmaxf"
            cast:List[str] = [f"(({width})({c}))" for c in arg_codes]
            return f"{minimum}({maximum}({cast[0]}, {cast[1]}), {cast[2]})", width

        if name == "mix":
            if len(args) != 3:
                self._error("gpu.mix(a, b, t) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self._opencl_syntax:
                    return f"mix({', '.join(arg_codes)})", vec_t
                return self._emit_vec_mix(arg_codes, arg_types, vec_t), vec_t
            a, b, t = arg_codes
            return f"(({a}) * (1.0f - ({t})) + ({b}) * ({t}))", args[0][1]

        if name == "mad":
            if len(args) != 3:
                self._error("gpu.mad(a, b, c) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self._opencl_syntax:
                    return f"mad({', '.join(arg_codes)})", vec_t
                return self._emit_vec_mad(arg_codes, arg_types, vec_t), vec_t
            a, b, c = arg_codes
            return f"((({a}) * ({b})) + ({c}))", args[0][1]

        if name == "saturate":
            if len(args) != 1:
                self._error("gpu.saturate(x) takes exactly 1 argument", node)
            code, t = args[0]
            if vec_t is not None:
                if self._opencl_syntax:
                    return f"clamp({code}, 0.0f, 1.0f)", vec_t
                return self._emit_vec_clamp([code], [t], vec_t), vec_t
            if self._opencl_syntax:
                return f"clamp({code}, 0.0f, 1.0f)", t
            # The pair follows the argument's width, the same rule the clamp branch
            # spells out: a double operand under the float spell fed the same
            # half-width mismatch a mixed `pow` call did.
            width:str = "double" if t == "double" else "float"
            minimum:str = "fmin" if width == "double" else "fminf"
            maximum:str = "fmax" if width == "double" else "fmaxf"
            zero:str = self._real_literal(0.0, width)
            one:str = self._real_literal(1.0, width)
            return f"{minimum}({maximum}(({width})({code}), {zero}), {one})", width

        if name == "step":
            if len(args) != 2:
                self._error("gpu.step(edge, x) takes exactly 2 arguments", node)
            (edge_code, edge_t), (x_code, x_t) = args
            if vec_t is not None and self.backend == "opencl":
                return f"step({edge_code}, {x_code})", vec_t
            if vec_t is not None:
                cls:object = _vec_registry()[vec_t]
                components:List[str] = _vec_field_names(cls)
                maker:str = f"make_{vec_t}"
                inner:str = ", ".join(
                    f"(({self._component(x_code, x_t, c)}) < ({self._component(edge_code, edge_t, c)})) ? 0.0f : 1.0f"
                    for c in components)
                return f"{maker}({inner})", vec_t
            return f"((({x_code}) < ({edge_code})) ? 0.0f : 1.0f)", "float"

        if name == "smoothstep":
            if len(args) != 3:
                self._error("gpu.smoothstep(edge0, edge1, x) takes exactly 3 arguments", node)
            (e0_code, e0_t), (e1_code, e1_t), (x_code, x_t) = args
            vec_t:object = next((t for t in (e0_t, e1_t, x_t) if is_vec(t)), None)
            for t in (e0_t, e1_t, x_t):
                if is_vec(t) and t != vec_t:
                    self._error(
                        f"gpu.smoothstep() vector operands must have the same type, "
                        f"got {vec_t} and {t}", node)
            if vec_t is not None:
                if self._opencl_syntax:
                    # The native name has a vector overload, but every argument must be
                    # that one vector type, so a scalar operand is broadcast by the cast.
                    def broadcast(arg_code:str, arg_t:object)->str:
                        return (arg_code if is_vec(arg_t) else f"({vec_t})({arg_code})")
                    return (f"smoothstep({broadcast(e0_code, e0_t)}, "
                            f"{broadcast(e1_code, e1_t)}, {broadcast(x_code, x_t)})"), vec_t
                element:str = self._vec_element_type(vec_t)
                terms:List[str] = []
                for component in _vec_field_names(_vec_registry()[vec_t]):
                    terms.append(self._smoothstep_expression(
                        self._component(e0_code, e0_t, component),
                        self._component(e1_code, e1_t, component),
                        self._component(x_code, x_t, component),
                        element))
                return self._vec_maker(vec_t, ", ".join(terms)), vec_t
            width:str = "double" if any(t == "double" for _, t in args) else "float"
            if self._opencl_syntax:
                # The native call takes one width throughout: mixed operands would leave
                # the compiler no gentype to resolve (OpenCL converts float to double
                # only inside arithmetic, not inside a call).
                return (f"smoothstep(({width})({e0_code}), ({width})({e1_code}), "
                        f"({width})({x_code}))"), width
            return self._smoothstep_expression(e0_code, e1_code, x_code, width), width

        if name == "dot":
            if len(args) != 2:
                self._error("gpu.dot(a, b) takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            if not (is_vec(lt) and is_vec(rt)):
                # GLSL and OpenCL define dot for vectors only, and pygf -- the host
                # implementation of this same function -- refuses a scalar pair. Reducing it
                # to a multiply here would make the two disagree about what `dot` means, so
                # the scalar spelling is refused with the expression it stands for.
                self._error(
                    "dot() takes two vectors; for scalars write `a * b`. The device and the "
                    "host interpreter share one implementation of the builtins, so a scalar "
                    "form that only one side accepted would make a kernel mean different "
                    "things per backend", node)
            if lt != rt:
                self._error(f"gpu.dot() operands must have the same type, got {lt} and {rt}", node)
            if self._opencl_syntax:
                return f"dot({lcode}, {rcode})", "float"
            return self._emit_vec_dot(lcode, rcode, lt), "float"

        if name in ("cross", "normalize", "length", "distance"):
            return self._emit_vec_builtin(name, args, node)

        if name == "select":
            if len(args) != 3:
                self._error("gpu.select(a, b, cond) takes exactly 3 arguments", node)
            (a_code, a_t), (b_code, b_t), (c_code, c_t) = args
            if a_t != b_t:
                self._error(f"gpu.select() operands must have the same type, got {a_t} and {b_t}", node)
            if is_vec(a_t):
                if not is_vec(c_t):
                    self._error("gpu.select() on vectors requires a vector condition", node)
                # opencl's native select treats int conditions implementation-defined (NVIDIA uses the MSB):
                # (cond == 0) yields a 0/-1 mask (MSB unambiguous), select(b, a, mask) means
                # cond nonzero -> b, cond == 0 -> a
                if self._opencl_syntax:
                    int_vec:str = self._get_int_vec_type(a_t)
                    return f"select({b_code}, {a_code}, ({c_code} == ({int_vec})(0)))", a_t
                return self._emit_vec_select(a_code, b_code, c_code, a_t), a_t
            if self._opencl_syntax:
                return f"select({b_code}, {a_code}, ({c_code} == 0))", a_t
            return f"(({c_code}) ? ({b_code}) : ({a_code}))", a_t

        if name in ("isequal", "isnotequal", "isgreater", "isgreaterequal", "isless", "islessequal", "isordered", "isunordered"):
            if len(args) != 2:
                self._error(f"gpu.{name}(a, b) takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            if is_vec(lt) or is_vec(rt):
                if not (is_vec(lt) and is_vec(rt)) or lt != rt:
                    self._error(f"gpu.{name}() vector operands must have the same type", node)
                if self._opencl_syntax:
                    # opencl vector relation functions use -1 (all-1 bits) for true; & 1 normalizes it to 1
                    return f"({name}({lcode}, {rcode}) & 1)", self._get_int_vec_type(lt)
                return self._emit_vec_relation(name, lcode, rcode, lt), self._get_int_vec_type(lt)
            if self._opencl_syntax:
                return f"{name}({lcode}, {rcode})", "int"
            return self._emit_scalar_relation(name, lcode, rcode), "int"

        if name in ("any", "all"):
            if len(args) != 1 or not is_vec(args[0][1]):
                self._error(f"gpu.{name}(cond_vec) takes exactly 1 vector argument", node)
            code, t = args[0]
            if self._opencl_syntax:
                return f"{name}({code})", "int"
            cls:object = _vec_registry()[t]
            components:List[str] = _vec_field_names(cls)
            joiner:str = ("|" if name == "any" else "&")
            expr:str = f"({joiner.join(f'({code}.{c})' for c in components)})"
            return f"(({expr}) != 0)", "int"

        if name in ("abs_diff", "hadd", "rhadd", "mad24", "mul24"):
            expected:int = (3 if name == "mad24" else 2)
            if len(args) != expected:
                self._error(f"gpu.{name}() invalid argument count", node)
            if any(is_vec(t) for _, t in args):
                if not all(is_vec(t) for _, t in args) or len({t for _, t in args}) != 1:
                    self._error(f"gpu.{name}() vector operands must all be the same vector type", node)
                # generate the integer function family uniformly per-component: NVIDIA's opencl implementation
                # unsigned-izes the return type (abs_diff/hadd/rhadd/mul24/mad24), so calling it directly on int
                # vectors fails to compile or drifts the result type; per-component expansion matches the cuda path and host semantics exactly
                if name == "abs_diff":
                    return self._emit_vec_absdiff(arg_codes[0], arg_codes[1], vec_t), vec_t
                if name == "hadd":
                    return self._emit_vec_hadd(arg_codes[0], arg_codes[1], vec_t), vec_t
                if name == "rhadd":
                    return self._emit_vec_rhadd(arg_codes[0], arg_codes[1], vec_t), vec_t
                if name == "mul24":
                    return self._vec_componentwise("*", arg_codes[0], vec_t, right=arg_codes[1]), vec_t
                return self._emit_vec_mad(arg_codes, arg_types, vec_t), vec_t
            if not all(is_int(t) for _, t in args):
                self._error(f"gpu.{name}() requires integer arguments", node)
            codes:List[str] = arg_codes
            if self._opencl_syntax:
                return f"{name}({', '.join(codes)})", args[0][1]
            if name == "abs_diff":
                return f"abs(({codes[0]}) - ({codes[1]}))", args[0][1]
            if name == "hadd":
                a, b = codes
                return f"((({a}) & ({b})) + ((({a}) ^ ({b})) >> 1))", args[0][1]
            if name == "rhadd":
                a, b = codes
                return f"((({a}) | ({b})) - ((({a}) ^ ({b})) >> 1))", args[0][1]
            if name == "mul24":
                return f"(({codes[0]}) * ({codes[1]}))", args[0][1]
            a, b, c = codes
            return f"((({a}) * ({b})) + ({c}))", args[0][1]

        self._error(f"unknown gpu builtin: gpu.{name}", node)


    def _smoothstep_expression(self, e0:str, e1:str, x:str, width:str)->str:
        """The clamp-and-fade as one expression, in the width the call promotes to.

        CUDA and HIP have no smoothstep intrinsic to call (their headers declare none),
        so the device reads the formula the GLSL name stands for -- the same shape the
        CUDA path already spells for `step` and `clamp`. The inner value is clamped by
        the pair the width picks (`fminf`/`fmaxf` for floats, `fmin`/`fmax` for
        doubles), and the constants take the width so the expression stays in it.
        """
        minimum:str = "fmin" if width == "double" else "fminf"
        maximum:str = "fmax" if width == "double" else "fmaxf"
        zero:str = self._real_literal(0.0, width)
        one:str = self._real_literal(1.0, width)
        t:str = (f"{minimum}({maximum}((({x}) - ({e0})) / (({e1}) - ({e0})), {zero}),"
                 f" {one})")
        three:str = self._real_literal(3.0, width)
        two:str = self._real_literal(2.0, width)
        return f"(({t}) * ({t}) * ({three} - ({two}) * ({t})))"
