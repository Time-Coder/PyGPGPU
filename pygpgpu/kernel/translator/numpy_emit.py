from __future__ import annotations
import ast
from typing import (List, NoReturn, Optional, cast)
import numpy as np
from .. import complex as complex_support
from .type_map import (
    DTYPE_MAP,
)
from .notation import (
    Translated,
)
from .vectors import (
    is_vec,
)
from .builtins import (
    COMPLEX_ABS_FUNCTIONS,
    COMPLEX_DEVICE_FUNCTIONS,
    COMPLEX_UNSUPPORTED_FUNCTIONS,
    _MATH_FUNCS,
    _math_device_name,
)
from .predicates import (
    is_float,
    is_image,
    is_int,
    is_ptr,
    promote,
)


class _NumpyEmitMixin:

    def _emit_numpy_elementwise_nodes(self, attr:str,
                                      node:ast.Call)->Optional[Translated]:
        """``_emit_numpy_elementwise`` for the calls that read their own argument syntax.

        ``np.clip`` treats a literal ``None`` bound as "no bound on this side" and
        ``np.where`` takes a condition plus two values, so both have to look at the source
        nodes: evaluating ``None`` as a device expression fails, and evaluating every
        argument up front would lose which ones were literal ``None``.
        """
        if attr == "clip":
            if len(node.args) < 2 or len(node.args) > 3:
                self._error("np.clip() takes 2 or 3 arguments", node)
            value_code, value_t = self._visit_expression(node.args[0])
            if is_ptr(value_t) or is_image(value_t):
                self._error_numpy_array_form("clip", node)
            bounds:List[Optional[Translated]] = []
            for bound_node in node.args[1:]:
                if isinstance(bound_node, ast.Constant) and bound_node.value is None:
                    bounds.append(None)
                    continue
                bound:Translated = self._visit_expression(bound_node)
                if is_ptr(bound[1]) or is_image(bound[1]):
                    self._error("np.clip() bounds must be scalars, not arrays", bound_node)
                bounds.append(bound)
            lo:Optional[Translated] = bounds[0] if bounds else None
            hi:Optional[Translated] = bounds[1] if len(bounds) > 1 else None
            if self._complex_dtype_of(value_t) is not None:
                # numpy's clip on a complex value is its lexicographic min/max, and a `None`
                # bound means the operation on that side is simply absent. The bounds keep
                # their own types, so a real one is widened to the complex value it stands
                # for (`np.clip(z, 0, 1)`) rather than being handed to a complex function.
                value:Translated = (value_code, value_t)
                if lo is None and hi is None:
                    return value
                if hi is None:
                    return self._emit_complex_builtin("max", [value, lo], node)
                if lo is None:
                    return self._emit_complex_builtin("min", [value, hi], node)
                return self._emit_complex_builtin("clamp", [value, lo, hi], node)
            # np.clip keeps an integer input an integer, so the element type is not forced
            # to float: `np.clip(300, 0, 255)` is 255, not a float.
            result_t:object = value_t if is_int(value_t) else "float"
            return self._emit_clip(value_code, result_t,
                                   lo[0] if lo is not None else None,
                                   hi[0] if hi is not None else None), result_t
        if attr == "where":
            if len(node.args) == 1:
                self._error(
                    "np.where(cond) with one argument returns indices, which a kernel "
                    "cannot represent; use np.where(cond, a, b)", node)
            if len(node.args) != 3:
                self._error("np.where() takes exactly 3 arguments in a kernel", node)
            args:List[Translated] = [self._visit_expression(a) for a in node.args]
            for code, t in args:
                if is_ptr(t) or is_image(t):
                    self._error_numpy_array_form("where", node)
            return self._emit_where(args, node)
        return None


    def _error_numpy_array_form(self, attr:str, node:ast.AST)->NoReturn:
        self._error(
            f"np.{attr}() on an array is a host-side element-wise operation "
            f"returning a new array, which a per-work-item kernel cannot represent. "
            f"Index the element first, e.g. np.{attr}(arr[i], ...)", node)


    def _emit_numpy_elementwise(self, attr:str, args:List[Translated],
                                node:ast.Call)->Optional[Translated]:
        """numpy functions that make sense on one element, translated into device code.

        A kernel body works on single values, so the array form of these functions cannot
        be represented (it would allocate); the element form is exactly the scalar rule
        python applies to the same call:

        * ``np.clip(v, lo, hi)`` -- clamp/min-max per backend, with None bounds omitted;
        * ``np.where(cond, a, b)`` -- a ?: expression.

        Returns None when ``attr`` is not one of them, so the caller can fall through to
        the math table (which rejects the name with its own message).
        """
        if attr in ("clip", "where", "abs", "fabs", "power", "arctan2",
                    "minimum", "maximum", "square", "reciprocal", "round",
                    "angle", "real", "imag", "conj", "conjugate", "sign"):
            for code, t in args:
                if is_ptr(t) or is_image(t):
                    self._error_numpy_array_form(attr, node)
        if attr in ("abs", "fabs"):
            # numpy has two spellings; the device picks by type, as python's abs does.
            code, t = args[0]
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is not None:
                # `np.abs(complex)` and `np.fabs(complex)` are both the modulus, a real.
                kind:str = self._complex_side(t, complex_dtype)
                self._require_complex(complex_dtype)
                element:str = "double" if kind == "c128" else "float"
                return f"{kind}_abs({code})", element
            if is_float(t):
                return f"fabs({code})", t
            if is_int(t):
                return f"abs({code})", t
            self._error(f"np.{attr}() requires a numeric argument", node)
        if attr == "square":
            code, t = args[0]
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is not None:
                # A complex square is the full product, not `x * x` spelled with a C
                # operator -- the same reason the `*` operator goes through the header.
                kind:str = self._complex_side(t, complex_dtype)
                self._require_complex(complex_dtype)
                return f"{kind}_square({code})", kind
            return f"(({code}) * ({code}))", t
        if attr == "reciprocal":
            code, t = args[0]
            complex_dtype = self._complex_dtype_of(t)
            if complex_dtype is not None:
                kind = self._complex_side(t, complex_dtype)
                self._require_complex(complex_dtype)
                return f"{kind}_reciprocal({code})", kind
            return f"(1.0f / ((float)({code})))", "float"
        if attr in ("angle", "real", "imag", "conj", "conjugate", "sign"):
            # The complex-only members and functions, spelled as numpy spells them. Each one
            # is the header's own operation, and each answers either a real or a complex
            # value -- `np.real`/`np.imag`/`np.angle` are the first kind, the rest the second.
            code, t = args[0]
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is None:
                if attr == "sign" and (is_float(t) or is_int(t)):
                    # `np.sign` on a real is the ordinary sign, which the builtin answers.
                    return self._emit_shared_math_builtin("sign", args, node)
                self._error(
                    f"np.{attr}() is a complex operation; here the argument is {t}", node)
            kind:str = self._complex_side(t, complex_dtype)
            self._require_complex(complex_dtype)
            element:str = "double" if kind == "c128" else "float"
            if attr == "real":
                return f"{kind}_real({code})", element
            if attr == "imag":
                return f"{kind}_imag({code})", element
            if attr == "angle":
                return f"{kind}_angle({code})", element
            if attr == "sign":
                # numpy's complex `sign` is the direction on the unit circle, and zero at
                # zero: a header function, because the division alone is 0/0 there.
                return f"{kind}_sign({code})", kind
            return f"{kind}_conjugate({code})", kind
        if attr == "minimum" or attr == "maximum":
            if len(args) != 2:
                self._error(f"np.{attr}() takes exactly 2 arguments", node)
            if any(self._complex_dtype_of(t) is not None for _, t in args):
                # numpy gives a complex value a lexicographic order, so its minimum and
                # maximum are defined; the header's comparison is what answers them.
                return self._emit_complex_builtin(
                    "min" if attr == "minimum" else "max", args, node)
            (lcode, lt), (rcode, rt) = args
            t:object = promote(lt, rt)
            pick:str = "min" if attr == "minimum" else "max"
            if is_float(t):
                # `fmin`/`fmax` exist on both backends (`fminf` is the C spelling OpenCL
                # does not provide); the plain min/max are the integer forms.
                return f"{'fmin' if pick == 'min' else 'fmax'}(({t})({lcode}), ({t})({rcode}))", t
            return f"{pick}(({t})({lcode}), ({t})({rcode}))", t
        if attr in ("power", "arctan2"):
            # Same operation as math.pow / math.atan2 under numpy's spelling. A complex base
            # or exponent is `np.power`'s own complex power, which is the header's `pow` --
            # the unary table below could not express it, since it takes two operands.
            if attr == "power" and any(self._complex_dtype_of(t) is not None
                                       for _, t in args):
                return self._emit_complex_builtin("pow", args, node)
            return self._emit_math_call({"power": "pow", "arctan2": "atan2"}[attr],
                                        args, node, module_name="np")
        if attr == "round":
            # numpy rounds halfway cases to even, which is python's round too. A complex
            # value is rounded per part -- and the *builtin* `round(z)` is refused, because
            # python's own round() has no complex form; `np.round(z)` is the spelling that
            # answers one, which is why this branch came before the builtin's. The numpy
            # spelling emits the header's function directly rather than going through the
            # builtin table, where `rint` is a pygf name that rounds through python's
            # `round()` and is therefore refused like it.
            if any(self._complex_dtype_of(t) is not None for _, t in args):
                code, t = args[0]
                dtype:np.dtype = cast(np.dtype, self._complex_dtype_of(t))
                kind:str = self._complex_side(t, dtype)
                self._require_complex(dtype)
                return f"{kind}_rint({code})", kind
            return self._emit_builtin("round", args, node)
        if attr == "where":
            return self._emit_where(args, node)
        return None


    def _emit_where(self, args:List[Translated],
                    node:ast.Call)->Translated:
        """`np.where(cond, a, b)` as a conditional expression."""
        cond_code, cond_t = args[0]
        then_code, then_t = args[1]
        else_code, else_t = args[2]
        if not is_int(cond_t) and cond_t != "bool":
            self._error(
                f"np.where() needs a boolean or integer condition, got {cond_t}", node)
        unified:object = promote(then_t, else_t)
        return (f"(({cond_code}) ? ({unified})({then_code}) : "
                f"({unified})({else_code}))"), unified


    def _emit_clip(self, value_code:str, result_t:object,
                   lo:Optional[str], hi:Optional[str])->str:
        """`np.clip` for one element, in the form the current backend provides.

        OpenCL C has a real `clamp` builtin for both integer and floating types, so that
        is used directly: one call, and the compiler recognizes it. CUDA C has no `clamp`
        function at all (verified with NVRTC: it reports `identifier "clamp" is
        undefined`), so the equivalent nesting of `fmin`/`fmax` is emitted instead. Both
        forms give the same numbers; this only keeps each backend's generated code in the
        shape that backend natively understands.

        A missing bound is simply left out, matching `np.clip(a, None, hi)`.
        """
        if lo is None and hi is None:
            return f"({value_code})"
        # Every cast below must use this backend's own spelling: CUDA C has no `uchar`
        # (see _device_type), and writing the translator's internal name here made a
        # clipped uint8 element fail to compile.
        element:str = self._device_type(result_t) if not is_vec(result_t) else result_t
        if self._opencl_syntax:
            # clamp(v, lo, hi) needs both bounds; with one bound the other side is the
            # type's own extreme, which min/max expresses directly and without a literal.
            if lo is not None and hi is not None:
                return f"clamp(({element})({value_code}), ({element})({lo}), ({element})({hi}))"
            if lo is not None:
                return f"max(({element})({value_code}), ({element})({lo}))"
            return f"min(({element})({value_code}), ({element})({hi}))"

        # CUDA/HIP: no clamp builtin. An integer element stays on min/max, because fminf's
        # overloads are ambiguous for integer arguments (NVRTC reports "more than one
        # instance of overloaded function fmax matches the argument list"); a float
        # element uses fmin/fmax, whose NaN rule (return the other operand) is what
        # np.clip does with a NaN input.
        pick_min:str = "min" if is_int(result_t) else "fmin"
        pick_max:str = "max" if is_int(result_t) else "fmax"
        code:str = f"({value_code})"
        if lo is not None:
            code = f"{pick_max}(({element})({code}), ({element})({lo}))"
        if hi is not None:
            code = f"{pick_min}(({element})({code}), ({element})({hi}))"
        return f"({code})"


    def _clip_bound(self, bound:Translated,
                    node:Optional[ast.expr], call:ast.Call)->Optional[str]:
        """One bound of np.clip: its code, or None when the bound is None."""
        if node is None:
            return None
        code, t = bound
        if is_ptr(t) or is_image(t):
            self._error(
                "np.clip() bounds must be scalars, not arrays", call)
        return code


    def _emit_complex_math(self, attr:str, args:List[Translated],
                           node:ast.Call, module_name:str)->Optional[Translated]:
        """A math/numpy function applied to a complex value, as the header's function.

        numpy defines the whole elementary set on a complex scalar -- `sqrt`, `exp`, `log`,
        the trigonometric and hyperbolic families and their inverses, `square`, `absolute`,
        `angle`, `conjugate` -- and refuses the real-only ones (`floor`, `ceil`, `trunc`,
        `fmod`, `hypot`, `cbrt`, `ldexp`, `atan2`), which have no complex meaning at all. The
        device mirrors that exactly: a name numpy refuses is refused here, with the reason
        named, rather than computed into something python never would.

        Returns None when neither argument is complex, so the ordinary scalar path answers.
        """
        complex_dtypes:List[Optional[np.dtype]] = [self._complex_dtype_of(t) for _, t in args]
        if not any(dtype is not None for dtype in complex_dtypes):
            return None
        if attr in ("isnan", "isinf", "isfinite"):
            # numpy defines all three for a complex value: a NaN or an infinity in *either*
            # part answers the first two, and `isfinite` needs both parts finite. The header
            # holds the predicate, and the answer is the number python's bool stands for --
            # the shape the real path below answers with as well.
            code, t = args[0]
            kind:str = self._complex_side(t, complex_dtypes[0])
            self._require_complex(complex_dtypes[0])
            return f"({kind}_{attr}({code}) != 0)", "bool"
        if attr in COMPLEX_ABS_FUNCTIONS:
            # A real result: `abs`, `np.abs`, `np.angle`, `xFabs` on a complex is the
            # magnitude or the phase, not a complex value.
            code, t = args[0]
            dtype:np.dtype = (self._complex_dtype_of(t) or complex_dtypes[0]  # type: ignore[assignment]
                              or np.dtype(np.complex128))
            kind:str = self._complex_side(t, dtype)
            self._require_complex(dtype)
            device:str = COMPLEX_ABS_FUNCTIONS[attr]
            element:str = "double" if kind == "c128" else "float"
            if device == "":
                return f"fabs({code})", element
            return f"{kind}_{device}({code})", element
        if attr in COMPLEX_UNSUPPORTED_FUNCTIONS:
            self._error(
                f"{module_name}.{attr}() has no complex meaning: numpy refuses it for a"
                f" complex input, and so does this package. Its complex counterpart is"
                f" {COMPLEX_UNSUPPORTED_FUNCTIONS[attr]}", node)
        if attr not in COMPLEX_DEVICE_FUNCTIONS:
            self._error(
                f"{module_name}.{attr}() is not defined for a complex value: the complex"
                f" functions are {', '.join(sorted(COMPLEX_DEVICE_FUNCTIONS))}", node)
        code, t = args[0]
        dtype = self._complex_side(t, self._complex_dtype_of(t))
        self._require_complex(np.dtype(np.complex128 if dtype == "c128" else np.complex64))
        return f"{dtype}_{COMPLEX_DEVICE_FUNCTIONS[attr]}({code})", dtype


    def _emit_math_call(self, attr:str, args:List[Translated], node:ast.Call, module_name:str="math")->Translated:
        complex_result:Optional[Translated] = self._emit_complex_math(attr, args, node,
                                                                      module_name)
        if complex_result is not None:
            return complex_result
        if attr not in _MATH_FUNCS:
            self._error(f"{module_name}.{attr} is not supported", node)
        ret_override:Optional[str] = _MATH_FUNCS[attr]
        arg_codes:List[str] = []
        ret:object = (ret_override or "float")
        vec_arg:Optional[str] = None
        for i, (code, t) in enumerate(args):
            if is_ptr(t):
                # python semantics: math.* only accepts scalars; np.*(arr) is a host-side element-wise
                # operation returning a new array, which cannot be represented in a work-item kernel -> require indexing to an element
                arr_desc:str = (node.args[i].id if isinstance(node.args[i], ast.Name) else "the argument")
                self._error(
                    f"{module_name}.{attr}() cannot be applied to the array '{arr_desc}': "
                    f"in python, {module_name}.{attr}(arr) would be an element-wise host operation "
                    f"returning a new array, which a per-work-item kernel cannot represent. "
                    f"Index the element first, e.g. {module_name}.{attr}(arr[i]).", node)
            if is_vec(t):
                vec_arg = (vec_arg or t)
            if ret_override is None and t == "double":
                ret = "double"
        if ret == "float" and module_name == "np" \
                and all(is_int(t) for _, t in args) and args:
            # numpy computes every one of these in float64 for an *integer* input
            # (`np.sqrt(2)` is a numpy float64), and the interpreter runs numpy -- so the
            # device has to as well. Spelling `sqrtf((float)2)` answered 1.41421354 where
            # numpy answers 1.4142135623730951. `math.*` keeps its own rule (python's
            # math.sqrt(2) is exact at the call, and the answer rounds like a float).
            ret = "double"

        if vec_arg is not None:
            # python semantics: math/np functions do not accept vector structs (np.sqrt(float4_value) raises TypeError).
            # OpenCL natively supports per-component vector math; CUDA implements it by per-component expansion.
            if ret_override is not None:
                self._error(
                    f"{module_name}.{attr}() on vector type {vec_arg} is not supported "
                    f"(it returns {ret_override} for scalars; apply it per-component, e.g. "
                    f"{attr}(v.x))", node)
            if len(args) != 1:
                self._error(f"{module_name}.{attr}() on vectors takes exactly 1 vector argument", node)
            if self._opencl_syntax:
                return f"{attr}({args[0][0]})", vec_arg
            return self._vec_componentwise(_math_device_name(attr, "cuda"), args[0][0], vec_arg), vec_arg

        if attr in ("degrees", "radians"):
            # No CUDA intrinsic for either; the conversion factor is the definition.
            factor:str = ("57.29577951308232f" if attr == "degrees"
                          else "0.017453292519943295f")
            if self._opencl_syntax:
                return f"{attr}({args[0][0]})", ret
            return f"(({args[0][0]}) * {factor})", ret

        if attr in ("isnan", "isinf", "isfinite"):
            # The predicate takes a float; casting it to the int result type is what made the
            # CUDA call ambiguous. The device answers 0/1, which is the number python's bool
            # stands for. (A complex value answered this in _emit_complex_math, above.)
            return f"({attr}((float)({args[0][0]})) != 0)", "bool"

        if ret_override == "int" and attr in ("floor", "ceil", "trunc"):
            # python's floor/ceil/trunc answer an int, but the *argument* is a float. Casting
            # the argument instead (`floor((int)x)`) picked the integer overload -- CUDA
            # rejects it as ambiguous, OpenCL has no such overload -- and it computed the
            # wrong number wherever it did resolve: `ceil(2.1)` became `ceil(2)`, so 2.
            code:str = args[0][0]
            device:str = _math_device_name(attr, self.backend)
            return f"((int){device}((float)({code})))", "int"

        for (code, t) in args:
            # Everything but the exact width is cast to the call's own width, INCLUDING a
            # float32 argument beside a double one. Leaving it uncast spelled bare
            # `pow(float, double)` -- two half-width candidates, and C++ refuses the call
            # ("more than one instance of overloaded function pow matches") where python's
            # math.pow is double throughout and simply widens. The all-double spelling
            # compiles and answers python's number.
            arg_codes.append(code if t == ret else f"(({ret})({code}))")
        c_func:str = _math_device_name(attr, self.backend, ret)
        return f"{c_func}({', '.join(arg_codes)})", ret


    def _emit_numpy_cast(self, name:str, args:List[Translated],
                         node:ast.Call)->Translated:
        """np.float32(x) / np.int32(x): a plain C cast to the matching device type."""
        if len(args) != 1:
            self._error(f"np.{name}() takes exactly 1 argument", node)
        code, source_t = args[0]
        if is_ptr(source_t) or is_vec(source_t) or is_image(source_t):
            self._error(f"np.{name}() cannot convert a {source_t}", node)
        target_t:object = DTYPE_MAP.get(name)
        if target_t is None:
            self._error(f"np.{name}() has no device equivalent", node)
        # A complex cast is not a C cast: the target and the source are structures, so the
        # conversion goes through the header's own widening and narrowing functions.
        target_complex:Optional[np.dtype] = complex_support.dtype_for_type_name(target_t)
        if target_complex is not None:
            source_complex:Optional[np.dtype] = self._complex_dtype_of(source_t)
            source_kind:str = self._complex_side(source_t, source_complex)
            if source_complex is not None:
                self._require_complex(target_complex)
                if source_kind == target_t or source_complex == target_complex:
                    return code, target_t
                self.complex_needs.update(complex_support.mixed_needs())
                converter:str = (f"{source_kind}_widen_{target_t}"
                                 if target_t == "c128" else f"{source_kind}_narrow_{target_t}")
                return f"{converter}({code})", target_t
            if not is_float(source_t) and not is_int(source_t):
                self._error(
                    f"np.{name}() cannot convert {source_t} to a complex value; convert a"
                    f" real or a complex number", node)
            # A real becomes the complex value with a zero imaginary part.
            self._require_complex(target_complex)
            element:str = "double" if target_t == "c128" else "float"
            return f"{target_t}_from_real(({element})({code}))", target_t
        return f"(({self._device_type(target_t)})({code}))", target_t
